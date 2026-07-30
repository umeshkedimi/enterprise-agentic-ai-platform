"""Registering and reading MCP servers.

The interesting decision here is what this module does *not* return. An MCP
server's credential goes in one direction only — out to that server, at call
time — so `auth_token` is accepted on write, encrypted, and never read back by
anything except the code that dials the connection.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.crypto import DecryptionError, EncryptionUnavailableError, decrypt, encrypt
from app.core.logging import get_logger
from app.core.net import UnsafeUrlError, validate_outbound_url
from app.models.mcp import McpServer
from app.services.errors import (
    CredentialStorageUnavailableError,
    NotFoundError,
    SlugAlreadyExistsError,
    UnsafeServerUrlError,
)

logger = get_logger(__name__)

_UPDATABLE_FIELDS = frozenset(
    {"name", "description", "url", "headers", "timeout_seconds", "enabled"}
)


def _validate_url(url: str, settings: Settings) -> None:
    try:
        validate_outbound_url(url, settings)
    except UnsafeUrlError as exc:
        raise UnsafeServerUrlError(str(exc)) from exc


def _encrypt_token(token: str, settings: Settings) -> str:
    try:
        return encrypt(token, settings)
    except EncryptionUnavailableError as exc:
        # The tenant's request is fine; the platform is not configured to hold a
        # secret safely. Refusing beats storing it in the clear.
        raise CredentialStorageUnavailableError(str(exc)) from exc


async def create_server(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    slug: str,
    name: str,
    url: str,
    description: str | None = None,
    auth_token: str | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: int,
    enabled: bool = True,
    settings: Settings | None = None,
) -> McpServer:
    settings = settings or get_settings()

    existing = await session.scalar(
        select(McpServer).where(McpServer.tenant_id == tenant_id, McpServer.slug == slug)
    )
    if existing is not None:
        raise SlugAlreadyExistsError(slug)

    _validate_url(url, settings)

    server = McpServer(
        tenant_id=tenant_id,
        slug=slug,
        name=name,
        description=description,
        url=url,
        auth_token_encrypted=_encrypt_token(auth_token, settings) if auth_token else None,
        headers=headers or {},
        timeout_seconds=timeout_seconds,
        enabled=enabled,
    )
    session.add(server)
    await session.commit()
    await session.refresh(server)
    logger.info(
        "mcp_server_registered",
        tenant_id=str(tenant_id),
        mcp_server_id=str(server.id),
        slug=slug,
        authenticated=server.auth_token_encrypted is not None,
    )
    return server


async def list_servers(session: AsyncSession, *, tenant_id: uuid.UUID) -> list[McpServer]:
    result = await session.scalars(
        select(McpServer)
        .where(McpServer.tenant_id == tenant_id)
        .order_by(McpServer.created_at.desc())
    )
    return list(result.all())


async def list_enabled_servers(
    session: AsyncSession, *, tenant_id: uuid.UUID
) -> list[McpServer]:
    """The servers a turn may actually reach.

    Separate from `list_servers` because a disabled server must stay visible to
    its owner in the API and invisible to the runtime — the same split `enabled`
    has on an agent.
    """
    result = await session.scalars(
        select(McpServer)
        .where(McpServer.tenant_id == tenant_id, McpServer.enabled.is_(True))
        .order_by(McpServer.slug)
    )
    return list(result.all())


async def get_server(
    session: AsyncSession, *, tenant_id: uuid.UUID, server_id: uuid.UUID
) -> McpServer:
    server = await session.get(McpServer, server_id)
    if server is None or server.tenant_id != tenant_id:
        raise NotFoundError(str(server_id))
    return server


async def update_server(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    server_id: uuid.UUID,
    changes: dict[str, Any],
    settings: Settings | None = None,
) -> McpServer:
    settings = settings or get_settings()
    server = await get_server(session, tenant_id=tenant_id, server_id=server_id)

    if changes.get("url"):
        _validate_url(changes["url"], settings)

    # Rotating the credential is an update like any other; clearing it is done by
    # sending an explicit empty string, so a partial update that omits the field
    # leaves the stored token alone rather than silently dropping it.
    if "auth_token" in changes:
        token = changes["auth_token"]
        server.auth_token_encrypted = _encrypt_token(token, settings) if token else None

    for field, value in changes.items():
        if field in _UPDATABLE_FIELDS:
            setattr(server, field, value)

    await session.commit()
    await session.refresh(server)
    logger.info("mcp_server_updated", tenant_id=str(tenant_id), mcp_server_id=str(server_id))
    return server


async def delete_server(
    session: AsyncSession, *, tenant_id: uuid.UUID, server_id: uuid.UUID
) -> None:
    server = await get_server(session, tenant_id=tenant_id, server_id=server_id)
    await session.delete(server)
    await session.commit()
    # Agents keep the deleted server's tool names in their allowlists. That is
    # the same stale-name case a retired built-in tool produces, and resolution
    # already skips an unresolvable name with a log rather than failing the turn.
    logger.info("mcp_server_deleted", tenant_id=str(tenant_id), mcp_server_id=str(server_id))


def auth_token_for(server: McpServer, settings: Settings) -> str | None:
    """Decrypt a server's credential for the duration of one connection.

    A failure to decrypt returns None rather than raising: the usual cause is a
    rotated encryption key, and the useful outcome is a server that fails to
    authenticate and is reported as unreachable — not a 500 on a chat request
    that happens to involve it.
    """
    if not server.auth_token_encrypted:
        return None
    try:
        return decrypt(server.auth_token_encrypted, settings)
    except (DecryptionError, EncryptionUnavailableError):
        logger.warning(
            "mcp_credential_unreadable",
            mcp_server_id=str(server.id),
            tenant_id=str(server.tenant_id),
        )
        return None
