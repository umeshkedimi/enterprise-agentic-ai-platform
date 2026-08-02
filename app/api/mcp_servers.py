import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import PageParams, get_current_tenant, page_params
from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.mcp import McpServer
from app.models.schemas import (
    McpDiscoveryResponse,
    McpServerCreate,
    McpServerResponse,
    McpServerUpdate,
    McpToolResponse,
    Page,
)
from app.models.tenant import Tenant
from app.services import mcp_service
from app.services.errors import (
    CredentialStorageUnavailableError,
    NotFoundError,
    SlugAlreadyExistsError,
    UnsafeServerUrlError,
)
from app.tools import mcp_tools

router = APIRouter(prefix="/mcp-servers", tags=["mcp"])

_NOT_FOUND = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found.")


def _to_response(server: McpServer) -> McpServerResponse:
    return McpServerResponse(
        id=server.id,
        slug=server.slug,
        name=server.name,
        description=server.description,
        url=server.url,
        # The fact of a credential, not the credential.
        authenticated=server.auth_token_encrypted is not None,
        headers=server.headers,
        timeout_seconds=server.timeout_seconds,
        enabled=server.enabled,
        created_at=server.created_at,
        updated_at=server.updated_at,
    )


def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SlugAlreadyExistsError):
        return HTTPException(
            status.HTTP_409_CONFLICT, "An MCP server with that slug already exists."
        )
    if isinstance(exc, UnsafeServerUrlError):
        # The tenant's problem, and the message says why — a blocked URL that
        # reported only "invalid" would be indistinguishable from a typo.
        return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    if isinstance(exc, CredentialStorageUnavailableError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The platform is not configured to store third-party credentials.",
        )
    return HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid MCP server configuration.")


@router.post("", response_model=McpServerResponse, status_code=status.HTTP_201_CREATED)
async def create_mcp_server(
    body: McpServerCreate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> McpServerResponse:
    """Register a server. Its tools become grantable to this tenant's agents.

    This POST is the whole integration: no code in this repository knows what the
    server's tools do, and nothing is deployed to add them.
    """
    try:
        server = await mcp_service.create_server(
            session,
            tenant_id=tenant.id,
            slug=body.slug,
            name=body.name,
            description=body.description,
            url=body.url,
            auth_token=body.auth_token,
            headers=body.headers,
            timeout_seconds=body.timeout_seconds,
            enabled=body.enabled,
        )
    except (SlugAlreadyExistsError, UnsafeServerUrlError, CredentialStorageUnavailableError) as exc:
        raise _to_http_error(exc) from exc
    return _to_response(server)


@router.get("", response_model=Page[McpServerResponse])
async def list_mcp_servers(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[McpServerResponse]:
    servers, has_more = await mcp_service.list_servers(
        session, tenant_id=tenant.id, limit=page.limit, offset=page.offset
    )
    return Page(
        items=[_to_response(s) for s in servers],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


@router.get("/{server_id}", response_model=McpServerResponse)
async def get_mcp_server(
    server_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> McpServerResponse:
    try:
        server = await mcp_service.get_server(
            session, tenant_id=tenant.id, server_id=server_id
        )
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
    return _to_response(server)


@router.get("/{server_id}/tools", response_model=McpDiscoveryResponse)
async def discover_mcp_tools(
    server_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> McpDiscoveryResponse:
    """Ask the server what it offers, under the names an allowlist has to use.

    A team owner cannot grant a tool they cannot name, and the namespaced form is
    not guessable from the server's own documentation. Unreachable is reported as
    a reachable=false body rather than a 502: "your server is down" is an answer
    to this question, not a failure to answer it.
    """
    try:
        server = await mcp_service.get_server(
            session, tenant_id=tenant.id, server_id=server_id
        )
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    # Bypasses the cache: this endpoint is what a team owner refreshes after
    # changing something on their own server, and a stale answer would send them
    # looking for a bug in the platform.
    discovery = await mcp_tools.discover(server, settings=get_settings(), use_cache=False)
    return McpDiscoveryResponse(
        server_id=server.id,
        slug=server.slug,
        reachable=discovery.error is None,
        tools=[McpToolResponse(name=t.name, description=t.description) for t in discovery.tools],
        error=discovery.error,
    )


@router.patch("/{server_id}", response_model=McpServerResponse)
async def update_mcp_server(
    server_id: uuid.UUID,
    body: McpServerUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> McpServerResponse:
    changes = body.model_dump(exclude_unset=True)
    try:
        server = await mcp_service.update_server(
            session, tenant_id=tenant.id, server_id=server_id, changes=changes
        )
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
    except (UnsafeServerUrlError, CredentialStorageUnavailableError) as exc:
        raise _to_http_error(exc) from exc
    return _to_response(server)


@router.delete("/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    server_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    try:
        await mcp_service.delete_server(session, tenant_id=tenant.id, server_id=server_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
