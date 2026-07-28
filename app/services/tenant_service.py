import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import generate_api_key, hash_api_key, key_prefix
from app.models.tenant import ApiKey, Tenant

logger = get_logger(__name__)


class SlugAlreadyExistsError(Exception):
    """Raised when a tenant slug collides — the API layer maps this to 409.

    A domain error, not an HTTPException: this service must stay callable from a
    worker or a test without importing the web framework.
    """


async def create_tenant(session: AsyncSession, *, slug: str, name: str) -> Tenant:
    existing = await session.scalar(select(Tenant).where(Tenant.slug == slug))
    if existing is not None:
        raise SlugAlreadyExistsError(slug)

    tenant = Tenant(slug=slug, name=name)
    session.add(tenant)
    await session.commit()
    await session.refresh(tenant)
    logger.info("tenant_created", tenant_id=str(tenant.id), slug=slug)
    return tenant


async def issue_api_key(
    session: AsyncSession, *, tenant_id: uuid.UUID, name: str
) -> tuple[ApiKey, str]:
    """Mint a key for a tenant. Returns the record and the plaintext.

    The plaintext is returned exactly once, to be surfaced to the caller and
    never stored: only its hash is persisted. If it is lost, it is unrecoverable
    and a new key must be issued — that is the point, not a limitation.
    """
    plaintext = generate_api_key()
    record = ApiKey(
        tenant_id=tenant_id,
        key_hash=hash_api_key(plaintext),
        key_prefix=key_prefix(plaintext),
        name=name,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    logger.info("api_key_issued", tenant_id=str(tenant_id), key_id=str(record.id))
    return record, plaintext


async def authenticate(session: AsyncSession, *, presented_key: str) -> Tenant | None:
    """Resolve a presented key to its tenant, or None if it is invalid.

    Looks the key up by hash (never by plaintext), rejects revoked keys, and
    records last_used_at for audit and stale-key detection.
    """
    digest = hash_api_key(presented_key)
    record = await session.scalar(
        select(ApiKey).where(ApiKey.key_hash == digest, ApiKey.revoked.is_(False))
    )
    if record is None:
        return None

    record.last_used_at = datetime.now(UTC)
    tenant = await session.get(Tenant, record.tenant_id)
    await session.commit()
    return tenant
