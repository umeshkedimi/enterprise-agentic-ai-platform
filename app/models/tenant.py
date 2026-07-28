import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Tenant(SQLModel, table=True):
    """An organization or team boundary.

    Every knowledge collection, agent, and document ultimately belongs to a
    tenant. This is the root of the isolation model: 'multi-tenant' is a
    property of the schema, not of a prompt. A query that forgets to filter by
    tenant is a data-leak bug, which is why tenancy lives in foreign keys the
    database enforces rather than in application conventions.
    """

    __tablename__ = "tenants"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # A stable, URL-safe handle ("hr", "finance") used in APIs and logs so we
    # never have to expose or memorize UUIDs. Unique across the platform.
    slug: str = Field(sa_column=Column(String(63), nullable=False, unique=True, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class ApiKey(SQLModel, table=True):
    """A credential that authenticates a caller *as* a tenant.

    We store only a SHA-256 hash of the key, never the plaintext: a database
    dump then leaks no usable credentials. The plaintext is shown exactly once,
    at creation. `key_prefix` keeps the first few characters so a key can be
    identified and revoked in a UI without ever revealing the secret.

    This is a deliberately simple bearer-token scheme — honest for a platform
    at this stage. It sits behind a single FastAPI dependency, so swapping it
    for OIDC/SSO later changes one resolver, not every endpoint.
    """

    __tablename__ = "api_keys"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    # Indexed because every authenticated request looks a key up by its hash.
    key_hash: str = Field(sa_column=Column(String(64), nullable=False, unique=True, index=True))
    key_prefix: str = Field(sa_column=Column(String(12), nullable=False))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    revoked: bool = Field(sa_column=Column(Boolean, nullable=False, default=False))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    last_used_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
