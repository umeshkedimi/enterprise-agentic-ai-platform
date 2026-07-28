import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Boolean as SABoolean
from sqlmodel import Field, SQLModel

# Defaults for an agent's execution policy. Conservative, and overridable per
# agent — they exist so a freshly-created agent is runnable without a caller
# having to understand every knob first.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_RETRIEVAL_TOP_K = 5


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Collection(SQLModel, table=True):
    """A named, tenant-scoped bucket of knowledge (e.g. "HR Policies").

    Documents belong to a collection, not merely to a tenant. This is what
    fences one team's retrieval from another's: an agent searches *its*
    collection, so Finance's runbooks can never surface in an HR answer even
    though both live in the same physical tables.
    """

    __tablename__ = "collections"
    # A slug is unique within a tenant, not globally — two tenants may each have
    # a "policies" collection without colliding.
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_collection_tenant_slug"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    slug: str = Field(sa_column=Column(String(63), nullable=False, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Agent(SQLModel, table=True):
    """An assistant expressed as configuration, not code.

    This row is the core of the "configuration, not deployment" thesis: the
    shared runtime reads an agent's prompt, model, tool allowlist, knowledge
    scope, and execution policy at request time, so onboarding a new assistant
    is an INSERT — no branch in the code, no redeploy. Two agents diverge in
    behavior entirely through the values in these columns.
    """

    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_agent_tenant_slug"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    slug: str = Field(sa_column=Column(String(63), nullable=False, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))

    # Chat model is per-agent runtime config and can vary freely — unlike the
    # embedding provider, which is pinned platform-wide. Stored as a string so a
    # new model is a config value, not a code change.
    model: str = Field(sa_column=Column(String(63), nullable=False, default=DEFAULT_MODEL))

    # Knowledge scope. Nullable on purpose: a pure-tool agent (no retrieval) is
    # valid. ON DELETE SET NULL so removing a collection disables retrieval for
    # its agents rather than cascading them out of existence.
    collection_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            ForeignKey("collections.id", ondelete="SET NULL"), nullable=True, index=True
        ),
    )

    # Execution policy. The tool allowlist is an allow-list by design: an agent
    # can invoke a tool only if it is named here, so capability is granted
    # explicitly per agent rather than inherited from what the platform happens
    # to support. Stored as JSON — a short list of tool names, queried as a whole.
    tool_allowlist: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    temperature: float = Field(
        sa_column=Column(Float, nullable=False, default=DEFAULT_TEMPERATURE)
    )
    max_output_tokens: int = Field(
        sa_column=Column(Integer, nullable=False, default=DEFAULT_MAX_OUTPUT_TOKENS)
    )
    retrieval_top_k: int = Field(
        sa_column=Column(Integer, nullable=False, default=DEFAULT_RETRIEVAL_TOP_K)
    )

    # A disabled agent stays configured but refuses to run — the platform way to
    # take an assistant offline without deleting its history or config.
    enabled: bool = Field(sa_column=Column(SABoolean, nullable=False, default=True))

    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=_utcnow),
    )
