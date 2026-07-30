"""An MCP server as configuration, the same way an agent is.

This is the thesis applied one level out. Chunk 3 made a *tool* something an agent
is granted by naming it; this makes the set of available tools itself a
configuration change. A team points the platform at an MCP server, and its tools
become grantable to that team's agents — no code in this repository knows what
those tools do, and adding a capability is a `POST` rather than a release.

Two things are deliberately absent from this table:

* **No stdio transport.** The MCP spec allows launching a server as a
  subprocess, and every hosted platform that offers it has handed its tenants
  arbitrary command execution on the platform's own machines. Remote HTTP only.
* **No plaintext credential.** The bearer token for somebody else's server is
  stored encrypted, because unlike an API key the platform issues, this one has
  to be replayed on every call and therefore cannot be hashed.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean as SABoolean
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

# How long a call to a third party may take before the turn gives up on it. A
# tool call sits inside a model turn a user is waiting on, so the ceiling is set
# by human patience rather than by what the remote server would like.
DEFAULT_MCP_TIMEOUT_SECONDS = 20

# Separates a server's slug from the tool's own name in the namespaced form the
# model sees. Two underscores rather than a dot because provider tool names are
# restricted to `[a-zA-Z0-9_-]` — a dot is rejected outright by OpenAI.
NAME_SEPARATOR = "__"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class McpServer(SQLModel, table=True):
    """A remote MCP server one tenant has registered.

    Tenant-scoped like everything else: the servers a tenant can reach are its
    own, so one team cannot grant its agents another team's integrations, and a
    tool name that resolves for one tenant resolves to nothing for the next.
    """

    __tablename__ = "mcp_servers"
    __table_args__ = (UniqueConstraint("tenant_id", "slug", name="uq_mcp_server_tenant_slug"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    # Also the namespace its tools appear under, which is why it is length-capped
    # more tightly than other slugs: `slug__tool_name` has to fit inside the
    # provider limit on function names.
    slug: str = Field(sa_column=Column(String(31), nullable=False, index=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    # Streamable HTTP endpoint. Validated on write against the SSRF guard, and
    # again on connect — a URL that resolved to a public address at registration
    # time can be re-pointed at a private one afterwards.
    url: str = Field(sa_column=Column(String(2048), nullable=False))

    # Fernet ciphertext, or NULL for a server that needs no credential. Never
    # returned by the API: it goes out over the wire to the MCP server and
    # nowhere else.
    auth_token_encrypted: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Non-secret headers a server needs for routing or versioning. Kept apart
    # from the credential so that what is displayable and what is not is a
    # property of the column rather than of a convention.
    headers: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))

    timeout_seconds: int = Field(
        sa_column=Column(Integer, nullable=False, default=DEFAULT_MCP_TIMEOUT_SECONDS)
    )
    enabled: bool = Field(sa_column=Column(SABoolean, nullable=False, default=True))

    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    # Doubles as the cache key for discovered tools: editing a server's URL or
    # credential must invalidate the tool list discovered under the old one.
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=_utcnow),
    )
