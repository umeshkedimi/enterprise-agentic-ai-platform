import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_TEMPERATURE,
)
from app.models.document import DocumentStatus
from app.models.mcp import DEFAULT_MCP_TIMEOUT_SECONDS

# A url-safe slug shape, reused wherever a caller names a resource. Constrained
# so a slug can be trusted downstream (URLs, logs) without escaping.
_SLUG = {"min_length": 1, "max_length": 63, "pattern": r"^[a-z0-9][a-z0-9-]*$"}


class TenantCreate(BaseModel):
    slug: str = Field(**_SLUG)
    name: str = Field(min_length=1, max_length=255)


class TenantResponse(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    created_at: datetime


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ApiKeyIssuedResponse(BaseModel):
    """Returned once, at creation. `api_key` is the plaintext and is never
    retrievable again — the platform stores only its hash."""

    id: uuid.UUID
    name: str
    key_prefix: str
    api_key: str
    created_at: datetime


class DocumentResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    filename: str
    status: DocumentStatus
    chunk_count: int
    uploaded_at: datetime
    error_message: str | None = None


class Citation(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    snippet: str
    score: float


class CollectionCreate(BaseModel):
    slug: str = Field(**_SLUG)
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class CollectionResponse(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    created_at: datetime


class McpServerCreate(BaseModel):
    # A tighter slug than elsewhere: this one becomes the namespace prefix on
    # every tool the server exposes, and `slug__tool_name` has to fit inside the
    # provider limit on function names.
    slug: str = Field(min_length=1, max_length=31, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    # Streamable HTTP only. The MCP spec also allows launching a server as a
    # subprocess, which on a multi-tenant platform is tenant-configurable command
    # execution — so there is no field here through which to ask for it.
    url: str = Field(min_length=1, max_length=2048)
    # Write-only. Stored encrypted and never returned; the platform has to replay
    # it to the server, so unlike an API key it cannot be hashed.
    auth_token: str | None = Field(default=None, min_length=1, max_length=4096)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=DEFAULT_MCP_TIMEOUT_SECONDS, ge=1, le=120)
    enabled: bool = True


class McpServerUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    # An empty string clears the stored credential; omitting the field leaves it
    # untouched. A partial update must not silently drop a secret it never saw.
    auth_token: str | None = Field(default=None, max_length=4096)
    headers: dict[str, str] | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=120)
    enabled: bool | None = None


class McpServerResponse(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    description: str | None
    url: str
    # Whether a credential is stored, never the credential. Enough to answer "is
    # this configured?" without the answer being usable.
    authenticated: bool
    headers: dict[str, str]
    timeout_seconds: int
    enabled: bool
    created_at: datetime
    updated_at: datetime


class McpToolResponse(BaseModel):
    """A tool discovered on a server, as an agent would have to name it.

    `name` is the namespaced form — that is the string a `tool_allowlist` entry
    has to match, and the reason this endpoint exists: a team owner needs to know
    what to grant before they can grant it.
    """

    name: str
    description: str


class McpDiscoveryResponse(BaseModel):
    server_id: uuid.UUID
    slug: str
    reachable: bool
    tools: list[McpToolResponse]
    # Present when `reachable` is false. A team owner debugging their own
    # integration needs to know whether the platform could not connect, could not
    # authenticate, or connected and was told nothing.
    error: str | None = None


class AgentCreate(BaseModel):
    slug: str = Field(**_SLUG)
    name: str = Field(min_length=1, max_length=255)
    system_prompt: str = Field(min_length=1)
    model: str = Field(default=DEFAULT_MODEL, max_length=63)
    # Knowledge scope. Optional: a pure-tool agent has no collection.
    collection_id: uuid.UUID | None = None
    tool_allowlist: list[str] = Field(default_factory=list)
    # Execution policy, range-checked so an agent cannot be created with
    # nonsensical sampling or token limits.
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1, le=32000)
    retrieval_top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, ge=1, le=50)
    enabled: bool = True


class AgentUpdate(BaseModel):
    """Partial update — only fields explicitly set are applied. Slug and tenant
    are immutable: an agent's identity does not change, you create a new one."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    system_prompt: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, max_length=63)
    collection_id: uuid.UUID | None = None
    tool_allowlist: list[str] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32000)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    enabled: bool | None = None


class AgentResponse(BaseModel):
    id: uuid.UUID
    slug: str
    name: str
    system_prompt: str
    model: str
    collection_id: uuid.UUID | None
    tool_allowlist: list[str]
    temperature: float
    max_output_tokens: int
    retrieval_top_k: int
    enabled: bool
    created_at: datetime
    updated_at: datetime


class CompletionTurn(BaseModel):
    """One caller-supplied turn. The system role is not accepted: an agent's
    instructions come from its stored configuration, not from the request."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AgentCompletionRequest(BaseModel):
    turns: list[CompletionTurn] = Field(min_length=1)


class TokenUsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AgentCompletionResponse(BaseModel):
    text: str
    # Which model and provider actually served the request — the point of the
    # multi-model layer is that this varies per agent, so it is reported back.
    model: str
    provider: str
    usage: TokenUsageResponse
    latency_ms: int


class AgentChatRequest(BaseModel):
    """One question for an orchestrated turn.

    Note what is absent: history. The platform stores the thread and replays it
    itself, so a caller supplies only the conversation to continue. That is a
    security boundary rather than a convenience — a client that supplies its own
    history can fabricate assistant turns, and a forged turn is indistinguishable
    from a real one by the time it reaches the model.

    Omitting `conversation_id` starts a new conversation; the id comes back on
    the response, so a first call needs no setup.
    """

    # Unknown fields are rejected rather than ignored. A client still sending the
    # `history` this endpoint used to accept has a real expectation about what
    # the model will see, and silently dropping it would look like the agent
    # forgetting things at random. A 422 says what changed.
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    conversation_id: uuid.UUID | None = None


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ConversationResponse(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    """One recorded turn, with the provenance of the answer it holds.

    Citations, tools, model, and tokens are stored per message rather than
    recomputed: they record what happened at the time, and an agent whose model
    or knowledge scope has since been reconfigured must not appear, in its own
    history, to have always been configured that way.
    """

    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation]
    tools_used: list[str]
    model: str | None
    provider: str | None
    usage: TokenUsageResponse
    latency_ms: int
    created_at: datetime


class AgentChatResponse(BaseModel):
    # Echoed on every turn — a caller that omitted it is being told which thread
    # its question landed in, and needs it to ask a follow-up.
    conversation_id: uuid.UUID
    answer: str
    # What the answer was grounded in. Empty for an agent with no knowledge
    # scope, and empty when retrieval legitimately found nothing — the answer
    # text says which, because an uncited claim is the thing worth noticing.
    citations: list[Citation]
    tools_used: list[str]
    model: str
    provider: str
    usage: TokenUsageResponse
    # End-to-end for the turn, so it exceeds the model latency by the cost of
    # retrieval. Keeping both is what makes a slow answer diagnosable.
    latency_ms: int
