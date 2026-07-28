import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_TEMPERATURE,
)
from app.models.document import DocumentStatus

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


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    citations: list[Citation]
    tools_used: list[str]
    latency_ms: int
