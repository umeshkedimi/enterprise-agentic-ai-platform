import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus


class TenantCreate(BaseModel):
    # Slug is the stable handle used in URLs and logs; constrain it to a
    # url-safe shape so it can be trusted downstream without escaping.
    slug: str = Field(min_length=1, max_length=63, pattern=r"^[a-z0-9][a-z0-9-]*$")
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


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    citations: list[Citation]
    tools_used: list[str]
    latency_ms: int
