import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlmodel import Field, SQLModel

EMBEDDING_DIM = 1536


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # A document lives inside a collection, which belongs to a tenant. Tenant
    # scope is reached through this FK rather than duplicated here, so there is
    # one source of truth for which team owns a document. ON DELETE CASCADE:
    # deleting a collection removes its documents (and their chunks) — the
    # knowledge is discarded with the scope that held it.
    collection_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    filename: str
    content_type: str
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED, sa_column=Column(String(20), nullable=False)
    )
    uploaded_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    error_message: str | None = None


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    chunk_index: int
    content: str = Field(sa_column=Column(Text, nullable=False))
    embedding: list[float] = Field(sa_column=Column(Vector(EMBEDDING_DIM), nullable=False))
    token_count: int
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
