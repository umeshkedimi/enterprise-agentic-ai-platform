import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, Computed, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    """A document belongs to a collection; a *version* of a document belongs to
    a `document_key` within it.

    Re-uploading is never an edit — ingestion has always written new chunk rows
    rather than mutating existing ones, because the evaluation judge reads a
    citation's chunk back by id and a surviving id has to still hold what the
    model actually read (see `app/services/evaluation_service.py`). Versioning
    extends that same append-only discipline one level up: a new upload sharing
    an old one's `document_key` is a new row, and `is_current` is the only thing
    that moves. The old row, and its chunks, are never deleted by a supersession
    — only excluded from retrieval — so an answer given last month can still be
    audited against exactly what it cited even after the policy changed.

    `document_key` is nullable and caller-supplied, not inferred from the
    filename: a filename match is fragile (a rename breaks it, two unrelated
    files can collide) and guessing identity is worse than not claiming it. A
    document uploaded without a key simply has no version history — it behaves
    exactly as before this existed.
    """

    __tablename__ = "documents"
    __table_args__ = (
        # At most one current version per (collection, key), enforced by
        # Postgres rather than by application code alone — the same posture
        # tenancy takes elsewhere in this platform: a guarantee worth having is
        # worth having at the layer that cannot be bypassed by a bug two
        # functions away. Partial: rows with no key, or already superseded,
        # never compete for this slot.
        Index(
            "uq_documents_current_version_per_key",
            "collection_id",
            "document_key",
            unique=True,
            postgresql_where=text("is_current AND document_key IS NOT NULL"),
        ),
    )

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
    # The logical identity a version history is grouped under. Null for a
    # document that was never given one — it is simply its own, permanent,
    # ungrouped current version.
    document_key: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    # Whether this is the version retrieval should surface. Only ever flipped
    # in pairs — see `document_service._supersede`, which retires the previous
    # current version in its own commit before promoting this one, so the two
    # updates never race to satisfy the partial unique index above.
    is_current: bool = Field(default=True, sa_column=Column(Boolean, nullable=False, default=True))
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
    # A Postgres GENERATED column (`to_tsvector('english', content)`), created
    # by raw SQL in the migration and GIN-indexed there too — the same shape
    # as the hand-written pgvector HNSW index, and for the same reason:
    # neither has a SQLModel/SQLAlchemy equivalent that autogenerate can
    # express. `Computed(...)` here is what tells the ORM to leave this
    # column out of every INSERT/UPDATE it issues — without it, SQLAlchemy
    # sends an explicit NULL for the column on every insert, which Postgres
    # refuses outright for a GENERATED ALWAYS column ("cannot insert a
    # non-DEFAULT value"). The expression text is never actually sent to the
    # database from here — the migration's raw SQL already created the
    # column this way — but Alembic's autogenerate does not manage a
    # computed column's *expression* even when one is declared, only its
    # presence, so this cannot drift into proposing to alter it.
    content_tsv: str | None = Field(
        default=None,
        sa_column=Column(
            TSVECTOR, Computed("to_tsvector('english', content)", persisted=True), nullable=True
        ),
    )
    token_count: int
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
