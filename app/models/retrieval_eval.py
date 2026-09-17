"""Ground truth for retrieval quality, and what a benchmark run found against it.

Every chunk that follows this one — better chunking, hybrid search, reranking —
is a claim that retrieval got *better*. Without a fixed set of questions and the
documents that ought to answer them, "better" is a feeling, not a number, and the
platform already has one deferred decision (the relevance floor) that sat idle
for two chunks specifically because nothing could tell whether a threshold would
help or hurt. This module is what lets every later change be judged against a
number instead of a guess.

Ground truth is recorded at the *document* level, not the chunk level. A golden
example says "the vacation policy document is relevant to this question," never
"chunk 7 of it is." Chunk identity is an ingestion implementation detail that
changes the moment chunking strategy changes — exactly the kind of change this
harness exists to evaluate — so ground truth that named a chunk would invalidate
itself the first time it did its job.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RetrievalGoldenExample(SQLModel, table=True):
    """One question a collection's retrieval is expected to answer correctly."""

    __tablename__ = "retrieval_golden_examples"
    __table_args__ = (
        # The same query recorded twice for one collection is not a second case,
        # it is a typo — re-adding it should fail loudly rather than silently
        # double-weight one question in every future average.
        UniqueConstraint(
            "collection_id", "query", name="uq_golden_example_collection_query"
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    collection_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    query: str = Field(sa_column=Column(Text, nullable=False))
    # Deliberately not a foreign key. A document named here can later be
    # deleted or superseded, and the case should keep meaning "this used to be
    # (or still is) the right answer" rather than disappear or block the
    # deletion — a benchmark run against a stale case simply scores it a miss,
    # which is itself a useful signal.
    relevant_document_ids: list[str] = Field(sa_column=Column(JSONB, nullable=False))
    notes: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class RetrievalBenchmarkRun(SQLModel, table=True):
    """One scored pass over every golden example on file for a collection.

    A row per run, not a running average, for the same reason `TurnEvaluation`
    is a row per judgement: a chunking change made next week must produce a new,
    comparable number sitting beside the old one, not overwrite the only
    record that the old behaviour ever existed.
    """

    __tablename__ = "retrieval_benchmark_runs"
    __table_args__ = (
        Index("ix_retrieval_benchmark_runs_collection_time", "collection_id", "created_at"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    collection_id: uuid.UUID = Field(
        sa_column=Column(ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    )
    # Free text, required. Forcing a caller to name what they're testing
    # ("baseline", "chunk-size-600", "hybrid-rrf") is what makes a later
    # calibration_service-style report legible instead of an undated pile of
    # numbers nobody can attribute to a change.
    label: str = Field(sa_column=Column(String(120), nullable=False))
    top_k: int = Field(sa_column=Column(Integer, nullable=False))
    cases: int = Field(sa_column=Column(Integer, nullable=False))
    # Mean of each case's recall (relevant documents found / relevant documents
    # expected) and mean reciprocal rank of the first relevant hit. Both are
    # plain arithmetic over `results`, recomputable by anyone reading the row —
    # the same discipline the groundedness judge uses for `score`.
    mean_recall: float = Field(sa_column=Column(Float, nullable=False))
    mrr: float = Field(sa_column=Column(Float, nullable=False))
    # Per-case breakdown: golden_example_id, query, relevant_document_ids,
    # retrieved_document_ids (ranked, may repeat a document across chunks),
    # recall, reciprocal_rank. This is what turns "mrr dropped to 0.61" into
    # something an engineer can act on instead of just distrust.
    results: list[dict] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, default=list)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
