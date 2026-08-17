"""The durable transcript of an agent conversation.

Two stores end up holding conversation data once a checkpointer is attached, and
they are not redundant — they answer different questions:

* These tables are the *product* record. They are tenant-scoped, indexed, and
  queryable: "list this team's conversations", "what did this agent answer, from
  which sources, at what token cost". This is what the API reads back, what
  hydrates history for the next turn, and what the evaluation harness in Chunk 7
  audits.
* The LangGraph checkpointer is the *execution* record. It holds the graph state
  of a run — scratchpad, pending tool calls, step counters — keyed by an opaque
  thread id, in tables owned by the library. It can resume a half-finished turn;
  it cannot tell you which tenant a thread belongs to.

The division of labour is deliberate: the transcript below is the single source
of truth for what the model is shown next turn. Nothing reads history out of a
checkpoint.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Conversation(SQLModel, table=True):
    """One ongoing thread between an end user and a configured agent.

    `tenant_id` is stored here rather than being reached through `agents` so the
    ownership check on every read is a direct filter, not a join. Tenancy is
    enforced in the query — a conversation id is a UUID a caller could guess at,
    and guessing it must not be enough.
    """

    __tablename__ = "conversations"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    # CASCADE, not SET NULL: a conversation is only meaningful next to the agent
    # config that produced it, and one whose agent is gone can never be resumed
    # — the prompt, model, and knowledge scope it ran under no longer exist.
    agent_id: uuid.UUID = Field(
        sa_column=Column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    title: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    # A rolling fold-in of everything older than the replay window, refreshed
    # lazily the first time a thread outgrows it. Nullable rather than
    # empty-string: most conversations never reach the window and should read as
    # having no summary, not an empty one.
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # The `seq` of the newest message already folded in. Everything at or below
    # this watermark is represented only in `summary`; everything above it is
    # still replayed verbatim by `load_history`. Null means nothing has been
    # summarized yet — the thread has never exceeded the window.
    summary_through_seq: int | None = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    # Bumped on every recorded turn, so "most recently active" is an index scan
    # rather than an aggregate over messages.
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class ConversationMessage(SQLModel, table=True):
    """One turn in a conversation, plus what it cost and what it was grounded in.

    Roles are user and assistant only — the same boundary the completion service
    enforces. The system prompt comes from agent config at request time and is
    never stored as a turn, so a replayed conversation cannot smuggle
    instructions back in as data.
    """

    __tablename__ = "conversation_messages"
    # Every read of this table is "the turns of one conversation, in order", so
    # one composite index matches that shape rather than indexing each column on
    # its own. Postgres walks it backwards for the newest-N history query.
    __table_args__ = (Index("ix_conversation_messages_thread", "conversation_id", "seq"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: uuid.UUID = Field(
        sa_column=Column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    )
    # Ordering key. Timestamps are not enough: a user turn and the assistant turn
    # answering it are written in the same transaction and can land on the same
    # microsecond, and history replayed in the wrong order is a subtly broken
    # conversation rather than an obvious one. A database-side sequence gives a
    # total order the application cannot get wrong.
    seq: int | None = Field(
        default=None, sa_column=Column(BigInteger, Identity(always=False), nullable=False)
    )
    role: str = Field(sa_column=Column(String(16), nullable=False))
    content: str = Field(sa_column=Column(Text, nullable=False))

    # Assistant-turn provenance. Citations are stored as JSONB rather than as
    # foreign keys to chunks on purpose: a citation records what the answer was
    # grounded in *at the time*, and re-indexing a document must not silently
    # rewrite the evidence for an answer already given.
    citations: list[dict] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, default=list)
    )
    tools_used: list[str] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, default=list)
    )

    # Which model actually served the turn — per-agent config can change between
    # turns, so this is history, not a copy of the current agent row.
    model: str | None = Field(default=None, sa_column=Column(String(63), nullable=True))
    provider: str | None = Field(default=None, sa_column=Column(String(31), nullable=True))
    prompt_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    completion_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    total_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    latency_ms: int = Field(sa_column=Column(Integer, nullable=False, default=0))

    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
