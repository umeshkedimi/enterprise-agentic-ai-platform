"""What the platform later judged about a turn it already served.

Kept in its own table rather than as columns on `conversation_messages`, and the
reason is that the two records answer to different clocks. A message row is
written once, inside the turn, and is the product's account of what happened. An
evaluation is written afterwards, possibly minutes later, possibly again next
week under a better rubric, and is the platform's opinion *about* what happened.
Folding an opinion into the record it judges means a rubric change rewrites
history, and it means the turn cannot be recorded until the judgement exists —
which would put an LLM call on the request path to store a row.

So evaluation is strictly downstream: it reads the transcript, writes here, and
nothing in the serving path waits for it. A judge outage costs the platform its
scores, never its answers.
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

# The checks a turn can be subjected to. A string rather than a DB enum: adding a
# check is a deploy, and a migration to widen an enum type for every new one is
# friction with nothing to show for it.
EVALUATOR_GROUNDEDNESS = "groundedness"

# Verdicts, in the order a reader should think about them.
#
# `abstained` is the one worth explaining. An agent that was shown nothing
# relevant and said so has done the right thing, and scoring that as a failed
# answer would train the platform's own metrics to punish the behaviour it
# exists to produce. It is tracked separately and excluded from the groundedness
# average, because an answer asserting nothing asserts nothing false — averaging
# it in as 1.0 would flatter a broken retriever.
VERDICT_SUPPORTED = "supported"
VERDICT_PARTIAL = "partial"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_ABSTAINED = "abstained"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_PARTIAL, VERDICT_UNSUPPORTED, VERDICT_ABSTAINED)

# Bumped when the rubric or the judge prompt changes in a way that makes two
# scores incomparable. Versioning it is what keeps a rubric change from looking
# like a regression in the platform: old rows keep the version they were scored
# under, and a chart can group by it instead of averaging across a discontinuity.
#
# v2: a live run against a real judge showed two ways a score came out too high.
# Remarks about what the sources *do not* contain were being counted as supported
# claims, and figures the answer computed for itself from a rule in the sources
# were being credited to the sources. Both are now called out in the rubric, and
# the first is additionally enforced platform-side — a claim marked supported
# while naming no source is not supported. Scores under v1 and v2 are not
# comparable, which is exactly what this constant is for.
RUBRIC_VERSION = "v2"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TurnEvaluation(SQLModel, table=True):
    """One judgement of one assistant turn under one rubric."""

    __tablename__ = "turn_evaluations"
    __table_args__ = (
        # Re-running the same rubric over the same turn is an idempotent
        # operation, not a second opinion — a backfill that is interrupted and
        # restarted must not double-count. A genuinely different rubric carries a
        # different version and therefore gets its own row, so history is kept
        # where it means something and not where it is noise.
        UniqueConstraint(
            "message_id", "evaluator", "rubric_version", name="uq_turn_evaluations_message_rubric"
        ),
        # The question a team owner asks is "how is *my agent* doing, lately",
        # which is this index and no join.
        Index("ix_turn_evaluations_agent_time", "agent_id", "created_at"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Denormalised from the message's conversation, for the same reason
    # `conversations.tenant_id` is: the ownership filter on every read must be a
    # predicate on this table, not a join through two others that a future query
    # could forget to write.
    tenant_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    agent_id: uuid.UUID = Field(
        sa_column=Column(ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    )
    # The assistant turn under judgement. CASCADE: a score for a deleted turn is
    # not evidence of anything, and keeping it would leave rows nothing can
    # explain. Deliberately not indexed on its own — the unique constraint above
    # leads with this column, so its btree already serves lookups by message.
    message_id: uuid.UUID = Field(
        sa_column=Column(
            ForeignKey("conversation_messages.id", ondelete="CASCADE"), nullable=False
        )
    )

    evaluator: str = Field(sa_column=Column(String(32), nullable=False))
    rubric_version: str = Field(sa_column=Column(String(16), nullable=False))

    # Nullable on purpose: an abstention has no claims to ground, and inventing a
    # number for it would corrupt every average computed over this column.
    score: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    verdict: str = Field(sa_column=Column(String(16), nullable=False))
    # The judge's reasoning, kept because a groundedness score nobody can audit
    # is a number nobody will act on. Truncated at write time.
    rationale: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    # The claim-by-claim breakdown the score was computed from: each entry is a
    # factual assertion lifted out of the answer, whether the sources carried it,
    # and which sources. This is the part an auditor actually reads — a bare 0.67
    # tells a team owner their agent is wrong a third of the time and nothing
    # about which third. It is also what makes the score reproducible: the number
    # is arithmetic over this list, not a float the model was asked to guess.
    claims: list[dict] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False, default=list)
    )

    # The retrieval signals as they stood for the judged turn, snapshotted rather
    # than re-derived. This is the whole point of the calibration work: the
    # platform's own confidence in a passage is `top_score`, and until it sits in
    # the same row as an independent judgement of the answer, there is no way to
    # find out what a given score is actually worth. Re-querying it later would
    # read a re-indexed corpus and answer a different question.
    retrieval_top_score: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    citation_count: int = Field(sa_column=Column(Integer, nullable=False, default=0))

    # What the judgement itself cost. Evaluation is an LLM workload like any
    # other and its bill is nobody's surprise if it is recorded from the start.
    judge_model: str | None = Field(default=None, sa_column=Column(String(63), nullable=True))
    judge_provider: str | None = Field(default=None, sa_column=Column(String(31), nullable=True))
    prompt_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    completion_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    total_tokens: int = Field(sa_column=Column(Integer, nullable=False, default=0))
    latency_ms: int = Field(sa_column=Column(Integer, nullable=False, default=0))

    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
