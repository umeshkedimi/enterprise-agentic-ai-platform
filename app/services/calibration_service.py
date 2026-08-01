"""What a retrieval score is actually worth.

The orchestration chunk deferred a decision: whether to drop retrieved chunks
below some similarity floor. It was deferred for a good reason — a cosine
threshold is specific to an embedding model and a corpus, and a floor set too
high silently converts working answers into "the documents do not cover this",
which is the failure nobody reports because it looks like the system working.
Guessing at it was strictly worse than leaving it out.

This module is the instrument that makes it decidable. Every evaluation row
carries two numbers about the same turn: `retrieval_top_score`, the platform's
own confidence in the best passage it found, and `score`, an independent
judgement of whether the answer that passage produced was actually supported.
Bucketed against each other they answer the question directly — *given that the
best match scored 0.62, how often was the resulting answer grounded?*

Two things this deliberately does not do. It does not apply a floor; nothing in
the serving path reads this, and turning a reading into a threshold is an
operator's decision made once and reviewed. And it does not suggest one from thin
data: below the sample minimums it returns no recommendation and says why,
because the entire point of deferring the decision was to stop it from being a
guess, and a guess dressed as an analysis is worse than the honest one it
replaced.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.evaluation import EVALUATOR_GROUNDEDNESS, RUBRIC_VERSION, TurnEvaluation

logger = get_logger(__name__)

# Upper bounds of each bucket; the last bucket runs from the final edge upward.
# Deliberately the interior boundaries of the `eaap_retrieval_top_score`
# histogram in `app/core/metrics.py`. That histogram shows how scores are
# distributed across all traffic, including the turns nobody has judged yet, and
# this report explains what those scores turned out to be worth. They only mean
# anything read side by side, and they cannot be if one is bucketed differently.
BUCKET_EDGES: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)

# A floor is only proposed when there is enough evidence to propose one. These
# are deliberately blunt: the failure this guards against is a threshold derived
# from nine samples and then left in place for a year.
DEFAULT_TARGET_GROUNDEDNESS = 0.8
MIN_BUCKET_SAMPLES = 20
MIN_TOTAL_SAMPLES = 100


@dataclass(frozen=True)
class CalibrationBucket:
    """One band of retrieval score, and how the answers in it held up."""

    lower: float
    # None on the top bucket, which is open-ended.
    upper: float | None
    evaluations: int
    # Abstentions are counted but not scored — see `app/models/evaluation.py`.
    # Keeping both numbers is what stops an agent that answers nothing from
    # looking perfectly grounded.
    graded: int
    mean_score: float | None
    abstentions: int


@dataclass(frozen=True)
class FloorRecommendation:
    """A proposed retrieval floor, or an explanation of why there isn't one."""

    floor: float | None
    rationale: str
    # Turns whose best match fell below the proposed floor. Under that floor they
    # would have retrieved nothing and abstained, so this is what the floor
    # costs — the number an operator actually has to weigh, and the one a
    # threshold picked off a chart never comes with.
    turns_that_would_abstain: int


@dataclass(frozen=True)
class CalibrationReport:
    agent_id: uuid.UUID | None
    rubric_version: str
    evaluations: int
    graded: int
    mean_score: float | None
    buckets: list[CalibrationBucket]
    recommendation: FloorRecommendation


def _bucket_index(column):
    """Map a score onto a bucket index, in SQL.

    Bucketed database-side rather than by loading rows and counting in Python:
    this table grows with every judged turn and a report that reads all of it is
    a report that stops working exactly when there is finally enough data to make
    it interesting.
    """
    return case(
        *[(column < edge, index) for index, edge in enumerate(BUCKET_EDGES)],
        else_=len(BUCKET_EDGES),
    )


def _bounds(index: int) -> tuple[float, float | None]:
    lower = 0.0 if index == 0 else BUCKET_EDGES[index - 1]
    upper = BUCKET_EDGES[index] if index < len(BUCKET_EDGES) else None
    return lower, upper


def recommend_floor(
    buckets: Sequence[CalibrationBucket],
    *,
    target: float = DEFAULT_TARGET_GROUNDEDNESS,
    min_bucket_samples: int = MIN_BUCKET_SAMPLES,
    min_total_samples: int = MIN_TOTAL_SAMPLES,
) -> FloorRecommendation:
    """Propose the score below which retrieved evidence stops being worth having.

    The rule is the simplest one that survives noisy data: walk up from the
    lowest bucket while the answers in it are, on average, worse than the target,
    and stop at the first band that clears it. The floor is the top edge of that
    losing run. Anything cleverer — fitting a curve, optimising a threshold —
    would read structure into a few hundred samples that is not reliably there.

    A bucket with too few samples to judge stops the walk rather than being
    assumed bad. The alternative sets a floor on the strength of a handful of
    rows, which is the exact mistake this whole module exists to avoid.
    """
    total = sum(b.evaluations for b in buckets)
    if total < min_total_samples:
        return FloorRecommendation(
            floor=None,
            rationale=(
                f"{total} evaluations on file; at least {min_total_samples} are needed "
                "before a floor is worth proposing."
            ),
            turns_that_would_abstain=0,
        )

    losing_run = 0
    for bucket in buckets:
        if bucket.graded < min_bucket_samples or bucket.mean_score is None:
            break
        if bucket.mean_score >= target:
            break
        losing_run += 1

    if losing_run == 0:
        return FloorRecommendation(
            floor=None,
            rationale=(
                "Even the lowest-scoring retrievals produced answers at or above the "
                f"{target:.0%} groundedness target, so a floor would discard evidence "
                "that is doing its job."
            ),
            turns_that_would_abstain=0,
        )

    floor = BUCKET_EDGES[losing_run - 1]
    cost = sum(b.evaluations for b in buckets[:losing_run])
    return FloorRecommendation(
        floor=floor,
        rationale=(
            f"Answers grounded in a best match below {floor:.2f} averaged under the "
            f"{target:.0%} groundedness target; at and above it they met it."
        ),
        turns_that_would_abstain=cost,
    )


async def calibration_report(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID | None = None,
    target: float = DEFAULT_TARGET_GROUNDEDNESS,
) -> CalibrationReport:
    """Aggregate this tenant's judged turns into score bands.

    Scoped to one rubric version. Averaging across rubric versions would compare
    numbers produced by different questions and read a change of measurement as a
    change in the thing measured.
    """
    index = _bucket_index(TurnEvaluation.retrieval_top_score)
    conditions = [
        TurnEvaluation.tenant_id == tenant_id,
        TurnEvaluation.evaluator == EVALUATOR_GROUNDEDNESS,
        TurnEvaluation.rubric_version == RUBRIC_VERSION,
        # A turn whose retrieval score was never recorded cannot say anything
        # about what a retrieval score is worth.
        TurnEvaluation.retrieval_top_score.is_not(None),
    ]
    if agent_id is not None:
        conditions.append(TurnEvaluation.agent_id == agent_id)

    rows = (
        await session.execute(
            select(
                index.label("bucket"),
                func.count().label("evaluations"),
                func.count(TurnEvaluation.score).label("graded"),
                func.avg(TurnEvaluation.score).label("mean_score"),
            )
            .where(*conditions)
            .group_by(index)
            .order_by(index)
        )
    ).all()

    by_index = {int(row.bucket): row for row in rows}
    buckets = []
    for i in range(len(BUCKET_EDGES) + 1):
        lower, upper = _bounds(i)
        row = by_index.get(i)
        evaluations = int(row.evaluations) if row else 0
        graded = int(row.graded) if row else 0
        buckets.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                evaluations=evaluations,
                graded=graded,
                mean_score=float(row.mean_score) if row and row.mean_score is not None else None,
                abstentions=evaluations - graded,
            )
        )

    total = sum(b.evaluations for b in buckets)
    graded_total = sum(b.graded for b in buckets)
    # Weighted by the graded count per bucket rather than averaging the bucket
    # means: a band holding four turns must not carry the same weight as one
    # holding four hundred.
    mean_score = (
        sum(b.mean_score * b.graded for b in buckets if b.mean_score is not None) / graded_total
        if graded_total
        else None
    )

    report = CalibrationReport(
        agent_id=agent_id,
        rubric_version=RUBRIC_VERSION,
        evaluations=total,
        graded=graded_total,
        mean_score=mean_score,
        buckets=buckets,
        recommendation=recommend_floor(buckets, target=target),
    )
    logger.info(
        "calibration_report_built",
        tenant_id=str(tenant_id),
        agent_id=str(agent_id) if agent_id else None,
        evaluations=total,
        graded=graded_total,
        mean_score=mean_score,
        suggested_floor=report.recommendation.floor,
    )
    return report
