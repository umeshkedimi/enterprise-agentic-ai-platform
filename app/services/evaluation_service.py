"""Judge a served answer against the evidence it was given.

This runs *after* a turn, over the transcript, and never inside one. The reason
is not performance alone, though a judge on the request path would roughly double
the latency and cost of every answer. It is that the two have to be able to fail
independently: an assistant whose evaluator is down is an assistant that still
works, and a scoring bug must never be able to take answers with it. The
transcript exists precisely so that judging is something the platform can do
later, twice, or under a better rubric.

The score is arithmetic, not a number the model was asked for. The judge
enumerates the claims in an answer and marks each supported or not; the platform
divides. Models are poor at producing calibrated floats and good at the concrete
question "does this sentence follow from that paragraph", so the design leans on
the second and computes the first. It also means every score can be recomputed
from the stored claims, and a reader who disagrees with a 0.5 can see exactly
which half.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics, tracing
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.agent import Agent
from app.models.conversation import Conversation, ConversationMessage
from app.models.document import Document, DocumentChunk
from app.models.evaluation import (
    EVALUATOR_GROUNDEDNESS,
    RUBRIC_VERSION,
    VERDICT_ABSTAINED,
    VERDICT_PARTIAL,
    VERDICT_SUPPORTED,
    VERDICT_UNSUPPORTED,
    TurnEvaluation,
)
from app.services.completion_service import Turn, complete
from app.services.errors import (
    EvaluationFailedError,
    EvaluationNotApplicableError,
    NotFoundError,
)
from app.services.evaluation_prompts import AUDIT_RUBRIC, build_audit_request, parse_audit_response

logger = get_logger(__name__)

_ASSISTANT = "assistant"

# The judge is given an `Agent` because that is what `complete()` takes, and
# routing it through the same function is deliberate: retries, provider
# credentials, sampling-parameter adaptation, token accounting, and tracing all
# live there, and a second path to a model is how a platform ends up with a
# workload it is not counting. This one is assembled per call and never stored —
# it belongs to the platform, not to a tenant, and there is no row for it.
_JUDGE_AGENT_ID = uuid.uuid5(uuid.NAMESPACE_URL, "eaap:evaluation-judge")

# Deterministic. A judge that returns a different verdict each time it is asked
# is not measuring the answer, it is measuring itself.
_JUDGE_TEMPERATURE = 0.0

# One claim's text, as stored. Long enough to be a quotable assertion, short
# enough that a judge which decides to paste an entire answer into one claim
# cannot make a row unreadable.
_CLAIM_MAX_CHARS = 1000


@dataclass(frozen=True)
class _Evidence:
    """The sources for one answer, renumbered as the answerer numbered them."""

    sources: list[tuple[int, str, str]]
    top_score: float | None
    citation_count: int
    # True when every cited chunk was still on disk and its full text recovered.
    # A citation snippet is a 240-character prefix of what the model actually
    # read, so auditing against snippets alone would mark supported claims
    # unsupported for no reason other than that the evidence was clipped.
    complete: bool


def _count_failure(reason: str) -> None:
    """One bucket per reason a judgement did not get written.

    The reasons are constants in this module, never a message from a provider or
    a model — the point of a label is that somebody can chart it a year from now,
    which needs the set of values to be finite and chosen here.
    """
    metrics.EVALUATION_FAILURES.labels(EVALUATOR_GROUNDEDNESS, reason).inc()


def _judge_agent(*, tenant_id: uuid.UUID, settings: Settings) -> Agent:
    return Agent(
        id=_JUDGE_AGENT_ID,
        # The real tenant, so the judge's token spend is attributable to whoever
        # it was spent auditing rather than disappearing into a platform bucket.
        tenant_id=tenant_id,
        slug="platform-evaluation-judge",
        name="Platform evaluation judge",
        system_prompt=AUDIT_RUBRIC,
        model=settings.evaluation_judge_model,
        collection_id=None,
        tool_allowlist=[],
        temperature=_JUDGE_TEMPERATURE,
        max_output_tokens=settings.evaluation_judge_max_output_tokens,
        retrieval_top_k=0,
        enabled=True,
    )


async def _load_target(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    message_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
) -> tuple[ConversationMessage, Conversation, Agent]:
    """Resolve the message, checking tenancy on the way rather than after.

    `conversation_id` is an additional predicate rather than a separate lookup
    and a comparison afterwards. A caller that addresses one of its own turns
    through the wrong thread gets the same not-found as a caller that invented
    the id, which is the point: any check performed after the row is in hand is
    a check somebody can forget to perform.
    """
    conditions = [
        ConversationMessage.id == message_id,
        # The ownership filter is part of the lookup, so a message id guessed
        # from another tenant is indistinguishable from one that does not exist.
        Conversation.tenant_id == tenant_id,
    ]
    if conversation_id is not None:
        conditions.append(ConversationMessage.conversation_id == conversation_id)

    row = (
        await session.execute(
            select(ConversationMessage, Conversation)
            .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
            .where(*conditions)
        )
    ).first()
    if row is None:
        raise NotFoundError(str(message_id))

    message, conversation = row
    if message.role != _ASSISTANT:
        _count_failure("not_applicable")
        raise EvaluationNotApplicableError("only assistant turns can be judged")

    agent = await session.get(Agent, conversation.agent_id)
    if agent is None:
        raise NotFoundError(str(conversation.agent_id))
    if agent.collection_id is None:
        # A pure-tool agent answers from tool results, which the platform does
        # not retain — there is no evidence to audit against, and a zero here
        # would be a statement about the rubric, not about the agent.
        _count_failure("not_applicable")
        raise EvaluationNotApplicableError("agent has no knowledge scope to be grounded in")

    return message, conversation, agent


async def _load_question(session: AsyncSession, message: ConversationMessage) -> str:
    """The user turn this answer replied to.

    Ordered by `seq` for the reason the column exists: both rows of a turn are
    written in one transaction and can share a timestamp, so a question found by
    time could be the wrong one.
    """
    question = await session.scalar(
        select(ConversationMessage.content)
        .where(
            ConversationMessage.conversation_id == message.conversation_id,
            ConversationMessage.seq < message.seq,
            ConversationMessage.role != _ASSISTANT,
        )
        .order_by(ConversationMessage.seq.desc())
        .limit(1)
    )
    # An assistant turn with nothing before it should not exist, but a transcript
    # is data and the auditor should not crash on a shape it did not expect.
    return question or ""


async def _gather_evidence(session: AsyncSession, message: ConversationMessage) -> _Evidence:
    """Rebuild the numbered sources the answer was written against.

    Full chunk text is read back by `chunk_id` rather than the stored snippet
    being used directly. The snippet is a 240-character prefix — enough to show a
    reader where a claim came from, nowhere near enough to decide whether it is
    supported. Re-reading is safe here because ingestion writes new chunk rows
    rather than editing existing ones, so a surviving id still holds the text the
    model saw; a deleted document loses its chunks, and those citations fall back
    to the frozen snippet, which is exactly what that snippet is for.
    """
    citations = message.citations or []
    if not citations:
        return _Evidence(sources=[], top_score=None, citation_count=0, complete=True)

    chunk_ids = [uuid.UUID(str(c["chunk_id"])) for c in citations if c.get("chunk_id")]
    rows = (
        await session.execute(
            select(DocumentChunk.id, DocumentChunk.content, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
    ).all()
    live = {row[0]: (row[1], row[2]) for row in rows}

    sources: list[tuple[int, str, str]] = []
    complete_evidence = True
    for number, citation in enumerate(citations, start=1):
        raw_id = citation.get("chunk_id")
        chunk_id = uuid.UUID(str(raw_id)) if raw_id else None
        recovered = live.get(chunk_id) if chunk_id else None
        if recovered is None:
            complete_evidence = False
            sources.append((number, "(document removed)", citation.get("snippet", "")))
        else:
            content, filename = recovered
            sources.append((number, filename, content))

    scores = [float(c["score"]) for c in citations if c.get("score") is not None]
    return _Evidence(
        sources=sources,
        # The best match of the search that produced this answer, frozen beside
        # the judgement of it. Two numbers in one row is the whole calibration
        # question: what a retrieval score of 0.62 was actually worth.
        top_score=max(scores) if scores else None,
        citation_count=len(citations),
        complete=complete_evidence,
    )


def _source_numbers(raw: Any) -> list[int]:
    """The source numbers a claim cites, keeping only what is actually a number.

    Models return `[1, 2]`, `["1", "2"]`, and occasionally `["source 1"]` for the
    same idea. The first two are the same answer and both are kept; the third is
    dropped rather than guessed at.
    """
    if not isinstance(raw, list):
        return []
    return [int(s) for s in raw if isinstance(s, int) or str(s).strip().isdigit()]


def _normalise_claims(raw: Sequence[Any], *, limit: int) -> list[dict]:
    """Coerce the judge's list into rows worth storing, discarding what it isn't.

    Everything here is defensive because the input is model output. A claim that
    is not an object, or carries no text, is dropped rather than stored as a
    malformed row that a chart would later have to special-case.
    """
    claims: list[dict] = []
    for entry in raw[:limit]:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("claim", "")).strip()
        if not text:
            continue
        claims.append(
            {
                "claim": text[:_CLAIM_MAX_CHARS],
                "supported": bool(entry.get("supported")),
                "sources": _source_numbers(entry.get("sources")),
            }
        )
    return claims


def _score(claims: Sequence[dict]) -> tuple[float | None, str]:
    """Turn the claim list into a score and a verdict, deterministically."""
    if not claims:
        # No claims is an abstention, and an abstention has no groundedness. See
        # `app/models/evaluation.py` for why this is a null and not a 1.0.
        return None, VERDICT_ABSTAINED

    supported = sum(1 for c in claims if c["supported"])
    score = supported / len(claims)
    if supported == len(claims):
        return score, VERDICT_SUPPORTED
    if supported == 0:
        return score, VERDICT_UNSUPPORTED
    return score, VERDICT_PARTIAL


async def _existing(
    session: AsyncSession, *, message_id: uuid.UUID
) -> TurnEvaluation | None:
    return await session.scalar(
        select(TurnEvaluation).where(
            TurnEvaluation.message_id == message_id,
            TurnEvaluation.evaluator == EVALUATOR_GROUNDEDNESS,
            TurnEvaluation.rubric_version == RUBRIC_VERSION,
        )
    )


async def evaluate_message(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    message_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
    settings: Settings | None = None,
    refresh: bool = False,
) -> TurnEvaluation:
    """Judge one assistant turn, or return the judgement already on file.

    Returning the stored row unless `refresh` is set is what makes a backfill
    cheap to retry: an interrupted run resumes without paying twice for the turns
    it already scored, and the unique constraint means the two cannot disagree
    about which row is current.
    """
    settings = settings or get_settings()
    message, conversation, agent = await _load_target(
        session, tenant_id=tenant_id, message_id=message_id, conversation_id=conversation_id
    )

    stored = await _existing(session, message_id=message_id)
    if stored is not None and not refresh:
        return stored

    question = await _load_question(session, message)
    evidence = await _gather_evidence(session, message)

    started = time.perf_counter()
    with tracing.span(
        "evaluation.groundedness",
        **{
            tracing.TENANT_ID: str(tenant_id),
            tracing.AGENT_ID: str(agent.id),
            tracing.CONVERSATION_ID: str(conversation.id),
            tracing.EVALUATION_CITATIONS: evidence.citation_count,
            tracing.EVALUATION_EVIDENCE_COMPLETE: evidence.complete,
        },
    ) as current:
        try:
            completion = await complete(
                agent=_judge_agent(tenant_id=tenant_id, settings=settings),
                turns=[
                    Turn(
                        role="user",
                        content=build_audit_request(
                            question=question, answer=message.content, sources=evidence.sources
                        ),
                    )
                ],
                # Counted apart from served turns. The judge's calls are long,
                # cheap, and unhurried, and averaging them into the serving path
                # would move the latency an operator pages on without a single
                # answer getting slower.
                workload=metrics.WORKLOAD_EVALUATION,
                settings=settings,
            )
        except Exception:
            _count_failure("provider")
            raise

        try:
            payload = parse_audit_response(completion.text)
        except ValueError as exc:
            logger.warning(
                "evaluation_unparseable",
                message_id=str(message_id),
                judge_model=completion.model,
                error=str(exc),
            )
            _count_failure("unreadable")
            raise EvaluationFailedError(str(exc)) from exc

        claims = _normalise_claims(payload["claims"], limit=settings.evaluation_max_claims)
        score, verdict = _score(claims)
        tracing.set_attributes(
            current,
            **{
                tracing.EVALUATION_VERDICT: verdict,
                tracing.EVALUATION_SCORE: score,
            },
        )

    rationale = str(payload.get("notes") or "")[: settings.evaluation_rationale_max_chars]
    evaluation = stored or TurnEvaluation(
        tenant_id=tenant_id,
        agent_id=agent.id,
        message_id=message_id,
        evaluator=EVALUATOR_GROUNDEDNESS,
        rubric_version=RUBRIC_VERSION,
    )
    evaluation.score = score
    evaluation.verdict = verdict
    evaluation.rationale = rationale or None
    evaluation.claims = claims
    evaluation.retrieval_top_score = evidence.top_score
    evaluation.citation_count = evidence.citation_count
    evaluation.judge_model = completion.model
    evaluation.judge_provider = completion.provider
    evaluation.prompt_tokens = completion.usage.prompt_tokens
    evaluation.completion_tokens = completion.usage.completion_tokens
    evaluation.total_tokens = completion.usage.total_tokens
    evaluation.latency_ms = int((time.perf_counter() - started) * 1000)

    session.add(evaluation)
    await session.commit()
    await session.refresh(evaluation)

    metrics.EVALUATIONS.labels(EVALUATOR_GROUNDEDNESS, verdict).inc()
    metrics.EVALUATION_DURATION.labels(EVALUATOR_GROUNDEDNESS).observe(
        evaluation.latency_ms / 1000
    )
    if score is not None:
        # Abstentions are deliberately absent rather than recorded as 1.0 — see
        # the histogram's own comment in `app/core/metrics.py`.
        metrics.EVALUATION_SCORE.labels(EVALUATOR_GROUNDEDNESS).observe(score)

    logger.info(
        "turn_evaluated",
        tenant_id=str(tenant_id),
        agent_id=str(agent.id),
        conversation_id=str(conversation.id),
        message_id=str(message_id),
        verdict=verdict,
        score=score,
        claims=len(claims),
        citations=evidence.citation_count,
        evidence_complete=evidence.complete,
        retrieval_top_score=evidence.top_score,
        judge_model=completion.model,
        total_tokens=completion.usage.total_tokens,
    )
    return evaluation


async def evaluate_conversation(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    conversation_id: uuid.UUID,
    settings: Settings | None = None,
    refresh: bool = False,
) -> list[TurnEvaluation]:
    """Judge every assistant turn in a thread, oldest first.

    Sequentially rather than gathered: this is background work with no user
    waiting on it, and a burst of concurrent judge calls would compete for
    provider rate limit with the serving path, which does have someone waiting.
    """
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None or conversation.tenant_id != tenant_id:
        raise NotFoundError(str(conversation_id))

    message_ids = list(
        (
            await session.scalars(
                select(ConversationMessage.id)
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.role == _ASSISTANT,
                )
                .order_by(ConversationMessage.seq)
            )
        ).all()
    )

    evaluations = []
    for message_id in message_ids:
        evaluations.append(
            await evaluate_message(
                session,
                tenant_id=tenant_id,
                message_id=message_id,
                settings=settings,
                refresh=refresh,
            )
        )
    return evaluations


async def list_evaluations(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    message_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
) -> list[TurnEvaluation]:
    """Every judgement on file for one turn, newest first.

    Scoped by conversation when the caller names one, for the same reason the
    write path is: a turn addressed through a thread it does not belong to must
    read as absent rather than as somebody else's row served under the wrong
    path.
    """
    conditions = [
        TurnEvaluation.tenant_id == tenant_id,
        TurnEvaluation.message_id == message_id,
    ]
    if conversation_id is not None:
        conditions.append(
            TurnEvaluation.message_id.in_(
                select(ConversationMessage.id).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            )
        )

    result = await session.scalars(
        select(TurnEvaluation).where(*conditions).order_by(TurnEvaluation.created_at.desc())
    )
    return list(result.all())
