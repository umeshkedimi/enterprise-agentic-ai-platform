"""Read the audit trail, and ask for turns to be judged.

Its own router rather than more routes on `app/api/agents.py`, which is already
the largest surface in the platform. The paths still nest under an agent and a
conversation because that is where the resource lives — an evaluation is *of* a
turn — but the code that serves them has nothing in common with configuring or
running an agent.

Everything here is tenant-scoped through the same `get_current_tenant` dependency
as the rest of the API. An evaluation quotes a turn's claims, which quote a
tenant's documents, so the ownership check is not a formality on this router: it
is the same content boundary the retrieval layer spends its effort on, one hop
further downstream.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_tenant
from app.db.session import get_db_session
from app.models.evaluation import TurnEvaluation
from app.models.schemas import (
    CalibrationBucketResponse,
    CalibrationReportResponse,
    EvaluationClaim,
    EvaluationRunRequest,
    FloorRecommendationResponse,
    TokenUsageResponse,
    TurnEvaluationResponse,
)
from app.models.tenant import Tenant
from app.services import agent_service, calibration_service, evaluation_service
from app.services.calibration_service import CalibrationReport
from app.services.errors import (
    DomainError,
    EvaluationFailedError,
    EvaluationNotApplicableError,
    ModelConfigurationError,
    NotFoundError,
    ProviderNotConfiguredError,
)

router = APIRouter(tags=["evaluation"])

_NOT_FOUND = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")


def _to_response(evaluation: TurnEvaluation) -> TurnEvaluationResponse:
    return TurnEvaluationResponse(
        id=evaluation.id,
        message_id=evaluation.message_id,
        evaluator=evaluation.evaluator,
        rubric_version=evaluation.rubric_version,
        score=evaluation.score,
        verdict=evaluation.verdict,  # type: ignore[arg-type]
        rationale=evaluation.rationale,
        claims=[EvaluationClaim(**c) for c in evaluation.claims],
        retrieval_top_score=evaluation.retrieval_top_score,
        citation_count=evaluation.citation_count,
        judge_model=evaluation.judge_model,
        judge_provider=evaluation.judge_provider,
        usage=TokenUsageResponse(
            prompt_tokens=evaluation.prompt_tokens,
            completion_tokens=evaluation.completion_tokens,
            total_tokens=evaluation.total_tokens,
        ),
        latency_ms=evaluation.latency_ms,
        created_at=evaluation.created_at,
    )


def _to_report_response(report: CalibrationReport) -> CalibrationReportResponse:
    return CalibrationReportResponse(
        agent_id=report.agent_id,
        rubric_version=report.rubric_version,
        evaluations=report.evaluations,
        graded=report.graded,
        mean_score=report.mean_score,
        buckets=[CalibrationBucketResponse(**vars(b)) for b in report.buckets],
        recommendation=FloorRecommendationResponse(**vars(report.recommendation)),
    )


def _to_http_error(exc: DomainError) -> HTTPException:
    """Map an evaluation failure onto the code that says whose problem it is.

    One mapping here differs from the serving path and the difference is the
    point: a `ModelConfigurationError` on `/chat` is 400, because the tenant
    typed the model name into their own agent. Here it is 503, because the model
    was named by the operator in `EVALUATION_JUDGE_MODEL` and there is nothing
    the tenant can do about it. The same exception, a different party at fault,
    and a status code that says so.
    """
    if isinstance(exc, EvaluationNotApplicableError):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc) or "Turn cannot be evaluated.")
    if isinstance(exc, ModelConfigurationError | ProviderNotConfiguredError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The evaluation judge is not correctly configured.",
        )
    if isinstance(exc, EvaluationFailedError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, "The judge returned no usable verdict.")
    # The upstream provider's message is withheld for the reason it is withheld
    # on the serving path: a provider error echoes the prompt, and this prompt
    # holds retrieved document text.
    return HTTPException(status.HTTP_502_BAD_GATEWAY, "Evaluation request failed.")


async def _require_agent(session: AsyncSession, *, tenant: Tenant, agent_id: uuid.UUID):
    try:
        return await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc


@router.post(
    "/agents/{agent_id}/conversations/{conversation_id}/messages/{message_id}/evaluations",
    response_model=TurnEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def evaluate_turn(
    agent_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    body: EvaluationRunRequest | None = None,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> TurnEvaluationResponse:
    """Judge one assistant turn against the evidence it was given.

    Synchronous, and it calls a model — so this is a slow endpoint by nature, not
    one that has been left unoptimised. Repeating it is cheap: without `refresh`
    it returns the judgement already on file rather than paying for a second one.
    """
    await _require_agent(session, tenant=tenant, agent_id=agent_id)
    body = body or EvaluationRunRequest()

    try:
        evaluation = await evaluation_service.evaluate_message(
            session,
            tenant_id=tenant.id,
            message_id=message_id,
            conversation_id=conversation_id,
            refresh=body.refresh,
        )
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
    except DomainError as exc:
        raise _to_http_error(exc) from exc

    return _to_response(evaluation)


@router.post(
    "/agents/{agent_id}/conversations/{conversation_id}/evaluations",
    response_model=list[TurnEvaluationResponse],
    status_code=status.HTTP_201_CREATED,
)
async def evaluate_conversation(
    agent_id: uuid.UUID,
    conversation_id: uuid.UUID,
    body: EvaluationRunRequest | None = None,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> list[TurnEvaluationResponse]:
    """Judge every assistant turn in a thread, oldest first.

    One model call per turn, run sequentially — a long thread takes a long time.
    A queue is the right home for this and is not built here; what is built is
    the property that makes a queue easy to add later, which is that re-running
    an evaluation is idempotent.
    """
    await _require_agent(session, tenant=tenant, agent_id=agent_id)
    body = body or EvaluationRunRequest()

    try:
        evaluations = await evaluation_service.evaluate_conversation(
            session, tenant_id=tenant.id, conversation_id=conversation_id, refresh=body.refresh
        )
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
    except DomainError as exc:
        raise _to_http_error(exc) from exc

    return [_to_response(e) for e in evaluations]


@router.get(
    "/agents/{agent_id}/conversations/{conversation_id}/messages/{message_id}/evaluations",
    response_model=list[TurnEvaluationResponse],
)
async def list_turn_evaluations(
    agent_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> list[TurnEvaluationResponse]:
    """Every judgement on file for one turn — one per rubric version.

    Reads only. A turn nobody has evaluated returns an empty list rather than a
    404: the turn exists, and "not judged yet" is a fact about the audit trail,
    not a missing resource.
    """
    await _require_agent(session, tenant=tenant, agent_id=agent_id)
    evaluations = await evaluation_service.list_evaluations(
        session, tenant_id=tenant.id, message_id=message_id, conversation_id=conversation_id
    )
    return [_to_response(e) for e in evaluations]


@router.get("/agents/{agent_id}/calibration", response_model=CalibrationReportResponse)
async def agent_calibration(
    agent_id: uuid.UUID,
    target: float = calibration_service.DEFAULT_TARGET_GROUNDEDNESS,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> CalibrationReportResponse:
    """What this agent's retrieval scores turned out to be worth."""
    agent = await _require_agent(session, tenant=tenant, agent_id=agent_id)
    report = await calibration_service.calibration_report(
        session, tenant_id=tenant.id, agent_id=agent.id, target=target
    )
    return _to_report_response(report)


@router.get("/evaluations/calibration", response_model=CalibrationReportResponse)
async def tenant_calibration(
    target: float = calibration_service.DEFAULT_TARGET_GROUNDEDNESS,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> CalibrationReportResponse:
    """The same reading across every agent in the tenant.

    Worth having separately because a floor is a property of the embedding model
    and the corpus rather than of one assistant, and a single agent rarely serves
    enough judged turns to fill the buckets on its own.
    """
    report = await calibration_service.calibration_report(
        session, tenant_id=tenant.id, target=target
    )
    return _to_report_response(report)
