"""Record ground truth for a collection's retrieval, and score it against it.

Its own router for the same reason `evaluations.py` is: this has nothing in
common with configuring a collection or an agent, even though every route nests
under `/collections/{collection_id}` because that is where the resource lives.
Tenant ownership of the collection is checked in the service layer on every
call, never assumed from the id in the path.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import PageParams, get_current_tenant, page_params
from app.db.session import get_db_session
from app.models.retrieval_eval import RetrievalBenchmarkRun, RetrievalGoldenExample
from app.models.schemas import (
    GoldenExampleCreate,
    GoldenExampleResponse,
    Page,
    RetrievalBenchmarkCaseResult,
    RetrievalBenchmarkRunResponse,
    RunBenchmarkRequest,
)
from app.models.tenant import Tenant
from app.services import retrieval_eval_service
from app.services.errors import NoGoldenExamplesError, NotFoundError

router = APIRouter(tags=["retrieval-evaluation"])

_COLLECTION_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
)


def _example_response(example: RetrievalGoldenExample) -> GoldenExampleResponse:
    return GoldenExampleResponse(
        id=example.id,
        collection_id=example.collection_id,
        query=example.query,
        relevant_document_ids=[uuid.UUID(d) for d in example.relevant_document_ids],
        notes=example.notes,
        created_at=example.created_at,
    )


def _run_response(run: RetrievalBenchmarkRun) -> RetrievalBenchmarkRunResponse:
    return RetrievalBenchmarkRunResponse(
        id=run.id,
        collection_id=run.collection_id,
        label=run.label,
        top_k=run.top_k,
        cases=run.cases,
        mean_recall=run.mean_recall,
        mrr=run.mrr,
        results=[
            RetrievalBenchmarkCaseResult(
                golden_example_id=uuid.UUID(r["golden_example_id"]),
                query=r["query"],
                relevant_document_ids=[uuid.UUID(d) for d in r["relevant_document_ids"]],
                retrieved_document_ids=[uuid.UUID(d) for d in r["retrieved_document_ids"]],
                recall=r["recall"],
                reciprocal_rank=r["reciprocal_rank"],
            )
            for r in run.results
        ],
        created_at=run.created_at,
    )


@router.post(
    "/collections/{collection_id}/golden-examples",
    response_model=GoldenExampleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_golden_example(
    collection_id: uuid.UUID,
    body: GoldenExampleCreate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> GoldenExampleResponse:
    try:
        example = await retrieval_eval_service.create_golden_example(
            session,
            tenant_id=tenant.id,
            collection_id=collection_id,
            query=body.query,
            relevant_document_ids=body.relevant_document_ids,
            notes=body.notes,
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc
    return _example_response(example)


@router.get(
    "/collections/{collection_id}/golden-examples",
    response_model=Page[GoldenExampleResponse],
)
async def list_golden_examples(
    collection_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[GoldenExampleResponse]:
    try:
        rows, has_more = await retrieval_eval_service.list_golden_examples(
            session,
            tenant_id=tenant.id,
            collection_id=collection_id,
            limit=page.limit,
            offset=page.offset,
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc
    return Page(
        items=[_example_response(e) for e in rows],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


@router.delete("/golden-examples/{golden_example_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_golden_example(
    golden_example_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    deleted = await retrieval_eval_service.delete_golden_example(
        session, tenant_id=tenant.id, golden_example_id=golden_example_id
    )
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Golden example not found.")


@router.post(
    "/collections/{collection_id}/retrieval-benchmark",
    response_model=RetrievalBenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def run_retrieval_benchmark(
    collection_id: uuid.UUID,
    body: RunBenchmarkRequest,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> RetrievalBenchmarkRunResponse:
    """Run every golden example on file through live retrieval, and score it.

    Synchronous by nature — it runs one embedding + search per case, so a
    collection with a hundred golden examples takes as long as a hundred
    searches. There is no serving traffic to protect here, so unlike the judge
    it is not worth a queue.
    """
    try:
        run = await retrieval_eval_service.run_benchmark(
            session,
            tenant_id=tenant.id,
            collection_id=collection_id,
            label=body.label,
            top_k=body.top_k,
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc
    except NoGoldenExamplesError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This collection has no golden examples to benchmark against.",
        ) from exc
    return _run_response(run)


@router.get(
    "/collections/{collection_id}/retrieval-benchmark",
    response_model=Page[RetrievalBenchmarkRunResponse],
)
async def list_retrieval_benchmark_runs(
    collection_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[RetrievalBenchmarkRunResponse]:
    try:
        rows, has_more = await retrieval_eval_service.list_benchmark_runs(
            session,
            tenant_id=tenant.id,
            collection_id=collection_id,
            limit=page.limit,
            offset=page.offset,
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc
    return Page(
        items=[_run_response(r) for r in rows],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )
