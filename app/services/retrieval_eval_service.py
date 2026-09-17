"""Score a collection's retrieval against a fixed set of questions with known answers.

This is the sibling of `evaluation_service.py`, one layer upstream. That module
judges whether a *served answer* was grounded in what was retrieved; this one
judges whether retrieval itself found the right thing to begin with. They stay
separate because they answer to different clocks in the same way `evaluation.py`
already explains for judged turns versus served ones: a golden example is
written once, by a human who knows the right answer, and a benchmark run reads
the collection as it stands today — it never waits on a served conversation to
exist.

Ground truth here is document-level, never chunk-level; see
`app/models/retrieval_eval.py` for why. A run's metrics — mean recall and mean
reciprocal rank — are plain arithmetic over the per-case results, recomputable
by anyone reading the stored row, the same discipline the groundedness judge
uses for its own score.
"""

import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Collection
from app.models.retrieval_eval import RetrievalBenchmarkRun, RetrievalGoldenExample
from app.services.errors import NoGoldenExamplesError, NotFoundError
from app.services.pagination import DEFAULT_PAGE_LIMIT, paginate, split_page
from app.services.retrieval_service import DEFAULT_TOP_K, RetrievedChunk, semantic_search

logger = get_logger(__name__)


@dataclass(frozen=True)
class _CaseResult:
    recall: float
    reciprocal_rank: float
    retrieved_document_ids: list[str]


async def _assert_collection_in_tenant(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID
) -> None:
    collection = await session.get(Collection, collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        raise NotFoundError(f"collection {collection_id}")


async def create_golden_example(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    query: str,
    relevant_document_ids: list[uuid.UUID],
    notes: str | None = None,
) -> RetrievalGoldenExample:
    await _assert_collection_in_tenant(session, tenant_id=tenant_id, collection_id=collection_id)

    example = RetrievalGoldenExample(
        tenant_id=tenant_id,
        collection_id=collection_id,
        query=query,
        relevant_document_ids=[str(d) for d in relevant_document_ids],
        notes=notes,
    )
    session.add(example)
    await session.commit()
    await session.refresh(example)
    return example


async def list_golden_examples(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> tuple[list[RetrievalGoldenExample], bool]:
    await _assert_collection_in_tenant(session, tenant_id=tenant_id, collection_id=collection_id)
    stmt = paginate(
        select(RetrievalGoldenExample)
        .where(RetrievalGoldenExample.collection_id == collection_id)
        .order_by(RetrievalGoldenExample.created_at.desc()),
        limit=limit,
        offset=offset,
    )
    result = await session.execute(stmt)
    return split_page(list(result.scalars().all()), limit)


async def delete_golden_example(
    session: AsyncSession, *, tenant_id: uuid.UUID, golden_example_id: uuid.UUID
) -> bool:
    """Delete a golden example, but only if it belongs to the calling tenant.

    Ownership checked by tenant_id on the row itself (denormalised at write
    time), not by a join through the collection — same reasoning as
    `TurnEvaluation.tenant_id`: the filter on every read has to be a predicate
    on this table, not a join a future query could forget to write.
    """
    example = await session.get(RetrievalGoldenExample, golden_example_id)
    if example is None or example.tenant_id != tenant_id:
        return False
    await session.delete(example)
    await session.commit()
    return True


def _score_case(chunks: list[RetrievedChunk], relevant_document_ids: set[str]) -> _CaseResult:
    """Recall and reciprocal rank for one query's retrieved chunks.

    Chunks are mapped onto their document, since ground truth is document-level.
    A document with several chunks in the top-k is only counted once — recall
    asks how many *relevant documents* were found, not how many of their chunks.
    """
    retrieved_document_ids = [str(c.document_id) for c in chunks]

    found: set[str] = set()
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_document_ids, start=1):
        if doc_id in relevant_document_ids and doc_id not in found:
            found.add(doc_id)
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    recall = len(found) / len(relevant_document_ids)
    return _CaseResult(
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        retrieved_document_ids=retrieved_document_ids,
    )


async def run_benchmark(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    label: str,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalBenchmarkRun:
    """Run every golden example on file for a collection through live retrieval.

    Sequential, not gathered: this is an on-demand diagnostic a person is
    waiting on, not background work, and there is no serving traffic here to
    protect from contention the way `evaluate_conversation` protects it from a
    burst of judge calls.
    """
    await _assert_collection_in_tenant(session, tenant_id=tenant_id, collection_id=collection_id)

    examples = (
        await session.scalars(
            select(RetrievalGoldenExample)
            .where(RetrievalGoldenExample.collection_id == collection_id)
            .order_by(RetrievalGoldenExample.created_at)
        )
    ).all()
    if not examples:
        raise NoGoldenExamplesError(str(collection_id))

    started = time.perf_counter()
    results: list[dict] = []
    for example in examples:
        chunks = await semantic_search(
            session, example.query, collection_id=collection_id, top_k=top_k
        )
        case = _score_case(chunks, set(example.relevant_document_ids))
        results.append(
            {
                "golden_example_id": str(example.id),
                "query": example.query,
                "relevant_document_ids": example.relevant_document_ids,
                "retrieved_document_ids": case.retrieved_document_ids,
                "recall": case.recall,
                "reciprocal_rank": case.reciprocal_rank,
            }
        )

    mean_recall = sum(r["recall"] for r in results) / len(results)
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results)

    run = RetrievalBenchmarkRun(
        tenant_id=tenant_id,
        collection_id=collection_id,
        label=label,
        top_k=top_k,
        cases=len(results),
        mean_recall=mean_recall,
        mrr=mrr,
        results=results,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    logger.info(
        "retrieval_benchmark_run",
        tenant_id=str(tenant_id),
        collection_id=str(collection_id),
        label=label,
        top_k=top_k,
        cases=len(results),
        mean_recall=mean_recall,
        mrr=mrr,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return run


async def list_benchmark_runs(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> tuple[list[RetrievalBenchmarkRun], bool]:
    await _assert_collection_in_tenant(session, tenant_id=tenant_id, collection_id=collection_id)
    stmt = paginate(
        select(RetrievalBenchmarkRun)
        .where(RetrievalBenchmarkRun.collection_id == collection_id)
        .order_by(RetrievalBenchmarkRun.created_at.desc()),
        limit=limit,
        offset=offset,
    )
    result = await session.execute(stmt)
    return split_page(list(result.scalars().all()), limit)
