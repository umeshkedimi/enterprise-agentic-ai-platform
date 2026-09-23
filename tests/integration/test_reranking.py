"""Reranking, real retrieval underneath a scripted model response.

Reuses the platform's own `VACATION_TEXT`/`EXPENSES_TEXT` pair and a
"vacation days" query — already the proven case elsewhere in this suite for
a deterministic, non-tied baseline order under the deterministic test
embedding. The reorder itself is the only thing scripted here; real upload,
real chunking, and real fusion produce the pre-rerank shortlist, and only the
reranker's own `litellm.acompletion` call is faked.
"""

import json
import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services.retrieval_service import semantic_search
from tests.integration.conftest import (
    EXPENSES_TEXT,
    VACATION_TEXT,
    create_collection,
    upload_document,
)

QUERY = "vacation days"


@pytest.fixture
def reranking_settings(monkeypatch):
    def _set(*, enabled: bool) -> None:
        monkeypatch.setenv("RERANKING_ENABLED", "true" if enabled else "false")
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def scripted_reranker(monkeypatch):
    """Fake the chat provider and return whatever `order` is queued next."""
    calls: list[dict] = []
    queue: list[str] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=queue.pop(0)))],
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=10, total_tokens=160),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls, queue


async def _seed(client, collection_id) -> tuple[str, str]:
    vacation_id = await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    expenses_id = await upload_document(client, collection_id, "expenses.txt", EXPENSES_TEXT)
    return vacation_id, expenses_id


async def test_reranking_overrides_the_fused_order(
    authed_client, fake_embeddings, reranking_settings, scripted_reranker
):
    client, tenant_id = authed_client
    collection_id = await create_collection(client, "hr")
    await _seed(client, collection_id)

    async with async_session_factory() as session:
        # Reranking is off by default, so this is the genuine pre-rerank
        # order — the vacation question ranking the vacation document first,
        # the same property `conftest.fake_embedding` is documented to hold.
        baseline = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=2
        )
    assert baseline[0].filename == "vacation.txt"

    reranking_settings(enabled=True)
    _, queue = scripted_reranker
    queue.append(json.dumps({"order": [2, 1]}))  # reverse whatever fusion produced

    async with async_session_factory() as session:
        reranked = await semantic_search(
            session,
            QUERY,
            collection_id=uuid.UUID(collection_id),
            top_k=2,
            tenant_id=tenant_id,
        )

    assert reranked[0].filename == "expenses.txt"
    assert reranked[1].filename == "vacation.txt"


async def test_an_unparseable_rerank_response_falls_back_to_the_fused_order(
    authed_client, fake_embeddings, reranking_settings, scripted_reranker
):
    client, tenant_id = authed_client
    collection_id = await create_collection(client, "hr")
    await _seed(client, collection_id)

    async with async_session_factory() as session:
        baseline = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=2
        )

    reranking_settings(enabled=True)
    _, queue = scripted_reranker
    queue.append("this is not JSON")

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session,
            QUERY,
            collection_id=uuid.UUID(collection_id),
            top_k=2,
            tenant_id=tenant_id,
        )

    # Degrades, doesn't fail the search — the same order it would have
    # returned without reranking at all.
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in baseline]


async def test_reranking_off_by_default_never_calls_the_provider(
    authed_client, fake_embeddings, scripted_reranker
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await _seed(client, collection_id)

    calls, _ = scripted_reranker

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=1
        )

    assert chunks
    assert calls == []


async def test_the_benchmark_harness_reflects_a_reranked_order(
    authed_client, fake_embeddings, reranking_settings, scripted_reranker
):
    """The same validation loop Chunks 10, 12, and 13 used: rerun the Chunk 8
    harness with a reranker that demotes the actually-relevant document, and
    watch `mrr` drop from a scripted reorder — proving the harness reads
    reranking's effect on ranking quality, not just on membership in top_k."""
    client, tenant_id = authed_client
    collection_id = await create_collection(client, "hr")
    vacation_id, _ = await _seed(client, collection_id)

    golden = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={"query": QUERY, "relevant_document_ids": [vacation_id]},
    )
    assert golden.status_code == 201, golden.text

    baseline = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "baseline", "top_k": 2},
    )
    assert baseline.json()["mrr"] == 1.0

    reranking_settings(enabled=True)
    _, queue = scripted_reranker
    queue.append(json.dumps({"order": [2, 1]}))  # demote the relevant document to #2

    reranked = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "reranked-worse", "top_k": 2},
    )
    assert reranked.status_code == 201, reranked.text
    assert reranked.json()["mrr"] == 0.5
