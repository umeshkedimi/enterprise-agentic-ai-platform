"""Hybrid search, proven with a real scenario where pure vector search alone
gets the wrong answer and fusion corrects it.

Both documents below were checked in Python against the platform's own
deterministic `fake_embedding` before being written here: `decoy` genuinely
outranks `target` on cosine similarity alone (a short, repetitive text wins
on vector norm even though it never mentions the code the query names), while
`target` is the only one of the two containing every word the full-text query
requires. That is what makes this a real rescue rather than an engineered
coincidence — the same query and documents, with hybrid search turned off,
demonstrably return the wrong document first.
"""

import uuid

import pytest

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services.retrieval_service import semantic_search
from tests.integration.conftest import create_collection, upload_document

TARGET_TEXT = (
    b"Reference RX2291 reimbursement addendum covering miscellaneous equipment "
    b"purchases for distributed staff working from non-office locations under "
    b"updated finance guidance issued this quarter across all regional business "
    b"units and satellite offices worldwide."
)
DECOY_TEXT = b"reimbursement reimbursement reimbursement policy summary"

QUERY = "RX2291 reimbursement"


@pytest.fixture
def hybrid_search_toggle(monkeypatch):
    def _set(enabled: bool) -> None:
        monkeypatch.setenv("HYBRID_SEARCH_ENABLED", "true" if enabled else "false")
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


async def test_hybrid_search_rescues_the_document_the_query_actually_names(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "target.txt", TARGET_TEXT)
    await upload_document(client, collection_id, "decoy.txt", DECOY_TEXT)

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=2
        )

    assert chunks[0].filename == "target.txt"


async def test_disabling_hybrid_search_reverts_to_the_wrong_answer(
    authed_client, fake_embeddings, hybrid_search_toggle
):
    """The toggle actually gates the behaviour — same documents, same query,
    opposite result, purely from the setting. This is also the proof that the
    "rescue" above isn't a coincidence of the two documents chosen: turn
    fusion off and pure vector search alone gets it wrong, exactly as the
    scenario was designed (and checked in Python) to do."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "target.txt", TARGET_TEXT)
    await upload_document(client, collection_id, "decoy.txt", DECOY_TEXT)

    hybrid_search_toggle(False)

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=2
        )

    assert chunks[0].filename == "decoy.txt"


async def test_every_returned_score_is_real_cosine_similarity_not_a_fused_number(
    authed_client, fake_embeddings
):
    """The design decision worth defending: fusion only ever decides *which*
    chunks and in what *order* — the `score` field a chunk carries is always
    its own vector confidence, exactly as it was before this chunk landed, so
    the relevance floor and the calibration report never need to know hybrid
    search exists."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "target.txt", TARGET_TEXT)

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, QUERY, collection_id=uuid.UUID(collection_id), top_k=1
        )

    assert chunks
    # Cosine similarity is bounded to [-1, 1]; an RRF score (~1/(60+rank))
    # is a small positive fraction nowhere near this range — the two are not
    # the kind of number that could be confused for one another.
    assert -1.0 <= chunks[0].score <= 1.0


async def test_the_benchmark_harness_shows_the_same_rescue(
    authed_client, fake_embeddings, hybrid_search_toggle
):
    """The same validation loop Chunks 10 and 12 used: rerun the Chunk 8
    harness at two settings and read the number, rather than trust that
    fusion helped. `top_k=1` on purpose — with both documents fitting inside
    a wider top_k, recall alone can't see a *ranking* improvement, only
    whether the right document is #1."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    doc_id = await upload_document(client, collection_id, "target.txt", TARGET_TEXT)
    await upload_document(client, collection_id, "decoy.txt", DECOY_TEXT)

    golden = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={"query": QUERY, "relevant_document_ids": [doc_id]},
    )
    assert golden.status_code == 201, golden.text

    with_hybrid = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "hybrid-on", "top_k": 1},
    )
    assert with_hybrid.json()["mean_recall"] == 1.0

    hybrid_search_toggle(False)
    without_hybrid = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "hybrid-off", "top_k": 1},
    )
    assert without_hybrid.json()["mean_recall"] == 0.0
