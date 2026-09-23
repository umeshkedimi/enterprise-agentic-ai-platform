"""Async ingestion: upload only queues, and the worker is what actually reads it.

`document_service`'s own tests (`test_documents.py`, `test_document_versioning.py`)
already cover what ingestion produces. This file is about the *shape* of getting
there: the request returns before any of that happens, a half-ingested document
is invisible to retrieval for free, and two workers polling the same queue never
claim the same row.
"""

import asyncio
import io
import uuid

from app.db.session import async_session_factory
from app.models.document import Document
from app.services.ingestion_worker import claim_next_document, process_next_document, run_worker
from app.services.retrieval_service import semantic_search
from tests.integration.conftest import create_collection, get_document, process_queued_documents

TEXT = b"Full-time employees accrue twenty-five days of paid annual leave each year."


async def _queue_upload(client, collection_id: str, filename: str = "policy.txt") -> str:
    """Upload without draining the queue — the raw, unprocessed 202 state."""
    r = await client.post(
        f"/collections/{collection_id}/documents",
        files={"file": (filename, io.BytesIO(TEXT), "text/plain")},
    )
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "uploaded"
    assert r.json()["chunk_count"] == 0
    return r.json()["id"]


async def test_upload_returns_before_any_ingestion_happens(authed_client, fake_embeddings):
    """The entire request-path cost of ingestion, post-Chunk-15: a row and a
    write, nothing that reads the file's content at all."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await _queue_upload(client, collection_id)
    # No assertion beyond the 202/uploaded/chunk_count==0 already checked in
    # `_queue_upload` — the point is that none of that required a worker to
    # have run at all.


async def test_a_queued_document_is_invisible_to_retrieval(authed_client, fake_embeddings):
    """No isolation work needed to make this safe: `semantic_search` already
    filters on `status == READY`, so a document still sitting in the queue
    simply doesn't exist yet as far as retrieval is concerned."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await _queue_upload(client, collection_id)

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, "vacation days", collection_id=uuid.UUID(collection_id)
        )
    assert chunks == []


async def test_the_worker_claims_and_processes_a_queued_document(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    document_id = await _queue_upload(client, collection_id)

    processed = await process_next_document(async_session_factory)
    assert processed is True

    entry = await get_document(client, collection_id, document_id)
    assert entry["status"] == "ready"
    assert entry["chunk_count"] > 0


async def test_processing_an_empty_queue_returns_false_and_does_nothing(
    authed_client, fake_embeddings
):
    processed = await process_next_document(async_session_factory)
    assert processed is False


async def test_raw_content_is_cleared_once_a_document_reaches_a_terminal_status(
    authed_client, fake_embeddings
):
    """The uploaded bytes exist only to get the document from queued to
    chunked — once that has happened, keeping them around serves nothing and
    only grows the table."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    document_id = await _queue_upload(client, collection_id)

    async with async_session_factory() as session:
        before = await session.get(Document, uuid.UUID(document_id))
        assert before.raw_content is not None

    await process_next_document(async_session_factory)

    async with async_session_factory() as session:
        after = await session.get(Document, uuid.UUID(document_id))
        assert after.raw_content is None


async def test_run_worker_processes_queued_work_within_a_bounded_number_of_iterations(
    authed_client, fake_embeddings
):
    """The real continuous loop, not just the one-shot claim/process function
    the other tests use — bounded so the test itself terminates."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    document_id = await _queue_upload(client, collection_id)

    await run_worker(async_session_factory, poll_interval_seconds=0.01, max_iterations=1)

    entry = await get_document(client, collection_id, document_id)
    assert entry["status"] == "ready"


async def test_two_workers_never_claim_the_same_document(authed_client, fake_embeddings):
    """`FOR UPDATE SKIP LOCKED` is the actual safety property async ingestion
    depends on for correctness under more than one worker process — proven
    directly: two claims racing for one queued row, run concurrently rather
    than sequentially, so the second one genuinely has to skip a row the
    first is still holding rather than simply finding it already gone."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    document_id = await _queue_upload(client, collection_id)

    async def _claim():
        async with async_session_factory() as session:
            claimed = await claim_next_document(session)
            return claimed.id if claimed else None

    first, second = await asyncio.gather(_claim(), _claim())
    claimed_ids = {first, second} - {None}

    assert claimed_ids == {uuid.UUID(document_id)}
    assert (first is None) != (second is None)


async def test_a_failed_document_never_blocks_the_next_one(authed_client, fake_embeddings):
    """One bad document in the queue costs itself, never the ones behind it."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    bad = await client.post(
        f"/collections/{collection_id}/documents",
        files={"file": ("empty.txt", io.BytesIO(b"     "), "text/plain")},
    )
    assert bad.status_code == 202, bad.text
    good_id = await _queue_upload(client, collection_id, "policy.txt")

    await process_queued_documents()

    bad_entry = await get_document(client, collection_id, bad.json()["id"])
    good_entry = await get_document(client, collection_id, good_id)
    assert bad_entry["status"] == "failed"
    assert good_entry["status"] == "ready"
