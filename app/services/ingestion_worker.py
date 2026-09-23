"""Claims queued documents and processes them, off the request path.

Ingestion used to run inside `POST /collections/{id}/documents` — extract,
chunk, embed, store, all before the response could be sent. That was the last
place this platform violated its own stated rule that nothing on a request
path waits on a model call (the same rule Chunk 7 enforced for the
evaluation judge, and Chunk 4 enforced for the checkpointer). A 25 MB PDF held
a database session and a provider connection for as long as extraction and
embedding took; the size cap bounded the damage, it never fixed the shape.

The queue is Postgres itself — `SELECT ... FOR UPDATE SKIP LOCKED` over
`documents.status = 'uploaded'` — not Celery, not a broker. Postgres is
already this platform's one hard dependency, and a broker earns its place at
a throughput this platform does not have; adding one here would be
infrastructure bought for a load that was never measured. `SKIP LOCKED` is
what makes the same query safe to run from more than one worker process at
once: a row another worker already claimed is invisible to this one rather
than something to queue behind.

Known, accepted gap: a worker that crashes between claiming a document
(flipping it to PROCESSING) and finishing it leaves that row stuck — nothing
currently re-queues an orphaned PROCESSING row. At this platform's scale, a
stuck document is visible (the same `status`/`error_message` a real failure
would show) and rare enough that automatic reclamation would be complexity
bought for a failure mode nobody has hit yet. Worth revisiting only once that
stops being true.
"""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.services.document_service import process_document

logger = get_logger(__name__)


async def claim_next_document(session: AsyncSession) -> Document | None:
    """Atomically claim the oldest queued document, or return `None` if
    there is nothing to do.

    The claim — flipping status to PROCESSING — is committed here, before
    this returns, so the row is visibly "being worked on" the instant it is
    taken rather than only once processing finishes.
    """
    document = await session.scalar(
        select(Document)
        .where(Document.status == DocumentStatus.UPLOADED)
        .order_by(Document.uploaded_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if document is None:
        return None
    document.status = DocumentStatus.PROCESSING
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def process_next_document(session_factory: async_sessionmaker[AsyncSession]) -> bool:
    """Claim and process one document, in its own session.

    The worker cannot use a caller's session — same rule streaming already
    established for the same reason: this runs in a different process, at a
    different time, than whatever request created the row it is about to
    read. Returns whether there was anything to do, which is what lets
    `run_worker` (and a test driving the queue directly) tell "processed
    one" apart from "queue was empty" without inspecting anything else.
    """
    async with session_factory() as session:
        document = await claim_next_document(session)
        if document is None:
            return False
        chunk_count = await process_document(session, document)
    logger.info(
        "document_ingested",
        document_id=str(document.id),
        status=document.status,
        chunks=chunk_count,
    )
    return True


async def run_worker(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    poll_interval_seconds: float,
    max_iterations: int | None = None,
) -> None:
    """Poll for queued documents, processing one at a time, forever.

    One document at a time, not gathered: a burst of concurrent embedding
    calls from a single worker would compete with itself for whatever rate
    limit the provider enforces, and nothing is waiting on this to hurry —
    a queue a few documents deep drains in the time it drains.

    `max_iterations` exists for tests: a bounded run that proves the loop
    itself — claim, process, sleep only when the queue was empty — actually
    works, without needing a real background process alive for the length
    of a test session. Production never sets it, and the loop does not
    return.
    """
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        processed = await process_next_document(session_factory)
        iterations += 1
        if not processed:
            await asyncio.sleep(poll_interval_seconds)
