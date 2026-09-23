import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Collection
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.chunking import XLSX_CONTENT_TYPE, chunk_text, count_tokens, extract_text
from app.services.embedding_service import embed_texts
from app.services.errors import NotFoundError
from app.services.pagination import DEFAULT_PAGE_LIMIT, paginate, split_page

logger = get_logger(__name__)

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
    XLSX_CONTENT_TYPE,
}


async def _assert_collection_in_tenant(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID
) -> None:
    collection = await session.get(Collection, collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        raise NotFoundError(f"collection {collection_id}")


async def _current_version_id(
    session: AsyncSession,
    *,
    collection_id: uuid.UUID,
    document_key: str,
    excluding: uuid.UUID | None = None,
) -> uuid.UUID | None:
    """The current version under a key, if any.

    `excluding` matters only when this is called from `process_document`: by
    then the row being processed already exists in the table (it was created
    at upload time), so a naive query would find *itself* if it happened to
    already be `is_current` — which is exactly the "first upload under a new
    key" case, see `create_upload`.
    """
    conditions = [
        Document.collection_id == collection_id,
        Document.document_key == document_key,
        Document.is_current.is_(True),
    ]
    if excluding is not None:
        conditions.append(Document.id != excluding)
    return await session.scalar(select(Document.id).where(*conditions))


async def _supersede(session: AsyncSession, *, previous_document_id: uuid.UUID) -> None:
    """Retire the previous current version, alone, in its own commit.

    Postgres checks the partial unique index on `(collection_id, document_key)
    WHERE is_current` per statement, not at end of transaction. Flipping the new
    row to current before this commit lands would put two current rows in front
    of that index at once and get rejected. Retiring first instead means the
    only intermediate state is zero current rows for a key, never two — and a
    beat where retrieval can't find either version of a document is nothing
    like a live contradiction it can find both, which is the failure this
    chunk exists to remove.

    Left unlocked deliberately: two re-uploads racing under the same key within
    the same instant is not a case this platform's synchronous, one-request-at-
    a-time ingestion needs to optimise for, and the partial unique index is
    still the backstop if it happens — the second commit fails closed with a
    constraint violation, caught by `upload_document`'s existing failure path,
    rather than silently producing two current versions.
    """
    previous = await session.get(Document, previous_document_id)
    if previous is not None:
        previous.is_current = False
        session.add(previous)
        await session.commit()


async def create_upload(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    filename: str,
    content_type: str,
    content: bytes,
    document_key: str | None = None,
) -> Document:
    """Record an upload and queue it — this is the entire request-path cost
    of ingestion now. Extraction, chunking, and embedding happen later, in
    the worker, never here: the collection ownership check and the write
    below are the only things this function does before returning.

    The collection is verified to belong to the calling tenant first, so a
    caller cannot queue work into another team's knowledge scope. The raw
    bytes are stored on the row itself (`raw_content`) because the worker
    that will read them runs in a different process, at a different time,
    than this one — they have to live somewhere durable a caller other than
    this one can reach, and Postgres is already the platform's one hard
    dependency for exactly this kind of thing.

    `document_key` is how a re-upload will supersede an earlier version
    instead of sitting beside it as an unrelated document retrieval can't
    tell apart from the one it's meant to replace — decided here only for
    whether *this* row starts as the current version (true iff nothing
    already holds that key); which row it actually supersedes, if any, is
    re-derived fresh when processing succeeds, in `process_document`, not
    carried across the gap between queuing and claiming.
    """
    await _assert_collection_in_tenant(
        session, tenant_id=tenant_id, collection_id=collection_id
    )

    previous_version_id = (
        await _current_version_id(session, collection_id=collection_id, document_key=document_key)
        if document_key
        else None
    )

    document = Document(
        collection_id=collection_id,
        filename=filename,
        content_type=content_type,
        status=DocumentStatus.UPLOADED,
        document_key=document_key,
        is_current=previous_version_id is None,
        raw_content=content,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def process_document(session: AsyncSession, document: Document) -> int:
    """Extract, chunk, embed, and store one already-claimed document.

    Called only by the ingestion worker, on a document whose status the
    worker has already moved to PROCESSING under its own claim (see
    `app/services/ingestion_worker.py`) — this function does not claim
    anything itself, so it must never be called against a row another
    worker might also be holding.

    On any failure the document is left FAILED with an error_message rather
    than raising — the worker logs and moves on to the next document; there
    is no request waiting on this one to fail loudly. `raw_content` is
    cleared on every terminal outcome, success or failure: once a document
    is chunked or abandoned, the original bytes serve no further purpose.
    """
    chunk_count = 0
    try:
        text = extract_text(document.raw_content, document.content_type)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document contained no extractable text.")

        embeddings = await embed_texts(chunks)

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk,
                    embedding=embedding,
                    token_count=count_tokens(chunk),
                )
            )
        chunk_count = len(chunks)

        if document.document_key:
            previous_version_id = await _current_version_id(
                session,
                collection_id=document.collection_id,
                document_key=document.document_key,
                excluding=document.id,
            )
            if previous_version_id is not None:
                await _supersede(session, previous_document_id=previous_version_id)
            document.is_current = True

        document.status = DocumentStatus.READY
        document.raw_content = None
        await session.commit()
    except Exception as exc:  # noqa: BLE001 - any failure here marks the document failed
        await session.rollback()
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:500]
        document.raw_content = None
        # is_current is left exactly as `create_upload` set it: True only if
        # this document held no key or was the first upload under one, in
        # which case there was nothing to supersede and nothing wrong with
        # leaving it current — a failed, chunkless document simply has
        # nothing for retrieval to find. False whenever a real previous
        # version exists, since supersession only ever happens after success
        # above — that previous version, untouched, is still current.
        session.add(document)
        await session.commit()
        chunk_count = 0
        logger.error("document_processing_failed", document_id=str(document.id), error=str(exc))

    return chunk_count


async def list_documents(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    current_only: bool = False,
) -> tuple[list[tuple[Document, int]], bool]:
    """List a collection's documents, newest upload first.

    `current_only` defaults to False: the API listing is a team owner's view
    into what's actually happened, including superseded versions, and hiding
    them would make a re-upload look like it silently vanished. The
    `list_documents` *tool* passes True — a model answering "what do you know
    about?" should describe the collection's current state, not its history.
    """
    await _assert_collection_in_tenant(
        session, tenant_id=tenant_id, collection_id=collection_id
    )
    conditions = [Document.collection_id == collection_id]
    if current_only:
        conditions.append(Document.is_current.is_(True))
    stmt = paginate(
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(*conditions)
        .group_by(Document.id)
        .order_by(Document.uploaded_at.desc()),
        limit=limit,
        offset=offset,
    )
    result = await session.execute(stmt)
    return split_page([(doc, count) for doc, count in result.all()], limit)


async def delete_document(
    session: AsyncSession, *, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> bool:
    """Delete a document, but only if it belongs to the calling tenant.

    Ownership is checked by joining through the document's collection: a
    document in another tenant's collection is indistinguishable from one that
    does not exist, so a cross-tenant delete returns False (→ 404).
    """
    document = await session.get(Document, document_id)
    if document is None:
        return False
    collection = await session.get(Collection, document.collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        return False
    await session.delete(document)
    await session.commit()
    return True
