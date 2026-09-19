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
    session: AsyncSession, *, collection_id: uuid.UUID, document_key: str
) -> uuid.UUID | None:
    return await session.scalar(
        select(Document.id).where(
            Document.collection_id == collection_id,
            Document.document_key == document_key,
            Document.is_current.is_(True),
        )
    )


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


async def upload_document(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    filename: str,
    content_type: str,
    content: bytes,
    document_key: str | None = None,
) -> tuple[Document, int]:
    """Create a Document in a collection and process it: extract, chunk, embed, store.

    The collection is verified to belong to the calling tenant first, so a
    caller cannot ingest into another team's knowledge scope. On any processing
    failure the document is left FAILED with an error_message rather than
    raising — callers decide how to surface that to the API layer.

    `document_key` is how a re-upload supersedes an earlier version instead of
    sitting beside it as an unrelated document that retrieval can't tell apart
    from the one it's meant to replace. A document uploaded under a key that
    already has a current version starts life as `is_current=False` — it only
    becomes the current version once ingestion actually succeeds, in
    `_supersede`, so a bad re-upload (empty file, unreadable PDF) fails without
    ever taking the working version off retrieval. Never deletes the version it
    replaces: its chunks stay in the table, unreadable to fresh retrieval but
    still recoverable by id, because an audit of an answer given under the old
    version still needs to read what it actually cited.
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
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    chunk_count = 0
    try:
        document.status = DocumentStatus.PROCESSING
        await session.commit()

        text = extract_text(content, content_type)
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

        if previous_version_id is not None:
            await _supersede(session, previous_document_id=previous_version_id)
            document.is_current = True

        document.status = DocumentStatus.READY
        await session.commit()
        await session.refresh(document)
    except Exception as exc:  # noqa: BLE001 - any failure here marks the document failed
        await session.rollback()
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:500]
        # is_current is already False here whenever this upload was superseding
        # something — it was never flipped, because that only happens after the
        # try block succeeds. The previous version, untouched, is still current.
        session.add(document)
        await session.commit()
        await session.refresh(document)
        chunk_count = 0
        logger.error("document_processing_failed", document_id=str(document.id), error=str(exc))

    return document, chunk_count


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
