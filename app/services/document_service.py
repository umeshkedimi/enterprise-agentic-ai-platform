import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Collection
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.chunking import chunk_text, count_tokens, extract_text
from app.services.embedding_service import embed_texts
from app.services.errors import NotFoundError
from app.services.pagination import DEFAULT_PAGE_LIMIT, paginate, split_page

logger = get_logger(__name__)

SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain", "text/markdown"}


async def _assert_collection_in_tenant(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID
) -> None:
    collection = await session.get(Collection, collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        raise NotFoundError(f"collection {collection_id}")


async def upload_document(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    collection_id: uuid.UUID,
    filename: str,
    content_type: str,
    content: bytes,
) -> tuple[Document, int]:
    """Create a Document in a collection and process it: extract, chunk, embed, store.

    The collection is verified to belong to the calling tenant first, so a
    caller cannot ingest into another team's knowledge scope. On any processing
    failure the document is left FAILED with an error_message rather than
    raising — callers decide how to surface that to the API layer.
    """
    await _assert_collection_in_tenant(
        session, tenant_id=tenant_id, collection_id=collection_id
    )

    document = Document(
        collection_id=collection_id,
        filename=filename,
        content_type=content_type,
        status=DocumentStatus.UPLOADED,
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

        document.status = DocumentStatus.READY
        await session.commit()
        await session.refresh(document)
    except Exception as exc:  # noqa: BLE001 - any failure here marks the document failed
        await session.rollback()
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)[:500]
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
) -> tuple[list[tuple[Document, int]], bool]:
    await _assert_collection_in_tenant(
        session, tenant_id=tenant_id, collection_id=collection_id
    )
    stmt = paginate(
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
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
