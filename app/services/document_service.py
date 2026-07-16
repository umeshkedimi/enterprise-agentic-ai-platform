import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.services.chunking import chunk_text, count_tokens, extract_text
from app.services.embedding_service import embed_texts

logger = get_logger(__name__)

SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain", "text/markdown"}


async def upload_document(
    session: AsyncSession,
    *,
    filename: str,
    content_type: str,
    content: bytes,
    tenant_id: str = "default",
) -> tuple[Document, int]:
    """Create a Document row and synchronously process it: extract, chunk, embed, store.

    On any failure the document is left in a FAILED state with an error_message
    rather than raising — callers decide how to surface that to the API layer.
    """
    document = Document(
        filename=filename,
        content_type=content_type,
        status=DocumentStatus.UPLOADED,
        tenant_id=tenant_id,
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
    session: AsyncSession, tenant_id: str = "default"
) -> list[tuple[Document, int]]:
    stmt = (
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.tenant_id == tenant_id)
        .group_by(Document.id)
        .order_by(Document.uploaded_at.desc())
    )
    result = await session.execute(stmt)
    return [(doc, count) for doc, count in result.all()]


async def delete_document(session: AsyncSession, document_id: uuid.UUID) -> bool:
    document = await session.get(Document, document_id)
    if document is None:
        return False
    await session.delete(document)
    await session.commit()
    return True
