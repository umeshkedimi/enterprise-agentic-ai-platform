import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_tenant
from app.db.session import get_db_session
from app.models.document import Document, DocumentStatus
from app.models.schemas import DocumentResponse
from app.models.tenant import Tenant
from app.services import document_service
from app.services.errors import NotFoundError

# Upload and listing are nested under a collection — a document only exists
# inside a knowledge scope. Deletion is addressed by document id directly, with
# tenant ownership verified in the service.
router = APIRouter(tags=["documents"])

_EXTENSION_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

_COLLECTION_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
)


def _resolve_content_type(filename: str, declared_content_type: str | None) -> str:
    """Trust the client's Content-Type only if it's one we support; otherwise
    fall back to the file extension. Clients like curl often send a generic
    application/octet-stream, so extension is the more reliable signal.
    """
    if declared_content_type in document_service.SUPPORTED_CONTENT_TYPES:
        return declared_content_type

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    inferred = _EXTENSION_CONTENT_TYPES.get(ext)
    if inferred is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type for '{filename}'. Supported: pdf, txt, markdown.",
        )
    return inferred


def _to_response(document: Document, chunk_count: int) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        collection_id=document.collection_id,
        filename=document.filename,
        status=document.status,
        chunk_count=chunk_count,
        uploaded_at=document.uploaded_at,
        error_message=document.error_message,
    )


@router.post(
    "/collections/{collection_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    content_type = _resolve_content_type(file.filename, file.content_type)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    try:
        document, chunk_count = await document_service.upload_document(
            session,
            tenant_id=tenant.id,
            collection_id=collection_id,
            filename=file.filename,
            content_type=content_type,
            content=content,
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc

    if document.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=document.error_message or "Document processing failed.",
        )

    return _to_response(document, chunk_count)


@router.get(
    "/collections/{collection_id}/documents", response_model=list[DocumentResponse]
)
async def list_documents(
    collection_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> list[DocumentResponse]:
    try:
        rows = await document_service.list_documents(
            session, tenant_id=tenant.id, collection_id=collection_id
        )
    except NotFoundError as exc:
        raise _COLLECTION_NOT_FOUND from exc
    return [_to_response(doc, count) for doc, count in rows]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    deleted = await document_service.delete_document(
        session, tenant_id=tenant.id, document_id=document_id
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
