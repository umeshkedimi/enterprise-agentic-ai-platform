import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.models.document import DocumentStatus
from app.models.schemas import DocumentResponse
from app.services import document_service

router = APIRouter(prefix="/documents", tags=["documents"])

_EXTENSION_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


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


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    content_type = _resolve_content_type(file.filename, file.content_type)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    document, chunk_count = await document_service.upload_document(
        session, filename=file.filename, content_type=content_type, content=content
    )

    if document.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=document.error_message or "Document processing failed.",
        )

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        chunk_count=chunk_count,
        uploaded_at=document.uploaded_at,
        error_message=document.error_message,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_db_session),
) -> list[DocumentResponse]:
    rows = await document_service.list_documents(session)
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            status=doc.status,
            chunk_count=count,
            uploaded_at=doc.uploaded_at,
            error_message=doc.error_message,
        )
        for doc, count in rows
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    deleted = await document_service.delete_document(session, document_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
