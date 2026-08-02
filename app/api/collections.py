import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import PageParams, get_current_tenant, page_params
from app.db.session import get_db_session
from app.models.schemas import CollectionCreate, CollectionResponse, Page
from app.models.tenant import Tenant
from app.services import collection_service
from app.services.errors import NotFoundError, SlugAlreadyExistsError

router = APIRouter(prefix="/collections", tags=["collections"])


def _to_response(collection) -> CollectionResponse:
    return CollectionResponse(
        id=collection.id,
        slug=collection.slug,
        name=collection.name,
        description=collection.description,
        created_at=collection.created_at,
    )


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CollectionCreate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> CollectionResponse:
    try:
        collection = await collection_service.create_collection(
            session,
            tenant_id=tenant.id,
            slug=body.slug,
            name=body.name,
            description=body.description,
        )
    except SlugAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A collection with slug '{body.slug}' already exists.",
        ) from exc
    return _to_response(collection)


@router.get("", response_model=Page[CollectionResponse])
async def list_collections(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[CollectionResponse]:
    collections, has_more = await collection_service.list_collections(
        session, tenant_id=tenant.id, limit=page.limit, offset=page.offset
    )
    return Page(
        items=[_to_response(c) for c in collections],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> CollectionResponse:
    try:
        collection = await collection_service.get_collection(
            session, tenant_id=tenant.id, collection_id=collection_id
        )
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
        ) from exc
    return _to_response(collection)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    try:
        await collection_service.delete_collection(
            session, tenant_id=tenant.id, collection_id=collection_id
        )
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
        ) from exc
