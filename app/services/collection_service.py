import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Collection
from app.services.errors import NotFoundError, SlugAlreadyExistsError
from app.services.pagination import DEFAULT_PAGE_LIMIT, paginate, split_page

logger = get_logger(__name__)


async def create_collection(
    session: AsyncSession, *, tenant_id: uuid.UUID, slug: str, name: str, description: str | None
) -> Collection:
    existing = await session.scalar(
        select(Collection).where(Collection.tenant_id == tenant_id, Collection.slug == slug)
    )
    if existing is not None:
        raise SlugAlreadyExistsError(slug)

    collection = Collection(tenant_id=tenant_id, slug=slug, name=name, description=description)
    session.add(collection)
    await session.commit()
    await session.refresh(collection)
    logger.info(
        "collection_created",
        tenant_id=str(tenant_id),
        collection_id=str(collection.id),
        slug=slug,
    )
    return collection


async def list_collections(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> tuple[list[Collection], bool]:
    stmt = paginate(
        select(Collection)
        .where(Collection.tenant_id == tenant_id)
        .order_by(Collection.created_at.desc()),
        limit=limit,
        offset=offset,
    )
    result = await session.scalars(stmt)
    return split_page(list(result.all()), limit)


async def get_collection(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID
) -> Collection:
    """Fetch a collection, enforcing tenant ownership.

    The tenant filter is the isolation boundary: a collection belonging to
    another tenant is indistinguishable from one that does not exist.
    """
    collection = await session.get(Collection, collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        raise NotFoundError(str(collection_id))
    return collection


async def delete_collection(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID
) -> None:
    collection = await get_collection(
        session, tenant_id=tenant_id, collection_id=collection_id
    )
    await session.delete(collection)
    await session.commit()
    logger.info(
        "collection_deleted", tenant_id=str(tenant_id), collection_id=str(collection_id)
    )
