import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Agent, Collection
from app.services.errors import NotFoundError, SlugAlreadyExistsError
from app.services.pagination import DEFAULT_PAGE_LIMIT, paginate, split_page

logger = get_logger(__name__)

# Fields a partial update is permitted to change. Slug and tenant are absent by
# design: identity is immutable, so an unknown or protected key in an update is
# simply ignored rather than silently reassigning an agent to another tenant.
_UPDATABLE_FIELDS = frozenset(
    {
        "name",
        "system_prompt",
        "model",
        "collection_id",
        "tool_allowlist",
        "temperature",
        "max_output_tokens",
        "retrieval_top_k",
        "enabled",
    }
)


async def _assert_collection_in_tenant(
    session: AsyncSession, *, tenant_id: uuid.UUID, collection_id: uuid.UUID | None
) -> None:
    """Reject an agent that points at a collection it does not own.

    Without this check a tenant could bind an agent to another tenant's
    collection and read across the isolation boundary — the exact failure
    tenancy exists to prevent. A cross-tenant id is treated as not-found.
    """
    if collection_id is None:
        return
    collection = await session.get(Collection, collection_id)
    if collection is None or collection.tenant_id != tenant_id:
        raise NotFoundError(f"collection {collection_id}")


async def create_agent(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    slug: str,
    name: str,
    system_prompt: str,
    model: str,
    collection_id: uuid.UUID | None,
    tool_allowlist: list[str],
    temperature: float,
    max_output_tokens: int,
    retrieval_top_k: int,
    enabled: bool,
) -> Agent:
    existing = await session.scalar(
        select(Agent).where(Agent.tenant_id == tenant_id, Agent.slug == slug)
    )
    if existing is not None:
        raise SlugAlreadyExistsError(slug)

    await _assert_collection_in_tenant(
        session, tenant_id=tenant_id, collection_id=collection_id
    )

    agent = Agent(
        tenant_id=tenant_id,
        slug=slug,
        name=name,
        system_prompt=system_prompt,
        model=model,
        collection_id=collection_id,
        tool_allowlist=tool_allowlist,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        retrieval_top_k=retrieval_top_k,
        enabled=enabled,
    )
    session.add(agent)
    await session.commit()
    await session.refresh(agent)
    logger.info("agent_created", tenant_id=str(tenant_id), agent_id=str(agent.id), slug=slug)
    return agent


async def list_agents(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> tuple[list[Agent], bool]:
    stmt = paginate(
        select(Agent).where(Agent.tenant_id == tenant_id).order_by(Agent.created_at.desc()),
        limit=limit,
        offset=offset,
    )
    result = await session.scalars(stmt)
    return split_page(list(result.all()), limit)


async def get_agent(
    session: AsyncSession, *, tenant_id: uuid.UUID, agent_id: uuid.UUID
) -> Agent:
    agent = await session.get(Agent, agent_id)
    if agent is None or agent.tenant_id != tenant_id:
        raise NotFoundError(str(agent_id))
    return agent


async def update_agent(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    changes: dict[str, Any],
) -> Agent:
    agent = await get_agent(session, tenant_id=tenant_id, agent_id=agent_id)

    if "collection_id" in changes:
        await _assert_collection_in_tenant(
            session, tenant_id=tenant_id, collection_id=changes["collection_id"]
        )

    for field, value in changes.items():
        if field in _UPDATABLE_FIELDS:
            setattr(agent, field, value)

    await session.commit()
    await session.refresh(agent)
    logger.info("agent_updated", tenant_id=str(tenant_id), agent_id=str(agent_id))
    return agent


async def delete_agent(
    session: AsyncSession, *, tenant_id: uuid.UUID, agent_id: uuid.UUID
) -> None:
    agent = await get_agent(session, tenant_id=tenant_id, agent_id=agent_id)
    await session.delete(agent)
    await session.commit()
    logger.info("agent_deleted", tenant_id=str(tenant_id), agent_id=str(agent_id))
