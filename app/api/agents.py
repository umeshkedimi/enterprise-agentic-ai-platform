import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_tenant
from app.db.session import get_db_session
from app.models.agent import Agent
from app.models.schemas import AgentCreate, AgentResponse, AgentUpdate
from app.models.tenant import Tenant
from app.services import agent_service
from app.services.errors import NotFoundError, SlugAlreadyExistsError

router = APIRouter(prefix="/agents", tags=["agents"])


def _to_response(agent: Agent) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        slug=agent.slug,
        name=agent.name,
        system_prompt=agent.system_prompt,
        model=agent.model,
        collection_id=agent.collection_id,
        tool_allowlist=agent.tool_allowlist,
        temperature=agent.temperature,
        max_output_tokens=agent.max_output_tokens,
        retrieval_top_k=agent.retrieval_top_k,
        enabled=agent.enabled,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


_NOT_FOUND = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found.")


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    body: AgentCreate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AgentResponse:
    try:
        agent = await agent_service.create_agent(
            session,
            tenant_id=tenant.id,
            slug=body.slug,
            name=body.name,
            system_prompt=body.system_prompt,
            model=body.model,
            collection_id=body.collection_id,
            tool_allowlist=body.tool_allowlist,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
            retrieval_top_k=body.retrieval_top_k,
            enabled=body.enabled,
        )
    except SlugAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An agent with slug '{body.slug}' already exists.",
        ) from exc
    except NotFoundError as exc:
        # The referenced collection does not exist in this tenant.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Referenced collection not found."
        ) from exc
    return _to_response(agent)


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> list[AgentResponse]:
    agents = await agent_service.list_agents(session, tenant_id=tenant.id)
    return [_to_response(a) for a in agents]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AgentResponse:
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
    return _to_response(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    body: AgentUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AgentResponse:
    changes = body.model_dump(exclude_unset=True)
    try:
        agent = await agent_service.update_agent(
            session, tenant_id=tenant.id, agent_id=agent_id, changes=changes
        )
    except NotFoundError as exc:
        # Either the agent or a newly-referenced collection is absent in tenant.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc) or "Not found."
        ) from exc
    return _to_response(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    try:
        await agent_service.delete_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc
