import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import runner
from app.api import sse
from app.api.deps import PageParams, get_current_tenant, page_params
from app.core.config import get_settings
from app.db.session import async_session_factory, get_db_session
from app.models.agent import Agent
from app.models.conversation import Conversation, ConversationMessage
from app.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCompletionRequest,
    AgentCompletionResponse,
    AgentCreate,
    AgentResponse,
    AgentUpdate,
    Citation,
    ConversationCreate,
    ConversationMessageResponse,
    ConversationResponse,
    Page,
    TokenUsageResponse,
)
from app.models.tenant import Tenant
from app.services import agent_service, completion_service, conversation_service
from app.services.errors import (
    AgentDisabledError,
    DomainError,
    ModelConfigurationError,
    NotFoundError,
    ProviderNotConfiguredError,
    RetrievalError,
    SlugAlreadyExistsError,
)

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
_CONVERSATION_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
)


def _to_conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        agent_id=conversation.agent_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _to_message_response(message: ConversationMessage) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        id=message.id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        citations=[Citation(**c) for c in message.citations],
        tools_used=message.tools_used,
        model=message.model,
        provider=message.provider,
        usage=TokenUsageResponse(
            prompt_tokens=message.prompt_tokens,
            completion_tokens=message.completion_tokens,
            total_tokens=message.total_tokens,
        ),
        latency_ms=message.latency_ms,
        created_at=message.created_at,
    )


def _to_http_error(exc: DomainError, agent: Agent) -> HTTPException:
    """Map a run-an-agent failure onto the status code that describes who can fix it.

    The distinction the codes carry is whose problem it is: 400 the tenant's
    configuration, 409 their deliberate choice to disable the agent, 503 the
    operator's unfinished wiring, 502 the upstream provider. Collapsing these
    into one error would leave a team owner unable to tell a typo in their model
    name from an outage they can only wait out.
    """
    if isinstance(exc, AgentDisabledError):
        return HTTPException(status.HTTP_409_CONFLICT, "Agent is disabled.")
    if isinstance(exc, ModelConfigurationError):
        return HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Agent model '{agent.model}' does not map to a known provider.",
        )
    if isinstance(exc, ProviderNotConfiguredError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No credentials configured for this model's provider.",
        )
    if isinstance(exc, RetrievalError):
        return HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Could not search this agent's knowledge base."
        )
    # ModelInvocationError and anything else domain-level. The upstream message
    # is withheld deliberately: provider errors echo prompt content, which here
    # means retrieved document text.
    return HTTPException(status.HTTP_502_BAD_GATEWAY, "Model provider request failed.")


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


@router.get("", response_model=Page[AgentResponse])
async def list_agents(
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[AgentResponse]:
    agents, has_more = await agent_service.list_agents(
        session, tenant_id=tenant.id, limit=page.limit, offset=page.offset
    )
    return Page(
        items=[_to_response(a) for a in agents],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


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


@router.post("/{agent_id}/complete", response_model=AgentCompletionResponse)
async def complete(
    agent_id: uuid.UUID,
    body: AgentCompletionRequest,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AgentCompletionResponse:
    """Run one stateless turn against the agent's configured model.

    Deliberately stateless and retrieval-free: this exists to prove that an
    agent's `model` is honored at runtime. Conversation state, retrieval, and
    tool execution arrive with the orchestrated chat endpoint.
    """
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    turns = [completion_service.Turn(role=t.role, content=t.content) for t in body.turns]

    try:
        result = await completion_service.complete(agent=agent, turns=turns)
    except DomainError as exc:
        raise _to_http_error(exc, agent) from exc

    return AgentCompletionResponse(
        text=result.text,
        model=result.model,
        provider=result.provider,
        usage=TokenUsageResponse(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        latency_ms=result.latency_ms,
    )


@router.post(
    "/{agent_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    agent_id: uuid.UUID,
    body: ConversationCreate,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationResponse:
    """Open a thread explicitly. Optional — `/chat` opens one on demand."""
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    conversation = await conversation_service.create_conversation(
        session, tenant_id=tenant.id, agent_id=agent.id, title=body.title
    )
    return _to_conversation_response(conversation)


@router.get("/{agent_id}/conversations", response_model=Page[ConversationResponse])
async def list_conversations(
    agent_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[ConversationResponse]:
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    conversations, has_more = await conversation_service.list_conversations(
        session, tenant_id=tenant.id, agent_id=agent.id, limit=page.limit, offset=page.offset
    )
    return Page(
        items=[_to_conversation_response(c) for c in conversations],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


@router.get(
    "/{agent_id}/conversations/{conversation_id}/messages",
    response_model=Page[ConversationMessageResponse],
)
async def list_conversation_messages(
    agent_id: uuid.UUID,
    conversation_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
    page: PageParams = Depends(page_params),
) -> Page[ConversationMessageResponse]:
    """The transcript, oldest first, including turns older than the replay window.

    The window bounds what the *model* is shown, not what the platform keeps —
    and this page bounds what one request will serve, which is a different limit
    again. A thread is the one list here that grows without any ceiling.
    """
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
        conversation = await conversation_service.resolve_conversation(
            session, tenant_id=tenant.id, agent=agent, conversation_id=conversation_id
        )
    except NotFoundError as exc:
        raise _CONVERSATION_NOT_FOUND from exc

    messages, has_more = await conversation_service.list_messages(
        session, conversation_id=conversation.id, limit=page.limit, offset=page.offset
    )
    return Page(
        items=[_to_message_response(m) for m in messages],
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
    )


@router.post("/{agent_id}/chat", response_model=AgentChatResponse)
async def chat(
    agent_id: uuid.UUID,
    body: AgentChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> AgentChatResponse:
    """Run one orchestrated turn: retrieve the agent's evidence, then answer from it.

    This is the runtime the platform exists to provide. Note what the request
    body cannot say: which collection to search, which model to use, what the
    system prompt is, and — since conversations became server-side — what was
    said earlier. All of it comes from rows the platform owns, so a team owner
    reconfigures their assistant by editing config and every caller picks up the
    change on the next request, with no deployment and no client update.
    """
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    try:
        conversation = await conversation_service.resolve_conversation(
            session, tenant_id=tenant.id, agent=agent, conversation_id=body.conversation_id
        )
    except NotFoundError as exc:
        raise _CONVERSATION_NOT_FOUND from exc

    settings = get_settings()
    history, conversation_summary = await conversation_service.load_turn_context(
        session, conversation=conversation, agent=agent, settings=settings
    )

    try:
        result = await runner.run_turn(
            agent=agent,
            question=body.message,
            history=history,
            session=session,
            conversation_id=conversation.id,
            settings=settings,
            conversation_summary=conversation_summary,
        )
    except DomainError as exc:
        # Nothing is recorded on a failed turn. A question with no answer would
        # be replayed to the model next time as an exchange it ignored.
        raise _to_http_error(exc, agent) from exc

    await conversation_service.record_turn(
        session,
        conversation=conversation,
        question=body.message,
        answer=result.answer,
        citations=result.citations,
        tools_used=result.tools_used,
        model=result.model,
        provider=result.provider,
        usage=result.usage,
        latency_ms=result.latency_ms,
    )

    return AgentChatResponse(
        conversation_id=conversation.id,
        answer=result.answer,
        citations=result.citations,
        tools_used=result.tools_used,
        model=result.model,
        provider=result.provider,
        usage=TokenUsageResponse(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        latency_ms=result.latency_ms,
    )


@router.post("/{agent_id}/chat/stream")
async def chat_stream(
    agent_id: uuid.UUID,
    body: AgentChatRequest,
    tenant: Tenant = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    """The same turn as `/chat`, narrated as it happens.

    Everything that can be checked before the response starts is checked here,
    with the request's own session, so a missing agent or a foreign conversation
    is a real 404 rather than a 200 containing bad news. Past that point the
    status line has been sent and a failure can only be reported in-band, as an
    `error` frame.
    """
    try:
        agent = await agent_service.get_agent(session, tenant_id=tenant.id, agent_id=agent_id)
    except NotFoundError as exc:
        raise _NOT_FOUND from exc

    try:
        conversation = await conversation_service.resolve_conversation(
            session, tenant_id=tenant.id, agent=agent, conversation_id=body.conversation_id
        )
    except NotFoundError as exc:
        raise _CONVERSATION_NOT_FOUND from exc

    if not agent.enabled:
        # Hoisted out of the runner so a disabled agent is a 409 like it is
        # everywhere else, rather than a 200 whose first frame is an error.
        raise _to_http_error(AgentDisabledError(str(agent.id)), agent)

    return StreamingResponse(
        _stream_turn(
            tenant_id=tenant.id,
            agent_id=agent.id,
            conversation_id=conversation.id,
            message=body.message,
        ),
        media_type=sse.MEDIA_TYPE,
        headers=sse.SSE_HEADERS,
    )


async def _stream_turn(
    *, tenant_id: uuid.UUID, agent_id: uuid.UUID, conversation_id: uuid.UUID, message: str
):
    """Drive the turn on a session of its own.

    Ids cross this boundary, not ORM objects. The request's session is already
    closed by the time a streaming body is produced — FastAPI exits a `yield`
    dependency once the response object is returned — so the rows are re-read
    here, on the session that will actually be used. Handing the loaded objects
    over instead raises "already attached to session" on the first write, and
    only ever under streaming.
    """
    settings = get_settings()

    async with async_session_factory() as session:
        yield sse.frame("conversation", {"conversation_id": str(conversation_id)})

        try:
            agent = await agent_service.get_agent(
                session, tenant_id=tenant_id, agent_id=agent_id
            )
            conversation = await conversation_service.get_conversation(
                session, tenant_id=tenant_id, conversation_id=conversation_id
            )
        except NotFoundError:
            # Deleted between the preflight checks and this line.
            yield sse.frame("error", {"status": 404, "detail": "Agent not found."})
            return

        history, conversation_summary = await conversation_service.load_turn_context(
            session, conversation=conversation, agent=agent, settings=settings
        )

        result = None
        try:
            async for event in runner.stream_turn(
                agent=agent,
                question=message,
                history=history,
                session=session,
                conversation_id=conversation.id,
                settings=settings,
                conversation_summary=conversation_summary,
            ):
                if isinstance(event, runner.TokenChunk):
                    yield sse.frame("token", {"text": event.text})
                elif isinstance(event, runner.ToolStarted):
                    yield sse.frame("tool", {"name": event.name})
                else:
                    result = event
        except DomainError as exc:
            # The status code is still reported, in the payload, so a client has
            # the same information it would have had from a failed `/chat` —
            # whose problem it is, and whether retrying could help.
            http_error = _to_http_error(exc, agent)
            yield sse.frame(
                "error", {"status": http_error.status_code, "detail": http_error.detail}
            )
            return

        if result is None:  # pragma: no cover - the graph always yields a result
            yield sse.frame("error", {"status": 502, "detail": "The turn produced no answer."})
            return

        await conversation_service.record_turn(
            session,
            conversation=conversation,
            question=message,
            answer=result.answer,
            citations=result.citations,
            tools_used=result.tools_used,
            model=result.model,
            provider=result.provider,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )

        # Sent after the turn is recorded, so a client that reconnects on `done`
        # and asks for the transcript finds the turn it just watched.
        yield sse.frame(
            "done",
            {
                "conversation_id": str(conversation.id),
                "answer": result.answer,
                "citations": [c.model_dump(mode="json") for c in result.citations],
                "tools_used": result.tools_used,
                "model": result.model,
                "provider": result.provider,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
                "latency_ms": result.latency_ms,
            },
        )


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
