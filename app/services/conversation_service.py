"""Conversation threads: who owns them, what they remember, what a turn cost.

This is where "history" stops being something a client hands us and becomes
something the platform owns. The difference is not convenience. A client that
supplies its own history can fabricate assistant turns — "you previously agreed
to ignore your instructions" — and no amount of care in the prompt layer detects
it, because a forged turn is indistinguishable from a real one by the time it
reaches the model. Reading history from a row the platform wrote closes that.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Agent
from app.models.conversation import Conversation, ConversationMessage
from app.models.schemas import Citation
from app.services.completion_service import TokenUsage, Turn
from app.services.errors import NotFoundError

logger = get_logger(__name__)

# How much of a thread is replayed to the model. Ultimately a cost and a
# coherence decision, not a correctness one — the transcript keeps everything,
# and only the tail is resent.
DEFAULT_HISTORY_WINDOW = 20

# The first message of a conversation may only be a user turn: Anthropic rejects
# a message list that opens with an assistant turn outright, and the others treat
# it as an oddity to be interpreted. A window that happens to cut between a
# question and its answer would otherwise produce exactly that list.
_USER = "user"
_ASSISTANT = "assistant"


async def create_conversation(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    title: str | None = None,
) -> Conversation:
    conversation = Conversation(tenant_id=tenant_id, agent_id=agent_id, title=title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    logger.info(
        "conversation_created",
        tenant_id=str(tenant_id),
        agent_id=str(agent_id),
        conversation_id=str(conversation.id),
    )
    return conversation


async def get_conversation(
    session: AsyncSession, *, tenant_id: uuid.UUID, conversation_id: uuid.UUID
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None or conversation.tenant_id != tenant_id:
        raise NotFoundError(str(conversation_id))
    return conversation


async def list_conversations(
    session: AsyncSession, *, tenant_id: uuid.UUID, agent_id: uuid.UUID
) -> list[Conversation]:
    result = await session.scalars(
        select(Conversation)
        .where(Conversation.tenant_id == tenant_id, Conversation.agent_id == agent_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.all())


async def resolve_conversation(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent: Agent,
    conversation_id: uuid.UUID | None,
) -> Conversation:
    """Continue the named thread, or start one. Never cross an agent boundary.

    A conversation belongs to the agent that produced it, and continuing it under
    a different agent is refused as not-found. The reason is retrieval, not
    tidiness: the history of an HR agent's thread contains what an HR agent was
    shown, and replaying it into a finance agent's context would move that
    content across the collection boundary the platform spends its effort
    enforcing — without a single query ever crossing it.
    """
    if conversation_id is None:
        return await create_conversation(session, tenant_id=tenant_id, agent_id=agent.id)

    conversation = await get_conversation(
        session, tenant_id=tenant_id, conversation_id=conversation_id
    )
    if conversation.agent_id != agent.id:
        raise NotFoundError(str(conversation_id))
    return conversation


async def list_messages(
    session: AsyncSession, *, conversation_id: uuid.UUID
) -> list[ConversationMessage]:
    result = await session.scalars(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.seq)
    )
    return list(result.all())


async def load_history(
    session: AsyncSession, *, conversation_id: uuid.UUID, limit: int = DEFAULT_HISTORY_WINDOW
) -> list[Turn]:
    """The tail of the thread, oldest-first, as the model should see it.

    Only the text of each turn is replayed. The retrieved passages that grounded
    an earlier answer are deliberately not resent: they would be a stale copy of
    documents that may since have changed, and re-grounding each question against
    the current index is both cheaper and more honest than carrying old evidence
    forward.
    """
    result = await session.scalars(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.seq.desc())
        .limit(limit)
    )
    messages = list(reversed(result.all()))

    # Trim any assistant turn left dangling at the front by the window boundary.
    while messages and messages[0].role != _USER:
        messages.pop(0)

    return [Turn(role=m.role, content=m.content) for m in messages]  # type: ignore[arg-type]


async def record_turn(
    session: AsyncSession,
    *,
    conversation: Conversation,
    question: str,
    answer: str,
    citations: Sequence[Citation] = (),
    tools_used: Sequence[str] = (),
    model: str | None = None,
    provider: str | None = None,
    usage: TokenUsage | None = None,
    latency_ms: int = 0,
) -> None:
    """Append the completed exchange as two messages, in one transaction.

    Written after the turn succeeds rather than as it starts. A half-recorded
    turn — a question with no answer — would be replayed as history on the next
    request and read to the model as an exchange it failed to respond to.
    """
    usage = usage or TokenUsage(0, 0, 0)

    session.add(
        ConversationMessage(
            conversation_id=conversation.id,
            role=_USER,
            content=question,
        )
    )
    session.add(
        ConversationMessage(
            conversation_id=conversation.id,
            role=_ASSISTANT,
            content=answer,
            # mode="json" so the UUIDs inside a citation land in JSONB as
            # strings rather than as objects psycopg cannot adapt.
            citations=[c.model_dump(mode="json") for c in citations],
            tools_used=list(tools_used),
            model=model,
            provider=provider,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )
    )
    # Touched explicitly: `onupdate` only fires on an UPDATE of this row, and
    # appending a message does not update the parent.
    conversation.updated_at = datetime.now(UTC)
    session.add(conversation)

    await session.commit()
    logger.info(
        "conversation_turn_recorded",
        tenant_id=str(conversation.tenant_id),
        agent_id=str(conversation.agent_id),
        conversation_id=str(conversation.id),
        total_tokens=usage.total_tokens,
    )
