"""The entry point everything outside orchestration calls to run a turn.

Callers — an HTTP handler today, a worker or an evaluation harness later — get a
plain async function and a result dataclass. They do not import LangGraph, do not
assemble state, and do not know how many nodes ran. That boundary is what lets
Chunks 4 through 7 change the graph's shape without touching a single caller.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import agent_graph
from app.agents.state import AgentState, OrchestrationContext
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.agent import Agent
from app.models.schemas import Citation
from app.services.completion_service import TokenUsage, Turn
from app.services.errors import AgentDisabledError
from app.services.retrieval_service import to_citations

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentTurnResult:
    """What one orchestrated turn produced, independent of how it was produced."""

    answer: str
    citations: list[Citation]
    tools_used: list[str]
    model: str
    provider: str
    usage: TokenUsage
    # Wall-clock for the whole graph run, which is deliberately not the model
    # latency the completion service reports: the gap between them is retrieval,
    # and keeping both is what makes a slow answer diagnosable.
    latency_ms: int
    retrieved_chunks: int = 0


async def run_turn(
    *,
    agent: Agent,
    question: str,
    history: Sequence[Turn] = (),
    session: AsyncSession,
    settings: Settings | None = None,
) -> AgentTurnResult:
    """Run one question through the orchestration graph under `agent`'s config.

    Raises the same domain errors as the completion service, plus RetrievalError.
    The API layer maps them; nothing here knows about HTTP.
    """
    settings = settings or get_settings()

    if not agent.enabled:
        # Checked before the graph rather than inside the completion node: a
        # disabled agent should not cost an embedding call and a vector scan on
        # its way to being refused.
        raise AgentDisabledError(str(agent.id))

    state: AgentState = {"question": question, "history": list(history), "tools_used": []}
    context = OrchestrationContext(agent=agent, session=session, settings=settings)

    started = time.perf_counter()
    # No blanket try/except here on purpose. Every failure the nodes can
    # anticipate is already a DomainError by the time it reaches this line, so
    # anything else is a bug — and turning a bug into a tidy 502 would hide it
    # behind a status code that says "the provider is having a bad day".
    final = await agent_graph.ainvoke(state, context=context)
    latency_ms = int((time.perf_counter() - started) * 1000)

    chunks = final.get("chunks", [])
    logger.info(
        "agent_turn_completed",
        agent_id=str(agent.id),
        tenant_id=str(agent.tenant_id),
        grounded=bool(chunks),
        chunks=len(chunks),
        model=final.get("model"),
        latency_ms=latency_ms,
    )

    return AgentTurnResult(
        answer=final.get("answer", ""),
        citations=to_citations(chunks),
        tools_used=final.get("tools_used", []),
        model=final.get("model", agent.model),
        provider=final.get("provider", ""),
        usage=final.get("usage", TokenUsage(0, 0, 0)),
        latency_ms=latency_ms,
        retrieved_chunks=len(chunks),
    )
