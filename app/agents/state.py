"""The two halves of a graph run: what is state, and what is only runtime.

LangGraph persists state and never persists context, and that split is a design
decision rather than a detail. Anything in `AgentState` must survive being
serialized to a checkpoint and read back by a different process on a later
request — so it holds conversation data and nothing else. Live handles (a
database session, resolved settings, the agent row itself) belong in
`OrchestrationContext`, which is passed per invocation and thrown away.

Putting the session in state would appear to work today, because Chunk 3 runs
without a checkpointer. It would fail the moment one is added: a resumed
conversation would deserialize a dead connection from whichever pod happened to
serve the first turn.
"""

from dataclasses import dataclass
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.agent import Agent
from app.services.completion_service import TokenUsage, ToolCall, Turn
from app.services.retrieval_service import RetrievedChunk


@dataclass(frozen=True)
class OrchestrationContext:
    """Per-invocation runtime dependencies. Never serialized.

    The agent row lives here rather than in state because it is configuration
    read at request time: a resumed conversation must pick up the agent's
    *current* prompt, model, and policy, not the copy that was in force when the
    conversation started. Config-as-data only holds if nothing caches it.
    """

    agent: Agent
    session: AsyncSession
    settings: Settings


class AgentState(TypedDict, total=False):
    """Everything one turn accumulates, in the order the graph fills it in."""

    # Input.
    question: str
    history: list[Turn]

    # Evidence for this turn: the automatic search first, then anything a
    # retrieval tool found. One list, because a citation should mean the same
    # thing whichever search produced it.
    chunks: list[RetrievedChunk]

    # Tool-call and tool-result messages accumulated while looping, in provider
    # shape. Replayed verbatim on each pass so the model sees what it asked for
    # and what came back.
    scratchpad: list[dict[str, Any]]

    # Tool calls the model has asked for and that have not run yet. Emptied by
    # the tools node, and the only thing the loop's exit condition consults —
    # the model stops asking, so the graph stops looping.
    pending_calls: list[ToolCall]

    # Completed generate→tools round trips. The cap on this is what makes the
    # loop terminate against a model that would otherwise keep calling tools.
    tool_steps: int
    tools_used: list[str]

    # Filled by the generate node once it answers instead of calling a tool.
    answer: str
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: int
