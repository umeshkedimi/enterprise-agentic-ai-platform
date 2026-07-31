"""The entry point everything outside orchestration calls to run a turn.

Callers — an HTTP handler today, a worker or an evaluation harness later — get a
plain async function and a result dataclass. They do not import LangGraph, do not
assemble state, and do not know how many nodes ran. That boundary is what lets
Chunks 4 through 7 change the graph's shape without touching a single caller.
"""

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import TOKEN_EVENT, TOOL_EVENT, get_agent_graph
from app.agents.state import AgentState, OrchestrationContext
from app.core import metrics, tracing
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
    # Wall-clock for the whole graph run, which is deliberately not the same as
    # the time spent waiting on the model: the gap between the two is retrieval
    # and tool execution, and keeping both is what makes a slow answer
    # diagnosable rather than merely slow.
    latency_ms: int
    model_latency_ms: int = 0
    retrieved_chunks: int = 0


def _turn_state(question: str, history: Sequence[Turn]) -> AgentState:
    """The starting state for one turn, with every turn-scoped key reset.

    Every field is written explicitly, including the ones that look like they
    could default. With a checkpointer attached, `ainvoke` *merges* the input
    into whatever the thread already holds, and a key that is simply absent
    keeps last turn's value: the previous question's retrieved chunks would be
    cited under this question, and `tool_steps` would start at the cap so the
    second turn in a conversation silently lost its tools.

    The rule this encodes: graph state is scratch for one turn. The conversation
    lives in the transcript, and arrives here as `history`.
    """
    return {
        "question": question,
        "history": list(history),
        "chunks": [],
        "scratchpad": [],
        "pending_calls": [],
        "tool_steps": 0,
        "tools_used": [],
        "answer": "",
        "usage": None,
        "latency_ms": 0,
    }


def _turn_attributes(agent: Agent, conversation_id: uuid.UUID | None, streamed: bool) -> dict:
    """What the turn's root span says about who it belongs to.

    Everything a metric label is forbidden to carry, a span carries here. That
    is the division of labour between the two: a counter says the platform's
    error rate rose, and a span says it rose for this tenant, this agent, and
    this conversation — because a span is sampled and expires, and a label is a
    series that never does.
    """
    return {
        tracing.TENANT_ID: str(agent.tenant_id),
        tracing.AGENT_ID: str(agent.id),
        tracing.AGENT_SLUG: agent.slug,
        tracing.CONVERSATION_ID: str(conversation_id) if conversation_id else None,
        tracing.COLLECTION_ID: str(agent.collection_id) if agent.collection_id else None,
        tracing.GEN_AI_REQUEST_MODEL: agent.model,
        tracing.STREAMED: streamed,
    }


def _record_turn_metrics(
    *, streamed: bool, started: float, error: BaseException | None = None
) -> None:
    """Count one turn, at the boundary that owns the whole of it.

    A turn is the unit an operator reasons about — it is what a user waited for
    and what a tenant is billed for — and it is not the same as a model call or
    an HTTP request. One turn can be several model calls; one HTTP request can
    be a turn that failed before the graph started.

    The error label is an exception class name. That is bounded by the code
    rather than by anything a caller can type, and it is the breakdown that says
    whose problem a failure is: `ProviderNotConfiguredError` is the operator's,
    `ModelConfigurationError` is a tenant's, `RetrievalError` is neither's until
    you look. A refusal that never entered the graph — a disabled agent — is
    deliberately absent: a turn that did not run is not a slow turn, and folding
    it in would pull the duration histogram toward zero.
    """
    label = str(streamed).lower()
    metrics.AGENT_TURNS.labels("error" if error else "ok", label).inc()
    metrics.AGENT_TURN_DURATION.labels(label).observe(time.perf_counter() - started)
    if error is not None:
        metrics.AGENT_TURN_ERRORS.labels(type(error).__name__).inc()


def _thread_config(conversation_id: uuid.UUID | None) -> dict:
    """Bind this run to a checkpoint thread, when there is one to bind to.

    The thread id is the conversation id, so a turn's execution trace is stored
    next to the conversation it belongs to and a run interrupted mid-tool-loop
    can be resumed rather than restarted. Without a conversation there is nothing
    stable to key on, and the run stays unpersisted.
    """
    if conversation_id is None:
        return {}
    return {"configurable": {"thread_id": str(conversation_id)}}


async def run_turn(
    *,
    agent: Agent,
    question: str,
    history: Sequence[Turn] = (),
    session: AsyncSession,
    conversation_id: uuid.UUID | None = None,
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

    context = OrchestrationContext(agent=agent, session=session, settings=settings)

    started = time.perf_counter()
    # Still no blanket try/except that *handles* anything. Every failure the
    # nodes can anticipate is already a DomainError by the time it reaches this
    # line, so anything else is a bug — and turning a bug into a tidy 502 would
    # hide it behind a status code that says "the provider is having a bad day".
    # This one counts and re-raises, which changes what an operator sees and not
    # what a caller does.
    try:
        with tracing.span("agent.turn", **_turn_attributes(agent, conversation_id, False)):
            final = await get_agent_graph().ainvoke(
                _turn_state(question, history),
                context=context,
                config=_thread_config(conversation_id),
            )
    except Exception as exc:
        _record_turn_metrics(streamed=False, started=started, error=exc)
        raise
    latency_ms = int((time.perf_counter() - started) * 1000)
    _record_turn_metrics(streamed=False, started=started)

    chunks = final.get("chunks", [])
    logger.info(
        "agent_turn_completed",
        agent_id=str(agent.id),
        tenant_id=str(agent.tenant_id),
        grounded=bool(chunks),
        chunks=len(chunks),
        model=final.get("model"),
        tools_used=final.get("tools_used", []),
        total_tokens=final.get("usage", TokenUsage(0, 0, 0)).total_tokens,
        latency_ms=latency_ms,
    )

    return _to_result(agent, final, latency_ms)


@dataclass(frozen=True)
class TokenChunk:
    """A fragment of the answer, as the provider produced it."""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    """A tool the model asked for, announced before it runs."""

    name: str


# What a streamed turn yields. The result arrives last and is the same object a
# non-streamed turn returns, so a caller can treat streaming as an early view of
# a turn rather than as a different kind of turn.
AgentStreamEvent = TokenChunk | ToolStarted | AgentTurnResult


async def stream_turn(
    *,
    agent: Agent,
    question: str,
    history: Sequence[Turn] = (),
    session: AsyncSession,
    conversation_id: uuid.UUID | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    """Run a turn, yielding fragments as they arrive and the result at the end.

    Same graph, same nodes, same error contract as `run_turn` — the difference is
    only that the answer is watched while it is produced. Domain errors propagate
    out of the iterator; a caller that has already started writing a response
    body has to report them in-band, which is the transport's problem and not
    this function's.
    """
    settings = settings or get_settings()

    if not agent.enabled:
        raise AgentDisabledError(str(agent.id))

    context = OrchestrationContext(
        agent=agent, session=session, settings=settings, stream=True
    )

    final: dict = {}
    started = time.perf_counter()
    try:
        # A span held open across `yield`s. Python has no per-generator context,
        # so the span stays current in the consumer between yields as well — and
        # that is harmless here, because each request runs in its own task with
        # its own copied context, and nothing downstream of the yield opens a
        # span of its own. The alternative, re-attaching the context on every
        # resumption, would buy nothing but a way to get it wrong.
        with tracing.span("agent.turn", **_turn_attributes(agent, conversation_id, True)):
            async for mode, payload in get_agent_graph().astream(
                _turn_state(question, history),
                context=context,
                config=_thread_config(conversation_id),
                # "values" alongside "custom" because the nodes narrate but do
                # not return; the last state snapshot is what the result is
                # built from.
                stream_mode=["custom", "values"],
            ):
                if mode == "values":
                    final = payload
                elif payload.get("type") == TOKEN_EVENT:
                    yield TokenChunk(text=payload["text"])
                elif payload.get("type") == TOOL_EVENT:
                    yield ToolStarted(name=payload["name"])
    except Exception as exc:
        _record_turn_metrics(streamed=True, started=started, error=exc)
        raise

    latency_ms = int((time.perf_counter() - started) * 1000)
    _record_turn_metrics(streamed=True, started=started)
    chunks = final.get("chunks", [])
    logger.info(
        "agent_turn_completed",
        agent_id=str(agent.id),
        tenant_id=str(agent.tenant_id),
        streamed=True,
        grounded=bool(chunks),
        chunks=len(chunks),
        model=final.get("model"),
        tools_used=final.get("tools_used", []),
        total_tokens=(final.get("usage") or TokenUsage(0, 0, 0)).total_tokens,
        latency_ms=latency_ms,
    )
    yield _to_result(agent, final, latency_ms)


def _to_result(agent: Agent, final: dict, latency_ms: int) -> AgentTurnResult:
    chunks = final.get("chunks", [])
    return AgentTurnResult(
        answer=final.get("answer", ""),
        citations=to_citations(chunks),
        tools_used=final.get("tools_used", []),
        model=final.get("model", agent.model),
        provider=final.get("provider", ""),
        usage=final.get("usage") or TokenUsage(0, 0, 0),
        latency_ms=latency_ms,
        model_latency_ms=final.get("latency_ms", 0),
        retrieved_chunks=len(chunks),
    )
