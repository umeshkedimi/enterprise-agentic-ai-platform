"""The orchestration graph: one compiled workflow, every agent.

This is where "onboarding an assistant is a configuration change" stops being a
claim about the data model and becomes true of the runtime. There is a single
graph for the whole platform, compiled once at import. It never branches on
tenant or agent identity — it branches on *configuration values*: an agent with a
`collection_id` routes through retrieval, one without goes straight to
generation. Adding a team adds a row, not a node.

The graph is a graph rather than a function because of what Chunks 4 through 7
attach to it: a checkpointer for conversation memory, node-level streaming, a
tool loop, and per-node spans for tracing. Each of those is an edge or a
compile-time argument on this structure, and none of them is a rewrite.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agents.prompts import (
    GROUNDING_DIRECTIVE,
    NO_RESULTS_DIRECTIVE,
    RETRY_SEARCH_DIRECTIVE,
    build_grounded_question,
)
from app.agents.state import AgentState, OrchestrationContext
from app.core.logging import get_logger
from app.models.agent import Agent
from app.services.completion_service import DeltaHandler, TokenUsage, Turn, complete
from app.services.errors import RetrievalError
from app.services.retrieval_service import semantic_search
from app.tools.registry import Tool, ToolContext
from app.tools.resolution import resolve_agent_tools

logger = get_logger(__name__)

RETRIEVE = "retrieve"
GENERATE = "generate"
TOOLS = "tools"

# Names for the two things a caller watching a turn can be told while it runs.
# A tool event exists because the pause it explains — a second retrieval, a
# reformulated search — is otherwise silence in the middle of an answer.
TOKEN_EVENT = "token"
TOOL_EVENT = "tool"

# How many generate→tools round trips one turn may make. A cap is not optional:
# a model that keeps calling tools would otherwise bill a tenant indefinitely and
# hold a database session open for as long as it kept going. Three is enough for
# search, refine, and answer — the pattern these tools exist for.
MAX_TOOL_STEPS = 3


async def retrieve_node(state: AgentState, runtime: Runtime[OrchestrationContext]) -> dict:
    """Search the agent's own collection for evidence bearing on the question.

    `collection_id` is the isolation boundary and it comes from the agent row, never
    from the request — a caller cannot widen the search by asking nicely, because
    there is no parameter through which to ask.
    """
    agent = runtime.context.agent
    try:
        chunks = await semantic_search(
            runtime.context.session,
            state["question"],
            collection_id=agent.collection_id,
            top_k=agent.retrieval_top_k,
        )
    except Exception as exc:  # noqa: BLE001 - OpenAI and driver errors are provider types
        # Translated here rather than in the runner because this is the only
        # frame that knows a failure means "no evidence gathered". Failing the
        # turn is the point: an agent configured to answer from documents must
        # not quietly answer without them.
        logger.warning(
            "retrieval_failed",
            agent_id=str(agent.id),
            collection_id=str(agent.collection_id),
            error=type(exc).__name__,
        )
        raise RetrievalError(str(exc)) from exc

    logger.info(
        "retrieval_completed",
        agent_id=str(agent.id),
        tenant_id=str(agent.tenant_id),
        collection_id=str(agent.collection_id),
        chunks=len(chunks),
        # The best score is the cheapest signal that retrieval is degrading:
        # an empty collection and a badly-matched one look identical in a
        # count, and quite different here.
        top_score=round(chunks[0].score, 4) if chunks else None,
    )
    return {"chunks": chunks}


def _directives(agent: Agent, state: AgentState, tools: list[Tool]) -> list[str]:
    chunks = state.get("chunks", [])
    if agent.collection_id is None:
        # No knowledge scope configured, so there is nothing to ground in and no
        # citation format to impose. The agent runs on its own prompt alone.
        return []
    if chunks:
        return [GROUNDING_DIRECTIVE]
    if tools:
        # Nothing found, but the model can search again with better terms — so
        # invite that rather than telling it to give up on the first miss.
        return [GROUNDING_DIRECTIVE, RETRY_SEARCH_DIRECTIVE]
    # Retrieval ran, came back empty, and there is no way to try again. Saying so
    # beats improvising from parametric memory under a prompt that promised
    # sources.
    return [GROUNDING_DIRECTIVE, NO_RESULTS_DIRECTIVE]


def _delta_writer(runtime: Runtime[OrchestrationContext]) -> DeltaHandler | None:
    """A sink for token fragments, or None to leave the call unstreamed.

    Returning None matters: streaming has a real cost (a chunked provider
    response, a reassembly pass) and a turn nobody is watching should not pay it.
    The events go through LangGraph's stream writer rather than out of the node
    directly, so a node stays a function of state that happens to narrate itself
    — the transport is the graph's problem, not the node's.
    """
    if not runtime.context.stream:
        return None

    writer = runtime.stream_writer

    def emit(text: str) -> None:
        writer({"type": TOKEN_EVENT, "text": text})

    return emit


def _add_usage(running: TokenUsage | None, latest: TokenUsage) -> TokenUsage:
    if running is None:
        return latest
    return TokenUsage(
        prompt_tokens=running.prompt_tokens + latest.prompt_tokens,
        completion_tokens=running.completion_tokens + latest.completion_tokens,
        total_tokens=running.total_tokens + latest.total_tokens,
    )


async def generate_node(state: AgentState, runtime: Runtime[OrchestrationContext]) -> dict:
    """Answer the question, or ask for a tool. Re-entered after each tool round."""
    agent = runtime.context.agent
    chunks = state.get("chunks", [])

    # Resolved per pass, so exhausting the step budget removes the tools from the
    # request rather than merely ignoring a call the model was still invited to
    # make. A model offered a tool it is not allowed to use will keep asking.
    #
    # Resolution became async once tools could live on another team's server. The
    # repeat cost across passes of one turn is a cache lookup, not a round trip —
    # and resolving per pass rather than once is what keeps the agent reading its
    # *current* configuration, the same reason the agent row travels in context.
    tools = (
        await resolve_agent_tools(
            agent=agent, session=runtime.context.session, settings=runtime.context.settings
        )
        if state.get("tool_steps", 0) < MAX_TOOL_STEPS
        else []
    )

    turns = [
        *state.get("history", []),
        Turn(role="user", content=build_grounded_question(state["question"], chunks)),
    ]

    completion = await complete(
        agent=agent,
        turns=turns,
        system_directives=_directives(agent, state, tools),
        tools=[t.to_schema() for t in tools],
        scratchpad=state.get("scratchpad", []),
        on_delta=_delta_writer(runtime),
        settings=runtime.context.settings,
    )

    update: dict = {
        "answer": completion.text,
        "model": completion.model,
        "provider": completion.provider,
        # Accumulated, not replaced. A tool loop makes one model call per pass
        # and the tenant pays for all of them, so reporting only the last would
        # understate the cost of exactly the turns that cost the most — and
        # would make the token metrics in Chunk 6 quietly wrong.
        "usage": _add_usage(state.get("usage"), completion.usage),
        "latency_ms": state.get("latency_ms", 0) + completion.latency_ms,
    }
    if completion.tool_calls:
        # The assistant's own tool-call message must be replayed verbatim next
        # pass, or the results sent back cannot be matched to the calls.
        update["scratchpad"] = [*state.get("scratchpad", []), completion.raw_message]
        update["pending_calls"] = completion.tool_calls
    return update


async def tools_node(state: AgentState, runtime: Runtime[OrchestrationContext]) -> dict:
    """Run the tools the model asked for and hand the results back to it."""
    agent = runtime.context.agent
    allowed = {
        t.name: t
        for t in await resolve_agent_tools(
            agent=agent, session=runtime.context.session, settings=runtime.context.settings
        )
    }

    # Shared across this round's calls so a retrieval tool can number its sources
    # from what the turn already holds, and so what it finds reaches the
    # citations. Seeded from state; merged back below.
    collected = list(state.get("chunks", []))
    context = ToolContext(agent=agent, session=runtime.context.session, chunks=collected)

    results: list[dict] = []
    used: list[str] = []
    for call in state.get("pending_calls", []):
        if runtime.context.stream:
            # Announced before it runs, not after. The point of the event is to
            # explain a pause a caller is already sitting through.
            runtime.stream_writer({"type": TOOL_EVENT, "name": call.name})
        tool = allowed.get(call.name)
        if tool is None:
            # The allowlist is enforced here, at execution, not only by omitting
            # the schema from the request. A model can hallucinate a tool name,
            # and a prompt injection can suggest one — neither may be enough to
            # run something this agent was not granted.
            logger.warning(
                "tool_call_denied",
                agent_id=str(agent.id),
                tenant_id=str(agent.tenant_id),
                tool=call.name,
            )
            output = f"Tool '{call.name}' is not available to this agent."
        else:
            try:
                output = await tool.handler(context, **tool.arguments_from(call.arguments))
            except Exception as exc:  # noqa: BLE001 - handlers touch DB and providers
                # Returned to the model rather than raised: a failing tool is a
                # fact the model can work around by answering from what it
                # already has, and losing the whole turn to it would be worse.
                logger.warning(
                    "tool_execution_failed",
                    agent_id=str(agent.id),
                    tool=call.name,
                    error=type(exc).__name__,
                )
                output = f"Tool '{call.name}' failed and returned no result."
            used.append(call.name)

        results.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {
        "scratchpad": [*state.get("scratchpad", []), *results],
        "chunks": collected,
        "tools_used": [*state.get("tools_used", []), *used],
        "tool_steps": state.get("tool_steps", 0) + 1,
        "pending_calls": [],
    }


def route_after_generate(state: AgentState, runtime: Runtime[OrchestrationContext]) -> str:
    """Loop back through tools only while the model is still asking for them.

    The step cap is re-checked here rather than relying on the generate node
    having withheld the schemas. Withholding them is the graceful half — it stops
    inviting calls the platform would refuse — but a model can emit a tool call
    it was never offered, and only this edge can guarantee the cycle terminates.
    """
    if state.get("pending_calls") and state.get("tool_steps", 0) < MAX_TOOL_STEPS:
        return TOOLS
    return END


def route_entry(state: AgentState, runtime: Runtime[OrchestrationContext]) -> str:
    """Retrieval is conditional on configuration, not on the question.

    An agent with no collection is a legitimate configuration — a pure-prompt or
    (from Chunk 5) pure-tool assistant — so it must not pay for an embedding call
    and a vector scan that can only return nothing.
    """
    return RETRIEVE if runtime.context.agent.collection_id is not None else GENERATE


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState, context_schema=OrchestrationContext)
    graph.add_node(RETRIEVE, retrieve_node)
    graph.add_node(GENERATE, generate_node)
    graph.add_node(TOOLS, tools_node)
    graph.add_conditional_edges(START, route_entry, {RETRIEVE: RETRIEVE, GENERATE: GENERATE})
    graph.add_edge(RETRIEVE, GENERATE)
    graph.add_conditional_edges(GENERATE, route_after_generate, {TOOLS: TOOLS, END: END})
    # The cycle is the difference between a pipeline and an agent: the model
    # decides how many passes its answer needs, bounded by MAX_TOOL_STEPS.
    graph.add_edge(TOOLS, GENERATE)
    return graph


# Compiled once per process: compilation validates the topology and builds the
# executor, and neither depends on the request. The per-request part is the
# context handed to `ainvoke`, which is exactly the part that is not cached.
#
# Two compilations exist because persistence is an operational dependency and
# the graph is not. A checkpointer needs a live connection pool, which a unit
# test, an import, or Alembic must not be made to open — so the default graph
# runs without one and everything below `app/api` stays callable offline. The
# application replaces it during startup, once the pool exists.
_default_graph = build_graph().compile()
_checkpointed_graph = None


def set_checkpointer(checkpointer) -> None:
    """Recompile the process graph against a checkpointer. Called once, at startup."""
    global _checkpointed_graph
    _checkpointed_graph = build_graph().compile(checkpointer=checkpointer)


def clear_checkpointer() -> None:
    global _checkpointed_graph
    _checkpointed_graph = None


def get_agent_graph():
    return _checkpointed_graph or _default_graph
