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
    build_grounded_question,
)
from app.agents.state import AgentState, OrchestrationContext
from app.core.logging import get_logger
from app.services.completion_service import Turn, complete
from app.services.errors import RetrievalError
from app.services.retrieval_service import semantic_search

logger = get_logger(__name__)

RETRIEVE = "retrieve"
GENERATE = "generate"


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


async def generate_node(state: AgentState, runtime: Runtime[OrchestrationContext]) -> dict:
    """Answer the question, grounded in whatever the retrieve node found."""
    agent = runtime.context.agent
    chunks = state.get("chunks", [])

    if agent.collection_id is None:
        # No knowledge scope configured, so there is nothing to ground in and no
        # citation format to impose. The agent runs on its own prompt alone.
        directives: list[str] = []
    elif chunks:
        directives = [GROUNDING_DIRECTIVE]
    else:
        # Retrieval ran and came back empty. Saying so beats letting the model
        # improvise from parametric memory under a prompt that promised sources.
        directives = [GROUNDING_DIRECTIVE, NO_RESULTS_DIRECTIVE]

    turns = [
        *state.get("history", []),
        Turn(role="user", content=build_grounded_question(state["question"], chunks)),
    ]

    completion = await complete(
        agent=agent,
        turns=turns,
        system_directives=directives,
        settings=runtime.context.settings,
    )

    return {
        "answer": completion.text,
        "model": completion.model,
        "provider": completion.provider,
        "usage": completion.usage,
        "latency_ms": completion.latency_ms,
    }


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
    graph.add_conditional_edges(START, route_entry, {RETRIEVE: RETRIEVE, GENERATE: GENERATE})
    graph.add_edge(RETRIEVE, GENERATE)
    graph.add_edge(GENERATE, END)
    return graph


# Compiled once per process: compilation validates the topology and builds the
# executor, and neither depends on the request. The per-request part is the
# context handed to `ainvoke`, which is exactly the part that is not cached.
agent_graph = build_graph().compile()
