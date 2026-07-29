"""Tools the platform ships with.

Both are scoped to the calling agent's own collection, taken from the agent row
rather than from an argument. A tool that accepted a collection id would hand the
model a parameter through which a prompt injection could cross the tenant
boundary — the isolation guarantee has to hold no matter what the model is
persuaded to ask for.

`search_knowledge_base` is what turns single-shot retrieval into something worth
calling orchestration: the graph always runs one search on the user's question,
but only the model can tell that "what changed in the policy?" needs a second
search on different terms once it has seen the first result.
"""

from app.core.logging import get_logger
from app.services import document_service
from app.services.retrieval_service import semantic_search
from app.tools.registry import Tool, ToolContext, register

logger = get_logger(__name__)

MAX_TOOL_TOP_K = 10
SNIPPET_LIMIT = 1500


async def search_knowledge_base(context: ToolContext, query: str = "", top_k: int = 0) -> str:
    agent = context.agent
    if agent.collection_id is None:
        return "This agent has no knowledge base configured; no documents are available."
    if not query.strip():
        return "The 'query' argument is required and must be a non-empty search phrase."

    limit = min(top_k or agent.retrieval_top_k, MAX_TOOL_TOP_K)
    results = await semantic_search(
        context.session, query, collection_id=agent.collection_id, top_k=limit
    )

    logger.info(
        "tool_search_completed",
        agent_id=str(agent.id),
        collection_id=str(agent.collection_id),
        chunks=len(results),
    )

    if not results:
        return f"No documents in the knowledge base matched '{query}'."

    # Numbering continues from what the turn has already gathered, and a chunk
    # already shown keeps the number it was first given. Reformulated searches
    # routinely return passages the automatic one found, and renumbering them
    # would silently invalidate a citation the model had already written —
    # while appending them again would cite one passage twice.
    seen = {c.chunk_id: i for i, c in enumerate(context.chunks, start=1)}
    blocks = []
    for chunk in results:
        index = seen.get(chunk.chunk_id)
        if index is None:
            context.chunks.append(chunk)
            index = len(context.chunks)
            seen[chunk.chunk_id] = index
        blocks.append(
            f'<source id="{index}" document="{chunk.filename}">\n'
            f"{chunk.content[:SNIPPET_LIMIT]}\n</source>"
        )
    return "\n\n".join(blocks)


async def list_documents(context: ToolContext) -> str:
    """Let an agent answer "what do you actually know about?" from fact.

    Without it the model guesses at its own coverage, which is the most
    confidently-wrong answer a retrieval assistant can give.
    """
    agent = context.agent
    if agent.collection_id is None:
        return "This agent has no knowledge base configured."

    documents = await document_service.list_documents(
        context.session, tenant_id=agent.tenant_id, collection_id=agent.collection_id
    )
    if not documents:
        return "The knowledge base is empty."

    lines = [f"- {doc.filename} ({count} sections, {doc.status.value})" for doc, count in documents]
    return "Documents available:\n" + "\n".join(lines)


register(
    Tool(
        name="search_knowledge_base",
        description=(
            "Search this agent's document collection for passages relevant to a query. "
            "Use it when the material already provided does not answer the question, or "
            "when a different phrasing would surface something the first search missed. "
            "Returns numbered sources you can cite."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search phrase. Describe the information you need rather than "
                        "repeating the user's wording — the search matches on meaning."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": f"How many passages to return (1-{MAX_TOOL_TOP_K}).",
                    "minimum": 1,
                    "maximum": MAX_TOOL_TOP_K,
                },
            },
            "required": ["query"],
        },
        handler=search_knowledge_base,
    )
)

register(
    Tool(
        name="list_documents",
        description=(
            "List the documents in this agent's knowledge base, with how many sections "
            "each was split into. Use it to answer questions about what you have access "
            "to, rather than guessing at your own coverage."
        ),
        parameters={"type": "object", "properties": {}},
        handler=list_documents,
    )
)
