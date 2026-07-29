"""The tool registry: what the platform can do, and which agents may do it.

Capability is granted, never inherited. The platform may register any number of
tools, but an agent can invoke only the ones named in its `tool_allowlist` —
which defaults to empty, so a newly-created agent can do nothing until someone
decides otherwise. That direction matters: the opposite default would mean every
tool added to the platform silently widened the reach of every agent already in
production, including agents whose owners never asked for it.

Tools are declared here and resolved per request, so granting one is a `PATCH`
on an agent rather than a deployment — the same property the rest of the platform
is built around.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.agent import Agent
from app.services.retrieval_service import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """What a handler is given: the agent it runs as, and a way to reach data.

    Deliberately narrower than the graph's own context, and defined here rather
    than imported from `app.agents`, so the dependency runs one way — the graph
    depends on the registry, never the reverse. A tool that needed the graph
    would be a sign it belonged in the graph.
    """

    agent: Agent
    session: AsyncSession
    # Evidence gathered so far this turn. A retrieval tool appends to it, which
    # is what puts tool-found sources into the turn's citations and keeps source
    # numbering continuous across the automatic search and any the model runs.
    chunks: list[RetrievedChunk]


# Handlers take the context plus the model's parsed arguments as keywords, and
# return text for the model to read.
ToolHandler = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    # Written for the model, not for a human reader: this is the only thing it
    # has to decide whether calling the tool is appropriate.
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_schema(self) -> dict[str, Any]:
        """Render as an OpenAI-style function schema, which LiteLLM translates
        to each provider's own tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def arguments_from(self, raw: str | None) -> dict[str, Any]:
        """Parse the model's argument blob and keep only declared parameters.

        The published schema is the contract, so anything outside it is dropped
        rather than forwarded. That matters twice over. A model inventing a
        plausible extra argument is common, and letting it through would raise a
        TypeError deep in the handler and cost the call for no reason. More
        importantly, a handler must never receive a parameter it did not declare
        — a suggested `collection_id` is exactly how an injected instruction
        would try to widen a tenant-scoped search, and this is the choke point
        that makes the attempt inert instead of merely unlikely to work.
        """
        declared = set(self.parameters.get("properties", {}))
        parsed = _parse_json_object(raw)
        extra = set(parsed) - declared
        if extra:
            logger.info("tool_arguments_dropped", tool=self.name, arguments=sorted(extra))
        return {k: v for k, v in parsed.items() if k in declared}


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    if tool.name in _REGISTRY:
        raise ValueError(f"Tool '{tool.name}' is already registered.")
    _REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def available_tools() -> list[str]:
    return sorted(_REGISTRY)


def resolve_tools(allowlist: list[str], *, agent_id: str = "") -> list[Tool]:
    """Resolve an agent's allowlist to the tools it may actually call.

    An allowlisted name the platform no longer registers is skipped with a log
    rather than raising. A tool can be retired for reasons that have nothing to
    do with the agents referencing it, and a stale name in stored configuration
    should not take a working assistant offline — the platform's job is to keep a
    configured agent runnable.
    """
    resolved: list[Tool] = []
    unknown: list[str] = []
    for name in allowlist:
        tool = _REGISTRY.get(name)
        if tool is None:
            unknown.append(name)
        else:
            resolved.append(tool)

    if unknown:
        logger.warning("tool_allowlist_unknown_entries", agent_id=agent_id, tools=unknown)
    return resolved


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    """Parse a model-supplied argument blob.

    Malformed JSON is a model mistake, not a platform fault, so it degrades to
    empty arguments and lets the handler's own validation produce an error the
    model can read and retry against.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
