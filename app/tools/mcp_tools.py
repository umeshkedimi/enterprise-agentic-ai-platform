"""Turning a remote MCP server into ordinary `Tool` objects.

The design goal is that nothing downstream learns MCP exists. Discovery returns
the same `Tool` dataclass `builtin.py` registers, with a handler that happens to
proxy to a third party — so the graph, the allowlist check, the argument filter,
and the citation logic are all unchanged, and the tool loop cannot tell a remote
tool from a local one.

What *is* different about a remote tool is trust, and that shapes three things
here:

* **Names are namespaced and validated.** A server names its own tools and the
  platform cannot assume the result is legal, let alone unique.
* **Descriptions are third-party text in the model's context.** That is
  unavoidable — a description is how a model decides to call a tool — but it is
  confined to the tool schema, never the system prompt, and length-capped.
* **A server being down is not a failed turn.** Discovery reports the failure and
  the agent runs with the tools it could reach.
"""

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.net import UnsafeUrlError, validate_outbound_url
from app.models.mcp import NAME_SEPARATOR, McpServer
from app.services import mcp_service
from app.tools.registry import Tool, ToolContext

logger = get_logger(__name__)

# Every provider constrains tool names, and they disagree on the ceiling —
# OpenAI allows 64 characters, Anthropic 128. The platform takes the smaller,
# because the model is per-agent configuration and can be changed by a `PATCH`
# after a tool has already been granted: a name that is legal only on today's
# provider is a time bomb in stored config.
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Third-party text that goes into every request this agent makes. Capped for
# tokens and to bound how much room a hostile description has to work in.
MAX_DESCRIPTION_CHARS = 1024

# A tool result is untrusted content the model will read. Capped so one remote
# response cannot fill the context window and push the actual sources out of it.
MAX_RESULT_CHARS = 4000

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True)
class Discovery:
    """What one server offered, or why it offered nothing."""

    tools: list[Tool] = field(default_factory=list)
    error: str | None = None


@asynccontextmanager
async def _connect(server: McpServer, settings: Settings):
    """Open a session to a server for the duration of one operation.

    Re-validated here rather than trusted from registration: a hostname that
    resolved to a public address when it was registered can be re-pointed at a
    private one afterwards, and DNS answers expire. The check that matters is the
    one immediately before the connection.

    Connections are per-operation rather than pooled. A pool keyed by tenant
    would be the optimisation, and it would also be a place for one tenant's
    session to be handed to another — not worth it until discovery shows up in a
    latency profile.
    """
    validate_outbound_url(server.url, settings)

    headers = dict(server.headers)
    token = mcp_service.auth_token_for(server, settings)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx2.AsyncClient(
        headers=headers, timeout=server.timeout_seconds, follow_redirects=False
    ) as http_client:
        transport = streamable_http_client(server.url, http_client=http_client)
        async with Client(transport, read_timeout_seconds=server.timeout_seconds) as client:
            yield client


def _namespaced(server: McpServer, tool_name: str) -> str | None:
    """The name the model and the allowlist both use, or None if unusable.

    The separator guarantees a remote tool can never collide with a built-in one:
    no built-in name contains `__`, so `search_knowledge_base` cannot be shadowed
    by a server that decides to offer a tool of that name.
    """
    name = f"{server.slug}{NAME_SEPARATOR}{tool_name}"
    if not NAME_PATTERN.match(name):
        # Dropped rather than sanitised. A rewritten name would not match what
        # the discovery endpoint told the team owner to put in their allowlist,
        # and sending an illegal name is not a failure of that one tool — the
        # provider rejects the whole request, taking the agent's other tools
        # down with it.
        return None
    return name


def _render(content: Any) -> str:
    """Flatten an MCP result into text the model can read.

    Non-text blocks are described rather than dropped silently: a model that
    asked for a chart and is told nothing came back will simply ask again.
    """
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[{getattr(block, 'type', 'unsupported')} content omitted]")
    rendered = "\n".join(parts).strip()
    if len(rendered) > MAX_RESULT_CHARS:
        # Truncation is announced, so the model treats a cut-off result as
        # partial rather than as the whole answer.
        return rendered[:MAX_RESULT_CHARS] + "\n[result truncated]"
    return rendered or "The tool returned no content."


def _make_handler(server: McpServer, remote_name: str, settings: Settings):
    async def handler(context: ToolContext, **arguments: Any) -> str:
        """Call the remote tool and return its output as text.

        `context` is unused, and that is the point: a remote tool gets no session,
        no collection id, and no access to the turn's evidence. Its entire input
        is the arguments the model supplied, already filtered to the schema the
        server itself published.
        """
        try:
            async with _connect(server, settings) as client:
                result = await client.call_tool(remote_name, arguments)
        except UnsafeUrlError as exc:
            logger.warning(
                "mcp_tool_blocked",
                mcp_server_id=str(server.id),
                tool=remote_name,
                reason=str(exc),
            )
            return f"Tool '{remote_name}' could not be called: its server is not reachable."
        except Exception as exc:  # noqa: BLE001 - transport, protocol, and timeout types
            # Returned to the model rather than raised, like a failing built-in:
            # a third party being down is a fact the model can work around, and
            # losing the whole turn to it would be worse.
            logger.warning(
                "mcp_tool_failed",
                mcp_server_id=str(server.id),
                tenant_id=str(server.tenant_id),
                tool=remote_name,
                error=type(exc).__name__,
            )
            return f"Tool '{remote_name}' failed and returned no result."

        if getattr(result, "is_error", False):
            # The server ran and refused. Its own message is the most useful
            # thing the model can be told, so it is passed through as content.
            logger.info(
                "mcp_tool_returned_error",
                mcp_server_id=str(server.id),
                tool=remote_name,
            )
            return f"Tool '{remote_name}' reported an error: {_render(result.content)}"

        logger.info(
            "mcp_tool_completed",
            mcp_server_id=str(server.id),
            tenant_id=str(server.tenant_id),
            tool=remote_name,
        )
        return _render(result.content)

    return handler


# Discovery is a round trip per server per model call, and a tool loop makes
# several model calls per turn. The cache key includes `updated_at` so editing a
# server's URL or rotating its credential invalidates what was discovered under
# the old configuration, without anything having to remember to clear it.
_CACHE: dict[tuple[str, str], tuple[float, Discovery]] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cache_key(server: McpServer) -> tuple[str, str]:
    return (str(server.id), server.updated_at.isoformat())


async def discover(
    server: McpServer, *, settings: Settings, use_cache: bool = True
) -> Discovery:
    """List a server's tools as platform `Tool` objects.

    Never raises. A server that is unreachable, misconfigured, or blocked comes
    back as a `Discovery` carrying the reason — because the caller is assembling
    the tools for a turn that should still happen, and an exception here would
    turn one team's broken integration into a failed chat request.
    """
    key = _cache_key(server)
    if use_cache:
        cached = _CACHE.get(key)
        if cached and cached[0] > monotonic():
            return cached[1]

    try:
        async with _connect(server, settings) as client:
            listed = await client.list_tools()
        discovery = Discovery(tools=_to_tools(server, listed.tools, settings))
    except UnsafeUrlError as exc:
        discovery = Discovery(error=str(exc))
    except Exception as exc:  # noqa: BLE001 - transport, protocol, and timeout types
        logger.warning(
            "mcp_discovery_failed",
            mcp_server_id=str(server.id),
            tenant_id=str(server.tenant_id),
            slug=server.slug,
            error=type(exc).__name__,
        )
        discovery = Discovery(error=f"{type(exc).__name__}: {exc}")

    # Failures are cached too, for the same TTL. A server that is down would
    # otherwise be dialled on every model call of every turn, adding its timeout
    # to each one — an outage on one integration becoming a latency problem for
    # every agent in the tenant.
    _CACHE[key] = (monotonic() + settings.mcp_tool_cache_ttl_seconds, discovery)
    return discovery


def _to_tools(server: McpServer, remote_tools: list[Any], settings: Settings) -> list[Tool]:
    tools: list[Tool] = []
    skipped: list[str] = []
    for remote in remote_tools:
        name = _namespaced(server, remote.name)
        if name is None:
            skipped.append(remote.name)
            continue
        tools.append(
            Tool(
                name=name,
                description=(remote.description or "")[:MAX_DESCRIPTION_CHARS],
                parameters=remote.input_schema or _EMPTY_SCHEMA,
                handler=_make_handler(server, remote.name, settings),
            )
        )

    if skipped:
        logger.warning(
            "mcp_tools_unusable_names",
            mcp_server_id=str(server.id),
            slug=server.slug,
            tools=skipped,
        )
    logger.info(
        "mcp_discovery_completed",
        mcp_server_id=str(server.id),
        tenant_id=str(server.tenant_id),
        slug=server.slug,
        tools=len(tools),
        skipped=len(skipped),
    )
    return tools
