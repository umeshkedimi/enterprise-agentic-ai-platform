"""Resolving one agent's allowlist to the tools it may actually call.

Chunk 3 could answer this from a dict in memory. It cannot any more: half the
answer now lives in another team's HTTP server. What has *not* changed is the
rule — an agent can call exactly the names in its `tool_allowlist`, and nothing
else — so the interesting work here is keeping that rule intact while the set of
resolvable names became tenant-scoped, remote, and occasionally unreachable.

Three properties this function is responsible for:

* **A remote name resolves only within its own tenant.** An allowlist entry of
  `otherteam__transfer_funds` resolves to nothing, because the servers consulted
  are the ones the agent's tenant registered. Namespacing is for legibility;
  tenancy is what makes it safe.
* **An agent that names no remote tools pays nothing.** No database read, no
  round trip. The cost of the integration falls on the agents configured to use
  it, not on every agent in the platform.
* **Order follows the allowlist.** The agent's configuration decides what the
  model sees first, rather than whichever server answered fastest.
"""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.agent import Agent
from app.models.mcp import NAME_SEPARATOR
from app.services import mcp_service
from app.tools import mcp_tools
from app.tools.registry import Tool, get_tool

logger = get_logger(__name__)


async def resolve_agent_tools(
    *, agent: Agent, session: AsyncSession, settings: Settings | None = None
) -> list[Tool]:
    """The tools `agent` may call, built-in and remote, in allowlist order.

    An allowlisted name that resolves to nothing is skipped with a log rather
    than raising, exactly as a retired built-in is. A tool can vanish for reasons
    that have nothing to do with the agents referencing it — a server deleted, an
    integration renamed — and a stale name in stored configuration must not take a
    working assistant offline.
    """
    settings = settings or get_settings()
    wanted = list(agent.tool_allowlist)

    resolved: dict[str, Tool] = {}
    for name in wanted:
        tool = get_tool(name)
        if tool is not None:
            resolved[name] = tool

    outstanding = [n for n in wanted if n not in resolved]
    if any(NAME_SEPARATOR in name for name in outstanding):
        resolved.update(
            await _resolve_remote(
                outstanding, tenant_id=agent.tenant_id, session=session, settings=settings
            )
        )

    unknown = [name for name in wanted if name not in resolved]
    if unknown:
        logger.warning(
            "tool_allowlist_unknown_entries", agent_id=str(agent.id), tools=unknown
        )

    return [resolved[name] for name in wanted if name in resolved]


async def _resolve_remote(
    outstanding: list[str], *, tenant_id, session: AsyncSession, settings: Settings
) -> dict[str, Tool]:
    """Discover only the servers this allowlist actually names.

    A tenant with a dozen registered servers and an agent granted one tool from
    one of them should cost one round trip, not twelve. The slug prefix of the
    unresolved names is exactly the filter for that.
    """
    wanted_slugs = {
        name.split(NAME_SEPARATOR, 1)[0] for name in outstanding if NAME_SEPARATOR in name
    }
    servers = [
        server
        for server in await mcp_service.list_enabled_servers(session, tenant_id=tenant_id)
        if server.slug in wanted_slugs
    ]
    if not servers:
        return {}

    # Concurrently, because these are independent third parties and a slow one
    # should not add its latency to the others'. `discover` never raises, so
    # gather needs no exception handling — a failed server contributes an empty
    # tool list and a logged reason.
    discoveries = await asyncio.gather(
        *(mcp_tools.discover(server, settings=settings) for server in servers)
    )

    outstanding_set = set(outstanding)
    return {
        tool.name: tool
        for discovery in discoveries
        for tool in discovery.tools
        # Discovery returns everything a server offers; the allowlist decides what
        # the agent gets. A server that adds fifty tools tomorrow widens nothing.
        if tool.name in outstanding_set
    }
