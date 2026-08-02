"""MCP integration, end to end against a real MCP server.

The server here is a genuine `MCPServer` with genuine tools, reached over the
SDK's in-memory transport. Only the *network hop* is removed — the protocol
handshake, the tool listing, the argument marshalling, and the result content
blocks are all real. That matters because everything this chunk gets wrong lives
in that layer: a tool name the provider will reject, a result shape that renders
to nothing, an error the model never sees.
"""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import litellm
import pytest
from httpx import ASGITransport, AsyncClient
from mcp import Client
from mcp.server.mcpserver import MCPServer

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services import mcp_service
from app.tools import mcp_tools
from tests.integration.conftest import _delete_tenant, create_agent

# A local address, so the SSRF guard has something that resolves without DNS.
LOCAL_URL = "http://127.0.0.1:8931/mcp"
FERNET_KEY = "0S0Vd6vzTHqPq7pEbFqoCH8pXjZ1yqNFYyoxRy_gN2E="


@pytest.fixture(autouse=True)
def clear_tool_cache():
    """Discovery is cached per server; tests must not inherit each other's."""
    mcp_tools.clear_cache()
    yield
    mcp_tools.clear_cache()


@pytest.fixture
def mcp_env(app, monkeypatch):
    """Settings a tenant needs to register a server at all.

    Private addresses are allowed here because the fixture server is local — the
    guard itself is tested separately, with the production default.
    """
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", FERNET_KEY)
    monkeypatch.setenv("MCP_ALLOW_PRIVATE_ADDRESSES", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build_mcp_server() -> MCPServer:
    server = MCPServer("weather")

    @server.tool()
    def forecast(city: str) -> str:
        """Return tomorrow's forecast for a city."""
        return f"Tomorrow in {city}: 18C, light rain."

    @server.tool()
    def failing() -> str:
        """A tool that always raises, to exercise the failure path."""
        raise RuntimeError("upstream exploded")

    return server


@pytest.fixture
def remote_server(monkeypatch):
    """Point the MCP client at an in-process server, and record what it was asked.

    `_connect` is the seam on purpose: everything above it — discovery,
    namespacing, caching, the proxy handler, the allowlist check — is the code
    under test, and everything below it is a socket.
    """
    server = build_mcp_server()
    connections: list[str] = []

    @asynccontextmanager
    async def _connect(srv, settings):
        connections.append(srv.slug)
        async with Client(server) as client:
            yield client

    monkeypatch.setattr(mcp_tools, "_connect", _connect)
    return connections


@pytest.fixture
def unreachable_server(monkeypatch):
    @asynccontextmanager
    async def _connect(srv, settings):
        raise ConnectionError("connection refused")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(mcp_tools, "_connect", _connect)


async def register_server(client, *, slug="weather", **overrides) -> dict:
    payload = {
        "slug": slug,
        "name": "Weather Service",
        "url": LOCAL_URL,
        **overrides,
    }
    r = await client.post("/mcp-servers", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def scripted_provider(monkeypatch, responses):
    """Fake the chat provider with a fixed script, recording each request."""
    seen: list[dict] = []

    async def _acompletion(**kwargs):
        seen.append(kwargs)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    return seen


def answer(text: str):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def tool_call(name: str, arguments: str):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                    model_dump=lambda: {"role": "assistant", "content": None},
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


# --- configuration plane ---------------------------------------------------


async def test_a_credential_goes_in_and_never_comes_back(authed_client, mcp_env):
    client, _ = authed_client
    created = await register_server(client, auth_token="secret-token-123")

    # The fact of a credential is reported; the credential is not.
    assert created["authenticated"] is True
    assert "auth_token" not in created

    listed = await client.get("/mcp-servers")
    body = listed.json()["items"][0]
    assert body["authenticated"] is True
    assert "secret-token-123" not in listed.text


async def test_a_url_that_resolves_inside_the_network_is_refused(app, authed_client, monkeypatch):
    client, _ = authed_client
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", FERNET_KEY)
    monkeypatch.setenv("MCP_ALLOW_PRIVATE_ADDRESSES", "false")
    get_settings.cache_clear()

    r = await client.post(
        "/mcp-servers",
        json={"slug": "metadata", "name": "Nope", "url": "http://127.0.0.1:80/mcp"},
    )

    # The platform dials this URL itself, from inside its own network, holding its
    # own cloud identity. A tenant pointing it at a loopback or link-local address
    # is asking the platform to read its own internals out as a tool result.
    assert r.status_code == 400
    assert "non-public" in r.json()["detail"]
    get_settings.cache_clear()


async def test_a_credential_is_refused_when_it_cannot_be_stored_safely(
    app, authed_client, monkeypatch
):
    client, _ = authed_client
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("MCP_ALLOW_PRIVATE_ADDRESSES", "true")
    get_settings.cache_clear()

    r = await client.post(
        "/mcp-servers",
        json={"slug": "weather", "name": "W", "url": LOCAL_URL, "auth_token": "t"},
    )

    # 503, not 400: the tenant's request is fine and the platform's wiring is not.
    # Refusing beats writing somebody else's secret to disk in the clear.
    assert r.status_code == 503
    get_settings.cache_clear()


async def test_a_server_with_no_credential_needs_no_encryption_key(
    app, authed_client, monkeypatch
):
    client, _ = authed_client
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("MCP_ALLOW_PRIVATE_ADDRESSES", "true")
    get_settings.cache_clear()

    created = await register_server(client)
    assert created["authenticated"] is False
    get_settings.cache_clear()


async def test_slugs_are_unique_per_tenant(authed_client, mcp_env):
    client, _ = authed_client
    await register_server(client)
    r = await client.post("/mcp-servers", json={"slug": "weather", "name": "W", "url": LOCAL_URL})
    # The slug is a namespace, so a duplicate would make two servers' tools
    # indistinguishable in an allowlist.
    assert r.status_code == 409


async def test_another_tenants_server_is_404(app, authed_client, admin_headers, mcp_env):
    client, _ = authed_client
    mine = await register_server(client)

    other = await client.post(
        "/tenants",
        json={"slug": f"o-{uuid.uuid4().hex[:8]}", "name": "Other"},
        headers=admin_headers,
    )
    other_id = uuid.UUID(other.json()["id"])
    key = await client.post(
        f"/tenants/{other_id}/keys", json={"name": "k"}, headers=admin_headers
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as intruder:
        intruder.headers["Authorization"] = f"Bearer {key.json()['api_key']}"
        r = await intruder.get(f"/mcp-servers/{mine['id']}")

    assert r.status_code == 404
    await _delete_tenant(other_id)


# --- discovery -------------------------------------------------------------


async def test_discovery_reports_the_names_an_allowlist_has_to_use(
    authed_client, mcp_env, remote_server
):
    client, _ = authed_client
    server = await register_server(client)

    r = await client.get(f"/mcp-servers/{server['id']}/tools")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reachable"] is True
    names = {t["name"] for t in body["tools"]}
    # Namespaced, because a remote server names its own tools and two servers can
    # both offer a `search`. The `__` separator also means a remote tool can never
    # shadow a built-in — no built-in name contains it.
    assert names == {"weather__forecast", "weather__failing"}


async def test_an_unreachable_server_is_an_answer_not_an_error(
    authed_client, mcp_env, unreachable_server
):
    client, _ = authed_client
    server = await register_server(client)

    r = await client.get(f"/mcp-servers/{server['id']}/tools")

    # 200 with reachable=false. "Your server is down" is an answer to "what does
    # my server offer", and a team owner debugging their own integration needs the
    # reason rather than a 502.
    assert r.status_code == 200, r.text
    assert r.json()["reachable"] is False
    assert "ConnectionError" in r.json()["error"]


# --- runtime ---------------------------------------------------------------


async def test_an_agent_can_call_a_tool_the_platform_knows_nothing_about(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(
        client, slug="weather-bot", collection_id=None, tool_allowlist=["weather__forecast"]
    )

    seen = scripted_provider(
        monkeypatch,
        [
            tool_call("weather__forecast", '{"city": "Berlin"}'),
            answer("Light rain tomorrow."),
        ],
    )

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "weather in Berlin?"})

    assert r.status_code == 200, r.text
    assert r.json()["tools_used"] == ["weather__forecast"]
    # The whole integration was a POST. Nothing in this repository knows what
    # `forecast` does, and the tool loop cannot tell it from a built-in.
    offered = [t["function"]["name"] for t in seen[0]["tools"]]
    assert offered == ["weather__forecast"]
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert "18C, light rain" in seen[1]["messages"][-1]["content"]


async def test_a_remote_tool_is_still_only_available_when_granted(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(client, slug="bare-bot", collection_id=None)

    seen = scripted_provider(
        monkeypatch,
        [tool_call("weather__forecast", '{"city": "Berlin"}'), answer("I cannot check that.")],
    )

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "weather in Berlin?"})

    assert r.status_code == 200, r.text
    # Never offered — the allowlist is empty, so registering a server widened
    # nothing for an agent nobody granted it to.
    assert "tools" not in seen[0]
    # And the call it invented anyway was refused at execution, not merely
    # unoffered. A server is not dialled for a tool the agent may not call.
    assert r.json()["tools_used"] == []
    assert "not available to this agent" in seen[1]["messages"][-1]["content"]
    assert remote_server == []


async def test_a_tool_name_does_not_resolve_across_tenants(
    app, authed_client, admin_headers, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)

    other = await client.post(
        "/tenants",
        json={"slug": f"o-{uuid.uuid4().hex[:8]}", "name": "Other"},
        headers=admin_headers,
    )
    other_id = uuid.UUID(other.json()["id"])
    key = await client.post(
        f"/tenants/{other_id}/keys", json={"name": "k"}, headers=admin_headers
    )

    seen = scripted_provider(monkeypatch, [answer("Nothing I can do.")])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as intruder:
        intruder.headers["Authorization"] = f"Bearer {key.json()['api_key']}"
        # The intruder knows the exact tool name and grants it to its own agent.
        agent_id = await create_agent(
            intruder, slug="thief", collection_id=None, tool_allowlist=["weather__forecast"]
        )
        r = await intruder.post(f"/agents/{agent_id}/chat", json={"message": "weather?"})

    assert r.status_code == 200, r.text
    # Namespacing is for legibility; tenancy is what makes it safe. The servers
    # consulted are the ones this agent's own tenant registered, so a known name
    # from another tenant resolves to nothing.
    assert "tools" not in seen[0]
    assert remote_server == []
    await _delete_tenant(other_id)


async def test_a_failing_remote_tool_does_not_fail_the_turn(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(
        client, slug="weather-bot", collection_id=None, tool_allowlist=["weather__failing"]
    )

    seen = scripted_provider(
        monkeypatch,
        [tool_call("weather__failing", "{}"), answer("That service is unavailable.")],
    )

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "try it"})

    assert r.status_code == 200, r.text
    # The server ran the tool and it raised. Its own message reaches the model,
    # which can then say so — losing the whole turn to a third party's bug would
    # be worse than answering around it.
    assert "reported an error" in seen[1]["messages"][-1]["content"]
    assert "upstream exploded" in seen[1]["messages"][-1]["content"]


async def test_an_unreachable_server_still_leaves_a_working_agent(
    authed_client, mcp_env, unreachable_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(
        client, slug="weather-bot", collection_id=None, tool_allowlist=["weather__forecast"]
    )

    seen = scripted_provider(monkeypatch, [answer("I answered without tools.")])

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hello"})

    # One team's broken integration must not be an outage for their assistant.
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "I answered without tools."
    assert "tools" not in seen[0]


async def test_a_disabled_server_is_visible_but_unreachable(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    server = await register_server(client)
    await client.patch(f"/mcp-servers/{server['id']}", json={"enabled": False})
    agent_id = await create_agent(
        client, slug="weather-bot", collection_id=None, tool_allowlist=["weather__forecast"]
    )

    seen = scripted_provider(monkeypatch, [answer("No tools today.")])
    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hello"})

    assert r.status_code == 200, r.text
    # Same split `enabled` has on an agent: still configured, still listed to its
    # owner, not reachable by the runtime.
    assert "tools" not in seen[0]
    assert remote_server == []
    assert (await client.get("/mcp-servers")).json()["items"][0]["enabled"] is False


async def test_an_agent_that_names_no_remote_tools_dials_nothing(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(
        client, slug="local-bot", collection_id=None, tool_allowlist=["list_documents"]
    )

    scripted_provider(monkeypatch, [answer("Hello.")])
    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hello"})

    assert r.status_code == 200, r.text
    # The cost of an integration falls on the agents configured to use it. A
    # built-in-only allowlist reads no rows and opens no connections, however many
    # servers the tenant has registered.
    assert remote_server == []


async def test_discovery_is_cached_across_the_calls_of_one_turn(
    authed_client, mcp_env, remote_server, provider_creds, monkeypatch
):
    client, _ = authed_client
    await register_server(client)
    agent_id = await create_agent(
        client, slug="weather-bot", collection_id=None, tool_allowlist=["weather__forecast"]
    )

    scripted_provider(
        monkeypatch,
        [tool_call("weather__forecast", '{"city": "Oslo"}'), answer("Rain.")],
    )
    await client.post(f"/agents/{agent_id}/chat", json={"message": "weather?"})

    # Tools are resolved on every model call and again in the tools node — four
    # resolutions this turn. Only the listing is cached, so the connections left
    # are the two model-call listings collapsing to one, plus the actual tool call.
    assert remote_server.count("weather") == 2


async def test_editing_a_server_invalidates_what_was_discovered_under_it(
    authed_client, mcp_env, remote_server
):
    client, tenant_id = authed_client
    created = await register_server(client)
    server_id = uuid.UUID(created["id"])

    async with async_session_factory() as session:
        server = await mcp_service.get_server(
            session, tenant_id=tenant_id, server_id=server_id
        )
        await mcp_tools.discover(server, settings=get_settings())
        await mcp_tools.discover(server, settings=get_settings())
        # Second call served from cache.
        assert remote_server.count("weather") == 1

        await client.patch(f"/mcp-servers/{created['id']}", json={"timeout_seconds": 5})
        session.expire_all()
        edited = await mcp_service.get_server(
            session, tenant_id=tenant_id, server_id=server_id
        )
        await mcp_tools.discover(edited, settings=get_settings())

    # The cache key carries `updated_at`, so a rotated credential or a changed URL
    # cannot serve tools discovered under the old configuration — and nothing has
    # to remember to clear it.
    assert remote_server.count("weather") == 2
