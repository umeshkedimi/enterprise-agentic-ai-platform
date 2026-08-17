"""The parts of MCP integration that are pure decisions, tested offline.

Namespacing, result rendering, URL safety, and credential storage each have a
failure mode that is invisible end to end: a name the provider rejects fails the
*whole* request rather than the one tool, a blocked URL is indistinguishable from
a typo, and a key that cannot decrypt looks like an authentication problem on
somebody else's server.
"""

import uuid
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.crypto import (
    DecryptionError,
    EncryptionUnavailableError,
    decrypt,
    encrypt,
)
from app.core.net import UnsafeUrlError, validate_outbound_url
from app.models.mcp import McpServer
from app.services import mcp_service
from app.tools import mcp_tools

FERNET_KEY = "0S0Vd6vzTHqPq7pEbFqoCH8pXjZ1yqNFYyoxRy_gN2E="
OTHER_KEY = "kA8lLb7hOe2fJmVQKZ9vT3xYsN1cRuWpD6gH0iB4eE0="

SETTINGS = Settings(credential_encryption_key=FERNET_KEY)
NO_KEY = Settings(credential_encryption_key="")
OPEN_NETWORK = Settings(mcp_allow_private_addresses=True)
CLOSED_NETWORK = Settings(mcp_allow_private_addresses=False)


def make_server(slug: str = "weather") -> McpServer:
    return McpServer(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        slug=slug,
        name="Weather",
        url="https://example.test/mcp",
        headers={},
        timeout_seconds=20,
        enabled=True,
    )


def remote(name: str, description: str = "d", schema: dict | None = None):
    return SimpleNamespace(name=name, description=description, input_schema=schema)


# --- namespacing -----------------------------------------------------------


def test_a_remote_tool_is_namespaced_under_its_server():
    tools = mcp_tools._to_tools(make_server(), [remote("forecast")], SETTINGS)
    assert [t.name for t in tools] == ["weather__forecast"]


def test_a_remote_tool_cannot_shadow_a_built_in():
    """The separator is the guarantee, not a convention.

    A server offering `search_knowledge_base` gets
    `weather__search_knowledge_base`, and no built-in name contains `__`, so the
    two namespaces cannot intersect however hostile the server is.
    """
    tools = mcp_tools._to_tools(make_server(), [remote("search_knowledge_base")], SETTINGS)
    assert tools[0].name == "weather__search_knowledge_base"
    assert tools[0].name != "search_knowledge_base"


@pytest.mark.parametrize(
    "name",
    [
        "get weather",  # space
        "weather.forecast",  # dot — legal in MCP, rejected by OpenAI
        "forecast/city",
        "f" * 60,  # namespaced length exceeds the 64-char provider limit
    ],
)
def test_a_name_the_provider_would_reject_is_dropped(name):
    """Dropped, not sanitised, and not passed through.

    An illegal function name does not fail that one tool — the provider rejects
    the entire request, which would take the agent's other tools down with it. And
    a rewritten name would no longer match what the discovery endpoint told the
    team owner to put in their allowlist.
    """
    tools = mcp_tools._to_tools(make_server(), [remote(name)], SETTINGS)
    assert tools == []


def test_a_usable_tool_survives_an_unusable_sibling():
    tools = mcp_tools._to_tools(
        make_server(), [remote("bad name"), remote("forecast")], SETTINGS
    )
    # One server's badly-named tool must not cost the agent the rest of them.
    assert [t.name for t in tools] == ["weather__forecast"]


def test_a_tool_with_no_schema_still_gets_a_valid_one():
    tools = mcp_tools._to_tools(make_server(), [remote("ping", schema=None)], SETTINGS)
    # Providers require an object schema; a server that omits one would otherwise
    # produce a request the provider rejects.
    assert tools[0].parameters == {"type": "object", "properties": {}}


def test_a_long_description_is_capped():
    tools = mcp_tools._to_tools(
        make_server(), [remote("forecast", description="x" * 5000)], SETTINGS
    )
    # Third-party text that ships with every request this agent makes: capped for
    # tokens, and to bound how much room a hostile description has to work in.
    assert len(tools[0].description) == mcp_tools.MAX_DESCRIPTION_CHARS


# --- cache serialization -----------------------------------------------------


def test_a_discovered_tool_round_trips_through_the_cache_encoding():
    """What gets written to Redis and read back must rebuild a working tool.

    The handler cannot itself survive the round trip — it closes over this
    process's HTTP client — so `remote_name` has to be recovered from the
    namespaced name instead. This is the test that would catch getting that
    prefix strip wrong.
    """
    server = make_server()
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    original = mcp_tools._to_tools(server, [remote("forecast", "d", schema)], SETTINGS)
    discovery = mcp_tools.Discovery(tools=original)

    loaded = mcp_tools._load(server, SETTINGS, mcp_tools._dump(discovery))

    assert [t.name for t in loaded.tools] == ["weather__forecast"]
    assert loaded.tools[0].description == "d"
    assert loaded.tools[0].parameters == schema
    assert loaded.tools[0].remote is True
    # A freshly built handler, not the original closure — serializing that was
    # never the point — but bound to the same remote tool name, which is what
    # this test would fail to catch if the prefix strip above were wrong.
    assert loaded.tools[0].handler is not original[0].handler


def test_a_cached_failure_round_trips_with_no_tools():
    loaded = mcp_tools._load(
        make_server(), SETTINGS, mcp_tools._dump(mcp_tools.Discovery(error="ETIMEDOUT"))
    )

    assert loaded.tools == []
    assert loaded.error == "ETIMEDOUT"


# --- result rendering ------------------------------------------------------


def test_text_blocks_are_joined():
    content = [SimpleNamespace(text="one", type="text"), SimpleNamespace(text="two", type="text")]
    assert mcp_tools._render(content) == "one\ntwo"


def test_a_non_text_block_is_described_rather_than_dropped():
    content = [SimpleNamespace(type="image", data="…")]
    # Silence would make the model ask again for something it already received.
    assert "image content omitted" in mcp_tools._render(content)


def test_an_empty_result_says_so():
    assert mcp_tools._render([]) == "The tool returned no content."


def test_an_oversized_result_is_truncated_out_loud():
    rendered = mcp_tools._render([SimpleNamespace(text="x" * 99999, type="text")])
    # Announced, so the model treats a cut-off result as partial rather than
    # complete — and one remote response cannot push the turn's own sources out
    # of the context window.
    assert rendered.endswith("[result truncated]")
    assert len(rendered) < 99999


# --- URL safety ------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/mcp",  # loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/mcp",
        "http://10.0.0.5/mcp",
    ],
)
def test_a_url_inside_the_network_is_refused(url):
    with pytest.raises(UnsafeUrlError):
        validate_outbound_url(url, CLOSED_NETWORK)


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "not-a-url"])
def test_a_scheme_the_platform_does_not_speak_is_refused(url):
    with pytest.raises(UnsafeUrlError):
        validate_outbound_url(url, CLOSED_NETWORK)


def test_local_development_can_opt_into_private_addresses():
    # The opt-in is explicit and defaults to off, so a deployment inherits the
    # safe behaviour and has to choose the weaker one.
    validate_outbound_url("http://127.0.0.1:8931/mcp", OPEN_NETWORK)


# --- credentials -----------------------------------------------------------


def test_a_credential_round_trips():
    assert decrypt(encrypt("secret", SETTINGS), SETTINGS) == "secret"


def test_ciphertext_does_not_contain_the_plaintext():
    assert "secret" not in encrypt("secret", SETTINGS)


def test_encryption_fails_closed_when_no_key_is_configured():
    # An unset secret must never read as "encryption not required" — the same
    # discipline the platform admin token has.
    with pytest.raises(EncryptionUnavailableError):
        encrypt("secret", NO_KEY)


def test_a_malformed_key_is_reported_as_unavailable_not_as_a_crash():
    with pytest.raises(EncryptionUnavailableError):
        encrypt("secret", Settings(credential_encryption_key="not-a-fernet-key"))


def test_a_rotated_key_cannot_read_the_old_ciphertext():
    ciphertext = encrypt("secret", SETTINGS)
    with pytest.raises(DecryptionError):
        decrypt(ciphertext, Settings(credential_encryption_key=OTHER_KEY))


def test_an_unreadable_credential_degrades_to_unauthenticated():
    server = make_server()
    server.auth_token_encrypted = encrypt("secret", SETTINGS)

    token = mcp_service.auth_token_for(server, Settings(credential_encryption_key=OTHER_KEY))

    # None rather than an exception: the usual cause is a rotated key, and the
    # useful outcome is a server that fails to authenticate and is reported
    # unreachable — not a 500 on every chat request that touches it.
    assert token is None


def test_a_server_with_no_credential_needs_no_key():
    assert mcp_service.auth_token_for(make_server(), NO_KEY) is None
