"""Model-routing and invocation tests.

The provider call is always faked: the suite must never need network access or
a real API key, and what is under test is the platform's adaptation of stored
agent configuration to a provider — not the provider itself.
"""

import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.core.config import Settings
from app.core.llm import UnknownModelError, resolve_model_route
from app.models.agent import Agent
from app.services import completion_service
from app.services.completion_service import Turn, complete
from app.services.errors import (
    AgentDisabledError,
    ModelConfigurationError,
    ModelInvocationError,
    ProviderNotConfiguredError,
)

# Current-generation Anthropic; rejects `temperature` with a 400.
ANTHROPIC_MODEL = "claude-sonnet-5"
# OpenAI; still accepts sampling parameters.
OPENAI_MODEL = "gpt-4o-mini"

SETTINGS = Settings(openai_api_key="sk-test-openai", anthropic_api_key="sk-test-anthropic")


def make_agent(*, model: str = ANTHROPIC_MODEL, enabled: bool = True) -> Agent:
    return Agent(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        slug="support",
        name="Support",
        system_prompt="You are the support assistant.",
        model=model,
        collection_id=None,
        tool_allowlist=[],
        temperature=0.2,
        max_output_tokens=512,
        retrieval_top_k=5,
        enabled=enabled,
    )


def fake_response(text: str = "hello", model: str = ANTHROPIC_MODEL) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


@pytest.fixture
def captured(monkeypatch):
    """Replace the provider call, recording the parameters it was given."""
    calls: list[dict] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs["model"])

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls


# --- routing -----------------------------------------------------------------


def test_anthropic_model_routes_to_anthropic_credentials():
    route = resolve_model_route(ANTHROPIC_MODEL, SETTINGS)
    assert route.provider == "anthropic"
    assert route.api_key == "sk-test-anthropic"


def test_openai_model_routes_to_openai_credentials():
    route = resolve_model_route(OPENAI_MODEL, SETTINGS)
    assert route.provider == "openai"
    assert route.api_key == "sk-test-openai"


def test_route_reports_sampling_support_per_model():
    # The whole reason temperature cannot be forwarded unconditionally.
    assert resolve_model_route(ANTHROPIC_MODEL, SETTINGS).supports_sampling_params is False
    assert resolve_model_route(OPENAI_MODEL, SETTINGS).supports_sampling_params is True


def test_unroutable_model_is_rejected():
    with pytest.raises(UnknownModelError):
        resolve_model_route("not-a-real-model", SETTINGS)


def test_missing_credential_yields_no_api_key():
    route = resolve_model_route(ANTHROPIC_MODEL, Settings(anthropic_api_key=""))
    assert route.api_key is None


# --- invocation --------------------------------------------------------------


async def test_agents_model_is_honored_at_runtime(captured):
    agent = make_agent(model=OPENAI_MODEL)
    result = await complete(agent=agent, turns=[Turn("user", "hi")], settings=SETTINGS)

    assert captured[0]["model"] == OPENAI_MODEL
    assert result.provider == "openai"


async def test_switching_the_model_switches_the_provider(captured):
    """Two agents differing only in a config string reach different providers."""
    for model, provider in ((OPENAI_MODEL, "openai"), (ANTHROPIC_MODEL, "anthropic")):
        result = await complete(
            agent=make_agent(model=model), turns=[Turn("user", "hi")], settings=SETTINGS
        )
        assert result.provider == provider

    assert [c["api_key"] for c in captured] == ["sk-test-openai", "sk-test-anthropic"]


async def test_system_prompt_leads_the_message_list(captured):
    agent = make_agent()
    await complete(
        agent=agent,
        turns=[Turn("user", "first"), Turn("assistant", "reply"), Turn("user", "second")],
        settings=SETTINGS,
    )

    assert captured[0]["messages"] == [
        {"role": "system", "content": agent.system_prompt},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]


async def test_temperature_sent_to_a_model_that_accepts_it(captured):
    await complete(
        agent=make_agent(model=OPENAI_MODEL), turns=[Turn("user", "hi")], settings=SETTINGS
    )
    assert captured[0]["temperature"] == 0.2


async def test_temperature_withheld_from_a_model_that_rejects_it(captured):
    # Sending it would be a 400; the agent must stay runnable regardless.
    await complete(
        agent=make_agent(model=ANTHROPIC_MODEL), turns=[Turn("user", "hi")], settings=SETTINGS
    )
    assert "temperature" not in captured[0]


async def test_execution_policy_is_applied(captured):
    await complete(agent=make_agent(), turns=[Turn("user", "hi")], settings=SETTINGS)
    assert captured[0]["max_tokens"] == 512


async def test_token_usage_is_surfaced(captured):
    result = await complete(agent=make_agent(), turns=[Turn("user", "hi")], settings=SETTINGS)
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 7)
    assert result.usage.total_tokens == 18


async def test_tool_only_response_yields_empty_text(monkeypatch):
    async def _no_content(**kwargs):
        response = fake_response()
        response.choices[0].message.content = None
        return response

    monkeypatch.setattr(litellm, "acompletion", _no_content)
    result = await complete(agent=make_agent(), turns=[Turn("user", "hi")], settings=SETTINGS)
    assert result.text == ""


# --- failure modes -----------------------------------------------------------


async def test_disabled_agent_refuses_to_run(captured):
    with pytest.raises(AgentDisabledError):
        await complete(
            agent=make_agent(enabled=False), turns=[Turn("user", "hi")], settings=SETTINGS
        )
    assert captured == []


async def test_unroutable_model_is_a_configuration_error(captured):
    with pytest.raises(ModelConfigurationError):
        await complete(
            agent=make_agent(model="not-a-real-model"),
            turns=[Turn("user", "hi")],
            settings=SETTINGS,
        )
    assert captured == []


async def test_missing_provider_credentials_is_distinct_from_bad_config(captured):
    with pytest.raises(ProviderNotConfiguredError):
        await complete(
            agent=make_agent(),
            turns=[Turn("user", "hi")],
            settings=Settings(anthropic_api_key=""),
        )
    assert captured == []


async def test_provider_failure_is_wrapped(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(litellm, "acompletion", _boom)
    with pytest.raises(ModelInvocationError):
        await complete(agent=make_agent(), turns=[Turn("user", "hi")], settings=SETTINGS)


def test_turn_roles_exclude_system():
    # The agent's instructions come from configuration; a caller must not be
    # able to supply its own system turn.
    assert completion_service.Turn.__annotations__["role"].__args__ == ("user", "assistant")


def _stream_chunk(content=None, usage=None):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=content))]
    )
    if usage is not None:
        chunk.usage = usage
    return chunk


async def test_streaming_returns_the_same_completion_it_streams(monkeypatch):
    """Streaming is an argument, not a second code path.

    The caller gets the identical `Completion` either way, which is what lets
    the graph, the tool loop, and token accounting stay unaware of how the
    answer was fetched.
    """
    from litellm.types.utils import Usage

    sent: list[dict] = []
    fragments: list[str] = []

    async def _acompletion(**kwargs):
        sent.append(kwargs)

        async def chunks():
            yield _stream_chunk(content="Hello ")
            yield _stream_chunk(content="world.")
            yield _stream_chunk(
                usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
            )

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    result = await complete(
        agent=make_agent(model=OPENAI_MODEL),
        turns=[Turn(role="user", content="hi")],
        on_delta=fragments.append,
        settings=SETTINGS,
    )

    assert fragments == ["Hello ", "world."]
    assert result.text == "Hello world."
    # Providers omit usage from a stream unless asked for it, so every streamed
    # turn would otherwise bill as zero tokens.
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert result.usage.total_tokens == 18


async def test_an_empty_stream_is_a_provider_failure(monkeypatch):
    async def _acompletion(**kwargs):
        async def chunks():
            return
            yield  # pragma: no cover - makes this an async generator

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    # Reported as an upstream failure rather than returned as an empty answer:
    # an agent that says nothing reads as a bad answer, not as an outage.
    with pytest.raises(ModelInvocationError):
        await complete(
            agent=make_agent(model=OPENAI_MODEL),
            turns=[Turn(role="user", content="hi")],
            on_delta=lambda _: None,
            settings=SETTINGS,
        )


async def test_a_turn_nobody_is_watching_is_not_streamed(monkeypatch, captured):
    await complete(
        agent=make_agent(model=OPENAI_MODEL),
        turns=[Turn(role="user", content="hi")],
        settings=SETTINGS,
    )
    # Streaming has a real cost — a chunked response and a reassembly pass — and
    # a turn with no consumer for the fragments should not pay it.
    assert "stream" not in captured[0]
