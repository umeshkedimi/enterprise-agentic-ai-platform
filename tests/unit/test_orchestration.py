"""Orchestration graph behaviour, with retrieval and the provider both faked.

The suite runs offline: `semantic_search` and `litellm.acompletion` are the only
two doors to the outside world, and both are patched. What is being asserted is
routing and prompt assembly — that configuration alone decides whether retrieval
runs, and that retrieved text never reaches the system role.
"""

import uuid

import litellm
import pytest

from app.agents import graph as graph_module
from app.agents.prompts import GROUNDING_DIRECTIVE, NO_RESULTS_DIRECTIVE
from app.agents.runner import run_turn
from app.core.config import Settings
from app.models.agent import Agent
from app.services.completion_service import Turn
from app.services.errors import AgentDisabledError, RetrievalError
from app.services.retrieval_service import RetrievedChunk


def make_agent(**overrides) -> Agent:
    defaults = dict(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        slug="support",
        name="Support Bot",
        system_prompt="You are the support assistant.",
        model="gpt-4o-mini",
        collection_id=uuid.uuid4(),
        tool_allowlist=[],
        temperature=0.2,
        max_output_tokens=512,
        retrieval_top_k=3,
        enabled=True,
    )
    return Agent(**{**defaults, **overrides})


def make_chunk(content: str = "Employees receive 25 days of leave.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="handbook.pdf",
        content=content,
        score=0.91,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="sk-test", anthropic_api_key="")


@pytest.fixture
def captured(monkeypatch):
    """Capture the exact params handed to the provider, and fake its reply."""
    calls: list[dict] = []

    async def fake_acompletion(**params):
        calls.append(params)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "25 days [1]."}}],
            usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
            model=params["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


@pytest.fixture
def searched(monkeypatch):
    """Record retrieval calls; the returned chunks are set per test."""
    calls: list[dict] = []
    result: list[RetrievedChunk] = []

    async def fake_search(session, query, *, collection_id, top_k):
        calls.append({"query": query, "collection_id": collection_id, "top_k": top_k})
        return list(result)

    monkeypatch.setattr(graph_module, "semantic_search", fake_search)
    return calls, result


async def test_agent_with_collection_retrieves_and_grounds(captured, searched, settings):
    calls, result = searched
    result.append(make_chunk())
    agent = make_agent()

    turn = await run_turn(
        agent=agent, question="How much leave?", session=None, settings=settings
    )

    assert len(calls) == 1
    assert calls[0]["collection_id"] == agent.collection_id
    assert calls[0]["top_k"] == 3
    assert turn.answer == "25 days [1]."
    assert turn.retrieved_chunks == 1


async def test_agent_without_collection_skips_retrieval(captured, searched, settings):
    calls, _ = searched

    turn = await run_turn(
        agent=make_agent(collection_id=None),
        question="Hello",
        session=None,
        settings=settings,
    )

    assert calls == []
    assert turn.citations == []
    # No knowledge scope means no grounding contract to impose on the model.
    system = captured[0]["messages"][0]["content"]
    assert GROUNDING_DIRECTIVE not in system


async def test_retrieved_text_goes_to_the_user_turn_not_the_system_prompt(
    captured, searched, settings
):
    _, result = searched
    result.append(make_chunk("SECRET: the vault code is 1234."))

    await run_turn(
        agent=make_agent(), question="What is the code?", session=None, settings=settings
    )

    messages = captured[0]["messages"]
    system, user = messages[0], messages[-1]
    # Document text is data, not instruction: it must never be elevated to the
    # system role, where a malicious upload would read as a platform directive.
    assert "vault code" not in system["content"]
    assert system["role"] == "system"
    assert "vault code" in user["content"]
    assert user["role"] == "user"
    assert "<sources>" in user["content"]


async def test_agent_prompt_precedes_platform_directives(captured, searched, settings):
    _, result = searched
    result.append(make_chunk())

    await run_turn(agent=make_agent(), question="q", session=None, settings=settings)

    system = captured[0]["messages"][0]["content"]
    assert system.startswith("You are the support assistant.")
    assert GROUNDING_DIRECTIVE in system


async def test_empty_retrieval_still_answers_but_is_told_to_admit_it(
    captured, searched, settings
):
    # `searched` is requested but no chunks are appended: the search runs and
    # legitimately finds nothing.
    turn = await run_turn(
        agent=make_agent(), question="Anything?", session=None, settings=settings
    )

    system = captured[0]["messages"][0]["content"]
    assert NO_RESULTS_DIRECTIVE in system
    assert turn.retrieved_chunks == 0
    # An empty search is a legitimate outcome, not an error.
    assert turn.answer


async def test_no_sources_block_when_retrieval_is_empty(captured, searched, settings):
    await run_turn(agent=make_agent(), question="Anything?", session=None, settings=settings)

    user = captured[0]["messages"][-1]["content"]
    assert user == "Anything?"


async def test_history_precedes_the_grounded_question(captured, searched, settings):
    _, result = searched
    result.append(make_chunk())

    await run_turn(
        agent=make_agent(),
        question="And for part-timers?",
        history=[Turn(role="user", content="How much leave?"), Turn("assistant", "25 days.")],
        session=None,
        settings=settings,
    )

    roles = [m["role"] for m in captured[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured[0]["messages"][1]["content"] == "How much leave?"


async def test_conversation_summary_precedes_sources_in_the_user_turn(
    captured, searched, settings
):
    _, result = searched
    result.append(make_chunk())

    await run_turn(
        agent=make_agent(),
        question="And for part-timers?",
        conversation_summary="Earlier, the user asked about full-time leave.",
        session=None,
        settings=settings,
    )

    user = captured[0]["messages"][-1]["content"]
    assert "<conversation_summary>" in user
    assert user.index("<conversation_summary>") < user.index("<sources>")
    assert "Earlier, the user asked about full-time leave." in user


async def test_no_summary_block_when_there_is_nothing_to_summarize(
    captured, searched, settings
):
    await run_turn(
        agent=make_agent(collection_id=None), question="Hello", session=None, settings=settings
    )

    user = captured[0]["messages"][-1]["content"]
    assert "<conversation_summary>" not in user


async def test_disabled_agent_refuses_before_spending_a_retrieval_call(
    captured, searched, settings
):
    calls, _ = searched

    with pytest.raises(AgentDisabledError):
        await run_turn(
            agent=make_agent(enabled=False), question="q", session=None, settings=settings
        )

    assert calls == []
    assert captured == []


async def test_retrieval_failure_fails_the_turn(captured, searched, settings, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("embeddings unavailable")

    monkeypatch.setattr(graph_module, "semantic_search", boom)

    # Answering anyway would turn a grounded assistant into an ungrounded one
    # without anyone noticing, so the turn fails instead.
    with pytest.raises(RetrievalError):
        await run_turn(agent=make_agent(), question="q", session=None, settings=settings)

    assert captured == []


async def test_citations_are_derived_from_what_was_retrieved(captured, searched, settings):
    _, result = searched
    chunk = make_chunk()
    result.append(chunk)

    turn = await run_turn(agent=make_agent(), question="q", session=None, settings=settings)

    assert len(turn.citations) == 1
    assert turn.citations[0].document_id == chunk.document_id
    assert turn.citations[0].chunk_id == chunk.chunk_id
    assert turn.citations[0].score == pytest.approx(0.91)


async def test_usage_and_provider_are_reported_from_the_completion(
    captured, searched, settings
):
    turn = await run_turn(agent=make_agent(), question="q", session=None, settings=settings)

    assert turn.usage.total_tokens == 48
    assert turn.provider == "openai"
    assert turn.latency_ms >= 0
