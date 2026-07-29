"""The tool loop: what an agent may call, how often, and what happens when it lies.

The provider is scripted rather than merely stubbed — each test hands back a
sequence of responses so a full generate → tools → generate cycle runs through
the real graph. What is under test is the platform's half of the contract: the
allowlist, the step cap, and the failure modes, none of which may depend on the
model behaving.
"""

import json
import uuid

import litellm
import pytest

from app.agents import graph as graph_module
from app.agents.graph import MAX_TOOL_STEPS
from app.agents.runner import run_turn
from app.core.config import Settings
from app.services.retrieval_service import RetrievedChunk
from app.tools import registry
from tests.unit.test_orchestration import make_agent, make_chunk


def tool_call_response(name: str, arguments: dict, *, call_id: str = "call_1"):
    return litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="gpt-4o-mini",
    )


def text_response(text: str):
    return litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": text}}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="gpt-4o-mini",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="sk-test")


@pytest.fixture
def script(monkeypatch):
    """Queue provider responses; returns (calls, queue). The last is repeated."""
    calls: list[dict] = []
    queue: list = []

    async def fake_acompletion(**params):
        calls.append(params)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls, queue


@pytest.fixture
def searched(monkeypatch):
    """Fake both the graph's automatic search and the search tool's."""
    results: dict[str, list[RetrievedChunk]] = {"initial": [], "tool": []}

    async def fake_search(session, query, *, collection_id, top_k):
        # The graph's first search runs on the raw question; anything after is
        # the model having reformulated it through the tool.
        key = "initial" if query == "How much leave?" else "tool"
        return list(results[key])

    monkeypatch.setattr(graph_module, "semantic_search", fake_search)
    monkeypatch.setattr("app.tools.builtin.semantic_search", fake_search)
    return results


async def test_agent_with_no_allowlist_is_offered_no_tools(script, searched, settings):
    calls, queue = script
    queue.append(text_response("Answer."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=[]), question="q", session=None, settings=settings
    )

    assert "tools" not in calls[0]
    assert turn.tools_used == []


async def test_allowlisted_tools_are_offered_as_schemas(script, searched, settings):
    calls, queue = script
    queue.append(text_response("Answer."))

    await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="q",
        session=None,
        settings=settings,
    )

    offered = [t["function"]["name"] for t in calls[0]["tools"]]
    # Only what was granted — the platform registers more than this.
    assert offered == ["search_knowledge_base"]
    assert "list_documents" in registry.available_tools()


async def test_a_tool_call_runs_and_the_model_answers_from_its_result(
    script, searched, settings
):
    calls, queue = script
    searched["tool"] = [make_chunk("Part-time staff accrue pro-rata leave.")]
    queue.append(tool_call_response("search_knowledge_base", {"query": "part-time leave"}))
    queue.append(text_response("Pro-rata [1]."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    assert turn.tools_used == ["search_knowledge_base"]
    assert turn.answer == "Pro-rata [1]."
    # Second pass replays the assistant's tool-call message and the result, so
    # the model can see what it asked for and what came back.
    scratchpad = calls[1]["messages"][-2:]
    assert scratchpad[0]["role"] == "assistant"
    assert scratchpad[1]["role"] == "tool"
    assert scratchpad[1]["tool_call_id"] == "call_1"
    assert "pro-rata" in scratchpad[1]["content"].lower()


async def test_evidence_found_by_a_tool_becomes_a_citation(script, searched, settings):
    _, queue = script
    chunk = make_chunk("Part-time staff accrue pro-rata leave.")
    searched["tool"] = [chunk]
    queue.append(tool_call_response("search_knowledge_base", {"query": "part-time"}))
    queue.append(text_response("Pro-rata [1]."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    # A source the model went looking for is as citable as one the platform
    # fetched automatically — otherwise a tool-assisted answer looks unsupported.
    assert [c.chunk_id for c in turn.citations] == [chunk.chunk_id]


async def test_tool_sources_are_numbered_after_the_automatic_ones(
    script, searched, settings
):
    calls, queue = script
    searched["initial"] = [make_chunk("Full-time: 25 days.")]
    searched["tool"] = [make_chunk("Part-time: pro-rata.")]
    queue.append(tool_call_response("search_knowledge_base", {"query": "part-time"}))
    queue.append(text_response("Both [1][2]."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    # Numbering continues rather than restarting, so a citation means the same
    # thing whichever search produced it.
    assert '<source id="2"' in calls[1]["messages"][-1]["content"]
    assert len(turn.citations) == 2


async def test_a_passage_found_twice_is_cited_once_and_keeps_its_number(
    script, searched, settings
):
    calls, queue = script
    shared = make_chunk("Full-time: 25 days.")
    searched["initial"] = [shared]
    # A reformulated search returns the passage the automatic one already found,
    # plus one it did not — the common case, not an edge case.
    searched["tool"] = [shared, make_chunk("Part-time: pro-rata.")]
    queue.append(tool_call_response("search_knowledge_base", {"query": "leave accrual"}))
    queue.append(text_response("Both [1][2]."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    assert len(turn.citations) == 2
    tool_result = calls[1]["messages"][-1]["content"]
    # Renumbering a passage would invalidate a citation the model already wrote.
    assert '<source id="1"' in tool_result
    assert '<source id="2"' in tool_result
    assert '<source id="3"' not in tool_result


async def test_a_tool_outside_the_allowlist_is_refused_at_execution(
    script, searched, settings
):
    calls, queue = script
    queue.append(tool_call_response("list_documents", {}))
    queue.append(text_response("I cannot do that."))

    turn = await run_turn(
        # Granted search only; the model asks for a registered tool it was
        # never offered — a hallucination, or an injection that suggested one.
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    assert turn.tools_used == []
    result = calls[1]["messages"][-1]
    assert result["role"] == "tool"
    assert "not available to this agent" in result["content"]


async def test_an_unregistered_name_in_the_allowlist_does_not_break_the_agent(
    script, searched, settings
):
    calls, queue = script
    queue.append(text_response("Answer."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base", "retired_tool"]),
        question="q",
        session=None,
        settings=settings,
    )

    # A tool can be retired for reasons unrelated to the agents naming it;
    # stale config must not take a working assistant offline.
    assert [t["function"]["name"] for t in calls[0]["tools"]] == ["search_knowledge_base"]
    assert turn.answer == "Answer."


async def test_the_loop_stops_at_the_step_cap(script, searched, settings):
    calls, queue = script
    # A model that never stops asking. Only the platform can end this.
    queue.append(tool_call_response("search_knowledge_base", {"query": "again"}))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    assert len(turn.tools_used) == MAX_TOOL_STEPS
    # One generate per tool round, plus the final pass that is offered no tools.
    assert len(calls) == MAX_TOOL_STEPS + 1
    assert "tools" not in calls[-1]


async def test_token_usage_covers_every_pass_of_the_loop(script, searched, settings):
    _, queue = script
    queue.append(tool_call_response("search_knowledge_base", {"query": "x"}))
    queue.append(text_response("Answer."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    # Two model calls at 15 tokens each. Reporting only the last would understate
    # exactly the turns that cost the most.
    assert turn.usage.total_tokens == 30
    assert turn.usage.prompt_tokens == 20


async def test_a_failing_tool_is_reported_to_the_model_not_raised(
    script, searched, settings, monkeypatch
):
    calls, queue = script
    queue.append(tool_call_response("search_knowledge_base", {"query": "x"}))
    queue.append(text_response("I could not look that up."))

    async def boom(session, query, *, collection_id, top_k):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("app.tools.builtin.semantic_search", boom)

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    # The turn survives: a broken tool is a fact the model can work around,
    # and losing the whole answer to it would be the worse outcome.
    assert turn.answer == "I could not look that up."
    assert "failed and returned no result" in calls[1]["messages"][-1]["content"]


async def test_malformed_tool_arguments_reach_the_handler_as_empty(
    script, searched, settings
):
    calls, queue = script
    bad = litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": "{not json",
                            },
                        }
                    ],
                }
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        model="gpt-4o-mini",
    )
    queue.append(bad)
    queue.append(text_response("Let me rephrase."))

    turn = await run_turn(
        agent=make_agent(tool_allowlist=["search_knowledge_base"]),
        question="How much leave?",
        session=None,
        settings=settings,
    )

    # The handler's own validation answers, so the model reads a correction it
    # can act on instead of the turn dying on a JSON parse error.
    assert "'query' argument is required" in calls[1]["messages"][-1]["content"]
    assert turn.answer == "Let me rephrase."


async def test_a_model_that_cannot_call_tools_still_answers(script, searched, settings):
    calls, queue = script
    queue.append(text_response("Answer."))

    turn = await run_turn(
        # `supports_function_calling` is unknown for an unmapped model, so the
        # platform withholds the schemas rather than sending ones the provider
        # would reject.
        agent=make_agent(
            model="openai/some-unmapped-model", tool_allowlist=["search_knowledge_base"]
        ),
        question="q",
        session=None,
        settings=settings,
    )

    assert "tools" not in calls[0]
    # Downgraded, not failed: an allowlist outlives any one model choice.
    assert turn.answer == "Answer."


async def test_search_tool_scope_comes_from_config_not_from_arguments(
    script, searched, settings, monkeypatch
):
    _, queue = script
    seen: list[uuid.UUID] = []

    async def spy(session, query, *, collection_id, top_k):
        seen.append(collection_id)
        return []

    monkeypatch.setattr("app.tools.builtin.semantic_search", spy)
    queue.append(
        tool_call_response(
            "search_knowledge_base",
            # A model persuaded to reach into another tenant's collection.
            {"query": "x", "collection_id": str(uuid.uuid4())},
        )
    )
    queue.append(text_response("Nothing found."))

    agent = make_agent(tool_allowlist=["search_knowledge_base"])
    await run_turn(agent=agent, question="How much leave?", session=None, settings=settings)

    # The extra argument is not in the schema and the handler has no parameter
    # for it, so the search is scoped by the agent row regardless of what the
    # model asked for. Isolation cannot be argued out of.
    assert seen == [agent.collection_id]
