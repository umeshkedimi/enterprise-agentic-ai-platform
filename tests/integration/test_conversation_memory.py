"""The graph run against a real Postgres checkpointer.

These tests exist because a checkpointer changes what `ainvoke` means. Without
one, every run starts from the state it was handed. With one, the input is
*merged* into whatever the thread already holds — which makes the second turn of
a conversation a genuinely different code path from the first, and one that no
amount of testing the first turn exercises.
"""

import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.agents import graph
from app.agents.checkpointer import start_checkpointer, stop_checkpointer
from app.core.config import get_settings
from tests.integration.conftest import (
    EXPENSES_TEXT,
    VACATION_TEXT,
    create_agent,
    create_collection,
    upload_document,
)


@pytest.fixture
async def checkpointed(app):
    """Run the graph against the real saver, and put it back afterwards.

    The application starts this in its lifespan; ASGITransport does not run
    lifespans, so tests that want persistence ask for it explicitly. That split
    is deliberate — nothing opens a connection pool by being imported.
    """
    await start_checkpointer(get_settings())
    assert graph._checkpointed_graph is not None, "checkpointer failed to start"
    yield
    await stop_checkpointer()


def _answer(text: str, tokens: int = 10):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2
        ),
    )


def _tool_call(name: str, arguments: str):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1", function=SimpleNamespace(name=name, arguments=arguments)
                        )
                    ],
                    model_dump=lambda: {"role": "assistant", "content": None},
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


async def test_a_turn_is_checkpointed_under_its_conversation(
    authed_client, fake_embeddings, provider_creds, checkpointed, monkeypatch
):
    client, _ = authed_client
    agent_id = await create_agent(client, slug="chatty", collection_id=None)

    monkeypatch.setattr(litellm, "acompletion", lambda **kw: _wrap(_answer("Hello.")))

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hi"})
    assert r.status_code == 200, r.text
    conversation_id = r.json()["conversation_id"]

    # The thread id is the conversation id, so a run's execution trace is stored
    # next to the conversation it belongs to rather than under an id only the
    # graph knows.
    state = await graph.get_agent_graph().aget_state(
        {"configurable": {"thread_id": conversation_id}}
    )
    assert state.values["question"] == "hi"
    assert state.values["answer"] == "Hello."


async def test_state_survives_a_round_trip_through_the_checkpoint(
    authed_client, fake_embeddings, provider_creds, checkpointed, monkeypatch
):
    """The state/context split, finally load-bearing.

    Every value in `AgentState` has to serialize. If a live handle — the
    `AsyncSession`, the `Settings`, the `Agent` row — had been kept in state
    rather than in `OrchestrationContext`, this is the test that would fail, and
    it would have failed only once a checkpointer existed to expose it.
    """
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    monkeypatch.setattr(litellm, "acompletion", lambda **kw: _wrap(_answer("Twenty-five [1].")))

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "How much leave?"})
    assert r.status_code == 200, r.text

    state = await graph.get_agent_graph().aget_state(
        {"configurable": {"thread_id": r.json()["conversation_id"]}}
    )
    # Read back from Postgres, not from memory: retrieved chunks and token usage
    # are dataclasses, and they come out the far side intact.
    chunk = state.values["chunks"][0]
    assert "twenty-five days" in chunk.content.lower()
    assert state.values["usage"].total_tokens == 20


async def test_a_second_turn_does_not_inherit_the_first_turns_evidence(
    authed_client, fake_embeddings, provider_creds, checkpointed, monkeypatch
):
    """The merge bite, pinned.

    `ainvoke` merges its input into the checkpointed state, so any turn-scoped
    key the runner does not explicitly reset keeps last turn's value. Left alone,
    turn two would cite turn one's passages, replay turn one's tool scratchpad,
    and start with `tool_steps` already at the cap — an agent that silently loses
    its tools after one question.
    """
    client, _ = authed_client
    collection_id = await create_collection(client, "policies")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    await upload_document(client, collection_id, "expenses.txt", EXPENSES_TEXT)
    agent_id = await create_agent(
        client,
        slug="policy-bot",
        collection_id=collection_id,
        tool_allowlist=["search_knowledge_base"],
        # One passage per search, so each turn's evidence is exactly one chunk
        # and a leak shows up as a citation count rather than as a subtlety.
        retrieval_top_k=1,
    )

    seen: list[dict] = []
    scripted = [
        _tool_call("search_knowledge_base", '{"query": "annual leave accrue"}'),
        _answer("Twenty-five days [1]."),
        _answer("Within thirty days [1]."),
    ]

    async def fake(**kwargs):
        seen.append(kwargs)
        return scripted[min(len(seen) - 1, len(scripted) - 1)]

    monkeypatch.setattr(litellm, "acompletion", fake)

    first = await client.post(
        f"/agents/{agent_id}/chat", json={"message": "vacation leave accrue"}
    )
    assert first.status_code == 200, first.text
    assert first.json()["tools_used"] == ["search_knowledge_base"]
    conversation_id = first.json()["conversation_id"]

    # Second turn, same thread, a question that retrieves the *other* document.
    second = await client.post(
        f"/agents/{agent_id}/chat",
        json={
            "message": "reimbursement receipts travel submitted",
            "conversation_id": conversation_id,
        },
    )

    assert second.status_code == 200, second.text
    body = second.json()
    # One citation, and it is this turn's. Two would mean the vacation passage
    # from turn one had survived in state and been cited under a question about
    # expenses.
    assert len(body["citations"]) == 1
    assert "reimbursement" in body["citations"][0]["snippet"].lower()
    assert body["tools_used"] == []
    # Usage is the turn's total, not the conversation's running total.
    assert body["usage"]["total_tokens"] == 20

    # The last request carries no trace of the first turn's tool exchange.
    last = seen[-1]["messages"]
    assert not any(m.get("role") == "tool" for m in last)
    # Tools are still offered, which they would not be if `tool_steps` had
    # carried over sitting at the cap.
    assert "tools" in seen[-1]


async def test_the_platform_runs_without_a_checkpointer(
    authed_client, fake_embeddings, provider_creds, monkeypatch
):
    """Memory is a feature, not a prerequisite.

    Startup logs and moves on when the checkpointer cannot be reached. The
    degraded mode is well-defined — turns run, history still replays from the
    transcript — and it beats refusing to serve traffic at all.
    """
    client, _ = authed_client
    agent_id = await create_agent(client, slug="chatty", collection_id=None)
    monkeypatch.setattr(litellm, "acompletion", lambda **kw: _wrap(_answer("Hello.")))

    assert graph._checkpointed_graph is None
    first = await client.post(f"/agents/{agent_id}/chat", json={"message": "hi"})
    conversation_id = first.json()["conversation_id"]
    second = await client.post(
        f"/agents/{agent_id}/chat",
        json={"message": "again", "conversation_id": conversation_id},
    )

    assert second.status_code == 200, second.text
    # History still reached the model — it came from the transcript, which is a
    # different store from the checkpoint and does not depend on it.
    assert len(second.json()["answer"]) > 0


async def test_a_broken_checkpointer_does_not_stop_startup(monkeypatch):
    settings = get_settings().model_copy(
        update={"database_url_sync": "postgresql+psycopg://nobody@127.0.0.1:1/none"}
    )
    await start_checkpointer(settings)
    try:
        assert graph._checkpointed_graph is None
    finally:
        await stop_checkpointer()


async def _wrap(value):
    return value


def test_turn_state_resets_every_turn_scoped_key():
    """The reset, asserted against the state schema rather than a snapshot.

    A field added to `AgentState` and forgotten here is precisely the bug this
    guards: it would work in every first turn and leak in every second one.
    """
    from app.agents.runner import _turn_state
    from app.agents.state import AgentState

    state = _turn_state("q", [])
    per_turn = set(AgentState.__annotations__) - {"model", "provider"}
    assert per_turn <= set(state), f"not reset: {per_turn - set(state)}"
    assert state["chunks"] == []
    assert state["scratchpad"] == []
    assert state["tool_steps"] == 0
    assert state["usage"] is None


def test_conversation_id_without_a_thread_is_not_persisted():
    from app.agents.runner import _thread_config

    assert _thread_config(None) == {}
    cid = uuid.uuid4()
    assert _thread_config(cid) == {"configurable": {"thread_id": str(cid)}}


async def test_a_window_overflow_folds_older_turns_into_a_summary(
    authed_client, fake_embeddings, provider_creds, monkeypatch
):
    """The rolling-summary upgrade to history, pinned end-to-end.

    A window of 4 is crossed on the fourth turn: turns one through three have
    left six raw messages on the transcript, one turn's worth (the first two)
    folds into a summary, and that turn's own generate call is the one that
    should carry it — not the raw window, which still replays only what fits.
    """
    monkeypatch.setenv("HISTORY_WINDOW", "4")
    get_settings.cache_clear()

    client, _ = authed_client
    agent_id = await create_agent(client, slug="chatty", collection_id=None)

    calls: list[dict] = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Twenty-five days [1]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(litellm, "acompletion", fake)

    conversation_id = None
    for i in range(4):
        body: dict = {"message": f"question {i}"}
        if conversation_id:
            body["conversation_id"] = conversation_id
        r = await client.post(f"/agents/{agent_id}/chat", json=body)
        assert r.status_code == 200, r.text
        conversation_id = r.json()["conversation_id"]

    get_settings.cache_clear()

    # Three plain turns, then a fourth that pays for one summarization call
    # ahead of its own generate call.
    assert len(calls) == 5

    summary_call = calls[3]
    assert "Summarize the conversation so far" in summary_call["messages"][0]["content"]
    summary_request = summary_call["messages"][-1]["content"]
    assert "<conversation>" in summary_request
    assert "question 0" in summary_request
    assert "<summary_so_far>" not in summary_request

    generate_call = calls[4]
    assert "<conversation_summary>" in generate_call["messages"][-1]["content"]
