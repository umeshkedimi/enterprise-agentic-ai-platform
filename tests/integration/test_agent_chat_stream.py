"""The streamed turn, over the real SSE endpoint.

The provider is faked at the streaming boundary — an async iterator of chunks in
LiteLLM's shape — rather than at ours, so what these tests exercise includes the
reassembly of those chunks into a completion. That reassembly is where a streamed
turn could quietly stop counting tokens or lose a tool call, and it is worth
having under test for exactly that reason.
"""

import json
from types import SimpleNamespace

import litellm
import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from tests.integration.conftest import (
    VACATION_TEXT,
    create_agent,
    create_collection,
    upload_document,
)


def _delta(content=None, tool_calls=None, usage=None) -> ModelResponseStream:
    """One streamed chunk, in LiteLLM's own types.

    Built from the real classes rather than a stand-in, because the code under
    test hands these straight to `stream_chunk_builder` — a duck-typed fake would
    only prove that a fake reassembles.
    """
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=content, tool_calls=tool_calls))]
    )
    if usage is not None:
        chunk.usage = usage
    return chunk


def _usage(prompt: int, completion: int) -> Usage:
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


@pytest.fixture
def streamed_provider(monkeypatch):
    """A provider that emits an answer word by word, like a real one does."""
    seen: list[dict] = []

    async def _acompletion(**kwargs):
        seen.append(kwargs)
        if not kwargs.get("stream"):
            return SimpleNamespace(
                model=kwargs["model"],
                choices=[SimpleNamespace(message=SimpleNamespace(content="Twenty-five days."))],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=4, total_tokens=14
                ),
            )

        async def chunks():
            for word in ("Twenty-five ", "days ", "per ", "year."):
                yield _delta(content=word)
            yield _delta(usage=_usage(10, 4))

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    return seen


async def _events(client, url: str, payload: dict) -> list[tuple[str, dict]]:
    """Read an SSE response into (event name, payload) pairs."""
    events: list[tuple[str, dict]] = []
    async with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200, await response.aread()
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[len("data: ") :])))
    return events


async def test_the_answer_arrives_in_fragments_then_whole(
    authed_client, fake_embeddings, provider_creds, streamed_provider
):
    client, _ = authed_client
    agent_id = await create_agent(client, slug="chatty", collection_id=None)

    events = await _events(client, f"/agents/{agent_id}/chat/stream", {"message": "hi"})
    names = [name for name, _ in events]

    # The thread id comes first, before any work: a client that drops mid-stream
    # can still ask for the transcript of the turn it started.
    assert names[0] == "conversation"
    assert names[-1] == "done"
    assert names.count("token") == 4

    tokens = "".join(p["text"] for n, p in events if n == "token")
    done = events[-1][1]
    assert tokens == "Twenty-five days per year."
    # The assembled answer is the fragments, not a second generation of them.
    assert done["answer"] == tokens
    # Usage survives reassembly. Providers omit it from a stream unless asked,
    # and a streamed turn that billed as zero tokens would be invisible.
    assert done["usage"]["total_tokens"] == 14


async def test_a_streamed_turn_is_recorded_like_any_other(
    authed_client, fake_embeddings, provider_creds, streamed_provider
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    events = await _events(
        client, f"/agents/{agent_id}/chat/stream", {"message": "How much leave?"}
    )
    done = events[-1][1]
    conversation_id = done["conversation_id"]

    assert done["citations"], "a grounded stream still cites its sources"

    messages = await client.get(f"/agents/{agent_id}/conversations/{conversation_id}/messages")
    body = messages.json()["items"]
    # Recorded before `done` is sent, so a client that reconnects on that event
    # finds the turn it just watched.
    assert [m["role"] for m in body] == ["user", "assistant"]
    assert body[1]["content"] == "Twenty-five days per year."
    assert body[1]["usage"]["total_tokens"] == 14


async def test_a_streamed_thread_is_replayed_on_the_next_turn(
    authed_client, fake_embeddings, provider_creds, streamed_provider
):
    client, _ = authed_client
    agent_id = await create_agent(client, slug="chatty", collection_id=None)

    first = await _events(client, f"/agents/{agent_id}/chat/stream", {"message": "one"})
    conversation_id = first[-1][1]["conversation_id"]
    await _events(
        client,
        f"/agents/{agent_id}/chat/stream",
        {"message": "two", "conversation_id": conversation_id},
    )

    # Memory and streaming are independent: the second streamed turn was sent the
    # first one's exchange, from the transcript.
    assert [m["role"] for m in streamed_provider[-1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


async def test_a_tool_call_is_announced_while_it_runs(
    authed_client, fake_embeddings, provider_creds, monkeypatch
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    agent_id = await create_agent(
        client,
        slug="hr-bot",
        collection_id=collection_id,
        tool_allowlist=["search_knowledge_base"],
    )

    calls = {"n": 0}

    async def _acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:

            async def tool_chunks():
                yield _delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=Function(
                                name="search_knowledge_base",
                                arguments='{"query": "annual leave"}',
                            ),
                        )
                    ]
                )
                yield _delta(usage=_usage(5, 5))

            return tool_chunks()

        async def answer_chunks():
            yield _delta(content="Twenty-five days.")
            yield _delta(usage=_usage(8, 4))

        return answer_chunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    events = await _events(
        client, f"/agents/{agent_id}/chat/stream", {"message": "time off?"}
    )
    names = [n for n, _ in events]

    # The pause while a second search runs is explained rather than silent, and
    # the tool call survived being reassembled from streamed deltas.
    assert "tool" in names
    assert events[names.index("tool")][1]["name"] == "search_knowledge_base"
    assert events[-1][1]["tools_used"] == ["search_knowledge_base"]
    # Both model calls are billed, exactly as in the unstreamed loop.
    assert events[-1][1]["usage"]["total_tokens"] == 22


async def test_a_failure_after_the_headers_is_an_error_frame(
    authed_client, provider_creds, monkeypatch
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    async def broken_embed(text: str):
        raise RuntimeError("embedding provider unavailable")

    from app.services import retrieval_service

    monkeypatch.setattr(retrieval_service, "embed_text", broken_embed)

    events = await _events(
        client, f"/agents/{agent_id}/chat/stream", {"message": "How much leave?"}
    )

    # A 200 whose body says 502. The status line was sent before retrieval ran,
    # so the code that says whose problem this is has to travel in the payload.
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 502
    assert not any(n == "done" for n, _ in events)


async def test_preflight_failures_are_still_real_status_codes(authed_client, provider_creds):
    client, _ = authed_client
    disabled = await create_agent(
        client, slug="off", collection_id=None, enabled=False
    )
    r = await client.post(f"/agents/{disabled}/chat/stream", json={"message": "hi"})
    # Everything checkable before the first byte is checked before the first
    # byte — a disabled agent is a 409 here exactly as it is on `/chat`.
    assert r.status_code == 409

    other = await create_agent(client, slug="mine", collection_id=None)
    foreign = await client.post(f"/agents/{other}/conversations", json={})
    stranger = await create_agent(client, slug="stranger", collection_id=None)
    r = await client.post(
        f"/agents/{stranger}/chat/stream",
        json={"message": "hi", "conversation_id": foreign.json()["id"]},
    )
    assert r.status_code == 404
