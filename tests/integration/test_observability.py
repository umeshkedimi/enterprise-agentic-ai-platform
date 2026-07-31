"""What a real turn leaves behind in the metrics, over the real endpoints.

The unit suite pins the rules — which labels are allowed, when a stream is
timed. This one pins that the rules are actually reached: that a chat request
moves the counters an operator would page on, that a UUID in a path does not
become a label, and that the scrape endpoint serves what it claims to.

Counters are process-global and accumulate across the session, so every
assertion here is a *delta* around the request under test. An absolute value
would pass or fail depending on which tests ran first.
"""

import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.core import metrics
from tests.integration.conftest import (
    VACATION_TEXT,
    create_agent,
    create_collection,
    upload_document,
)


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Twenty-five days [1]."))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=6, total_tokens=126),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls


def sample(name: str, labels: dict | None = None) -> float:
    """A counter's current value, treating "never observed" as zero."""
    return metrics.REGISTRY.get_sample_value(name, labels or {}) or 0.0


# --- the scrape endpoint ----------------------------------------------------


async def test_metrics_are_served_without_a_key(client):
    r = await client.get("/metrics")

    # Unauthenticated on purpose, and safe *because* of the label rules: nothing
    # in this body says which tenant did anything. Prometheus is awkward at
    # carrying credentials, and aggregating the subjects away is what makes the
    # question moot rather than merely unasked.
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "eaap_http_requests_total" in r.text


async def test_the_scrape_endpoint_stays_out_of_the_public_api(client):
    r = await client.get("/openapi.json")
    # An operator surface, not part of what a team owner integrates against.
    assert "/metrics" not in r.json()["paths"]


async def test_metrics_can_be_switched_off(client, monkeypatch):
    from app.api import metrics as metrics_api
    from app.core.config import Settings

    monkeypatch.setattr(metrics_api, "get_settings", lambda: Settings(metrics_enabled=False))
    r = await client.get("/metrics")

    # 404, not 403: a deployment that collects some other way should look like
    # one that never had the endpoint, not like one that is refusing you.
    assert r.status_code == 404


# --- what a turn records ----------------------------------------------------


async def test_a_chat_turn_moves_the_numbers_an_operator_watches(
    authed_client, fake_embeddings, provider_creds, captured
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    turn_labels = {"outcome": "ok", "streamed": "false"}
    token_labels = {"provider": "openai", "model": "gpt-4o-mini", "kind": "prompt"}
    before = (
        sample("eaap_agent_turns_total", turn_labels),
        sample("eaap_llm_tokens_total", token_labels),
        sample("eaap_retrieval_requests_total", {"outcome": "ok"}),
        sample("eaap_retrieval_top_score_count"),
    )

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "How much leave?"})
    assert r.status_code == 200, r.text

    after = (
        sample("eaap_agent_turns_total", turn_labels),
        sample("eaap_llm_tokens_total", token_labels),
        sample("eaap_retrieval_requests_total", {"outcome": "ok"}),
        sample("eaap_retrieval_top_score_count"),
    )

    assert after[0] - before[0] == 1  # one turn
    assert after[1] - before[1] == 120  # prompt tokens, as billed
    assert after[2] - before[2] == 1  # one vector search
    assert after[3] - before[3] == 1  # one best-match score, for the floor question


async def test_a_failed_turn_is_counted_as_the_error_it_was(
    authed_client, fake_embeddings, provider_creds, monkeypatch
):
    async def _explode(**kwargs):
        raise RuntimeError("provider is having a bad day")

    monkeypatch.setattr(litellm, "acompletion", _explode)

    client, _ = authed_client
    agent_id = await create_agent(client, slug="doomed", collection_id=None)

    labels = {"error": "ModelInvocationError"}
    before = sample("eaap_agent_turn_errors_total", labels)

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hi"})
    assert r.status_code == 502

    # The breakdown that says whose problem a failure is. An error rate alone
    # cannot distinguish an upstream outage from a tenant's typo in a model name.
    assert sample("eaap_agent_turn_errors_total", labels) - before == 1


async def test_a_turn_the_agent_refused_is_not_counted_as_a_slow_turn(
    authed_client, provider_creds
):
    client, _ = authed_client
    agent_id = await create_agent(client, slug="off", collection_id=None, enabled=False)

    before = sample("eaap_agent_turn_duration_seconds_count", {"streamed": "false"})
    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hi"})
    assert r.status_code == 409

    # A turn that never entered the graph is not a turn. Folding a refusal in
    # would drag the latency histogram toward zero with requests that did no work.
    assert sample("eaap_agent_turn_duration_seconds_count", {"streamed": "false"}) == before


# --- label cardinality, over real URLs --------------------------------------


async def test_a_uuid_in_a_path_does_not_become_a_metric_label(authed_client, provider_creds):
    client, _ = authed_client
    first = await create_agent(client, slug="one", collection_id=None)
    second = await create_agent(client, slug="two", collection_id=None)

    labels = {"method": "GET", "route": "/agents/{agent_id}", "status": "200"}
    before = sample("eaap_http_requests_total", labels)

    await client.get(f"/agents/{first}")
    await client.get(f"/agents/{second}")

    # Two different agents, one series. Labelling by path instead would mint a
    # permanent series per agent, per conversation, and per document — a
    # cardinality bomb that a perfectly well-behaved client sets off.
    assert sample("eaap_http_requests_total", labels) - before == 2

    body = (await client.get("/metrics")).text
    assert first not in body


async def test_the_streamed_turn_is_timed_over_the_real_endpoint(
    authed_client, provider_creds, monkeypatch
):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    async def _acompletion(**kwargs):
        async def chunks():
            for word in ("Twenty-five ", "days."):
                yield ModelResponseStream(
                    choices=[StreamingChoices(index=0, delta=Delta(content=word))]
                )
            last = ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content=None))]
            )
            last.usage = Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
            yield last

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    client, _ = authed_client
    agent_id = await create_agent(client, slug="streamer", collection_id=None)

    route = {"method": "POST", "route": "/agents/{agent_id}/chat/stream"}
    before = sample("eaap_http_request_duration_seconds_count", route)
    before_turns = sample("eaap_agent_turns_total", {"outcome": "ok", "streamed": "true"})

    async with client.stream(
        "POST", f"/agents/{agent_id}/chat/stream", json={"message": "hi"}
    ) as response:
        assert response.status_code == 200
        body = [line async for line in response.aiter_lines()]

    assert any("event: done" in line for line in body)
    # Recorded once the last frame is written, not when the handler returned —
    # the SSE endpoint is the one place where the difference is the whole
    # duration of the answer.
    assert sample("eaap_http_request_duration_seconds_count", route) - before == 1
    assert (
        sample("eaap_agent_turns_total", {"outcome": "ok", "streamed": "true"}) - before_turns == 1
    )


async def test_a_request_for_nothing_lands_in_a_single_bucket(client):
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = sample("eaap_http_requests_total", labels)

    for path in ("/wp-login.php", "/.env", f"/{uuid.uuid4()}"):
        assert (await client.get(path)).status_code == 404

    # Otherwise anyone on the internet can add series to the platform's TSDB by
    # requesting URLs that do not exist.
    assert sample("eaap_http_requests_total", labels) - before == 3
