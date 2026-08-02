"""End-to-end evaluation: a real turn, judged, over the real transcript.

Only the two network calls are faked — embeddings and the provider. The chat
turn that produces the material under audit runs for real, which means the
citations the judge is shown were produced by an actual pgvector search over
actually-chunked text, and the evidence it reads is recovered from the chunk
rows by the id the citation recorded. That path is the one worth exercising: the
snippet stored on a citation is a 240-character prefix, and a judge fed snippets
instead of chunks would fail answers for lack of evidence the model actually had.
"""

import json
import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.core import metrics
from tests.integration.conftest import create_agent, create_collection, upload_document

HANDBOOK = (
    b"Annual leave. Permanent staff receive twenty-five days of paid annual leave "
    b"each calendar year, in addition to public holidays. Leave accrues monthly and "
    b"unused days do not carry over beyond March of the following year."
)


def judge_reply(payload: dict, *, model: str = "gpt-4o-mini"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
        usage=SimpleNamespace(prompt_tokens=400, completion_tokens=60, total_tokens=460),
    )


def one_supported(notes: str = ""):
    """A minimal well-formed verdict: one claim, carried by source 1.

    Naming the source is not padding. The platform treats "supported" with an
    empty source list as unsupported, because that combination is the judge
    agreeing with the answer rather than checking it.
    """
    return {"claims": [{"claim": "a", "supported": True, "sources": [1]}], "notes": notes}


def answer_reply(text: str, *, model: str = "gpt-4o-mini"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=8, total_tokens=128),
    )


@pytest.fixture
def scripted_provider(monkeypatch):
    """Serve queued responses in order, recording every call.

    The chat turn and the judge both go through `litellm.acompletion` — the judge
    deliberately shares the serving path's client rather than having one of its
    own — so a single queue covers both, and the recorded calls are how the tests
    check what the judge was actually shown.
    """
    queue: list = []
    calls: list[dict] = []

    async def _acompletion(**kwargs):
        calls.append(kwargs)
        if not queue:
            raise AssertionError("provider called more times than the test scripted")
        return queue.pop(0)

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    return SimpleNamespace(queue=queue, calls=calls)


async def served_turn(client, scripted_provider, *, answer="Twenty-five days [1]."):
    """Run one real grounded turn and return `(agent_id, conversation_id, message_id)`."""
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "handbook.txt", HANDBOOK)
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    scripted_provider.queue.append(answer_reply(answer))
    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "how much annual leave"})
    assert r.status_code == 200, r.text
    conversation_id = r.json()["conversation_id"]
    assert r.json()["citations"], "the turn under audit must have retrieved something"

    r = await client.get(f"/agents/{agent_id}/conversations/{conversation_id}/messages")
    assert r.status_code == 200, r.text
    assistant = [m for m in r.json()["items"] if m["role"] == "assistant"]
    return agent_id, conversation_id, assistant[-1]["id"]


def evaluation_url(agent_id, conversation_id, message_id):
    return (
        f"/agents/{agent_id}/conversations/{conversation_id}"
        f"/messages/{message_id}/evaluations"
    )


# --- The happy path ----------------------------------------------------------


async def test_a_grounded_answer_scores_supported(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(
        judge_reply(
            {
                "claims": [
                    {
                        "claim": "Staff receive twenty-five days of annual leave.",
                        "supported": True,
                        "sources": [1],
                    }
                ],
                "notes": "The handbook states this directly.",
            }
        )
    )
    r = await client.post(evaluation_url(agent_id, conversation_id, message_id))

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == "supported"
    assert body["score"] == 1.0
    assert body["claims"][0]["sources"] == [1]
    assert body["citation_count"] == 1
    # Snapshotted from the citation the turn actually produced, which is what
    # makes the calibration report possible at all.
    assert body["retrieval_top_score"] is not None
    assert body["rationale"] == "The handbook states this directly."


async def test_the_judge_reads_full_chunk_text_not_the_stored_snippet(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """The bug this design exists to avoid.

    A citation snippet is a 240-character prefix. The handbook chunk runs past
    that, and the sentence about carry-over sits in the tail — so if the judge
    were fed snippets it would be auditing against evidence the answering model
    never had to work with, and marking supported claims unsupported.
    """
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(judge_reply({"claims": [], "notes": "n/a"}))
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    audit_prompt = scripted_provider.calls[-1]["messages"][-1]["content"]
    assert "unused days do not carry over" in audit_prompt
    # ...and the whole chunk arrived, not the clipped copy.
    assert len(audit_prompt) > 240


async def test_the_rubric_is_the_only_thing_in_the_judges_system_role(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """The boundary the serving path enforces, checked at the judge.

    Retrieved document text and a model-written answer are both data. If either
    could reach the system role, a document could rewrite the rubric auditing it.
    """
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(judge_reply({"claims": [], "notes": ""}))
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    messages = scripted_provider.calls[-1]["messages"]
    system = next(m["content"] for m in messages if m["role"] == "system")
    assert "grounding auditor" in system
    assert "twenty-five days" not in system.lower()
    assert "Twenty-five days [1]." not in system


async def test_an_ungrounded_answer_scores_unsupported(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(
        client, scripted_provider, answer="Staff get unlimited leave and a company car."
    )

    scripted_provider.queue.append(
        judge_reply(
            {
                "claims": [
                    {"claim": "Leave is unlimited.", "supported": False, "sources": []},
                    {"claim": "Staff get a company car.", "supported": False, "sources": []},
                ],
                "notes": "Neither claim appears in the sources.",
            }
        )
    )
    r = await client.post(evaluation_url(agent_id, conversation_id, message_id))

    assert r.json()["verdict"] == "unsupported"
    assert r.json()["score"] == 0.0


async def test_a_partly_grounded_answer_records_which_half_failed(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """The reason claims are stored rather than just a float."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(
        judge_reply(
            {
                "claims": [
                    {"claim": "Twenty-five days.", "supported": True, "sources": [1]},
                    {"claim": "Leave carries over indefinitely.", "supported": False},
                ],
                "notes": "The second claim contradicts the handbook.",
            }
        )
    )
    r = await client.post(evaluation_url(agent_id, conversation_id, message_id))

    body = r.json()
    assert body["verdict"] == "partial"
    assert body["score"] == 0.5
    failed = [c for c in body["claims"] if not c["supported"]]
    assert failed[0]["claim"] == "Leave carries over indefinitely."


async def test_an_abstention_scores_null_and_is_not_a_failure(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """An agent that correctly says it does not know must not read as wrong."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(
        client, scripted_provider, answer="The available documents do not cover this."
    )

    scripted_provider.queue.append(
        judge_reply({"claims": [], "notes": "The answer asserts nothing."})
    )
    r = await client.post(evaluation_url(agent_id, conversation_id, message_id))

    assert r.json()["verdict"] == "abstained"
    assert r.json()["score"] is None


# --- Idempotency and refresh -------------------------------------------------


async def test_re_evaluating_returns_the_stored_row_without_calling_the_judge(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """What makes an interrupted backfill cheap to restart."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)
    url = evaluation_url(agent_id, conversation_id, message_id)

    scripted_provider.queue.append(
        judge_reply(one_supported("first"))
    )
    first = await client.post(url)
    calls_after_first = len(scripted_provider.calls)

    second = await client.post(url)

    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    # The queue is empty; a second judge call would have raised in the fixture.
    assert len(scripted_provider.calls) == calls_after_first


async def test_refresh_re_judges_the_same_row(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)
    url = evaluation_url(agent_id, conversation_id, message_id)

    scripted_provider.queue.append(
        judge_reply({"claims": [{"claim": "a", "supported": False}], "notes": "first pass"})
    )
    first = await client.post(url)
    assert first.json()["verdict"] == "unsupported"

    scripted_provider.queue.append(
        judge_reply(one_supported("second pass"))
    )
    second = await client.post(url, json={"refresh": True})

    assert second.json()["verdict"] == "supported"
    # The same row, updated: the unique constraint means one rubric yields one
    # current judgement, not a pile of them.
    assert second.json()["id"] == first.json()["id"]

    listed = await client.get(url)
    assert len(listed.json()) == 1


# --- Refusals ----------------------------------------------------------------


async def test_an_unreadable_verdict_is_a_502_and_writes_nothing(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """A missing judgement must not become a fabricated one in the audit trail."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)
    url = evaluation_url(agent_id, conversation_id, message_id)

    scripted_provider.queue.append(
        SimpleNamespace(
            model="gpt-4o-mini",
            choices=[SimpleNamespace(message=SimpleNamespace(content="I cannot do that."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    r = await client.post(url)

    assert r.status_code == 502
    assert (await client.get(url)).json() == []


async def test_a_user_turn_cannot_be_evaluated(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, _ = await served_turn(client, scripted_provider)

    r = await client.get(f"/agents/{agent_id}/conversations/{conversation_id}/messages")
    user_message_id = next(m["id"] for m in r.json()["items"] if m["role"] == "user")

    r = await client.post(evaluation_url(agent_id, conversation_id, user_message_id))
    assert r.status_code == 409


async def test_an_agent_with_no_knowledge_scope_cannot_be_judged_for_grounding(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """A rubric about document grounding has nothing to say about a tool agent."""
    client, _ = authed_client
    agent_id = await create_agent(client, slug="tools-only", collection_id=None)

    scripted_provider.queue.append(answer_reply("Sure."))
    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "hello"})
    conversation_id = r.json()["conversation_id"]

    r = await client.get(f"/agents/{agent_id}/conversations/{conversation_id}/messages")
    message_id = next(m["id"] for m in r.json()["items"] if m["role"] == "assistant")

    r = await client.post(evaluation_url(agent_id, conversation_id, message_id))
    assert r.status_code == 409


async def test_a_turn_addressed_through_the_wrong_conversation_is_not_found(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """The conversation in the path is a filter, not decoration."""
    client, _ = authed_client
    agent_id, _, message_id = await served_turn(client, scripted_provider)

    r = await client.post(f"/agents/{agent_id}/conversations", json={})
    other_conversation = r.json()["id"]

    r = await client.post(evaluation_url(agent_id, other_conversation, message_id))
    assert r.status_code == 404


async def test_evaluations_require_auth(client):
    r = await client.post(evaluation_url(uuid.uuid4(), uuid.uuid4(), uuid.uuid4()))
    assert r.status_code == 401


async def test_an_unjudged_turn_lists_empty_rather_than_404(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    r = await client.get(evaluation_url(agent_id, conversation_id, message_id))

    assert r.status_code == 200
    assert r.json() == []


# --- Whole threads and calibration -------------------------------------------


async def test_evaluating_a_conversation_judges_every_assistant_turn(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, _ = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(answer_reply("It accrues monthly [1]."))
    r = await client.post(
        f"/agents/{agent_id}/chat",
        json={"message": "does leave accrue", "conversation_id": conversation_id},
    )
    assert r.status_code == 200, r.text

    for _ in range(2):
        scripted_provider.queue.append(
            judge_reply(one_supported())
        )
    r = await client.post(f"/agents/{agent_id}/conversations/{conversation_id}/evaluations")

    assert r.status_code == 201, r.text
    assert len(r.json()) == 2
    assert {e["verdict"] for e in r.json()} == {"supported"}


async def test_calibration_reports_the_bands_without_recommending_from_thin_data(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """One evaluation is a data point, not a threshold."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(
        judge_reply(one_supported())
    )
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    r = await client.get(f"/agents/{agent_id}/calibration")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["evaluations"] == 1
    assert body["graded"] == 1
    assert body["mean_score"] == 1.0
    assert sum(b["evaluations"] for b in body["buckets"]) == 1
    assert body["recommendation"]["floor"] is None
    assert "at least" in body["recommendation"]["rationale"]


async def test_the_judges_tokens_are_counted_apart_from_served_turns(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """Otherwise running the audit harness reads as a serving-latency regression.

    The judge deliberately goes through the same `complete()` as the serving
    path, so that it cannot become a workload nobody is counting. `workload` is
    what keeps that decision from costing an operator the metric they page on.
    """
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    def tokens(workload: str) -> float:
        return (
            metrics.REGISTRY.get_sample_value(
                "eaap_llm_tokens_total",
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "workload": workload,
                    "kind": "prompt",
                },
            )
            or 0.0
        )

    before_serving, before_evaluation = tokens("serving"), tokens("evaluation")

    scripted_provider.queue.append(
        judge_reply(one_supported())
    )
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    # The judge's 400 prompt tokens landed on the evaluation series...
    assert tokens("evaluation") - before_evaluation == 400
    # ...and none of them on the one that describes answering a user.
    assert tokens("serving") == before_serving


async def test_a_verdict_is_counted_and_an_abstention_leaves_the_score_alone(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    """An answer that asserts nothing must not report as perfectly grounded."""
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(
        client, scripted_provider, answer="The available documents do not cover this."
    )

    def score_count() -> float:
        return (
            metrics.REGISTRY.get_sample_value(
                "eaap_evaluation_score_count", {"evaluator": "groundedness"}
            )
            or 0.0
        )

    def verdicts(verdict: str) -> float:
        return (
            metrics.REGISTRY.get_sample_value(
                "eaap_evaluations_total",
                {"evaluator": "groundedness", "verdict": verdict},
            )
            or 0.0
        )

    before_scores, before_abstentions = score_count(), verdicts("abstained")

    scripted_provider.queue.append(judge_reply({"claims": [], "notes": ""}))
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    assert verdicts("abstained") - before_abstentions == 1
    # Absent from the score histogram entirely, rather than recorded as 1.0.
    assert score_count() == before_scores


async def test_tenant_calibration_sees_the_same_row_as_agent_calibration(
    authed_client, fake_embeddings, provider_creds, scripted_provider
):
    client, _ = authed_client
    agent_id, conversation_id, message_id = await served_turn(client, scripted_provider)

    scripted_provider.queue.append(
        judge_reply(one_supported())
    )
    await client.post(evaluation_url(agent_id, conversation_id, message_id))

    per_agent = (await client.get(f"/agents/{agent_id}/calibration")).json()
    tenant_wide = (await client.get("/evaluations/calibration")).json()

    assert tenant_wide["agent_id"] is None
    assert tenant_wide["evaluations"] == per_agent["evaluations"] == 1
