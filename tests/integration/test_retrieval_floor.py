"""The relevance floor, applied end to end: real pgvector scores, real filtering.

Rather than depend on the exact numeric score the deterministic lexical
embedding produces for a given phrase (fragile — a wording change would shift
it), these tests use floor values *outside* cosine similarity's actual range
([-1, 1]): 1.1 is unreachable, so it deterministically drops everything a real
search finds; -1.1 is deterministically a no-op. What's under test is the
filtering and its side effects (metrics, the abstention path, the benchmark
harness), not the embedding's arithmetic.
"""

import uuid
from types import SimpleNamespace

import litellm
import pytest

from app.core import metrics
from app.core.config import get_settings
from app.db.session import async_session_factory
from app.services import retrieval_service
from tests.integration.conftest import (
    VACATION_TEXT,
    create_agent,
    create_collection,
    upload_document,
)


def sample(name: str, labels: dict | None = None) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.fixture
def relevance_floor(monkeypatch):
    """Set the floor via env, the same channel an operator would use."""

    def _set(value: float) -> None:
        monkeypatch.setenv("RETRIEVAL_RELEVANCE_FLOOR", str(value))
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


@pytest.fixture
def captured(monkeypatch):
    """Fake the chat provider and record exactly what it was asked to do."""
    calls: list[dict] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="I don't have that."))],
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=5, total_tokens=85),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls


async def test_a_floor_above_the_maximum_possible_score_excludes_everything(
    authed_client, fake_embeddings, relevance_floor
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    relevance_floor(1.1)

    async with async_session_factory() as session:
        chunks = await retrieval_service.semantic_search(
            session, "vacation days", collection_id=uuid.UUID(collection_id)
        )
    assert chunks == []


async def test_a_floor_below_the_minimum_possible_score_is_a_no_op(
    authed_client, fake_embeddings, relevance_floor
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    relevance_floor(-1.1)

    async with async_session_factory() as session:
        chunks = await retrieval_service.semantic_search(
            session, "vacation days", collection_id=uuid.UUID(collection_id)
        )
    assert len(chunks) == 1


async def test_no_floor_configured_behaves_exactly_as_before(authed_client, fake_embeddings):
    """The default: unset means unchanged, the same guarantee every other
    additive chunk in this platform has made."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    async with async_session_factory() as session:
        chunks = await retrieval_service.semantic_search(
            session, "vacation days", collection_id=uuid.UUID(collection_id)
        )
    assert len(chunks) == 1


async def test_a_floor_that_rejects_everything_counts_as_a_floor_abstention(
    authed_client, fake_embeddings, relevance_floor
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    before = sample("eaap_retrieval_floor_abstentions_total")
    relevance_floor(1.1)

    async with async_session_factory() as session:
        await retrieval_service.semantic_search(
            session, "vacation days", collection_id=uuid.UUID(collection_id)
        )

    assert sample("eaap_retrieval_floor_abstentions_total") - before == 1


async def test_a_genuinely_empty_search_is_not_counted_as_a_floor_abstention(
    authed_client, fake_embeddings, relevance_floor
):
    """The counter means "found something and distrusted it", not "found
    nothing" — a collection with nothing in it is a different story entirely
    and must not inflate the same number."""
    client, _ = authed_client
    collection_id = await create_collection(client, "empty")

    before = sample("eaap_retrieval_floor_abstentions_total")
    relevance_floor(1.1)

    async with async_session_factory() as session:
        chunks = await retrieval_service.semantic_search(
            session, "anything", collection_id=uuid.UUID(collection_id)
        )

    assert chunks == []
    assert sample("eaap_retrieval_floor_abstentions_total") - before == 0


async def test_chat_from_an_agent_whose_floor_rejects_everything_abstains(
    authed_client, fake_embeddings, provider_creds, relevance_floor, captured
):
    """The whole point: an agent stops using evidence it doesn't trust, and the
    existing "documents don't cover this" path — built for a genuinely empty
    search — handles a floor-triggered one identically, with no graph changes."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    relevance_floor(1.1)
    r = await client.post(
        f"/agents/{agent_id}/chat", json={"message": "How many vacation days do I get?"}
    )

    assert r.status_code == 200, r.text
    assert r.json()["citations"] == []
    # The no-results directive is platform-authored and lands in the system
    # message, appended after the agent's own prompt — never in the user turn.
    system_message = captured[0]["messages"][0]["content"]
    assert "do not cover it" in system_message


async def test_the_benchmark_harness_reflects_the_configured_floor(
    authed_client, fake_embeddings, relevance_floor
):
    """The payoff named in the roadmap: the Chunk 8 harness runs through the
    same `semantic_search`, so a floor an operator sets shows up as a recall
    change here without the harness knowing the floor exists."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    doc_id = await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    golden = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={"query": "vacation days", "relevant_document_ids": [doc_id]},
    )
    assert golden.status_code == 201, golden.text

    baseline = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark", json={"label": "no-floor"}
    )
    assert baseline.json()["mean_recall"] == 1.0

    relevance_floor(1.1)
    floored = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark", json={"label": "floor-1.1"}
    )
    assert floored.json()["mean_recall"] == 0.0
