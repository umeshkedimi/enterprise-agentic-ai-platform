"""Paging over real rows, through the real endpoints.

The properties worth pinning here are the ones a unit test cannot see: that the
window reaches the database rather than trimming a full result in Python, that
`has_more` is honest at the boundary, and that an out-of-range request is
refused rather than quietly clamped.
"""

from types import SimpleNamespace

import litellm
import pytest

from app.services.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from tests.integration.conftest import create_agent, create_collection


@pytest.fixture
def faked_provider(monkeypatch, provider_creds):
    async def _fake_acompletion(**kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Twenty-five days [1]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)


async def _make_agents(client, count: int) -> None:
    for i in range(count):
        await create_agent(client, slug=f"bot-{i}", collection_id=None)


async def test_a_page_stops_at_the_limit_and_says_there_is_more(authed_client):
    client, _ = authed_client
    await _make_agents(client, 5)

    r = await client.get("/agents", params={"limit": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert body["has_more"] is True


async def test_the_last_page_reports_no_more(authed_client):
    client, _ = authed_client
    await _make_agents(client, 5)

    body = (await client.get("/agents", params={"limit": 2, "offset": 4})).json()
    assert len(body["items"]) == 1
    assert body["has_more"] is False


async def test_a_page_that_exactly_exhausts_the_rows_is_not_more(authed_client):
    """The boundary the probe row exists to get right."""
    client, _ = authed_client
    await _make_agents(client, 4)

    body = (await client.get("/agents", params={"limit": 4})).json()
    assert len(body["items"]) == 4
    assert body["has_more"] is False


async def test_paging_walks_every_row_exactly_once(authed_client):
    client, _ = authed_client
    await _make_agents(client, 7)

    seen: list[str] = []
    offset = 0
    while True:
        body = (await client.get("/agents", params={"limit": 3, "offset": offset})).json()
        seen.extend(a["slug"] for a in body["items"])
        if not body["has_more"]:
            break
        offset += 3

    assert sorted(seen) == sorted(f"bot-{i}" for i in range(7))


async def test_an_empty_list_is_a_page_not_an_error(authed_client):
    client, _ = authed_client
    body = (await client.get("/agents")).json()
    assert body == {"items": [], "limit": DEFAULT_PAGE_LIMIT, "offset": 0, "has_more": False}


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": MAX_PAGE_LIMIT + 1},
        {"offset": -1},
    ],
)
async def test_an_out_of_range_window_is_refused_not_clamped(authed_client, params):
    """A caller that asked for 10,000 rows must not be handed 200 and left to
    conclude it has seen everything."""
    client, _ = authed_client
    r = await client.get("/agents", params=params)
    assert r.status_code == 422, r.text


async def test_the_transcript_pages_in_order(authed_client, fake_embeddings, faked_provider):
    client, _ = authed_client
    coll_id = await create_collection(client, "hr")
    agent_id = await create_agent(client, slug="hr-bot", collection_id=coll_id)

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "First?"})
    assert r.status_code == 200, r.text
    conversation_id = r.json()["conversation_id"]
    await client.post(
        f"/agents/{agent_id}/chat",
        json={"message": "Second?", "conversation_id": conversation_id},
    )

    url = f"/agents/{agent_id}/conversations/{conversation_id}/messages"
    first = (await client.get(url, params={"limit": 2})).json()
    assert [m["role"] for m in first["items"]] == ["user", "assistant"]
    assert first["items"][0]["content"] == "First?"
    assert first["has_more"] is True

    second = (await client.get(url, params={"limit": 2, "offset": 2})).json()
    assert second["items"][0]["content"] == "Second?"
    assert second["has_more"] is False


async def test_documents_and_collections_page_too(authed_client, fake_embeddings):
    """The envelope is the shape of every list, not a special case for agents."""
    client, _ = authed_client
    for i in range(3):
        await create_collection(client, f"coll-{i}")

    body = (await client.get("/collections", params={"limit": 2})).json()
    assert len(body["items"]) == 2 and body["has_more"] is True

    body = (await client.get("/mcp-servers", params={"limit": 2})).json()
    assert body["items"] == [] and body["has_more"] is False
