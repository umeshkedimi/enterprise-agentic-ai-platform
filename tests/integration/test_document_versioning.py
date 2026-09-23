"""End-to-end document versioning: a re-upload supersedes, it never contradicts.

Only the embedding and chat-provider calls are faked. Everything else — the
queue-and-process ingestion flow, the two-commit supersession inside
`process_document`, the real pgvector query now filtering on `is_current`, and
the partial unique index itself — runs against a real Postgres, because the
property worth pinning is that the database, not just the service function,
refuses two current versions of one document.

Upload is now two steps, not one: `POST .../documents` only queues (202,
`status: "uploaded"`), and `process_queued_documents()` (from conftest) is
what actually runs the worker's own claim-and-process code against it. Tests
that care what a document looked like *before* processing — whether this
upload's `is_current` reflects something it is about to supersede — check the
raw 202 response; tests that care what happens *after* re-fetch through
`get_document`.
"""

import io
import uuid
from types import SimpleNamespace

import litellm
import pytest
from sqlalchemy.exc import IntegrityError

from app.db.session import async_session_factory
from app.models.agent import Agent
from app.models.document import Document, DocumentStatus
from app.tools.builtin import list_documents as list_documents_tool
from app.tools.registry import ToolContext
from tests.integration.conftest import (
    create_agent,
    create_collection,
    get_document,
    process_queued_documents,
)

VACATION_V1 = (
    b"Vacation policy. Full-time employees accrue twenty-five days of paid annual "
    b"leave each year."
)
VACATION_V2 = (
    b"Vacation policy, revised. Full-time employees accrue thirty days of paid annual "
    b"leave each year."
)
WHITESPACE_ONLY = b"     "


@pytest.fixture
def captured(monkeypatch):
    """Fake the chat provider and record exactly what it was asked to do."""
    calls: list[dict] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="Thirty days [1]."))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=6, total_tokens=126),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls


async def upload(client, collection_id, filename, body, *, document_key=None):
    data = {"document_key": document_key} if document_key else {}
    return await client.post(
        f"/collections/{collection_id}/documents",
        data=data,
        files={"file": (filename, io.BytesIO(body), "text/plain")},
    )


async def test_reupload_under_the_same_key_supersedes_the_previous_version(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    first = await upload(
        client, collection_id, "vacation.txt", VACATION_V1, document_key="vacation"
    )
    assert first.status_code == 202, first.text
    # Nothing to supersede yet — the first upload under a key is current the
    # moment it's queued, not only once it finishes processing.
    assert first.json()["is_current"] is True
    assert first.json()["document_key"] == "vacation"

    second = await upload(
        client, collection_id, "vacation.txt", VACATION_V2, document_key="vacation"
    )
    assert second.status_code == 202, second.text
    # Not current yet — a previous version already holds that slot, and
    # supersession only happens once this upload actually succeeds.
    assert second.json()["is_current"] is False

    await process_queued_documents()

    listed = await client.get(f"/collections/{collection_id}/documents")
    by_id = {d["id"]: d for d in listed.json()["items"]}
    # Both versions are still on file — versioning never deletes — but only one
    # is current.
    assert by_id[first.json()["id"]]["is_current"] is False
    assert by_id[second.json()["id"]]["is_current"] is True


async def test_uploads_without_a_document_key_never_supersede_each_other(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    first = await upload(client, collection_id, "a.txt", VACATION_V1)
    second = await upload(client, collection_id, "b.txt", VACATION_V2)

    assert first.json()["is_current"] is True
    assert second.json()["is_current"] is True
    assert first.json()["document_key"] is None


async def test_a_failed_reupload_leaves_the_previous_version_current(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    good = await upload(
        client, collection_id, "vacation.txt", VACATION_V1, document_key="vacation"
    )
    assert good.status_code == 202, good.text
    await process_queued_documents()
    good_entry = await get_document(client, collection_id, good.json()["id"])
    assert good_entry["status"] == "ready"
    assert good_entry["is_current"] is True

    # Whitespace-only content extracts to no chunks, which process_document
    # treats as a processing failure — the same path a corrupt PDF would hit.
    # It queues and returns 202 exactly like a good upload would; there is no
    # request left by the time it actually fails.
    bad = await upload(
        client, collection_id, "vacation.txt", WHITESPACE_ONLY, document_key="vacation"
    )
    assert bad.status_code == 202, bad.text
    await process_queued_documents()

    listed = await client.get(f"/collections/{collection_id}/documents")
    by_id = {d["id"]: d for d in listed.json()["items"]}
    # The good version was never superseded — supersession only happens after
    # the new upload reaches READY, which this one never did.
    assert by_id[good.json()["id"]]["is_current"] is True
    failed_entries = [d for d in by_id.values() if d["status"] == "failed"]
    assert len(failed_entries) == 1
    assert failed_entries[0]["is_current"] is False


async def test_superseded_version_is_not_retrieved(
    authed_client, fake_embeddings, provider_creds, captured
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload(client, collection_id, "vacation.txt", VACATION_V1, document_key="vacation")
    await process_queued_documents()
    await upload(client, collection_id, "vacation.txt", VACATION_V2, document_key="vacation")
    await process_queued_documents()

    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)
    r = await client.post(
        f"/agents/{agent_id}/chat", json={"message": "How many vacation days do I get?"}
    )
    assert r.status_code == 200, r.text

    prompt = captured[0]["messages"][-1]["content"]
    assert "thirty" in prompt.lower()
    assert "twenty-five" not in prompt.lower()


async def test_list_documents_tool_shows_only_the_current_version(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload(
        client, collection_id, "vacation_v1.txt", VACATION_V1, document_key="vacation"
    )
    await process_queued_documents()
    await upload(
        client, collection_id, "vacation_v2.txt", VACATION_V2, document_key="vacation"
    )
    await process_queued_documents()
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    async with async_session_factory() as session:
        agent = await session.get(Agent, uuid.UUID(agent_id))
        result = await list_documents_tool(ToolContext(agent=agent, session=session, chunks=[]))

    assert "vacation_v2.txt" in result
    assert "vacation_v1.txt" not in result


async def test_the_partial_unique_index_refuses_two_current_versions(authed_client):
    """The guarantee holds even if application logic is bypassed entirely.

    Two rows inserted directly, both current, under the same collection and
    key — the same shape a bug in `document_service` would have to produce to
    cause the contradiction this chunk exists to prevent.
    """
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    async with async_session_factory() as session:
        session.add(
            Document(
                collection_id=uuid.UUID(collection_id),
                filename="a.txt",
                content_type="text/plain",
                status=DocumentStatus.READY,
                document_key="dupe",
                is_current=True,
            )
        )
        session.add(
            Document(
                collection_id=uuid.UUID(collection_id),
                filename="b.txt",
                content_type="text/plain",
                status=DocumentStatus.READY,
                document_key="dupe",
                is_current=True,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
