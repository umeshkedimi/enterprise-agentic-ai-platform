"""Document endpoints — auth and collection-scoping guarantees.

The ingestion pipeline itself (chunk/embed) is covered elsewhere; these tests
pin the tenant-isolation behaviour added when documents were scoped to
collections, without calling the embedding API.
"""

import io
import uuid

import pytest

from app.core.config import get_settings
from tests.integration.conftest import create_collection


def _txt_file():
    return {"file": ("note.txt", io.BytesIO(b"hello world"), "text/plain")}


@pytest.fixture
def tiny_upload_limit(app, monkeypatch):
    """Shrink the upload ceiling so the limit can be crossed with a few KB.

    Same shape as `provider_creds`: settings are cached, so an env override only
    takes effect once the cache is dropped — and has to be dropped again on the
    way out so the next test reads the real default.
    """
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()
    yield 1024
    get_settings.cache_clear()


async def test_upload_requires_auth(client):
    r = await client.post(f"/collections/{uuid.uuid4()}/documents", files=_txt_file())
    assert r.status_code == 401


async def test_list_requires_auth(client):
    r = await client.get(f"/collections/{uuid.uuid4()}/documents")
    assert r.status_code == 401


async def test_upload_to_missing_collection_is_404(authed_client):
    client, _ = authed_client
    r = await client.post(f"/collections/{uuid.uuid4()}/documents", files=_txt_file())
    assert r.status_code == 404


async def test_list_missing_collection_is_404(authed_client):
    client, _ = authed_client
    r = await client.get(f"/collections/{uuid.uuid4()}/documents")
    assert r.status_code == 404


async def test_list_empty_collection_is_ok(authed_client):
    client, _ = authed_client
    coll_id = (
        await client.post("/collections", json={"slug": "empty", "name": "Empty"})
    ).json()["id"]
    r = await client.get(f"/collections/{coll_id}/documents")
    assert r.status_code == 200
    assert r.json()["items"] == []


async def test_upload_over_the_size_limit_is_413(authed_client, tiny_upload_limit, fake_embeddings):
    """An oversized upload is refused, and refused before any ingestion happens.

    `fake_embeddings` is requested precisely so that the test would fail loudly
    if the limit were checked after the pipeline ran — a real embedding call
    would raise instead of returning a 413.
    """
    client, _ = authed_client
    coll_id = await create_collection(client, "bounded")

    r = await client.post(
        f"/collections/{coll_id}/documents",
        files={"file": ("big.txt", io.BytesIO(b"x" * (tiny_upload_limit + 1)), "text/plain")},
    )
    assert r.status_code == 413, r.text
    assert "maximum upload size" in r.json()["detail"]

    # Nothing was stored: a refused upload leaves no half-ingested document.
    listed = await client.get(f"/collections/{coll_id}/documents")
    assert listed.json()["items"] == []


async def test_upload_under_the_size_limit_still_ingests(
    authed_client, tiny_upload_limit, fake_embeddings
):
    client, _ = authed_client
    coll_id = await create_collection(client, "within-bounds")

    r = await client.post(
        f"/collections/{coll_id}/documents",
        files={"file": ("small.txt", io.BytesIO(b"vacation policy applies"), "text/plain")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "ready"


async def test_delete_missing_document_is_404(authed_client):
    client, _ = authed_client
    r = await client.delete(f"/documents/{uuid.uuid4()}")
    assert r.status_code == 404
