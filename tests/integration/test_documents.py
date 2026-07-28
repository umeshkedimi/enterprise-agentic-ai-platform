"""Document endpoints — auth and collection-scoping guarantees.

The ingestion pipeline itself (chunk/embed) is covered elsewhere; these tests
pin the tenant-isolation behaviour added when documents were scoped to
collections, without calling the embedding API.
"""

import io
import uuid


def _txt_file():
    return {"file": ("note.txt", io.BytesIO(b"hello world"), "text/plain")}


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
    assert r.json() == []


async def test_delete_missing_document_is_404(authed_client):
    client, _ = authed_client
    r = await client.delete(f"/documents/{uuid.uuid4()}")
    assert r.status_code == 404
