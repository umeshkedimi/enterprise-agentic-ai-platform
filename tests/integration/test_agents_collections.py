"""Collection and agent CRUD, and the tenant-isolation guarantees around them."""

import uuid


async def test_collection_crud_roundtrip(authed_client):
    client, _ = authed_client

    r = await client.post("/collections", json={"slug": "policies", "name": "HR Policies"})
    assert r.status_code == 201, r.text
    coll_id = r.json()["id"]

    assert (await client.get("/collections")).json()[0]["id"] == coll_id
    assert (await client.get(f"/collections/{coll_id}")).json()["name"] == "HR Policies"
    assert (await client.delete(f"/collections/{coll_id}")).status_code == 204
    assert (await client.get(f"/collections/{coll_id}")).status_code == 404


async def test_collection_slug_unique_per_tenant(authed_client):
    client, _ = authed_client
    body = {"slug": "dup", "name": "First"}
    assert (await client.post("/collections", json=body)).status_code == 201
    assert (await client.post("/collections", json=body)).status_code == 409


async def test_agent_requires_auth(client):
    assert (await client.get("/agents")).status_code == 401
    assert (await client.post("/agents", json={})).status_code == 401


async def test_agent_create_with_collection_and_defaults(authed_client):
    client, _ = authed_client
    coll_id = (
        await client.post("/collections", json={"slug": "kb", "name": "KB"})
    ).json()["id"]

    r = await client.post(
        "/agents",
        json={
            "slug": "hr-helper",
            "name": "HR Helper",
            "system_prompt": "You answer HR questions.",
            "collection_id": coll_id,
        },
    )
    assert r.status_code == 201, r.text
    agent = r.json()
    # Execution-policy defaults applied by the model.
    assert agent["model"] == "claude-sonnet-5"
    assert agent["temperature"] == 0.2
    assert agent["retrieval_top_k"] == 5
    assert agent["enabled"] is True
    assert agent["collection_id"] == coll_id


async def test_agent_partial_update(authed_client):
    client, _ = authed_client
    agent_id = (
        await client.post(
            "/agents",
            json={"slug": "a1", "name": "A1", "system_prompt": "p"},
        )
    ).json()["id"]

    r = await client.patch(f"/agents/{agent_id}", json={"enabled": False, "temperature": 0.9})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["temperature"] == 0.9
    # Untouched fields are preserved.
    assert body["name"] == "A1"


async def test_agent_cannot_reference_foreign_collection(authed_client, client, admin_headers):
    """An agent must not bind to another tenant's collection — the core
    isolation guarantee. A cross-tenant collection id reads as not-found."""
    agent_client, _ = authed_client

    # A second, unrelated tenant with its own collection.
    other_slug = f"other-{uuid.uuid4().hex[:8]}"
    other_id = (
        await client.post(
            "/tenants", json={"slug": other_slug, "name": "Other"}, headers=admin_headers
        )
    ).json()["id"]
    other_key = (
        await client.post(
            f"/tenants/{other_id}/keys", json={"name": "k"}, headers=admin_headers
        )
    ).json()["api_key"]
    foreign_coll_id = (
        await client.post(
            "/collections",
            json={"slug": "secret", "name": "Secret"},
            headers={"Authorization": f"Bearer {other_key}"},
        )
    ).json()["id"]

    # The first tenant's agent may not point at the second tenant's collection.
    r = await agent_client.post(
        "/agents",
        json={
            "slug": "sneaky",
            "name": "Sneaky",
            "system_prompt": "p",
            "collection_id": foreign_coll_id,
        },
    )
    assert r.status_code == 404, r.text

    # Cleanup the extra tenant (cascades its collection).
    from tests.integration.conftest import _delete_tenant

    await _delete_tenant(uuid.UUID(other_id))


async def test_agents_are_isolated_between_tenants(authed_client, app, admin_headers):
    """A tenant lists only its own agents, never another tenant's."""
    from httpx import ASGITransport, AsyncClient

    client_a, _ = authed_client
    await client_a.post("/agents", json={"slug": "mine", "name": "Mine", "system_prompt": "p"})

    other_slug = f"iso-{uuid.uuid4().hex[:8]}"
    other_id = (
        await client_a.post(  # admin call works on any client
            "/tenants", json={"slug": other_slug, "name": "Iso"}, headers=admin_headers
        )
    ).json()["id"]
    other_key = (
        await client_a.post(
            f"/tenants/{other_id}/keys", json={"name": "k"}, headers=admin_headers
        )
    ).json()["api_key"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client_b:
        client_b.headers["Authorization"] = f"Bearer {other_key}"
        assert (await client_b.get("/agents")).json() == []

    from tests.integration.conftest import _delete_tenant

    await _delete_tenant(uuid.UUID(other_id))
