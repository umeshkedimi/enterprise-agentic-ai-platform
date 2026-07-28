"""End-to-end tenant identity flow. Fixtures live in conftest.py."""

import uuid


async def test_create_tenant_requires_admin_token(client):
    r = await client.post("/tenants", json={"slug": "no-auth", "name": "X"})
    assert r.status_code == 403


async def test_duplicate_slug_conflicts(client, tenant, admin_headers):
    slug, _ = tenant
    r = await client.post("/tenants", json={"slug": slug, "name": "Dup"}, headers=admin_headers)
    assert r.status_code == 409


async def test_minted_key_authenticates_as_its_tenant(client, tenant, admin_headers):
    slug, tenant_id = tenant
    r = await client.post(f"/tenants/{tenant_id}/keys", json={"name": "k"}, headers=admin_headers)
    assert r.status_code == 201
    api_key = r.json()["api_key"]
    assert api_key.startswith("eaap_sk_")

    r = await client.get("/tenants/me", headers={"Authorization": f"Bearer {api_key}"})
    assert r.status_code == 200
    assert r.json()["slug"] == slug


async def test_missing_and_invalid_keys_are_rejected(client):
    assert (await client.get("/tenants/me")).status_code == 401
    bad = {"Authorization": "Bearer eaap_sk_not_a_real_key"}
    assert (await client.get("/tenants/me", headers=bad)).status_code == 401


async def test_key_for_missing_tenant_is_404(client, admin_headers):
    r = await client.post(
        f"/tenants/{uuid.uuid4()}/keys", json={"name": "k"}, headers=admin_headers
    )
    assert r.status_code == 404
