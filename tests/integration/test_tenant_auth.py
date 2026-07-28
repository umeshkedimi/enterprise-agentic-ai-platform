"""End-to-end tenant identity flow against a live Postgres.

Requires `docker compose up -d postgres` and a migrated database.
Run with: uv run pytest tests/integration -v
"""

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.models.tenant import Tenant

ADMIN_TOKEN = "test-admin-token"
ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


@pytest.fixture
def app():
    # Enable admin endpoints for the test, then rebuild the cached settings so
    # require_platform_admin sees the token. Restore afterwards.
    os.environ["PLATFORM_ADMIN_TOKEN"] = ADMIN_TOKEN
    get_settings.cache_clear()
    from app.main import create_app

    yield create_app()

    os.environ.pop("PLATFORM_ADMIN_TOKEN", None)
    get_settings.cache_clear()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


@pytest.fixture
async def tenant(client):
    slug = f"test-{uuid.uuid4().hex[:8]}"
    r = await client.post("/tenants", json={"slug": slug, "name": "Test Co"}, headers=ADMIN_HEADERS)
    assert r.status_code == 201, r.text
    tenant_id = uuid.UUID(r.json()["id"])
    yield slug, tenant_id
    # Cascade-deletes the tenant's api_keys too.
    async with async_session_factory() as s:
        obj = await s.get(Tenant, tenant_id)
        if obj:
            await s.delete(obj)
            await s.commit()


async def test_create_tenant_requires_admin_token(client):
    r = await client.post("/tenants", json={"slug": "no-auth", "name": "X"})
    assert r.status_code == 403


async def test_duplicate_slug_conflicts(client, tenant):
    slug, _ = tenant
    r = await client.post("/tenants", json={"slug": slug, "name": "Dup"}, headers=ADMIN_HEADERS)
    assert r.status_code == 409


async def test_minted_key_authenticates_as_its_tenant(client, tenant):
    slug, tenant_id = tenant
    r = await client.post(f"/tenants/{tenant_id}/keys", json={"name": "k"}, headers=ADMIN_HEADERS)
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


async def test_key_for_missing_tenant_is_404(client):
    r = await client.post(
        f"/tenants/{uuid.uuid4()}/keys", json={"name": "k"}, headers=ADMIN_HEADERS
    )
    assert r.status_code == 404
