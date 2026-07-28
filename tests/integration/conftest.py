"""Shared fixtures for integration tests against a live Postgres.

Requires `docker compose up -d postgres` and a migrated database.
"""

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.models.tenant import Tenant

ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-Token": ADMIN_TOKEN}


@pytest.fixture
def app():
    # Enable admin endpoints for the test, then rebuild cached settings so
    # require_platform_admin observes the token. Restore afterwards.
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


async def _delete_tenant(tenant_id: uuid.UUID) -> None:
    # Cascade-deletes the tenant's api_keys, collections, and agents.
    async with async_session_factory() as s:
        obj = await s.get(Tenant, tenant_id)
        if obj:
            await s.delete(obj)
            await s.commit()


@pytest.fixture
async def tenant(client, admin_headers):
    slug = f"test-{uuid.uuid4().hex[:8]}"
    r = await client.post("/tenants", json={"slug": slug, "name": "Test Co"}, headers=admin_headers)
    assert r.status_code == 201, r.text
    tenant_id = uuid.UUID(r.json()["id"])
    yield slug, tenant_id
    await _delete_tenant(tenant_id)


@pytest.fixture
async def authed_client(app, tenant, admin_headers):
    """A client that authenticates as a freshly-provisioned tenant.

    Yields (client_with_bearer_key, tenant_id) — the common starting point for
    exercising tenant-scoped endpoints.
    """
    _, tenant_id = tenant
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(f"/tenants/{tenant_id}/keys", json={"name": "k"}, headers=admin_headers)
        assert r.status_code == 201, r.text
        c.headers["Authorization"] = f"Bearer {r.json()['api_key']}"
        yield c, tenant_id
