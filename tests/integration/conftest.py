"""Shared fixtures for integration tests against a live Postgres.

Requires `docker compose up -d postgres` and a migrated database.

Only two things are ever faked: the embedding call and the chat provider. Every
other layer — chunking, storage, the HNSW-indexed similarity query, prompt
assembly, tenancy filters, error mapping — runs for real, because the properties
worth pinning here are about isolation boundaries, and a test that stubbed the
query would assert the stub.
"""

import hashlib
import io
import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.models.document import EMBEDDING_DIM
from app.models.tenant import Tenant
from app.services import document_service, retrieval_service

ADMIN_TOKEN = "test-admin-token"

VACATION_TEXT = (
    b"Vacation policy. Full-time employees accrue twenty-five days of paid annual "
    b"leave each year. Vacation days carry over for a maximum of five days."
)
EXPENSES_TEXT = (
    b"Expense policy. Reimbursement claims for travel must be submitted within "
    b"thirty days. Receipts are required for any expense above fifty dollars."
)


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


def fake_embedding(text: str) -> list[float]:
    """A deterministic lexical embedding: tokens hashed into dimensions.

    Real enough for cosine similarity to rank a vacation question above an
    expenses document, which is the only property these tests lean on. Values
    are stable across runs, so a retrieval assertion cannot flake.
    """
    vector = [0.0] * EMBEDDING_DIM
    for token in text.lower().split():
        token = token.strip(".,;:!?()")
        if not token:
            continue
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vector[index] += 1.0
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector] if norm else vector


@pytest.fixture
def fake_embeddings(monkeypatch):
    async def embed_texts(texts: list[str]) -> list[list[float]]:
        return [fake_embedding(t) for t in texts]

    async def embed_text(text: str) -> list[float]:
        return fake_embedding(text)

    # Patched at the point of use: both modules import the function by name, so
    # patching the embedding module itself would rebind nothing.
    monkeypatch.setattr(document_service, "embed_texts", embed_texts)
    monkeypatch.setattr(retrieval_service, "embed_text", embed_text)


@pytest.fixture
def provider_creds(app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def create_collection(client, slug: str) -> str:
    r = await client.post("/collections", json={"slug": slug, "name": slug.title()})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def upload_document(client, collection_id: str, filename: str, body: bytes) -> None:
    r = await client.post(
        f"/collections/{collection_id}/documents",
        files={"file": (filename, io.BytesIO(body), "text/plain")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "ready", r.text


async def create_agent(client, *, slug: str, collection_id: str | None, **overrides) -> str:
    payload = {
        "slug": slug,
        "name": "Policy Bot",
        "system_prompt": "You are the HR policy assistant.",
        "model": "gpt-4o-mini",
        "collection_id": collection_id,
        **overrides,
    }
    r = await client.post("/agents", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]
