"""HTML pages and tables (CSV/XLSX), ingested and actually retrievable.

Unlike the txt/pdf fixtures elsewhere, these tests send the real MIME type for
each file — the `upload_document` conftest helper hardcodes `text/plain`,
which is itself a *supported* type, so reusing it here would upload an HTML
page as plain text and never touch the new extractor. Only the embedding call
is faked; extraction, chunking, storage, and the real pgvector query all run,
because the property under test is that a table's column/value pairing and an
HTML page's real content — not its chrome — are what actually gets embedded.
"""

import io
import uuid
from types import SimpleNamespace

import litellm
import pytest
from openpyxl import Workbook

from app.db.session import async_session_factory
from app.services.chunking import XLSX_CONTENT_TYPE
from app.services.retrieval_service import semantic_search
from tests.integration.conftest import create_agent, create_collection

HTML_PAGE = b"""
<html>
  <head><style>body { color: red; }</style></head>
  <body>
    <nav>Home | Benefits | Contact</nav>
    <header>Acme Intranet</header>
    <main>
      <h1>Remote Work Policy</h1>
      <p>Employees may work remotely up to three days per week with manager approval.</p>
    </main>
    <footer>&copy; 2026 Acme Corp</footer>
    <script>trackPageView();</script>
  </body>
</html>
"""

LEAVE_CSV = b"Employee,Days\nAlex,25\nJordan,20\nSam,15\n"


def _leave_xlsx() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Leave"
    for row in [["Employee", "Days"], ["Alex", 25], ["Jordan", 20], ["Sam", 15]]:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def captured(monkeypatch):
    """Fake the chat provider and record exactly what it was asked to do."""
    calls: list[dict] = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        answer = SimpleNamespace(content="Yes, up to three days [1].")
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=answer)],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=8, total_tokens=128),
        )

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    return calls


async def upload(client, collection_id: str, filename: str, body: bytes, content_type: str):
    r = await client.post(
        f"/collections/{collection_id}/documents",
        files={"file": (filename, io.BytesIO(body), content_type)},
    )
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "ready", r.text
    return r.json()["id"]


async def test_an_html_page_is_ingested_with_boilerplate_stripped(
    authed_client, fake_embeddings, provider_creds, captured
):
    """Real retrieval, not just extraction: search for the actual content and
    confirm the chrome never made it into what got embedded."""
    client, _ = authed_client
    collection_id = await create_collection(client, "intranet")
    await upload(client, collection_id, "remote-work.html", HTML_PAGE, "text/html")
    agent_id = await create_agent(client, slug="hr-bot", collection_id=collection_id)

    r = await client.post(f"/agents/{agent_id}/chat", json={"message": "Can I work remotely?"})
    assert r.status_code == 200, r.text
    citations = r.json()["citations"]
    assert citations, "expected the remote-work policy to be retrieved"
    snippet = citations[0]["snippet"].lower()
    assert "remote" in snippet
    assert "home | benefits" not in snippet
    assert "trackpageview" not in snippet


async def test_a_csv_row_keeps_its_column_labels_through_retrieval(
    authed_client, fake_embeddings
):
    """The point of table-aware extraction: a retrieved chunk about Jordan's
    leave has to say "Days: 20" attached to "Employee: Jordan", not just drop
    both names and numbers into the same unlabeled block of text."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload(client, collection_id, "leave.csv", LEAVE_CSV, "text/csv")

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, "Jordan Days", collection_id=uuid.UUID(collection_id)
        )
    assert chunks
    assert "Employee: Jordan" in chunks[0].content
    assert "Days: 20" in chunks[0].content


async def test_an_xlsx_row_keeps_its_column_labels_through_retrieval(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    await upload(client, collection_id, "leave.xlsx", _leave_xlsx(), XLSX_CONTENT_TYPE)

    async with async_session_factory() as session:
        chunks = await semantic_search(
            session, "Jordan Days", collection_id=uuid.UUID(collection_id)
        )
    assert chunks
    assert "Sheet: Leave" in chunks[0].content
    assert "Employee: Jordan" in chunks[0].content
    assert "Days: 20" in chunks[0].content


async def test_a_csv_uploaded_with_a_generic_browser_content_type_still_resolves_by_extension(
    authed_client, fake_embeddings
):
    """Real browsers routinely send application/octet-stream for a type they
    don't recognise — the extension fallback exists precisely for this."""
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    doc_id = await upload(
        client, collection_id, "leave.csv", LEAVE_CSV, "application/octet-stream"
    )

    listed = await client.get(f"/collections/{collection_id}/documents")
    entry = next(d for d in listed.json()["items"] if d["id"] == doc_id)
    assert entry["status"] == "ready"
    assert entry["chunk_count"] > 0
