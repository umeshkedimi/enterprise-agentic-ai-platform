"""A large table, ingested for real, stays retrievable row by row.

The unit suite already proves the chunker's guarantee in isolation (a row
never splits, and the old algorithm demonstrably could). This is the
same claim through the real pipeline: `document_service.process_document` →
the real `chunk_text` → real embeddings (faked) → real pgvector storage →
the Chunk 8 benchmark harness, which is exactly the validation loop the RAG
roadmap promised — rerun the harness after a chunking change and read the
number, rather than trust that it helped.
"""

from tests.integration.conftest import create_collection, get_document, upload_document

LEAVE_CSV_HEADER = b"Employee,Days,Department,Location,Manager\n"


def _wide_leave_csv(n: int) -> bytes:
    rows = [
        f"Person{i},{i},Engineering,Remote,Person{i + 1}\n".encode() for i in range(n)
    ]
    return LEAVE_CSV_HEADER + b"".join(rows)


async def test_facts_scattered_across_a_large_table_are_all_recallable(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    doc_id = await upload_document(
        client, collection_id, "leave.csv", _wide_leave_csv(80), content_type="text/csv"
    )
    # More than one chunk, or this test is not exercising a boundary at all.
    entry = await get_document(client, collection_id, doc_id)
    assert entry["chunk_count"] > 1

    # Golden examples spread across the document — early, middle, late row —
    # so a fact that landed near a chunk boundary is exactly as likely to be
    # tested as one that didn't.
    for target in (5, 40, 75):
        r = await client.post(
            f"/collections/{collection_id}/golden-examples",
            json={
                "query": f"Person{target} Days Department",
                "relevant_document_ids": [doc_id],
            },
        )
        assert r.status_code == 201, r.text

    run = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "structure-aware-chunking", "top_k": 3},
    )
    assert run.status_code == 201, run.text
    body = run.json()
    assert body["cases"] == 3
    assert body["mean_recall"] == 1.0
    assert body["mrr"] == 1.0
