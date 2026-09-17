"""End-to-end retrieval benchmarking: real chunks, real pgvector, real scoring.

Only the embedding call is faked (`fake_embeddings`), the same deterministic
lexical stand-in the rest of the suite uses — real enough that a vacation
question ranks the vacation document above the expenses one. Everything else —
chunking, storage, the HNSW-indexed similarity query, and the recall/MRR
arithmetic — runs for real.
"""

import uuid

from httpx import ASGITransport, AsyncClient

from tests.integration.conftest import (
    EXPENSES_TEXT,
    VACATION_TEXT,
    _delete_tenant,
    create_collection,
    upload_document,
)


async def test_benchmark_scores_retrieval_against_golden_examples(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    vacation_id = await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)
    await upload_document(client, collection_id, "expenses.txt", EXPENSES_TEXT)

    golden = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={
            "query": "How many vacation days do employees accrue?",
            "relevant_document_ids": [vacation_id],
        },
    )
    assert golden.status_code == 201, golden.text

    run = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark",
        json={"label": "baseline", "top_k": 3},
    )
    assert run.status_code == 201, run.text
    body = run.json()

    # The vacation question matches the vacation document far better than the
    # expenses one under the deterministic lexical embedding, so it should be
    # the top hit: perfect recall, reciprocal rank of 1.
    assert body["cases"] == 1
    assert body["mean_recall"] == 1.0
    assert body["mrr"] == 1.0
    assert body["results"][0]["relevant_document_ids"] == [vacation_id]
    assert body["results"][0]["retrieved_document_ids"][0] == vacation_id

    listed = await client.get(f"/collections/{collection_id}/retrieval-benchmark")
    assert listed.status_code == 200, listed.text
    assert listed.json()["items"][0]["label"] == "baseline"


async def test_benchmark_refuses_without_golden_examples(authed_client, fake_embeddings):
    client, _ = authed_client
    collection_id = await create_collection(client, "empty")

    r = await client.post(
        f"/collections/{collection_id}/retrieval-benchmark", json={"label": "baseline"}
    )

    # A run with zero cases would still produce a mean_recall and an mrr —
    # numbers that look real and mean nothing. Refused rather than fabricated.
    assert r.status_code == 409, r.text


async def test_golden_example_requires_at_least_one_relevant_document(
    authed_client, fake_embeddings
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    r = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={"query": "anything", "relevant_document_ids": []},
    )

    assert r.status_code == 422, r.text


async def test_golden_examples_can_be_listed_and_deleted(authed_client, fake_embeddings):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")
    doc_id = await upload_document(client, collection_id, "vacation.txt", VACATION_TEXT)

    created = await client.post(
        f"/collections/{collection_id}/golden-examples",
        json={"query": "leave policy", "relevant_document_ids": [doc_id]},
    )
    example_id = created.json()["id"]

    listed = await client.get(f"/collections/{collection_id}/golden-examples")
    assert [e["id"] for e in listed.json()["items"]] == [example_id]

    deleted = await client.delete(f"/golden-examples/{example_id}")
    assert deleted.status_code == 204, deleted.text

    listed_again = await client.get(f"/collections/{collection_id}/golden-examples")
    assert listed_again.json()["items"] == []


async def test_another_tenants_collection_is_404_for_golden_examples(
    app, authed_client, admin_headers
):
    client, _ = authed_client
    collection_id = await create_collection(client, "hr")

    other = await client.post(
        "/tenants",
        json={"slug": f"o-{uuid.uuid4().hex[:8]}", "name": "Other"},
        headers=admin_headers,
    )
    other_id = uuid.UUID(other.json()["id"])
    key = await client.post(
        f"/tenants/{other_id}/keys", json={"name": "k"}, headers=admin_headers
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as intruder:
        intruder.headers["Authorization"] = f"Bearer {key.json()['api_key']}"
        r = await intruder.post(
            f"/collections/{collection_id}/golden-examples",
            json={"query": "anything", "relevant_document_ids": [str(uuid.uuid4())]},
        )

    assert r.status_code == 404, r.text
    await _delete_tenant(other_id)
