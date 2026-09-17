"""Scoring one golden example's retrieved chunks against its expected documents.

`_score_case` is pure — no DB, no embeddings — so these pin the arithmetic
directly: recall and reciprocal rank are computed the same way a reader of a
stored `RetrievalBenchmarkRun.results` row would recompute them by hand.
"""

import uuid

from app.services.retrieval_eval_service import _score_case
from app.services.retrieval_service import RetrievedChunk

DOC_A = uuid.uuid4()
DOC_B = uuid.uuid4()
DOC_C = uuid.uuid4()


def chunk(document_id: uuid.UUID, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=document_id,
        filename="doc.txt",
        content="text",
        score=score,
    )


def test_relevant_document_at_rank_one_is_a_perfect_case():
    result = _score_case([chunk(DOC_A), chunk(DOC_B)], {str(DOC_A)})
    assert result.recall == 1.0
    assert result.reciprocal_rank == 1.0


def test_relevant_document_further_down_discounts_reciprocal_rank():
    result = _score_case([chunk(DOC_B), chunk(DOC_C), chunk(DOC_A)], {str(DOC_A)})
    assert result.recall == 1.0
    assert result.reciprocal_rank == 1.0 / 3


def test_no_relevant_document_retrieved_scores_zero():
    result = _score_case([chunk(DOC_B), chunk(DOC_C)], {str(DOC_A)})
    assert result.recall == 0.0
    assert result.reciprocal_rank == 0.0


def test_partial_recall_when_only_some_relevant_documents_are_found():
    result = _score_case([chunk(DOC_A), chunk(DOC_C)], {str(DOC_A), str(DOC_B)})
    assert result.recall == 0.5
    # Rank of the first relevant hit, DOC_A, regardless of the miss beside it.
    assert result.reciprocal_rank == 1.0


def test_a_document_repeated_across_chunks_counts_once_toward_recall():
    """Multiple chunks of the same relevant document must not inflate recall
    past 1.0 for a single-document case, and only the first occurrence sets
    the reciprocal rank."""
    result = _score_case([chunk(DOC_B), chunk(DOC_A), chunk(DOC_A)], {str(DOC_A)})
    assert result.recall == 1.0
    assert result.reciprocal_rank == 0.5


def test_retrieved_document_ids_are_kept_raw_including_duplicates():
    result = _score_case([chunk(DOC_A), chunk(DOC_A)], {str(DOC_A)})
    assert result.retrieved_document_ids == [str(DOC_A), str(DOC_A)]
