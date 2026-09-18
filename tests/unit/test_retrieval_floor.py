"""The relevance floor's actual arithmetic, pinned without a database.

`_above_floor` is the whole decision: everything else in `semantic_search` is
metrics, tracing, and SQL plumbing around this one filter.
"""

import uuid

from app.services.retrieval_service import RetrievedChunk, _above_floor


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="doc.txt",
        content="text",
        score=score,
    )


def test_no_floor_configured_returns_everything_unchanged():
    chunks = [chunk(0.9), chunk(0.1)]
    assert _above_floor(chunks, None) == chunks


def test_chunks_below_the_floor_are_dropped():
    chunks = [chunk(0.9), chunk(0.5), chunk(0.2)]
    assert _above_floor(chunks, 0.6) == [chunks[0]]


def test_a_score_exactly_at_the_floor_is_kept():
    """The floor is a minimum to clear, not a strict threshold to beat — a
    chunk scoring exactly the configured value is evidence, not noise."""
    exact = chunk(0.6)
    assert _above_floor([exact], 0.6) == [exact]


def test_everything_below_the_floor_returns_an_empty_list():
    assert _above_floor([chunk(0.1), chunk(0.2)], 0.6) == []


def test_an_empty_input_stays_empty_regardless_of_the_floor():
    assert _above_floor([], 0.6) == []
