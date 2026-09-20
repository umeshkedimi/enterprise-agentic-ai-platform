"""The actual claim Chunk 12 makes: a chunk boundary falls between structural
units — paragraphs, table rows — never inside one, and never overflows its
token budget even right after an overlap reset.

`chunk_text` had no dedicated tests before this file; the v1 algorithm was
only ever exercised indirectly, through documents uploaded in integration
tests. These pin the packing logic itself, offline.
"""

import pytest

from app.services.chunking import _encoding, chunk_text, count_tokens


def test_prose_paragraphs_that_fit_are_never_split():
    paragraphs = [f"Paragraph number {i} has a few words in it." for i in range(6)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size_tokens=30, chunk_overlap_tokens=5)

    for paragraph in paragraphs:
        assert any(paragraph in chunk for chunk in chunks), paragraph


def test_a_table_row_is_never_split_across_a_chunk_boundary():
    """The whole point: `_render_table_rows` joins rows with a single
    newline and no blank lines between them — exactly the shape that made a
    row a target for v1's raw token-position slicing."""
    rows = [f"Employee: Person{i}, Days: {i}" for i in range(40)]
    table = "\n".join(rows)

    chunks = chunk_text(table, chunk_size_tokens=50, chunk_overlap_tokens=10)

    assert len(chunks) > 1, "the table should have needed more than one chunk"
    for row in rows:
        assert any(row in chunk for chunk in chunks), f"{row!r} was not kept intact"


def test_no_chunk_exceeds_the_token_budget_even_right_after_an_overlap_reset():
    """Pins the fix for a real edge case found while writing this: an
    overlap tail seeded into a new chunk, plus the next unit, can together
    exceed the budget — the packer has to drop the overlap rather than let
    the chunk grow past what was asked for."""
    rows = [f"Employee: Person{i}, Days: {i}" for i in range(60)]
    table = "\n".join(rows)
    chunk_size = 50

    chunks = chunk_text(table, chunk_size_tokens=chunk_size, chunk_overlap_tokens=20)

    for chunk in chunks:
        assert count_tokens(chunk) <= chunk_size


def test_consecutive_chunks_share_at_least_one_row_at_the_boundary():
    rows = [f"Employee: Person{i}, Days: {i}" for i in range(30)]
    table = "\n".join(rows)

    chunks = chunk_text(table, chunk_size_tokens=40, chunk_overlap_tokens=15)

    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        a_rows = set(a.split("\n\n"))
        b_rows = set(b.split("\n\n"))
        assert a_rows & b_rows, "expected at least one row to carry across the boundary"


def test_a_single_paragraph_too_large_to_fit_falls_back_to_token_slicing():
    """Nothing structural to respect here — one giant line, no table rows,
    no paragraph breaks — so the only remaining strategy is v1's own."""
    long_paragraph = " ".join(f"word{i}" for i in range(200))

    chunks = chunk_text(long_paragraph, chunk_size_tokens=30, chunk_overlap_tokens=5)

    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 30


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("something", chunk_size_tokens=10, chunk_overlap_tokens=10)


def _legacy_chunk_text(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    """v1's exact algorithm, preserved only here to prove what it used to do.

    Pure positional token-window slicing, no awareness of paragraph or row
    boundaries at all — this is what every chunk in the platform's collections
    was built with before this chunk landed.
    """
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        piece = _encoding.decode(tokens[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end == len(tokens):
            break
        start += step
    return chunks


def test_the_old_algorithm_demonstrably_broke_rows_the_new_one_does_not():
    """The concrete "prove it beats the naive window" comparison the roadmap
    asked for. Wide rows and a tight overlap — the exact conditions under
    which v1's positional slicing has no row-boundary luck to fall back on,
    unlike the shorter rows used in the tests above (where a generous
    default overlap happens to recover most splits by accident, masking the
    underlying flaw rather than fixing it)."""
    rows = [
        f"Employee: Person{i}, Days: {i}, Department: Engineering, "
        f"Location: Remote, Manager: Person{i + 1}, StartDate: 2020-01-{(i % 28) + 1:02d}"
        for i in range(60)
    ]
    table = "\n".join(rows)

    legacy_chunks = _legacy_chunk_text(table, chunk_size_tokens=100, chunk_overlap_tokens=10)
    new_chunks = chunk_text(table, chunk_size_tokens=100, chunk_overlap_tokens=10)

    legacy_broken = [r for r in rows if not any(r in c for c in legacy_chunks)]
    new_broken = [r for r in rows if not any(r in c for c in new_chunks)]

    assert legacy_broken, "expected the naive algorithm to lose at least one row here"
    assert new_broken == []
