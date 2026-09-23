"""Reading a reranker's answer back, and what makes it unusable.

`parse_rerank_response` is the entire contract between the platform and the
model: the same discipline `evaluation_prompts.parse_audit_response` already
established (tolerate a fence, tolerate a preamble, raise on anything the
platform cannot act on rather than guessing at a partial reordering).
"""

import json

import pytest

from app.services.reranking_prompts import build_rerank_request, parse_rerank_response


def test_a_clean_json_object_parses():
    text = json.dumps({"order": [3, 1, 2]})
    assert parse_rerank_response(text, expected_ids={1, 2, 3}) == [3, 1, 2]


def test_a_fenced_response_is_tolerated():
    text = "```json\n" + json.dumps({"order": [2, 1]}) + "\n```"
    assert parse_rerank_response(text, expected_ids={1, 2}) == [2, 1]


def test_a_preamble_before_the_object_is_tolerated():
    text = 'Here is the ranking:\n{"order": [1, 2]}'
    assert parse_rerank_response(text, expected_ids={1, 2}) == [1, 2]


def test_unparseable_text_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_rerank_response("not json at all", expected_ids={1, 2})


def test_a_missing_order_key_raises():
    with pytest.raises(ValueError, match="`order`"):
        parse_rerank_response(json.dumps({"ranking": [1, 2]}), expected_ids={1, 2})


def test_a_non_integer_entry_raises():
    with pytest.raises(ValueError, match="non-integer"):
        parse_rerank_response(json.dumps({"order": [1, "two"]}), expected_ids={1, 2})


def test_a_dropped_passage_raises():
    """The reranker must account for every passage it was given — silently
    dropping one is not a valid reordering, it is missing evidence."""
    with pytest.raises(ValueError, match="permutation"):
        parse_rerank_response(json.dumps({"order": [1]}), expected_ids={1, 2})


def test_a_duplicated_passage_raises():
    with pytest.raises(ValueError, match="permutation"):
        parse_rerank_response(json.dumps({"order": [1, 1]}), expected_ids={1, 2})


def test_an_invented_passage_number_raises():
    with pytest.raises(ValueError, match="permutation"):
        parse_rerank_response(json.dumps({"order": [1, 2, 3]}), expected_ids={1, 2})


def test_the_request_numbers_passages_the_way_they_were_given():
    request = build_rerank_request(
        query="How many vacation days?",
        passages=[(1, "Full-time staff accrue 25 days."), (2, "Contractors are not eligible.")],
    )
    assert '<passage id="1">' in request
    assert '<passage id="2">' in request
    assert "How many vacation days?" in request
