"""The probe row: how a page knows there is another one behind it."""

from sqlalchemy import select

from app.models.agent import Agent
from app.services.pagination import MAX_PAGE_LIMIT, paginate, split_page


def _sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_paginate_asks_for_one_row_more_than_requested():
    sql = _sql(paginate(select(Agent), limit=25, offset=100))
    assert "LIMIT 26" in sql
    assert "OFFSET 100" in sql


def test_split_page_drops_the_probe_row_and_reports_it():
    rows = list(range(11))
    items, has_more = split_page(rows, 10)
    assert items == list(range(10))
    assert has_more is True


def test_a_full_page_with_nothing_behind_it_is_not_more():
    """Exactly `limit` rows means the probe row was never there."""
    items, has_more = split_page(list(range(10)), 10)
    assert items == list(range(10))
    assert has_more is False


def test_a_short_page_is_the_last_page():
    items, has_more = split_page([1, 2], 10)
    assert items == [1, 2]
    assert has_more is False


def test_an_empty_page_is_the_last_page():
    assert split_page([], 10) == ([], False)


def test_the_ceiling_is_a_bound_not_a_suggestion():
    """The probe row means a page never costs more than one row over the cap."""
    sql = _sql(paginate(select(Agent), limit=MAX_PAGE_LIMIT, offset=0))
    assert f"LIMIT {MAX_PAGE_LIMIT + 1}" in sql
