"""One page of a list, and how the platform decides there is another.

Every list a tenant can grow — documents, conversations, transcripts — was
returning every row it had. That is a query whose cost is set by the caller's
history rather than by the request, and the first tenant to notice is the one
whose thread got long enough to time out.

The page is described by `limit`/`offset` rather than an opaque cursor because
these lists are ordered by columns a caller can already see, and an offset a
client can compute is one it can also debug. The tradeoff is the usual one: a
row inserted at the head of a descending sort between two requests shifts the
window. That is acceptable for browsing config and history, and it is worth
saying out loud rather than implying otherwise with a cursor that would only
look more rigorous.
"""

from sqlalchemy import Select

DEFAULT_PAGE_LIMIT = 50
# The ceiling a caller may ask for. Not a guess about what a client needs — a
# bound on the largest response one request can cost the platform.
MAX_PAGE_LIMIT = 200


def paginate(stmt: Select, *, limit: int, offset: int) -> Select:
    """Window a statement, asking for one row more than the caller wants.

    That extra row is how the platform answers "is there another page" without
    a second `COUNT(*)` over the whole table on every request. It never reaches
    the caller — `split_page` drops it and reports its existence instead.
    """
    return stmt.limit(limit + 1).offset(offset)


def split_page[T](rows: list[T], limit: int) -> tuple[list[T], bool]:
    """Trim the probe row off a windowed result and report whether it was there."""
    return rows[:limit], len(rows) > limit
