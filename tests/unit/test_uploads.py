"""The bounded read behind the upload size limit.

The endpoint has a fast path that trusts the multipart parser's own byte count,
so the integration tests never reach the read loop. This covers the case that
fast path cannot answer: an upload whose size the parser did not record.
"""

import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.documents import _read_within_limit


def _upload(body: bytes) -> UploadFile:
    # size=None is the point: it is what a client sending chunked
    # transfer-encoding leaves behind, and it is why the read loop counts.
    return UploadFile(file=io.BytesIO(body), filename="doc.txt", size=None)


async def test_reads_a_file_within_the_limit():
    assert await _read_within_limit(_upload(b"hello"), 1024) == b"hello"


async def test_reads_a_file_exactly_at_the_limit():
    body = b"x" * 1024
    assert await _read_within_limit(_upload(body), 1024) == body


async def test_refuses_one_byte_over_the_limit():
    with pytest.raises(HTTPException) as exc:
        await _read_within_limit(_upload(b"x" * 1025), 1024)
    assert exc.value.status_code == 413


async def test_oversized_upload_is_not_buffered_whole():
    """The refusal happens while reading, not after.

    A stream that would fail on the third chunk proves the loop stopped early:
    if the implementation read the upload whole before measuring it, this raises
    the sentinel error instead of returning a 413.
    """

    class ExplodingStream(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        def read(self, size: int = -1) -> bytes:
            self.reads += 1
            if self.reads > 2:
                raise AssertionError("read past the limit")
            return b"x" * (1024 * 1024)

    with pytest.raises(HTTPException) as exc:
        await _read_within_limit(
            UploadFile(file=ExplodingStream(), filename="big.txt", size=None),
            1024 * 1024,
        )
    assert exc.value.status_code == 413
