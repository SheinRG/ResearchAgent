"""
Tests for the upload size guard.

The guard exists to bound memory, not just to reject big files: the naive
`await file.read()` then `len(content) > MAX` check has already materialised the
whole body before it can refuse it, which is the failure mode that matters on a
512MB instance. So these tests assert on *how much was read*, not only on the
resulting 413.
"""

import pytest
from fastapi import HTTPException

from app.routers.upload import _read_within_limit, CHUNK_SIZE


class FakeUpload:
    """
    Minimal stand-in for UploadFile that records how many bytes were handed out.

    `read(n)` honours the requested size, the way Starlette's SpooledTemporaryFile
    -backed UploadFile does, so `bytes_served` reflects what a real request would
    have pulled into memory.
    """

    def __init__(self, payload: bytes):
        self._payload = payload
        self._pos = 0
        self.bytes_served = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._payload[self._pos:]
        else:
            chunk = self._payload[self._pos:self._pos + size]
        self._pos += len(chunk)
        self.bytes_served += len(chunk)
        return chunk


@pytest.mark.asyncio
async def test_small_file_is_returned_intact():
    """A file under the limit comes back byte-identical."""
    payload = b"hello world" * 100
    upload = FakeUpload(payload)
    assert await _read_within_limit(upload, 1024 * 1024) == payload


@pytest.mark.asyncio
async def test_empty_file_is_allowed_through():
    """Zero bytes is not a size violation — extraction rejects it later."""
    assert await _read_within_limit(FakeUpload(b""), 1024) == b""


@pytest.mark.asyncio
async def test_file_exactly_at_limit_is_accepted():
    """The limit is inclusive: exactly MAX bytes is fine, MAX + 1 is not."""
    payload = b"x" * 2048
    assert await _read_within_limit(FakeUpload(payload), 2048) == payload

    with pytest.raises(HTTPException) as exc:
        await _read_within_limit(FakeUpload(payload + b"x"), 2048)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_oversized_file_is_rejected_without_being_fully_read():
    """
    The regression that motivated this: a body far larger than the limit must
    not be pulled into memory in full before being refused. We allow one chunk
    of overshoot — that's the read that detects the breach.
    """
    limit = 64 * 1024
    upload = FakeUpload(b"x" * (limit * 50))

    with pytest.raises(HTTPException) as exc:
        await _read_within_limit(upload, limit)

    assert exc.value.status_code == 413
    assert upload.bytes_served <= limit + CHUNK_SIZE
