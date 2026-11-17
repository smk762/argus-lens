"""Tests for the shared atomic model-file download (no network: ``httpx.stream`` is faked)."""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest

from argus_lens.backends import _download
from argus_lens.backends._download import stream_download


class _FakeResponse:
    """Stand-in for a streamed httpx response carrying a fixed body."""

    def __init__(self, body: bytes) -> None:
        """Store the body served by :meth:`iter_bytes`."""
        self._body = body
        self.headers = {"content-length": str(len(body))}

    def raise_for_status(self) -> None:
        """Succeed: the fake response is always a 200."""

    def iter_bytes(self, chunk_size: int) -> Any:
        """Yield the body in *chunk_size* pieces."""
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Serve a fixed body from every ``httpx.stream`` call; return that body."""
    body = b"model-bytes" * 100

    @contextlib.contextmanager
    def _stream(*_a: Any, **_kw: Any) -> Any:
        """Yield a fake streamed response."""
        yield _FakeResponse(body)

    monkeypatch.setattr(httpx, "stream", _stream)
    return body


def test_download_gets_umask_mode_not_mkstemp_0600(tmp_path: Path, fake_stream: bytes) -> None:
    """The final file is readable as a normally created file would be, not left owner-only."""
    old = os.umask(0o022)
    try:
        dest = tmp_path / "model.onnx"
        stream_download("http://x/model.onnx", dest, event="t", chunk_size=64)
    finally:
        os.umask(old)
    assert dest.read_bytes() == fake_stream
    assert stat.S_IMODE(dest.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".*.part"))  # temp file renamed, not left behind


def test_download_is_fsynced_before_it_appears(
    tmp_path: Path, fake_stream: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data is fsynced before the rename makes it visible under the real name."""
    dest = tmp_path / "model.onnx"
    seen: list[bool] = []
    real_fsync = os.fsync

    def _fsync(fd: int) -> None:
        """Record whether *dest* already existed at each fsync, then fsync for real."""
        seen.append(dest.exists())
        real_fsync(fd)

    monkeypatch.setattr(_download.os, "fsync", _fsync)
    stream_download("http://x/model.onnx", dest, event="t", chunk_size=64)
    assert seen and seen[0] is False  # first fsync is the temp file's data, before the rename
