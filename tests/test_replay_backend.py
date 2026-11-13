"""Tests for the replay backend — recorded captions from the cortex store (#45).

No Postgres required: a fake connection/cursor is injected so the SQL lookup,
the sha256 keying, and the engine short-circuit are all exercised in-process.
"""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image

from argus_lens.backends.replay import ReplayBackend, ReplayMiss
from argus_lens.engine import ArgusLens

# (final_caption, variants, raw_tags, raw_prose, backend, profile, metadata, version)
_RECORDED_ROW = (
    "a weathered fisherman mends his net at dawn",
    {"training": "sks_person, fisherman, net, dawn", "identity": "a weathered fisherman"},
    "1girl, solo",  # deliberately arbitrary — replay returns it verbatim
    "a weathered fisherman mends his net at dawn",
    "hybrid",
    {"target_category": "identity", "target_style": "photo"},
    {"note": "seeded by the tape"},
    2,
)


class _FakeCursor:
    """Minimal psycopg-cursor stand-in backed by an in-memory row table."""

    def __init__(self, rows: dict[tuple[str, str], tuple]) -> None:
        """Store the in-memory row table keyed by ``(column, value)``."""
        self._rows = rows
        self._result: tuple | None = None

    def __enter__(self) -> _FakeCursor:
        """Enter the cursor context (psycopg cursors are context managers)."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the cursor context; nothing to clean up."""
        return None

    def execute(self, sql: str, params: tuple) -> None:
        """Resolve the canned row for whichever key column the SQL filters on."""
        # The lookup SQL filters on either a.sha256 or a.uri — detect which.
        column = "sha256" if "a.sha256" in sql else "uri"
        self._result = self._rows.get((column, params[0]))

    def fetchone(self) -> tuple | None:
        """Return the row matched by the last :meth:`execute`, or ``None``."""
        return self._result


class _FakeConn:
    """Connection stand-in that hands out cursors over a shared row table."""

    def __init__(self, rows: dict[tuple[str, str], tuple]) -> None:
        """Store the shared row table and mark the connection open."""
        self._rows = rows
        self.closed = False

    def cursor(self) -> _FakeCursor:
        """Return a fresh cursor over the shared row table."""
        return _FakeCursor(self._rows)

    def close(self) -> None:
        """Mark the connection closed (asserted by the ownership test)."""
        self.closed = True


def _png_bytes(color: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    """Return PNG bytes for a tiny solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _backend_with(rows: dict[tuple[str, str], tuple]) -> ReplayBackend:
    """Build a ReplayBackend wired to an injected fake connection."""
    return ReplayBackend(connection=_FakeConn(rows))


def test_lookup_by_sha256_returns_recorded_result_verbatim() -> None:
    """A sha256 hit returns the recorded caption verbatim with replay provenance stamped."""
    sha = "deadbeef" * 8
    backend = _backend_with({("sha256", sha): _RECORDED_ROW})

    result = backend.lookup(sha256=sha)

    assert result is not None
    assert result.final_caption == _RECORDED_ROW[0]
    assert result.caption_variants == _RECORDED_ROW[1]
    assert result.raw_tags == "1girl, solo"
    assert result.raw_prose == _RECORDED_ROW[3]
    assert result.backend_name == "hybrid"
    assert result.selected_category == "identity"
    # Replay provenance is stamped so consumers can tell it from a live caption.
    assert result.metadata["replay"]["source"] == "cortex-lineage"
    assert result.metadata["replay"]["asset_sha256"] == sha
    assert result.metadata["replay"]["caption_version"] == 2
    # Recorded metadata is preserved alongside the replay stamp.
    assert result.metadata["note"] == "seeded by the tape"


def test_lookup_falls_back_to_uri_when_sha_absent() -> None:
    """With no sha row (or no sha given), the lookup resolves by asset uri."""
    backend = _backend_with({("uri", "immich://abc"): _RECORDED_ROW})

    assert backend.lookup(sha256="notpresent" * 6) is None  # no sha row, no uri given
    result = backend.lookup(sha256=None, uri="immich://abc")
    assert result is not None
    assert result.final_caption == _RECORDED_ROW[0]


def test_lookup_prefers_sha_over_uri() -> None:
    """When both keys match, the sha256 row wins over the uri row."""
    sha = "a" * 64
    sha_row = _RECORDED_ROW
    uri_row = ("wrong caption",) + _RECORDED_ROW[1:]
    backend = _backend_with({("sha256", sha): sha_row, ("uri", "file:///x.jpg"): uri_row})

    result = backend.lookup(sha256=sha, uri="file:///x.jpg")
    assert result is not None
    assert result.final_caption == sha_row[0]


def test_lookup_miss_returns_none() -> None:
    """An asset absent under both keys yields None (the engine turns this into a miss)."""
    backend = _backend_with({})
    assert backend.lookup(sha256="f" * 64) is None
    assert backend.lookup(sha256=None, uri="nope") is None


def test_is_available_reflects_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Availability tracks a configured DSN or an injected connection."""
    monkeypatch.delenv("CORTEX_PG_URL", raising=False)
    assert ReplayBackend().is_available() is False
    assert "CORTEX_PG_URL" in (ReplayBackend().availability_reason() or "")

    assert _backend_with({}).is_available() is True  # injected connection
    assert ReplayBackend(dsn="postgresql://x/y").is_available() is True


def test_caption_image_is_unsupported() -> None:
    """The pixel-level caption path is unsupported and raises."""
    backend = _backend_with({})
    with pytest.raises(NotImplementedError):
        backend.caption_image(Image.new("RGB", (4, 4)))


def test_unload_leaves_injected_connection_alone() -> None:
    """unload() never closes a connection the backend does not own."""
    conn = _FakeConn({})
    backend = ReplayBackend(connection=conn)
    backend.unload()
    assert conn.closed is False  # we don't own an injected connection


def test_engine_short_circuits_to_recorded_caption() -> None:
    """The engine returns the recorded result and bypasses the assembly pipeline."""
    data = _png_bytes()
    sha = hashlib.sha256(data).hexdigest()
    backend = _backend_with({("sha256", sha): _RECORDED_ROW})

    engine = ArgusLens(backend=backend)
    result = engine.caption(data)

    # Verbatim recorded output — the assembly pipeline was bypassed, so the
    # arbitrary raw_tags survive untouched instead of being normalised.
    assert result.final_caption == _RECORDED_ROW[0]
    assert result.raw_tags == "1girl, solo"
    assert result.metadata["replay"]["asset_sha256"] == sha


def test_engine_raises_replay_miss_for_unknown_asset() -> None:
    """An unknown asset surfaces as ReplayMiss from the engine."""
    backend = _backend_with({})  # empty tape
    engine = ArgusLens(backend=backend)
    with pytest.raises(ReplayMiss):
        engine.caption(_png_bytes())


def test_engine_pil_input_has_no_sha_and_misses() -> None:
    """A pre-decoded PIL input carries no sha256, so replay misses cleanly."""
    # A pre-decoded PIL image carries no original-bytes sha256, so replay can't
    # key it — a clean miss rather than a wrong match.
    backend = _backend_with({("sha256", "x" * 64): _RECORDED_ROW})
    engine = ArgusLens(backend=backend)
    with pytest.raises(ReplayMiss):
        engine.caption(Image.new("RGB", (8, 8)))


def test_engine_batch_replays_each_by_name() -> None:
    """Batch replay keys each recorded caption to its provided name."""
    red, blue = _png_bytes((200, 0, 0)), _png_bytes((0, 0, 200))
    rows = {
        ("sha256", hashlib.sha256(red).hexdigest()): _RECORDED_ROW,
        ("sha256", hashlib.sha256(blue).hexdigest()): ("a cold blue sky",) + _RECORDED_ROW[1:],
    }
    engine = ArgusLens(backend=_backend_with(rows))

    results = engine.caption_batch([red, blue], names=["red.png", "blue.png"])

    assert set(results) == {"red.png", "blue.png"}
    assert results["red.png"].final_caption == _RECORDED_ROW[0]
    assert results["blue.png"].final_caption == "a cold blue sky"


# -- server wiring ------------------------------------------------------------

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.server import create_app  # noqa: E402


def _replay_client(rows: dict[tuple[str, str], tuple]) -> TestClient:
    """A TestClient whose server engine replays from an injected fake connection."""
    return TestClient(create_app(default_backend=_backend_with(rows)))


def test_server_caption_returns_recorded_caption() -> None:
    """POST /caption returns the recorded caption for a known asset."""
    data = _png_bytes()
    sha = hashlib.sha256(data).hexdigest()
    client = _replay_client({("sha256", sha): _RECORDED_ROW})

    resp = client.post("/caption", files={"file": ("photo.jpg", data, "image/jpeg")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["final_caption"] == _RECORDED_ROW[0]
    assert body["metadata"]["replay"]["asset_sha256"] == sha


def test_server_caption_missing_asset_is_404() -> None:
    """POST /caption maps a replay miss to HTTP 404."""
    client = _replay_client({})  # empty tape
    resp = client.post("/caption", files={"file": ("photo.jpg", _png_bytes(), "image/jpeg")})
    assert resp.status_code == 404
    assert "no recorded caption" in resp.json()["detail"]


def test_server_caption_rejects_invalid_image() -> None:
    """POST /caption rejects a non-image upload with HTTP 400."""
    client = _replay_client({})
    resp = client.post("/caption", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400


def _truncated_jpeg() -> bytes:
    """A JPEG with a valid header but truncated scan data.

    It passes ``Image.verify()`` (a header-only check) yet raises on a full
    decode — the case that regressed the upload endpoints to HTTP 500 when they
    validated with ``verify()`` instead of a real decode.
    """
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buf, format="JPEG", quality=95)
    full = buf.getvalue()
    return full[: len(full) // 2]


def test_server_caption_rejects_truncated_image() -> None:
    """POST /caption rejects a header-valid-but-undecodable image with 400, not 500."""
    client = _replay_client({})
    resp = client.post("/caption", files={"file": ("x.jpg", _truncated_jpeg(), "image/jpeg")})
    assert resp.status_code == 400


def test_server_batch_skips_undecodable_and_keeps_good() -> None:
    """POST /caption/batch skips an undecodable upload and still returns the good ones."""
    good = _png_bytes()
    sha = hashlib.sha256(good).hexdigest()
    client = _replay_client({("sha256", sha): _RECORDED_ROW})

    resp = client.post(
        "/caption/batch",
        files=[
            ("files", ("good.png", good, "image/png")),
            ("files", ("bad.jpg", _truncated_jpeg(), "image/jpeg")),
        ],
    )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert "good.png" in results
    assert results["good.png"]["final_caption"] == _RECORDED_ROW[0]
    assert "bad.jpg" not in results  # undecodable file skipped, not a 500


def test_server_stream_miss_reports_uploaded_name() -> None:
    """POST /caption/stream labels a replay miss with the uploaded filename, not "bytes"."""
    client = _replay_client({})  # empty tape → every asset misses
    resp = client.post(
        "/caption/stream",
        files=[("files", ("portrait.jpg", _png_bytes(), "image/jpeg"))],
    )

    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["name"] == "portrait.jpg"  # the caller's name, not the ingest "bytes"
    assert "no recorded caption" in lines[0]["error"]


# -- per-item miss handling for batch/stream (#48) ----------------------------


def test_engine_batch_on_miss_skips_and_keeps_good() -> None:
    """With an on_miss handler, caption_batch skips the missing image and captions the rest."""
    red = _png_bytes((200, 0, 0))
    blue = _png_bytes((0, 0, 200))  # deliberately not on the tape
    rows = {("sha256", hashlib.sha256(red).hexdigest()): _RECORDED_ROW}
    engine = ArgusLens(backend=_backend_with(rows))

    misses: list[str] = []
    results = engine.caption_batch(
        [red, blue],
        names=["red.png", "blue.png"],
        on_miss=lambda name, exc: misses.append(name),
    )

    assert set(results) == {"red.png"}  # the good image survived
    assert results["red.png"].final_caption == _RECORDED_ROW[0]
    assert misses == ["blue.png"]  # the miss was reported, not raised


def test_engine_batch_raises_miss_without_on_miss() -> None:
    """Without a handler, a batch miss still raises (fail-fast default)."""
    red = _png_bytes((200, 0, 0))
    blue = _png_bytes((0, 0, 200))
    rows = {("sha256", hashlib.sha256(red).hexdigest()): _RECORDED_ROW}
    engine = ArgusLens(backend=_backend_with(rows))

    with pytest.raises(ReplayMiss):
        engine.caption_batch([red, blue], names=["red.png", "blue.png"])


def test_server_batch_reports_miss_and_keeps_good() -> None:
    """POST /caption/batch returns the recorded caption and reports the miss under errors."""
    known = _png_bytes((1, 2, 3))
    unknown = _png_bytes((9, 9, 9))  # not on the tape
    sha = hashlib.sha256(known).hexdigest()
    client = _replay_client({("sha256", sha): _RECORDED_ROW})

    resp = client.post(
        "/caption/batch",
        files=[
            ("files", ("known.png", known, "image/png")),
            ("files", ("unknown.png", unknown, "image/png")),
        ],
    )

    assert resp.status_code == 200  # not 404 — one miss no longer fails the batch
    body = resp.json()
    assert body["results"]["known.png"]["final_caption"] == _RECORDED_ROW[0]
    assert "unknown.png" not in body["results"]
    assert "no recorded caption" in body["errors"]["unknown.png"]


def test_server_stream_reports_miss_and_continues() -> None:
    """POST /caption/stream reports a miss as one line and keeps captioning the rest."""
    known = _png_bytes((1, 2, 3))
    unknown = _png_bytes((9, 9, 9))  # not on the tape
    sha = hashlib.sha256(known).hexdigest()
    client = _replay_client({("sha256", sha): _RECORDED_ROW})

    resp = client.post(
        "/caption/stream",
        files=[
            ("files", ("known1.png", known, "image/png")),
            ("files", ("miss.png", unknown, "image/png")),
            ("files", ("known2.png", known, "image/png")),
        ],
    )

    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert [line["name"] for line in lines] == ["known1.png", "miss.png", "known2.png"]
    assert lines[0]["final_caption"] == _RECORDED_ROW[0]
    assert "no recorded caption" in lines[1]["error"]
    assert lines[2]["final_caption"] == _RECORDED_ROW[0]  # stream did not stop at the miss
