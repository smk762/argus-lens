"""Tests for the /immich/* endpoints (album listing, pull, caption stream)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.connectors.immich import ImmichSink, ImmichSource  # noqa: E402
from argus_lens.server import create_app  # noqa: E402

NOT_CONFIGURED = "Immich is not configured: set IMMICH_URL and IMMICH_API_KEY"


class _StubBackend(CaptionBackend):
    """Caption backend stub that returns a fixed caption without any model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op load."""
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption for any image."""
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        """No-op unload."""
        pass


def _png_bytes(size=(8, 8), color=(120, 120, 120)) -> bytes:
    """Return PNG-encoded bytes for a solid-color test image."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _client(source_root: Path | None = None) -> TestClient:
    """FastAPI test client wired to the stub caption backend."""
    return TestClient(create_app(default_backend=_StubBackend(), source_root=str(source_root) if source_root else None))


def _events(resp) -> list[dict]:
    """Parse an NDJSON response body into a list of event dicts."""
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


@pytest.fixture
def immich_env(monkeypatch) -> None:
    """Point the Immich endpoints at a fake server via env vars."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.local")
    monkeypatch.setenv("IMMICH_API_KEY", "key")


@pytest.fixture
def album_assets(monkeypatch) -> list[dict]:
    """Stub ImmichSource.list_album_assets with a two-image album."""
    assets = [{"id": "a1", "name": "01.jpg"}, {"id": "a2", "name": "02.jpg"}]
    monkeypatch.setattr(ImmichSource, "list_album_assets", lambda self, album_id: list(assets))
    return assets


# --- configuration guard ---


def test_immich_endpoints_return_503_when_unconfigured(monkeypatch, tmp_path: Path) -> None:
    """All /immich endpoints return 503 with a fixed detail until env vars are set."""
    monkeypatch.delenv("IMMICH_URL", raising=False)
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    client = _client(tmp_path)

    pull_body = {"album_id": "al1", "dest_folder": "x"}
    caption_body = {"album_id": "al1"}
    for resp in (
        client.get("/immich/albums"),
        client.post("/immich/pull", json=pull_body),
        client.post("/immich/caption/stream", json=caption_body),
    ):
        assert resp.status_code == 503
        assert resp.json()["detail"] == NOT_CONFIGURED


def test_immich_requires_both_env_vars(monkeypatch) -> None:
    """A URL without an API key (or vice versa) is still unconfigured."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.local")
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    assert _client().get("/immich/albums").status_code == 503


# --- GET /immich/albums ---


def test_immich_albums_lists_albums(immich_env, monkeypatch) -> None:
    """Returns the connector's album list under an `albums` key."""
    albums = [{"id": "al1", "name": "Trip", "asset_count": 2}]
    monkeypatch.setattr(ImmichSource, "list_albums", lambda self: list(albums))
    resp = _client().get("/immich/albums")
    assert resp.status_code == 200
    assert resp.json() == {"albums": albums}


def test_immich_albums_maps_http_errors_to_502(immich_env, monkeypatch) -> None:
    """An unreachable Immich server surfaces as 502, not a 500 crash."""

    def _boom(self):
        """Simulate a connection failure to Immich."""
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ImmichSource, "list_albums", _boom)
    resp = _client().get("/immich/albums")
    assert resp.status_code == 502
    assert "Immich request failed" in resp.json()["detail"]


def test_immich_albums_maps_garbled_payload_to_502(immich_env, monkeypatch) -> None:
    """A 200 response that isn't the expected JSON (e.g. an HTML login page) is a 502, not a 500."""

    def _bad(self):
        """Simulate resp.json() choking on a non-JSON body."""
        raise json.JSONDecodeError("Expecting value", "<html>", 0)

    monkeypatch.setattr(ImmichSource, "list_albums", _bad)
    resp = _client().get("/immich/albums")
    assert resp.status_code == 502
    assert "Immich request failed" in resp.json()["detail"]


# --- POST /immich/pull ---


def test_immich_pull_downloads_and_skips_existing(immich_env, album_assets, monkeypatch, tmp_path: Path) -> None:
    """Downloads album originals into the dest folder, skipping same-name files."""
    dest = tmp_path / "trip"
    dest.mkdir()
    (dest / "01.jpg").write_bytes(b"already here")  # pre-existing: skipped, not clobbered
    monkeypatch.setattr(ImmichSource, "fetch_original", lambda self, ref: _png_bytes())

    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert [e["done"] for e in progress] == [1, 2]
    assert all(e["total"] == 2 for e in progress)
    assert [e["name"] for e in progress] == ["01.jpg", "02.jpg"]
    assert events[-1] == {"type": "complete", "folder": "trip", "downloaded": 1, "skipped": 1, "failed": 0}
    assert (dest / "01.jpg").read_bytes() == b"already here"
    assert (dest / "02.jpg").read_bytes() == _png_bytes()


def test_immich_pull_filters_to_requested_asset_ids(immich_env, album_assets, monkeypatch, tmp_path: Path) -> None:
    """When asset_ids is given, only those album assets are pulled."""
    monkeypatch.setattr(ImmichSource, "fetch_original", lambda self, ref: _png_bytes())
    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "asset_ids": ["a2"], "dest_folder": "trip"})
    assert resp.status_code == 200
    events = _events(resp)
    assert [e["name"] for e in events if e["type"] == "progress"] == ["02.jpg"]
    assert events[-1]["downloaded"] == 1
    assert not (tmp_path / "trip" / "01.jpg").exists()


def test_immich_pull_reports_per_asset_failures(immich_env, album_assets, monkeypatch, tmp_path: Path) -> None:
    """A failing download is reported on its progress line and counted, without aborting."""

    def _fetch(self, ref):
        """Fail for the first asset only."""
        if ref.id == "a1":
            raise httpx.ConnectError("boom")
        return _png_bytes()

    monkeypatch.setattr(ImmichSource, "fetch_original", _fetch)
    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert "error" in progress[0]
    assert "error" not in progress[1]
    assert events[-1] == {"type": "complete", "folder": "trip", "downloaded": 1, "skipped": 0, "failed": 1}


def test_immich_pull_duplicate_basename_in_request_is_a_failure(immich_env, monkeypatch, tmp_path: Path) -> None:
    """Two assets sharing a basename in one request: the second is a collision error, not a skip."""
    assets = [{"id": "a1", "name": "same.jpg"}, {"id": "a2", "name": "sub/same.jpg"}]
    monkeypatch.setattr(ImmichSource, "list_album_assets", lambda self, album_id: list(assets))
    monkeypatch.setattr(ImmichSource, "fetch_original", lambda self, ref: _png_bytes())

    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert "error" not in progress[0]
    assert "collision" in progress[1]["error"]
    assert events[-1] == {"type": "complete", "folder": "trip", "downloaded": 1, "skipped": 0, "failed": 1}


def test_immich_pull_requires_source_root(immich_env) -> None:
    """Returns 400 when the app has no configured source root."""
    resp = _client().post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    assert resp.status_code == 400
    assert "source root" in resp.json()["detail"]


def test_immich_pull_rejects_path_traversal(immich_env, tmp_path: Path) -> None:
    """A dest_folder escaping the source root is rejected with 400."""
    root = tmp_path / "root"
    root.mkdir()
    resp = _client(root).post("/immich/pull", json={"album_id": "al1", "dest_folder": "../outside"})
    assert resp.status_code == 400
    assert not (tmp_path / "outside").exists()


def test_immich_pull_rejects_file_dest_folder(immich_env, album_assets, tmp_path: Path) -> None:
    """A dest_folder that is an existing regular file is a 400, not a 500."""
    (tmp_path / "cat.jpg").write_bytes(b"file")
    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "cat.jpg"})
    assert resp.status_code == 400
    assert "not a directory" in resp.json()["detail"]


def test_immich_pull_empty_asset_ids_pulls_nothing(immich_env, album_assets, monkeypatch, tmp_path: Path) -> None:
    """asset_ids=[] is an explicit empty selection, not 'the whole album'."""

    def _fail(self, ref):
        """Nothing may be fetched for an empty selection."""
        raise AssertionError("nothing should be fetched for an empty selection")

    monkeypatch.setattr(ImmichSource, "fetch_original", _fail)
    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "asset_ids": [], "dest_folder": "trip"})
    assert resp.status_code == 200
    assert _events(resp) == [{"type": "complete", "folder": "trip", "downloaded": 0, "skipped": 0, "failed": 0}]


def test_immich_pull_unknown_asset_ids_is_404(immich_env, album_assets, tmp_path: Path) -> None:
    """Requesting ids that are not in the album is a 404, not a silent zero-work success."""
    resp = _client(tmp_path).post(
        "/immich/pull", json={"album_id": "al1", "asset_ids": ["a2", "ghost"], "dest_folder": "trip"}
    )
    assert resp.status_code == 404
    assert "ghost" in resp.json()["detail"]


def test_immich_pull_failed_download_leaves_no_file_behind(
    immich_env, album_assets, monkeypatch, tmp_path: Path
) -> None:
    """A failed download leaves neither the target nor a .part temp file, so re-pulling retries it."""

    def _fetch(self, ref):
        """Fail for the first asset only."""
        if ref.id == "a1":
            raise httpx.ConnectError("boom")
        return _png_bytes()

    monkeypatch.setattr(ImmichSource, "fetch_original", _fetch)
    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    assert _events(resp)[-1]["failed"] == 1
    dest = tmp_path / "trip"
    assert not (dest / "01.jpg").exists()
    assert (dest / "02.jpg").exists()
    assert not list(dest.glob("*.part"))


def test_immich_pull_warns_on_uncaptionable_extension(immich_env, monkeypatch, tmp_path: Path) -> None:
    """Pulled originals /caption/folder cannot walk (e.g. HEIC) carry a warning on their progress line."""
    assets = [{"id": "a1", "name": "IMG_0001.HEIC"}, {"id": "a2", "name": "02.jpg"}]
    monkeypatch.setattr(ImmichSource, "list_album_assets", lambda self, album_id: list(assets))
    monkeypatch.setattr(ImmichSource, "fetch_original", lambda self, ref: _png_bytes())

    resp = _client(tmp_path).post("/immich/pull", json={"album_id": "al1", "dest_folder": "trip"})
    progress = [e for e in _events(resp) if e["type"] == "progress"]
    assert "not captionable" in progress[0]["warning"]
    assert ".heic" in progress[0]["warning"]
    assert "warning" not in progress[1]


# --- POST /immich/caption/stream ---


def test_immich_caption_stream_progress_then_complete(immich_env, album_assets, monkeypatch) -> None:
    """Streams a captioned progress line per asset, then a completion summary."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "trigger_word": "sks_person"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert [e["done"] for e in progress] == [1, 2]
    assert all(e["total"] == 2 for e in progress)
    assert [e["asset_id"] for e in progress] == ["a1", "a2"]
    assert [e["name"] for e in progress] == ["01.jpg", "02.jpg"]
    assert all(e["final_caption"] for e in progress)
    assert events[-1] == {"type": "complete", "total": 2, "captioned": 2, "failed": 0}


def test_immich_caption_stream_writes_back_when_asked(immich_env, album_assets, monkeypatch) -> None:
    """write_back=true pushes each final caption to Immich via ImmichSink.write."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))
    written: list[tuple[str, list[str], str]] = []

    def _record(self, ref, *, keywords, description=""):
        """Record write-back calls instead of hitting the network."""
        written.append((ref.id, keywords, description))

    monkeypatch.setattr(ImmichSink, "write", _record)

    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "write_back": True})
    events = _events(resp)
    assert events[-1]["captioned"] == 2
    assert [w[0] for w in written] == ["a1", "a2"]
    # the stub backend emits prose only, so keywords fall back to []
    assert all(w[1] == [] for w in written)
    progress = [e for e in events if e["type"] == "progress"]
    assert [w[2] for w in written] == [e["final_caption"] for e in progress]


def test_immich_caption_stream_no_write_back_by_default(immich_env, album_assets, monkeypatch) -> None:
    """Without write_back, ImmichSink.write is never called."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))

    def _fail(self, ref, *, keywords, description=""):
        """Fail loudly if write-back happens when not requested."""
        raise AssertionError("write_back should not be called")

    monkeypatch.setattr(ImmichSink, "write", _fail)
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1"})
    assert _events(resp)[-1]["captioned"] == 2


def test_immich_caption_stream_reports_per_asset_errors(immich_env, album_assets, monkeypatch) -> None:
    """A failing fetch is an error progress line and a failed count, not an aborted stream."""

    def _fetch(self, ref):
        """Fail for the first asset only."""
        if ref.id == "a1":
            raise httpx.ConnectError("boom")
        return Image.new("RGB", (8, 8), (120, 120, 120))

    monkeypatch.setattr(ImmichSource, "fetch_image", _fetch)
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1"})
    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert "error" in progress[0] and "final_caption" not in progress[0]
    assert progress[1]["final_caption"]
    assert events[-1] == {"type": "complete", "total": 2, "captioned": 1, "failed": 1}


def test_immich_caption_stream_rejects_write_xmp(immich_env, album_assets, monkeypatch) -> None:
    """write_xmp is a 400: Immich assets are in-memory and have no local path for a sidecar."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "write_xmp": True})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "write_xmp" in detail
    assert "/immich/pull" in detail  # points at the supported route to XMP sidecars


def test_immich_caption_stream_filters_to_requested_asset_ids(immich_env, album_assets, monkeypatch) -> None:
    """When asset_ids is given, only those album assets are captioned."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "asset_ids": ["a2"]})
    events = _events(resp)
    assert [e["asset_id"] for e in events if e["type"] == "progress"] == ["a2"]
    assert events[-1]["total"] == 1


def test_immich_caption_stream_empty_asset_ids_captions_nothing(immich_env, album_assets, monkeypatch) -> None:
    """asset_ids=[] captions nothing rather than the whole album (or mass write-back)."""

    def _fail(self, ref):
        """Nothing may be fetched for an empty selection."""
        raise AssertionError("nothing should be fetched for an empty selection")

    monkeypatch.setattr(ImmichSource, "fetch_image", _fail)
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "asset_ids": [], "write_back": True})
    assert _events(resp) == [{"type": "complete", "total": 0, "captioned": 0, "failed": 0}]


def test_immich_caption_stream_unknown_asset_ids_is_404(immich_env, album_assets) -> None:
    """Requesting ids that are not in the album is a 404, not a silent zero-work success."""
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "asset_ids": ["ghost"]})
    assert resp.status_code == 404
    assert "ghost" in resp.json()["detail"]


def test_immich_caption_stream_write_back_failure_keeps_caption(immich_env, album_assets, monkeypatch) -> None:
    """A failed write-back still reports the computed caption on the progress line (and counts as failed)."""
    monkeypatch.setattr(ImmichSource, "fetch_image", lambda self, ref: Image.new("RGB", (8, 8), (120, 120, 120)))

    def _boom(self, ref, *, keywords, description=""):
        """Simulate Immich rejecting the write-back."""
        raise httpx.ConnectError("tags rejected")

    monkeypatch.setattr(ImmichSink, "write", _boom)
    resp = _client().post("/immich/caption/stream", json={"album_id": "al1", "write_back": True})
    events = _events(resp)
    progress = [e for e in events if e["type"] == "progress"]
    assert all("write_back failed" in e["error"] for e in progress)
    assert all(e["final_caption"] for e in progress)
    assert events[-1] == {"type": "complete", "total": 2, "captioned": 0, "failed": 2}
