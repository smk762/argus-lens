"""Recording live captions into the cortex lineage store (argus_lens.lineage)."""

from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("argus_cortex.store")

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.backends.replay import ReplayBackend  # noqa: E402
from argus_lens.engine import ArgusLens  # noqa: E402
from argus_lens.lineage import LineageRecorder, recorder_from_env  # noqa: E402


class _StubBackend(CaptionBackend):
    """Prose backend stub: a fixed caption, no model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op load."""

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption for any image."""
        return "a person standing in a garden"

    def unload(self) -> None:
        """No-op unload."""


class _FakeStore:
    """In-memory LineageStore: records calls and how many transactions they ran in."""

    def __init__(self, fail: bool = False) -> None:
        self.assets: list = []
        self.captions: list = []
        self.transactions = 0
        self.schema = 0
        self.closed = False
        self.fail = fail

    def ensure_schema(self) -> None:
        self.schema += 1

    @contextlib.contextmanager
    def transaction(self):
        self.transactions += 1
        yield

    def record_asset(self, asset) -> str:
        if self.fail:
            raise RuntimeError("db down")
        self.assets.append(asset)
        return f"asset-{len(self.assets)}"

    def record_caption(self, asset_id: str, caption) -> str:
        self.captions.append((asset_id, caption))
        return f"caption-{len(self.captions)}"

    def close(self) -> None:
        self.closed = True


class _FakeBlobs:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.puts: list[bytes] = []

    @property
    def is_enabled(self) -> bool:  # a property on cortex's BlobStore, not a method
        return self.enabled

    def put_content_addressed(self, data: bytes, *, content_type: str | None = None) -> str:
        self.puts.append(data)
        return hashlib.sha256(data).hexdigest()

    def close(self) -> None:
        pass


def _png(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def _engine(store: _FakeStore, blobs: _FakeBlobs | None = None) -> ArgusLens:
    return ArgusLens(backend=_StubBackend(), recorder=LineageRecorder(store, blobs))


def test_caption_of_a_file_records_asset_and_caption(tmp_path: Path) -> None:
    """A file caption becomes one source_asset (path + original-bytes sha256) and one caption row."""
    data = _png()
    path = tmp_path / "cat.png"
    path.write_bytes(data)
    store = _FakeStore()

    result = _engine(store).caption(path, trigger_word="sks", target_style="anime", target_backend="flux")

    assert store.schema == 1 and store.transactions == 1
    (asset,) = store.assets
    assert asset.uri == str(path)
    assert asset.sha256 == hashlib.sha256(data).hexdigest()  # the key replay looks up
    ((asset_id, caption),) = store.captions
    assert asset_id == "asset-1"
    assert caption.final_caption == result.final_caption
    assert caption.variants == result.caption_variants
    assert caption.backend == "stub"
    assert caption.params == {"trigger_word": "sks"}
    # argus-lightbox rebuilds a CaptionTargetProfile from these keys.
    assert caption.profile["target_style"] == "anime"
    assert caption.profile["target_backend"] == "flux"
    assert caption.profile["target_category"] == "identity"


def test_batch_and_stream_record_each_upload_under_its_name() -> None:
    """Uploads have no path, so the row's uri names the upload; every live path records."""
    store = _FakeStore()
    engine = _engine(store)
    engine.caption_batch([_png("red"), _png("blue")], names=["a.png", "b.png"])
    list(engine.caption_stream([_png("green")], names=["c.png"]))
    assert [a.uri for a in store.assets] == ["upload:a.png", "upload:b.png", "upload:c.png"]
    assert len(store.captions) == 3


def test_original_bytes_go_to_the_blob_store_when_enabled() -> None:
    """With a blob store configured the original bytes are stored under their sha256."""
    store, blobs = _FakeStore(), _FakeBlobs()
    data = _png()
    _engine(store, blobs).caption(data)
    assert blobs.puts == [data]


def test_disabled_blob_store_is_skipped() -> None:
    """A NullBlobStore (CORTEX_S3_* unset) is not written to."""
    blobs = _FakeBlobs(enabled=False)
    _engine(_FakeStore(), blobs).caption(_png())
    assert blobs.puts == []


def test_pil_input_is_not_recorded() -> None:
    """A pre-decoded PIL image has no original bytes, so no sha256 replay could find."""
    store = _FakeStore()
    _engine(store).caption(Image.new("RGB", (4, 4)))
    assert store.assets == [] and store.captions == []


def test_a_failed_write_never_fails_the_caption() -> None:
    """A database error is logged and the caption is still returned."""
    store = _FakeStore(fail=True)
    result = _engine(store).caption(_png())
    assert result.final_caption
    assert store.captions == []


def test_replay_never_records(monkeypatch) -> None:
    """Replayed captions are served from the tape and must not be written back into it."""
    data = _png()
    sha = hashlib.sha256(data).hexdigest()
    recorded = ArgusLens(backend=_StubBackend()).caption(data)
    backend = ReplayBackend(connection=object())
    monkeypatch.setattr(backend, "lookup", lambda sha256=None, uri=None: recorded if sha256 == sha else None)
    store = _FakeStore()
    ArgusLens(backend=backend, recorder=LineageRecorder(store)).caption(data)
    assert store.assets == [] and store.captions == []


def test_close_releases_the_store() -> None:
    """Engine.close() closes the recorder's pool."""
    store = _FakeStore()
    _engine(store).close()
    assert store.closed


def test_recorder_from_env_is_opt_in(monkeypatch) -> None:
    """Off unless LENS_RECORD_LINEAGE is on; on without CORTEX_PG_URL (or a junk value) is an error."""
    monkeypatch.delenv("LENS_RECORD_LINEAGE", raising=False)
    monkeypatch.setenv("CORTEX_PG_URL", "postgresql://x/y")  # replay's URL alone never turns recording on
    assert recorder_from_env() is None
    monkeypatch.setenv("LENS_RECORD_LINEAGE", "off")
    assert recorder_from_env() is None

    monkeypatch.setenv("LENS_RECORD_LINEAGE", "enabled")
    with pytest.raises(ValueError, match="neither true nor false"):
        recorder_from_env()
    monkeypatch.setenv("LENS_RECORD_LINEAGE", "1")
    monkeypatch.delenv("CORTEX_PG_URL")
    with pytest.raises(ValueError, match="CORTEX_PG_URL is unset"):
        recorder_from_env()


def test_server_records_captions_but_not_the_v1_router(monkeypatch) -> None:
    """create_app hands the recorder to the captioning engine only, never to the Frigate /v1 engines."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import argus_lens.openai_compat as compat
    from argus_lens.server import create_app

    seen: list[dict] = []
    real_router = compat.create_openai_router

    def _spy(**kwargs):
        seen.append(kwargs)
        return real_router(**kwargs)

    monkeypatch.setattr("argus_lens.server.create_openai_router", _spy)
    store = _FakeStore()
    client = TestClient(create_app(default_backend=_StubBackend(), recorder=LineageRecorder(store)))
    assert "recorder" not in seen[0]

    resp = client.post("/caption", files={"file": ("cat.png", _png(), "image/png")})
    assert resp.status_code == 200
    assert len(store.captions) == 1
