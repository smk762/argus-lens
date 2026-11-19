"""Uploads are capped per file by ``LENS_MAX_UPLOAD_MB`` (argus_cortex.limits)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import DEFAULT_MAX_UPLOAD_MB, _max_upload_bytes, create_app  # noqa: E402

# 64 KB: a small PNG fits, a padded one doesn't.
_CAP_MB = "0.0625"
_CAP_BYTES = 64 * 1024


class _StubBackend(CaptionBackend):
    """CPU-only stub backend that returns a fixed caption without loading a model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        """No-op."""


def _png(pad: int = 0) -> bytes:
    """A tiny PNG, optionally followed by *pad* trailing bytes (PIL ignores them)."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue() + b"\0" * pad


@pytest.fixture
def client(monkeypatch, tmp_path: Path) -> TestClient:
    """A TestClient with a 64 KB upload cap and a source root (for the manifest routes)."""
    monkeypatch.setenv("LENS_MAX_UPLOAD_MB", _CAP_MB)
    monkeypatch.delenv("LENS_READ_ONLY", raising=False)
    return TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))


def test_cap_defaults_to_50_mb(monkeypatch) -> None:
    """Unset, the cap is DEFAULT_MAX_UPLOAD_MB (50 MB); set, it is read as MB."""
    monkeypatch.delenv("LENS_MAX_UPLOAD_MB", raising=False)
    assert DEFAULT_MAX_UPLOAD_MB == 50
    assert _max_upload_bytes() == 50 * 1024 * 1024
    monkeypatch.setenv("LENS_MAX_UPLOAD_MB", _CAP_MB)
    assert _max_upload_bytes() == _CAP_BYTES


@pytest.mark.parametrize("value", ["0", "-1", "lots", "nan", "inf"])
def test_bad_cap_fails_at_startup(monkeypatch, value) -> None:
    """A cap that isn't a positive number refuses to start rather than running uncapped."""
    monkeypatch.setenv("LENS_MAX_UPLOAD_MB", value)
    with pytest.raises(ValueError):
        create_app(default_backend=_StubBackend())


def test_caption_under_cap_works(client) -> None:
    """An upload under the cap captions as before."""
    resp = client.post("/caption", files={"file": ("a.png", _png(), "image/png")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["final_caption"]


def test_caption_over_cap_is_413(client) -> None:
    """An upload over the cap is a 413, not a caption."""
    resp = client.post("/caption", files={"file": ("big.png", _png(pad=_CAP_BYTES), "image/png")})
    assert resp.status_code == 413
    assert "larger than" in resp.json()["detail"]


@pytest.mark.parametrize("route", ["/caption/batch", "/caption/stream"])
def test_multi_file_routes_cap_each_file(client, route) -> None:
    """The multi-file routes accept files under the cap and refuse the request when one is over."""
    ok = client.post(
        route, files=[("files", ("a.png", _png(), "image/png")), ("files", ("b.png", _png(), "image/png"))]
    )
    assert ok.status_code == 200, ok.text

    over = client.post(
        route,
        files=[("files", ("a.png", _png(), "image/png")), ("files", ("big.png", _png(pad=_CAP_BYTES), "image/png"))],
    )
    assert over.status_code == 413


@pytest.mark.parametrize("route", ["/caption/manifest", "/caption/manifest/stream"])
def test_manifest_over_cap_is_413(client, route) -> None:
    """An oversized manifest is refused before it is parsed."""
    row = json.dumps({"abs_path": "missing.png", "pad": "x" * 1000}) + "\n"
    manifest = (row * (_CAP_BYTES // len(row) + 2)).encode()
    resp = client.post(
        route,
        files={"manifest": ("manifest.jsonl", manifest, "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 413


def test_manifest_under_cap_is_read(client) -> None:
    """A manifest under the cap is parsed and captioned row by row as before."""
    manifest = (json.dumps({"abs_path": "missing.png"}) + "\n").encode()
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", manifest, "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["total"] == 1
