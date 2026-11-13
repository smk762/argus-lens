"""Tests for /health write capabilities and read-only mode (#78) and opt-in batch buckets (#79)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import create_app  # noqa: E402


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


def _png(path: Path) -> None:
    """Write a small grey PNG at the path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (120, 120, 120)).save(path)


def _manifest(paths: list[Path]) -> bytes:
    """A curator manifest with one row per image path."""
    rows = [{"rel_path": p.name, "abs_path": str(p), "target_profile": {}} for p in paths]
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Start each test with no read-only or Immich configuration in the environment."""
    for var in ("LENS_READ_ONLY", "IMMICH_URL", "IMMICH_API_KEY", "LENS_SOURCE_PATH"):
        monkeypatch.delenv(var, raising=False)


def _app(tmp_path: Path, **kwargs) -> TestClient:
    """A test client over the stub backend, rooted at *tmp_path*."""
    return TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path), **kwargs))


# --- /health capabilities (#78) -------------------------------------------------


def test_health_capabilities_default_writable_without_immich(tmp_path: Path) -> None:
    """A default server is writable, and Immich is off until both env vars are set."""
    body = _app(tmp_path).get("/health").json()
    assert body["read_only"] is False
    assert body["immich"] is False
    assert body["immich_write_back"] is False


def test_health_capabilities_with_immich(tmp_path: Path, monkeypatch) -> None:
    """With Immich configured, write-back is advertised unless the server is read-only."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.local")
    monkeypatch.setenv("IMMICH_API_KEY", "k")
    assert _app(tmp_path).get("/health").json()["immich_write_back"] is True
    body = _app(tmp_path, read_only=True).get("/health").json()
    assert body["immich"] is True
    assert body["read_only"] is True
    assert body["immich_write_back"] is False


def test_immich_needs_both_env_vars(tmp_path: Path, monkeypatch) -> None:
    """A URL without an API key does not count as Immich being configured."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.local")
    assert _app(tmp_path).get("/health").json()["immich"] is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_read_only_from_env(tmp_path: Path, monkeypatch, value: str) -> None:
    """LENS_READ_ONLY accepts the usual on-flag spellings."""
    monkeypatch.setenv("LENS_READ_ONLY", value)
    assert _app(tmp_path).get("/health").json()["read_only"] is True


def test_explicit_read_only_false_overrides_env(tmp_path: Path, monkeypatch) -> None:
    """An explicit read_only=False wins over the environment."""
    monkeypatch.setenv("LENS_READ_ONLY", "1")
    assert _app(tmp_path, read_only=False).get("/health").json()["read_only"] is False


# --- read-only refusals (#78) ---------------------------------------------------


def test_read_only_refuses_folder_sidecar_writes(tmp_path: Path) -> None:
    """A read-only server refuses a folder caption that would write sidecars, and writes nothing."""
    _png(tmp_path / "a.png")
    client = _app(tmp_path, read_only=True)
    resp = client.post("/caption/folder", json={"folder": "."})
    assert resp.status_code == 403
    assert not (tmp_path / "a.txt").exists()
    resp = client.post("/caption/folder", json={"folder": ".", "write_sidecar": False, "write_xmp": True})
    assert resp.status_code == 403
    assert not (tmp_path / "a.xmp").exists()


def test_read_only_allows_folder_caption_without_writes(tmp_path: Path) -> None:
    """With writes off, a read-only server still returns captions."""
    _png(tmp_path / "a.png")
    resp = _app(tmp_path, read_only=True).post("/caption/folder", json={"folder": ".", "write_sidecar": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["captioned"] == 1
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize("route", ["/caption/manifest", "/caption/manifest/stream"])
def test_read_only_refuses_manifest_sidecar_writes(tmp_path: Path, route: str) -> None:
    """Both manifest endpoints refuse sidecar writes on a read-only server."""
    img = tmp_path / "a.png"
    _png(img)
    client = _app(tmp_path, read_only=True)
    files = {"manifest": ("m.jsonl", _manifest([img]), "application/x-ndjson")}
    assert client.post(route, files=files).status_code == 403
    assert not (tmp_path / "a.txt").exists()
    files = {"manifest": ("m.jsonl", _manifest([img]), "application/x-ndjson")}
    assert client.post(route, files=files, data={"write_sidecar": "false"}).status_code == 200


def test_read_only_refuses_immich_pull_and_write_back(tmp_path: Path, monkeypatch) -> None:
    """Immich pulls (disk writes) and write-back are refused before any Immich call."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.invalid")
    monkeypatch.setenv("IMMICH_API_KEY", "k")
    client = _app(tmp_path, read_only=True)
    assert client.post("/immich/pull", json={"album_id": "x", "dest_folder": "d"}).status_code == 403
    resp = client.post("/immich/caption/stream", json={"album_id": "x", "write_back": True})
    assert resp.status_code == 403


# --- include_buckets on batch rows (#79) ---------------------------------------


def test_folder_rows_omit_buckets_by_default(tmp_path: Path) -> None:
    """Batch rows stay small unless buckets are asked for."""
    _png(tmp_path / "a.png")
    row = _app(tmp_path).post("/caption/folder", json={"folder": ".", "write_sidecar": False}).json()["results"][0]
    assert "metadata" not in row


def test_folder_rows_include_buckets_when_asked(tmp_path: Path) -> None:
    """include_buckets adds metadata.buckets split by tag/prose source."""
    _png(tmp_path / "a.png")
    body = (
        _app(tmp_path)
        .post("/caption/folder", json={"folder": ".", "write_sidecar": False, "include_buckets": True})
        .json()
    )
    buckets = body["results"][0]["metadata"]["buckets"]
    assert isinstance(buckets, dict)
    assert buckets  # the stub's comma-separated caption lands in at least one source
    for by_cat in buckets.values():
        assert all(isinstance(frags, list) for frags in by_cat.values())


@pytest.mark.parametrize("route", ["/caption/manifest", "/caption/manifest/stream"])
def test_manifest_rows_include_buckets_when_asked(tmp_path: Path, route: str) -> None:
    """Both manifest endpoints carry metadata.buckets per row when include_buckets is set."""
    img = tmp_path / "a.png"
    _png(img)
    files = {"manifest": ("m.jsonl", _manifest([img]), "application/x-ndjson")}
    resp = _app(tmp_path).post(route, files=files, data={"write_sidecar": "false", "include_buckets": "true"})
    assert resp.status_code == 200, resp.text
    row = json.loads(resp.text.splitlines()[0]) if route.endswith("/stream") else resp.json()["results"][0]
    assert "buckets" in row["metadata"]


def test_read_only_unrecognised_value_stays_off(tmp_path: Path, monkeypatch) -> None:
    """A typo in LENS_READ_ONLY leaves the server writable (env_flag warns in the log)."""
    monkeypatch.setenv("LENS_READ_ONLY", "enabled")
    assert _app(tmp_path).get("/health").json()["read_only"] is False
