"""Tests for POST /caption/folder and GET /folders (local-folder captioning)."""

from __future__ import annotations

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
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        """No-op."""
        pass


def _png(path: Path, size: int = 64) -> None:
    """Write a small grey PNG at the path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (120, 120, 120)).save(path)


def test_caption_folder_non_recursive_writes_sidecars(tmp_path: Path) -> None:
    """Non-recursive captioning writes .txt sidecars for top-level images and skips subfolders."""
    _png(tmp_path / "01.jpg")
    _png(tmp_path / "02.png")
    _png(tmp_path / "sub" / "03.jpg")  # ignored when non-recursive

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert body["captioned"] == 2
    assert (tmp_path / "01.txt").read_text().strip()
    assert not (tmp_path / "sub" / "03.txt").exists()


def test_caption_folder_recursive(tmp_path: Path) -> None:
    """Recursive captioning includes subfolder images; write_sidecar=False skips sidecar files."""
    _png(tmp_path / "01.jpg")
    _png(tmp_path / "sub" / "03.jpg")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post(
        "/caption/folder",
        json={"folder": str(tmp_path), "recursive": True, "write_sidecar": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    rels = {r["rel_path"] for r in body["results"]}
    assert rels == {"01.jpg", str(Path("sub") / "03.jpg")}
    assert not (tmp_path / "01.txt").exists()  # sidecars disabled


def test_caption_folder_accepts_relative_path(tmp_path: Path) -> None:
    """A folder relative to the source root resolves inside it."""
    _png(tmp_path / "sub" / "01.jpg")
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": "sub"})
    assert resp.status_code == 200
    assert resp.json()["captioned"] == 1


def test_caption_folder_rejects_missing_dir(tmp_path: Path) -> None:
    """POST /caption/folder returns 400 for a nonexistent folder under the root."""
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path / "nope")})
    assert resp.status_code == 400
    assert "not a directory" in resp.json()["detail"]


def test_caption_folder_requires_source_root(tmp_path: Path) -> None:
    """POST /caption/folder returns 400 when no source root is configured."""
    _png(tmp_path / "01.jpg")
    client = TestClient(create_app(default_backend=_StubBackend()))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path)})
    assert resp.status_code == 400
    assert "source root" in resp.json()["detail"]


def test_caption_folder_rejects_path_outside_root(tmp_path: Path) -> None:
    """Absolute or traversal paths outside the source root are rejected with 400."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    _png(root / "01.jpg")
    _png(outside / "02.jpg")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(root)))
    assert client.post("/caption/folder", json={"folder": str(outside)}).status_code == 400
    assert client.post("/caption/folder", json={"folder": "../outside"}).status_code == 400
    assert not (outside / "02.txt").exists()


def test_caption_folder_reports_sidecar_stem_collision(tmp_path: Path) -> None:
    """Same-stem images (cat.jpg + cat.png) don't silently overwrite one sidecar."""
    _png(tmp_path / "cat.jpg")
    _png(tmp_path / "cat.png")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 1
    assert body["captioned"] + body["failed"] == body["total"] == 2
    assert "collision" in body["errors"][0]["error"]
    assert (tmp_path / "cat.txt").read_text().strip()


def test_caption_folder_write_xmp_writes_xmp_sidecars(tmp_path: Path) -> None:
    """write_xmp=true writes an <image>.xmp sidecar per image, independent of write_sidecar."""
    _png(tmp_path / "01.jpg")
    _png(tmp_path / "02.png")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post(
        "/caption/folder",
        json={"folder": str(tmp_path), "write_sidecar": False, "write_xmp": True},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["captioned"] == 2
    assert body["xmp_written"] == 2
    assert {r["xmp_path"] for r in body["results"]} == {
        str(tmp_path / "01.jpg.xmp"),
        str(tmp_path / "02.png.xmp"),
    }
    doc = (tmp_path / "01.jpg.xmp").read_text(encoding="utf-8")
    assert "dc:description" in doc
    assert body["results"][0]["final_caption"] in doc
    assert not (tmp_path / "01.txt").exists()  # write_sidecar was off


def test_caption_folder_write_xmp_off_by_default(tmp_path: Path) -> None:
    """Without write_xmp, only the .txt sidecar is written and xmp_written is 0."""
    _png(tmp_path / "01.jpg")
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["xmp_written"] == 0
    assert "xmp_path" not in body["results"][0]
    assert not (tmp_path / "01.jpg.xmp").exists()
    assert (tmp_path / "01.txt").exists()


def test_caption_folder_write_xmp_overwrites_existing_sidecar(tmp_path: Path) -> None:
    """A pre-existing .xmp is replaced, matching the .txt sidecar overwrite semantics."""
    _png(tmp_path / "01.jpg")
    stale_xmp = tmp_path / "01.jpg.xmp"
    stale_xmp.write_text("<stale/>", encoding="utf-8")
    stale_txt = tmp_path / "01.txt"
    stale_txt.write_text("stale caption", encoding="utf-8")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    resp = client.post("/caption/folder", json={"folder": str(tmp_path), "write_xmp": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 0
    assert body["xmp_written"] == 1
    # both sidecars were replaced with the fresh caption
    assert "stale" not in stale_xmp.read_text(encoding="utf-8")
    assert body["results"][0]["final_caption"] in stale_xmp.read_text(encoding="utf-8")
    assert stale_txt.read_text(encoding="utf-8") == body["results"][0]["final_caption"]


def test_folders_browse(tmp_path: Path) -> None:
    """GET /folders lists subfolders with image counts and rejects path traversal with 400."""
    _png(tmp_path / "personA" / "01.jpg")
    _png(tmp_path / "personA" / "02.jpg")
    _png(tmp_path / "personB" / "01.jpg")

    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    root = client.get("/folders").json()
    assert root["parent"] is None
    assert {f["name"] for f in root["folders"]} == {"personA", "personB"}
    person_a = next(f for f in root["folders"] if f["name"] == "personA")
    assert person_a["image_count"] == 2

    sub = client.get("/folders", params={"path": "personA"}).json()
    assert sub["parent"] == ""
    assert sub["direct_image_count"] == 2
    assert client.get("/folders", params={"path": "../../etc"}).status_code == 400


def test_folders_requires_source_root() -> None:
    """GET /folders returns 400 when the app was created without a source root."""
    client = TestClient(create_app(default_backend=_StubBackend()))
    assert client.get("/folders").status_code == 400
