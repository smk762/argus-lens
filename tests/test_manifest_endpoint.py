"""Tests for POST /caption/manifest (argus-curator handoff ingest)."""

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
from argus_lens.server import create_app  # noqa: E402


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


def _png(path: Path, size: int = 64) -> None:
    """Write a solid-gray square PNG to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (120, 120, 120)).save(path)


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client wired to the stub caption backend."""
    return TestClient(create_app(default_backend=_StubBackend()))


def _manifest_bytes(rows: list[dict]) -> bytes:
    """Encode manifest rows as newline-delimited JSON bytes."""
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def test_manifest_captions_and_writes_sidecars(client: TestClient, tmp_path: Path) -> None:
    """Captions every manifest row and writes a .txt sidecar next to each image."""
    img_a = tmp_path / "personA" / "01.jpg"
    img_b = tmp_path / "personB" / "02.jpg"
    _png(img_a)
    _png(img_b)

    rows = [
        {
            "rel_path": "personA/01.jpg",
            "abs_path": str(img_a),
            "target_profile": {
                "target_style": "photo",
                "target_backend": "sdxl",
                "checkpoint": None,
                "target_category": "identity",
            },
            "primary_face_cluster": "face_1",
            "score": 0.9,
            "similar_group": 1,
        },
        {
            "rel_path": "personB/02.jpg",
            "abs_path": str(img_b),
            "target_profile": {
                "target_style": "anime",
                "target_backend": "flux-dev-1",
                "checkpoint": None,
                "target_category": "wardrobe",
            },
            "primary_face_cluster": "face_2",
            "score": 0.8,
            "similar_group": 2,
        },
    ]

    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"trigger_word": "sks_person", "write_sidecar": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert body["captioned"] == 2
    assert body["failed"] == 0
    assert img_a.with_suffix(".txt").exists()
    assert img_b.with_suffix(".txt").read_text().strip()


def test_manifest_reports_missing_image(client: TestClient, tmp_path: Path) -> None:
    """Counts a nonexistent image as failed and reports it in the errors list."""
    rows = [{"rel_path": "gone.jpg", "abs_path": str(tmp_path / "gone.jpg"), "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 0
    assert body["failed"] == 1
    assert body["errors"][0]["rel_path"] == "gone.jpg"


def test_manifest_rejects_bad_json(client: TestClient) -> None:
    """Returns 400 when a manifest line is not valid JSON."""
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"{not json}\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400


def test_manifest_rejects_non_object_rows(client: TestClient) -> None:
    """Valid JSON that is not an object (null, scalar, array) is a 400, not a crash."""
    for payload in (b"null\n", b"42\n", b"[1, 2]\n"):
        resp = client.post(
            "/caption/manifest",
            files={"manifest": ("manifest.jsonl", io.BytesIO(payload), "application/x-ndjson")},
        )
        assert resp.status_code == 400, payload
        assert "not a JSON object" in resp.json()["detail"]
    # The streaming endpoint shares the parser.
    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"null\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400


def test_manifest_sidecar_failure_counts_once(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    """A caption that succeeds but whose sidecar write fails is failed-only, not double-counted."""
    img = tmp_path / "01.jpg"
    _png(img)
    real_write_text = Path.write_text

    def _failing_write_text(self: Path, *args, **kwargs):
        """Simulate a read-only filesystem for .txt sidecars only."""
        if self.suffix == ".txt":
            raise OSError("read-only file system")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _failing_write_text)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["captioned"] == 0
    assert body["failed"] == 1
    assert "sidecar write failed" in body["errors"][0]["error"]


def test_manifest_reports_sidecar_stem_collision(client: TestClient, tmp_path: Path) -> None:
    """Two same-stem images map to one .txt sidecar; the second is an error, not an overwrite."""
    img_a = tmp_path / "cat.jpg"
    img_b = tmp_path / "cat.png"
    _png(img_a)
    _png(img_b)
    rows = [
        {"rel_path": "cat.jpg", "abs_path": str(img_a), "target_profile": {}},
        {"rel_path": "cat.png", "abs_path": str(img_b), "target_profile": {}},
    ]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 1
    assert "collision" in body["errors"][0]["error"]


def test_manifest_stream_yields_progress_then_complete(client: TestClient, tmp_path: Path) -> None:
    """Streams one NDJSON progress event per row (including failures) then a complete event."""
    img = tmp_path / "personA" / "01.jpg"
    _png(img)
    rows = [
        {"rel_path": "personA/01.jpg", "abs_path": str(img), "target_profile": {}},
        {"rel_path": "gone.jpg", "abs_path": str(tmp_path / "gone.jpg"), "target_profile": {}},
    ]

    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"trigger_word": "sks_person", "write_sidecar": "true"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    progress = [e for e in events if e["type"] == "progress"]
    assert [e["done"] for e in progress] == [1, 2]
    assert all(e["total"] == 2 for e in progress)
    assert progress[0]["rel_path"] == "personA/01.jpg"
    assert progress[0]["final_caption"]
    assert progress[1]["rel_path"] == "gone.jpg"
    assert "error" in progress[1]

    assert events[-1] == {"type": "complete", "total": 2, "captioned": 1, "failed": 1, "xmp_written": 0}
    assert img.with_suffix(".txt").read_text().strip()


def test_manifest_stream_skips_sidecar_when_disabled(client: TestClient, tmp_path: Path) -> None:
    """Does not write a .txt sidecar when write_sidecar is false."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]

    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events[-1]["captioned"] == 1
    assert not img.with_suffix(".txt").exists()


def test_manifest_write_xmp_writes_xmp_sidecars(client: TestClient, tmp_path: Path) -> None:
    """write_xmp=true writes an <image>.xmp sidecar carrying the caption, independent of write_sidecar."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]

    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "write_xmp": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["captioned"] == 1
    assert body["xmp_written"] == 1
    xmp = tmp_path / "01.jpg.xmp"
    assert body["results"][0]["xmp_path"] == str(xmp)
    doc = xmp.read_text(encoding="utf-8")
    assert "dc:description" in doc
    assert body["results"][0]["final_caption"] in doc
    assert not img.with_suffix(".txt").exists()  # write_sidecar was off


def test_manifest_write_xmp_off_by_default(client: TestClient, tmp_path: Path) -> None:
    """Without write_xmp, no .xmp sidecar is written and xmp_written reports 0."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["xmp_written"] == 0
    assert "xmp_path" not in body["results"][0]
    assert not (tmp_path / "01.jpg.xmp").exists()


def test_manifest_write_xmp_reports_duplicate_row_collision(client: TestClient, tmp_path: Path) -> None:
    """The same abs_path listed twice is an xmp collision error, not a silent overwrite."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [
        {"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}},
        {"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}},
    ]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "write_xmp": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 1
    assert body["xmp_written"] == 1
    assert "collision" in body["errors"][0]["error"]


def test_manifest_write_xmp_collision_detected_across_path_spellings(client: TestClient, tmp_path: Path) -> None:
    """The same image spelled two ways ('sub/../') is one collision error, not a silent double write."""
    img = tmp_path / "01.jpg"
    _png(img)
    (tmp_path / "sub").mkdir()
    alias = str(tmp_path / "sub" / ".." / "01.jpg")
    rows = [
        {"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}},
        {"rel_path": "01.jpg-alias", "abs_path": alias, "target_profile": {}},
    ]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "write_xmp": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 1
    assert body["xmp_written"] == 1
    assert "collision" in body["errors"][0]["error"]


def test_manifest_write_xmp_overwrite_false_protects_existing_sidecar(client: TestClient, tmp_path: Path) -> None:
    """xmp_overwrite=false turns a pre-existing .xmp into a per-image error instead of replacing it."""
    img = tmp_path / "01.jpg"
    _png(img)
    existing = tmp_path / "01.jpg.xmp"
    existing.write_text("<precious/>", encoding="utf-8")
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "write_xmp": "true", "xmp_overwrite": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["failed"] == 1
    assert "already exists" in body["errors"][0]["error"]
    assert existing.read_text(encoding="utf-8") == "<precious/>"


def test_manifest_stream_write_xmp(client: TestClient, tmp_path: Path) -> None:
    """The stream reports xmp_path per progress line and an xmp_written total on complete."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]

    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "write_xmp": "true"},
    )
    assert resp.status_code == 200, resp.text
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    progress = [e for e in events if e["type"] == "progress"]
    assert progress[0]["xmp_path"] == str(tmp_path / "01.jpg.xmp")
    assert events[-1] == {"type": "complete", "total": 1, "captioned": 1, "failed": 0, "xmp_written": 1}
    assert (tmp_path / "01.jpg.xmp").read_text(encoding="utf-8").startswith("<?xpacket")


def test_manifest_stream_rejects_bad_json(client: TestClient) -> None:
    """The streaming endpoint also returns 400 for invalid manifest JSON."""
    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"{not json}\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400


def _v2_row(exported_path: str, abs_path: str, **extra: object) -> dict:
    """Build a manifest 2.0 row (exported_path + manifest_version)."""
    return {
        "manifest_version": "2.0",
        "rel_path": exported_path,
        "abs_path": abs_path,
        "exported_path": exported_path,
        "target_profile": {},
        "score": 0.9,
        "similar_group": 1,
        **extra,
    }


def test_manifest_v2_prefers_exported_path_over_stale_abs_path(client: TestClient, tmp_path: Path) -> None:
    """With export_root, a 2.0 row is read from export_root/exported_path even when abs_path is gone (mode=move)."""
    export_root = tmp_path / "export"
    img = export_root / "personA" / "01.jpg"
    _png(img)
    moved_away = tmp_path / "source" / "personA" / "01.jpg"  # never created: the move-mode source

    rows = [_v2_row("personA/01.jpg", str(moved_away))]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"export_root": str(export_root)},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["captioned"] == 1
    assert body["failed"] == 0
    assert img.with_suffix(".txt").read_text().strip()  # sidecar lands next to the exported image
    assert not moved_away.with_suffix(".txt").exists()


def test_manifest_v2_without_export_root_falls_back_to_abs_path(client: TestClient, tmp_path: Path) -> None:
    """A 2.0 row still captions via abs_path when no export_root is supplied (copy/symlink on a shared volume)."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [_v2_row("de-collided-01.jpg", str(img))]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["captioned"] == 1


def test_manifest_v1_rows_ignore_export_root(client: TestClient, tmp_path: Path) -> None:
    """Pre-2.0 rows (no exported_path) keep using abs_path even when export_root is supplied."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [{"rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "export_root": str(tmp_path / "elsewhere")},
    )
    # export_root must exist even if no row ends up using it
    assert resp.status_code == 400
    (tmp_path / "elsewhere").mkdir()
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false", "export_root": str(tmp_path / "elsewhere")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["captioned"] == 1


def test_manifest_rejects_unsupported_major_version(client: TestClient, tmp_path: Path) -> None:
    """A manifest_version outside the 1.x/2.x majors is a 400 naming the line, not a misread."""
    img = tmp_path / "01.jpg"
    _png(img)
    rows = [_v2_row("01.jpg", str(img), manifest_version="3.0")]
    for endpoint in ("/caption/manifest", "/caption/manifest/stream"):
        resp = client.post(
            endpoint,
            files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        )
        assert resp.status_code == 400, endpoint
        assert "unsupported manifest_version" in resp.json()["detail"]
        assert "line 1" in resp.json()["detail"]


def test_manifest_accepts_1x_and_2x_versions(client: TestClient, tmp_path: Path) -> None:
    """Explicit 1.x and 2.x manifest_version values are both accepted."""
    img = tmp_path / "01.jpg"
    _png(img)
    for version in ("1.0", "2.0", "2.1"):
        rows = [{"manifest_version": version, "rel_path": "01.jpg", "abs_path": str(img), "target_profile": {}}]
        resp = client.post(
            "/caption/manifest",
            files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
            data={"write_sidecar": "false"},
        )
        assert resp.status_code == 200, version
        assert resp.json()["captioned"] == 1


def test_manifest_rejects_missing_export_root_dir(client: TestClient, tmp_path: Path) -> None:
    """An export_root that is not an existing directory is one clear 400, not N per-row read errors."""
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes([])), "application/x-ndjson")},
        data={"export_root": str(tmp_path / "nope")},
    )
    assert resp.status_code == 400
    assert "export_root is not a directory" in resp.json()["detail"]


def test_manifest_row_with_only_exported_path_errors_without_root(client: TestClient) -> None:
    """A row carrying only exported_path (no abs_path) is a per-row error pointing at export_root."""
    rows = [{"rel_path": "01.jpg", "exported_path": "01.jpg", "target_profile": {}}]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"write_sidecar": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["failed"] == 1
    assert "export_root" in body["errors"][0]["error"]


def test_manifest_stream_uses_export_root(client: TestClient, tmp_path: Path) -> None:
    """The streaming endpoint resolves 2.0 rows against export_root too."""
    export_root = tmp_path / "export"
    img = export_root / "01.jpg"
    _png(img)
    rows = [_v2_row("01.jpg", str(tmp_path / "moved-away" / "01.jpg"))]

    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"export_root": str(export_root)},
    )
    assert resp.status_code == 200, resp.text
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events[-1]["captioned"] == 1
    assert img.with_suffix(".txt").read_text().strip()


def test_manifest_v2_exported_path_escaping_export_root_is_per_row_error(client: TestClient, tmp_path: Path) -> None:
    """An absolute or ``..`` exported_path is confined under export_root: it is a per-row error, never an escape."""
    export_root = tmp_path / "export"
    export_root.mkdir()
    outside = tmp_path / "outside.jpg"  # a real image living OUTSIDE the export root
    _png(outside)

    # (1) absolute exported_path would discard export_root; (2) ``..`` would escape it.
    for bad_exported in (str(outside), "../outside.jpg"):
        rows = [_v2_row(bad_exported, str(tmp_path / "moved" / "01.jpg"))]
        resp = client.post(
            "/caption/manifest",
            files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
            data={"export_root": str(export_root)},
        )
        assert resp.status_code == 200, (bad_exported, resp.text)
        body = resp.json()
        assert body["failed"] == 1, bad_exported
        assert body["captioned"] == 0, bad_exported
        assert "export root" in body["errors"][0]["error"], bad_exported
        # the confined join never reads or writes next to the outside target
        assert not outside.with_suffix(".txt").exists(), bad_exported


def test_manifest_v2_nonstring_exported_path_falls_back_to_abs_path(client: TestClient, tmp_path: Path) -> None:
    """A non-string exported_path degrades to abs_path instead of failing an otherwise-valid row."""
    export_root = tmp_path / "export"
    export_root.mkdir()
    img = tmp_path / "real.jpg"
    _png(img)
    rows = [
        {
            "manifest_version": "2.0",
            "rel_path": "real.jpg",
            "abs_path": str(img),
            "exported_path": 123,  # non-string: not a usable locator -> fall back to abs_path
            "target_profile": {},
        }
    ]
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(_manifest_bytes(rows)), "application/x-ndjson")},
        data={"export_root": str(export_root), "write_sidecar": "false"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["captioned"] == 1
