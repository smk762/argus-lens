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

    assert events[-1] == {"type": "complete", "total": 2, "captioned": 1, "failed": 1}
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


def test_manifest_stream_rejects_bad_json(client: TestClient) -> None:
    """The streaming endpoint also returns 400 for invalid manifest JSON."""
    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"{not json}\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400
