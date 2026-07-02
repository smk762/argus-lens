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
    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        pass


def _png(path: Path, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (120, 120, 120)).save(path)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(default_backend=_StubBackend()))


def _manifest_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def test_manifest_captions_and_writes_sidecars(client: TestClient, tmp_path: Path) -> None:
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
    resp = client.post(
        "/caption/manifest",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"{not json}\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400


def test_manifest_stream_yields_progress_then_complete(client: TestClient, tmp_path: Path) -> None:
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
    resp = client.post(
        "/caption/manifest/stream",
        files={"manifest": ("manifest.jsonl", io.BytesIO(b"{not json}\n"), "application/x-ndjson")},
    )
    assert resp.status_code == 400
