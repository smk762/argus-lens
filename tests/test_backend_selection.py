"""Per-request backend selection on the caption endpoints, and naming the default (#107)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens import engine as engine_mod  # noqa: E402
from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import create_app  # noqa: E402


class _StubBackend(CaptionBackend):
    """The server default: a fixed caption, no model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op load."""

    def caption_image(self, image: Image.Image) -> str:
        """Return the default backend's fixed caption."""
        return "a person, plain studio background"

    def unload(self) -> None:
        """No-op unload."""


class _AltBackend(_StubBackend):
    """A second registered backend a request can switch to."""

    name = "alt"
    instances = 0

    def __init__(self) -> None:
        """Count constructions, so engine reuse is observable."""
        type(self).instances += 1

    def caption_image(self, image: Image.Image) -> str:
        """Return a caption distinguishable from the default's."""
        return "a dog on a beach"


class _OfflineBackend(_StubBackend):
    """A registered backend that reports itself unavailable."""

    name = "offline"

    def is_available(self) -> bool:
        """Never available."""
        return False

    def availability_reason(self) -> str | None:
        """Say why, as a real backend would."""
        return "API key not configured (set OFFLINE_API_KEY)"


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the alt/offline stubs alongside the real backends."""
    monkeypatch.delenv("ARGUS_ALLOWED_BACKENDS", raising=False)
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    engine_mod._register_backends()
    monkeypatch.setitem(engine_mod._BACKEND_REGISTRY, "alt", _AltBackend)
    monkeypatch.setitem(engine_mod._BACKEND_REGISTRY, "offline", _OfflineBackend)
    _AltBackend.instances = 0


def _client(**kwargs) -> TestClient:
    """A test client over a stub-default server with the refiner off."""
    return TestClient(create_app(default_backend=_StubBackend(), refiner=None, recorder=None, **kwargs))


def _png() -> io.BytesIO:
    """A tiny PNG upload."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def _caption(client: TestClient, **data):
    """POST /caption with one image and the given form fields."""
    return client.post("/caption", files={"file": ("x.png", _png(), "image/png")}, data=data)


def test_default_backend_is_named_without_captioning() -> None:
    """/health and /backends name the default before any caption runs."""
    client = _client()
    health = client.get("/health").json()
    assert health["backend"] == "stub"
    assert health["allowed_backends"] is None
    body = client.get("/backends").json()
    assert body["default"] == "stub"
    assert body["allowed"] is None
    # The injected default is listed even though it isn't in the registry.
    assert body["backends"]["stub"]["available"] is True
    assert body["backends"]["alt"]["allowed"] is True


def test_omitted_backend_uses_the_default() -> None:
    """No ``backend`` field captions with the server default, as before."""
    resp = _caption(_client())
    assert resp.status_code == 200
    assert resp.json()["backend_name"] == "stub"


def test_named_backend_captions_and_says_so() -> None:
    """``backend=alt`` is captioned by alt, and its engine is reused afterwards."""
    client = _client()
    first = _caption(client, backend="alt")
    assert first.status_code == 200, first.text
    assert first.json()["backend_name"] == "alt"
    assert "dog" in first.json()["final_caption"]
    assert _caption(client, backend="alt").status_code == 200
    # One instance for the availability check + engine; not one per request.
    assert _AltBackend.instances == 1
    # The default is untouched by the switch.
    assert _caption(client).json()["backend_name"] == "stub"


def test_naming_the_default_explicitly_uses_the_default_engine() -> None:
    """``backend=<default>`` is the default engine, not a second copy."""
    resp = _caption(_client(), backend="stub")
    assert resp.status_code == 200
    assert resp.json()["backend_name"] == "stub"


def test_unknown_backend_is_a_400() -> None:
    """An unregistered name is refused, never silently replaced by the default."""
    resp = _caption(_client(), backend="nope")
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]
    assert "Unknown backend" in resp.json()["detail"]


def test_unavailable_backend_is_a_400_carrying_its_reason() -> None:
    """An unavailable backend's own ``reason`` reaches the client."""
    resp = _caption(_client(), backend="offline")
    assert resp.status_code == 400
    assert "OFFLINE_API_KEY" in resp.json()["detail"]
    # ...and /backends reported the same reason up front.
    entry = _client().get("/backends").json()["backends"]["offline"]
    assert entry["available"] is False
    assert "OFFLINE_API_KEY" in entry["reason"]


def test_allowlist_refuses_unlisted_backends() -> None:
    """A pinned server refuses an available-but-unlisted backend; the default stays allowed."""
    client = _client(allowed_backends=["alt"])
    assert _caption(client, backend="alt").status_code == 200
    assert _caption(client).status_code == 200  # default, not listed, still works
    resp = _caption(client, backend="offline")
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["detail"]
    body = client.get("/backends").json()
    assert body["allowed"] == ["alt", "stub"]
    assert body["backends"]["alt"]["allowed"] is True
    assert body["backends"]["wd14"]["allowed"] is False
    assert client.get("/health").json()["allowed_backends"] == ["alt", "stub"]


def test_allowlist_from_env_pins_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARGUS_ALLOWED_BACKENDS=<default> disables switching altogether."""
    monkeypatch.setenv("ARGUS_ALLOWED_BACKENDS", "stub")
    client = _client()
    assert _caption(client, backend="alt").status_code == 400
    assert _AltBackend.instances == 0  # refused before anything was built


def test_stream_refuses_a_bad_backend_before_streaming() -> None:
    """The streaming endpoint answers 400, not a 200 stream of per-line errors."""
    resp = _client().post(
        "/caption/stream", files=[("files", ("x.png", _png(), "image/png"))], data={"backend": "offline"}
    )
    assert resp.status_code == 400


def test_batch_and_stream_honour_backend() -> None:
    """Multipart batch endpoints caption with the requested backend."""
    client = _client()
    batch = client.post("/caption/batch", files=[("files", ("a.png", _png(), "image/png"))], data={"backend": "alt"})
    assert batch.json()["results"]["a.png"]["backend_name"] == "alt"
    stream = client.post("/caption/stream", files=[("files", ("a.png", _png(), "image/png"))], data={"backend": "alt"})
    line = json.loads(stream.text.splitlines()[0])
    assert line["backend_name"] == "alt"


def test_folder_endpoint_honours_backend(tmp_path: Path) -> None:
    """JSON-bodied endpoints take ``backend`` too (here /caption/folder)."""
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    client = TestClient(
        create_app(default_backend=_StubBackend(), refiner=None, recorder=None, source_root=str(tmp_path))
    )
    resp = client.post("/caption/folder", json={"folder": ".", "write_sidecar": True, "backend": "alt"})
    assert resp.status_code == 200, resp.text
    assert "dog" in (tmp_path / "a.txt").read_text()
    bad = client.post("/caption/folder", json={"folder": ".", "backend": "offline"})
    assert bad.status_code == 400


def test_admin_unload_frees_switched_engines_too() -> None:
    """/admin/unload frees every pooled engine, and /health lists what's resident."""
    client = _client()
    assert _caption(client, backend="alt").status_code == 200
    assert client.get("/health").json()["gpu"]["resident"] == ["alt"]
    assert client.post("/admin/unload").json()["unloaded"] is True
    assert client.get("/health").json()["gpu"]["resident"] == []


def test_stream_disambiguates_duplicate_filenames() -> None:
    """Like /caption/batch, two uploads sharing a filename stream under distinct names."""
    files = [("files", ("a.png", _png(), "image/png")), ("files", ("a.png", _png(), "image/png"))]
    stream = _client().post("/caption/stream", files=files)
    assert [json.loads(line)["name"] for line in stream.text.splitlines()] == ["a.png", "a.png#2"]
