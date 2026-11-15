"""POST /caption/url refuses internal URLs and non-URL paths (#110)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens import url_fetch  # noqa: E402
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


@pytest.fixture
def client(monkeypatch):
    """A TestClient over a stub backend, with URL policy env cleared."""
    monkeypatch.delenv("LENS_URL_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("LENS_URL_FETCH", raising=False)
    return TestClient(create_app(default_backend=_StubBackend()))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/api/tags",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://10.0.0.1/",
    ],
)
def test_internal_url_is_400_without_connecting(client, monkeypatch, url):
    """A loopback/link-local/private URL is a 400 and no HTTP request is sent."""
    sent = []
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: sent.append(a) or pytest.fail("connected"))
    resp = client.post("/caption/url", json={"image_url": url})
    assert resp.status_code == 400
    assert "non-public" in resp.json()["detail"]
    assert sent == []


def test_local_path_is_400(client, tmp_path: Path):
    """A filesystem path is never read: image_url must be http(s)."""
    img = tmp_path / "secret.png"
    Image.new("RGB", (8, 8)).save(img)
    resp = client.post("/caption/url", json={"image_url": str(img)})
    assert resp.status_code == 400
    assert "http(s)" in resp.json()["detail"]


def test_allowed_host_captions(client, monkeypatch):
    """A host named in LENS_URL_ALLOWED_HOSTS captions as before."""
    monkeypatch.setenv("LENS_URL_ALLOWED_HOSTS", "127.0.0.1")
    called = []

    def fake_fetch(url):
        called.append(url_fetch.check_url(url))
        import io

        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(url_fetch, "fetch_url", fake_fetch)
    resp = client.post("/caption/url", json={"image_url": "http://127.0.0.1:8080/a.png"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["final_caption"]
    assert called == [("127.0.0.1", ["127.0.0.1"], 8080)]
