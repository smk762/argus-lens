"""POST /caption/url refuses internal URLs and non-URL paths (#110)."""

from __future__ import annotations

import functools
import io
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import httpx  # noqa: E402
from argus_cortex import net as cortex_net  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import create_app  # noqa: E402
from tests._netguard import block_outbound  # noqa: E402


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
    attempts = block_outbound(monkeypatch)
    resp = client.post("/caption/url", json={"image_url": url})
    assert resp.status_code == 400
    assert "non-public" in resp.json()["detail"]
    assert attempts == []


def test_local_path_is_400(client, tmp_path: Path):
    """A filesystem path is never read: image_url must be http(s)."""
    img = tmp_path / "secret.png"
    Image.new("RGB", (8, 8)).save(img)
    resp = client.post("/caption/url", json={"image_url": str(img)})
    assert resp.status_code == 400
    assert "http(s)" in resp.json()["detail"]


def _png() -> bytes:
    """A tiny PNG."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def _stub_net(monkeypatch, handler, dns: dict[str, list[str]]) -> list[httpx.Request]:
    """Route cortex's fetch through *handler* (``transport=``) and *dns* (``resolve=``).

    Returns the list of requests that reached the transport.
    """
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def resolve(host: str, port: int) -> list[str]:
        if host not in dns:
            raise OSError(f"cannot resolve {host}")
        return dns[host]

    monkeypatch.setattr(
        cortex_net,
        "fetch",
        functools.partial(cortex_net.fetch, resolve=resolve, transport=httpx.MockTransport(record)),
    )
    return seen


def test_allowed_host_captions(client, monkeypatch):
    """A host named in LENS_URL_ALLOWED_HOSTS captions as before, pinned to its checked address."""
    monkeypatch.setenv("LENS_URL_ALLOWED_HOSTS", "immich.lan")
    seen = _stub_net(monkeypatch, lambda r: httpx.Response(200, content=_png()), {"immich.lan": ["192.168.1.20"]})
    resp = client.post("/caption/url", json={"image_url": "http://immich.lan:8080/a.png"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["final_caption"]
    assert [(str(r.url), r.headers["host"]) for r in seen] == [("http://192.168.1.20:8080/a.png", "immich.lan:8080")]


def test_connect_failure_detail_is_generic(client, monkeypatch):
    """A connection failure is a 400 naming only the host, never the address or raw error."""

    def refuse(request: httpx.Request) -> httpx.Response:
        """Fail the connection with an error naming an internal address."""
        raise httpx.ConnectError("[Errno 111] Connection refused (10.0.0.5:6333)", request=request)

    _stub_net(monkeypatch, refuse, {"example.com": ["93.184.215.14"]})
    resp = client.post("/caption/url", json={"image_url": "https://example.com/a.png"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "10.0.0.5" not in detail and "93.184.215.14" not in detail and "Errno" not in detail


def test_transport_error_detail_is_generic(client, monkeypatch):
    """A raw fetch error escaping the helper (which can name internal hosts/ports) is not echoed back."""

    def boom(url, **kwargs):
        """Raise a transport error naming an internal address."""
        raise httpx.ReadError("[Errno 104] Connection reset (10.0.0.5:6333)")

    monkeypatch.setattr(cortex_net, "fetch", boom)
    resp = client.post("/caption/url", json={"image_url": "https://example.com/a.png"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "could not fetch image_url"


def test_http_status_error_reports_only_the_status(client, monkeypatch):
    """A non-2xx response is reported by status code alone."""
    _stub_net(
        monkeypatch,
        lambda r: httpx.Response(404, text="404 for http://internal:9000/x"),
        {"example.com": ["93.184.215.14"]},
    )
    resp = client.post("/caption/url", json={"image_url": "https://example.com/a.png"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "image_url returned HTTP 404"


def test_undecodable_body_is_a_generic_400(client, monkeypatch):
    """A body that isn't an image gets a fixed message, not PIL's error text."""
    _stub_net(
        monkeypatch,
        lambda r: httpx.Response(200, content=b"<html>not an image</html>"),
        {"example.com": ["93.184.215.14"]},
    )
    resp = client.post("/caption/url", json={"image_url": "https://example.com/a.png"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "image_url did not return a decodable image"
