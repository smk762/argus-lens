"""Cross-site write guard + CORS allow-list on the server (argus_cortex.server.WriteGuard)."""

from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import create_app  # noqa: E402

_STUDIO = "http://localhost:3000"
_EVIL = "https://evil.example"


class _StubBackend(CaptionBackend):
    """Caption backend stub that returns a fixed caption without any model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op load."""

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption for any image."""
        return "a person"

    def unload(self) -> None:
        """No-op unload."""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Keep host env from leaking into the allow-list or the admin gate."""
    monkeypatch.delenv("LENS_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)


def _client(**kwargs) -> TestClient:
    return TestClient(create_app(default_backend=_StubBackend(), **kwargs))


def _post(client: TestClient, origin: str | None = None) -> int:
    """POST a write (/admin/unload) from *origin*; ``None`` sends no Origin, like curl."""
    headers = {"Origin": origin} if origin else {}
    return client.post("/admin/unload", headers=headers).status_code


def test_originless_and_same_host_writes_pass_without_cors() -> None:
    """curl/curator/Frigate send no Origin, and a UI proxied onto this host matches Host: both write."""
    client = _client()
    assert _post(client) == 200
    assert _post(client, "http://testserver") == 200  # TestClient's Host


def test_cross_site_write_refused_even_with_cors_off() -> None:
    """CORS off doesn't stop a no-preflight form POST; the guard does."""
    client = _client()
    resp = client.post("/admin/unload", headers={"Origin": _EVIL})
    assert resp.status_code == 403
    assert "cross-site" in resp.json()["detail"]


def test_bare_cors_trusts_only_the_localhost_studio() -> None:
    """--cors allows (and lets write) the :3000 dev frontend, not every origin."""
    client = _client(cors=True)
    assert _post(client, _STUDIO) == 200
    assert _post(client, _EVIL) == 403
    allowed = client.get("/health", headers={"Origin": _STUDIO}).headers
    assert allowed["access-control-allow-origin"] == _STUDIO
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": _EVIL}).headers


def test_named_origins_from_flag_and_env(monkeypatch) -> None:
    """--cors-origin / LENS_CORS_ORIGINS replace the localhost default."""
    lan = "http://192.168.1.5:3000"
    assert _post(_client(cors_origins=[lan]), lan) == 200
    assert _post(_client(cors_origins=[lan]), _STUDIO) == 403
    monkeypatch.setenv("LENS_CORS_ORIGINS", f" {lan} , ")
    assert _post(_client(), lan) == 200


def test_wildcard_reads_anywhere_but_never_writes_cross_site() -> None:
    """--cors-any grants credential-less reads to any origin, but no cross-site writes."""
    client = _client(cors_allow_any=True)
    headers = client.get("/health", headers={"Origin": _EVIL}).headers
    assert headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in headers
    assert _post(client, _EVIL) == 403
    assert _post(client, _STUDIO) == 403  # the localhost default doesn't apply under the wildcard


def test_star_in_list_is_wildcard_but_keeps_named_writers() -> None:
    """A '*' co-listed with a real origin is the wildcard, and the named origin can still write."""
    client = _client(cors_origins=["*", _STUDIO])
    assert client.get("/health", headers={"Origin": _EVIL}).headers["access-control-allow-origin"] == "*"
    assert _post(client, _STUDIO) == 200
    assert _post(client, _EVIL) == 403


def test_guard_does_not_buffer_streams() -> None:
    """A trusted write to a streaming endpoint still gets its streamed response."""
    client = _client(cors=True)
    resp = client.post(
        "/caption/stream",
        headers={"Origin": _STUDIO},
        files=[("files", ("a.png", _png(), "image/png"))],
    )
    assert resp.status_code == 200
    assert resp.text.strip()


def _png() -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()
