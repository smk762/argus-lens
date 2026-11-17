"""The /v1 (Frigate) router accepts inline images only; it never fetches URLs."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import argus_lens.openai_compat as compat  # noqa: E402
from tests._netguard import block_outbound  # noqa: E402


def _body(url: str) -> dict:
    """A Frigate-shaped chat completion request carrying *url* as the image."""
    return {
        "model": "argus-wd14",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
    }


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/", "https://intranet.local/x.jpg"])
def test_remote_image_url_is_refused_without_fetching(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """An http(s) image_url is a 400 and no outbound request is made (SSRF)."""
    attempts = block_outbound(monkeypatch)
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body(url))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "remote_url_not_supported"
    assert attempts == []


def test_outbound_guard_catches_sync_and_async_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard used above trips on sync, async and raw-socket fetches, but not on TestClient."""
    import asyncio
    import socket

    import httpx

    attempts = block_outbound(monkeypatch)
    with pytest.raises(pytest.fail.Exception):
        httpx.get("http://192.0.2.1/x.jpg")

    async def _async_fetch() -> None:
        """Fetch with an AsyncClient, as an async handler would."""
        async with httpx.AsyncClient() as ac:
            await ac.get("http://192.0.2.1/x.jpg")

    with pytest.raises(pytest.fail.Exception):
        asyncio.run(_async_fetch())
    with pytest.raises(pytest.fail.Exception):
        socket.create_connection(("192.0.2.1", 80))
    assert len(attempts) == 3
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    assert TestClient(app).get("/v1/models").status_code == 200
    assert len(attempts) == 3


def test_data_uri_still_reaches_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inline form Frigate sends is decoded and captioned as before."""
    from argus_lens.types import CaptionResult

    seen = {}

    class _Engine:
        """Engine stand-in recording what it was asked to caption."""

        def caption(self, pil):
            """Record the image size and return a fixed result."""
            seen["size"] = pil.size
            return CaptionResult(final_caption="a red square")

    monkeypatch.setattr(compat._EnginePool, "get", lambda self, model_id: _Engine())
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body(url))
    assert resp.status_code == 200, resp.text
    assert seen["size"] == (4, 4)
    assert resp.json()["choices"][0]["message"]["content"] == "a red square"


def test_oversized_data_uri_is_413_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A data: URI over the size cap is refused before base64 decoding."""
    monkeypatch.setattr(compat, "_MAX_B64_CHARS", 16)
    monkeypatch.setattr(compat.base64, "b64decode", lambda *a, **k: pytest.fail("decoded oversized payload"))
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body("data:image/png;base64," + "A" * 32))
    assert resp.status_code == 413
    assert resp.json()["detail"]["error"]["code"] == "image_too_large"


def test_invalid_base64_data_uri_is_400() -> None:
    """Malformed base64 is a 400 invalid_image, not a server error."""
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body("data:image/png;base64,@@not*base64@@"))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "invalid_image"


def test_line_wrapped_base64_data_uri_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """MIME-style line breaks inside the base64 payload are tolerated, as before strict validation."""
    from argus_lens.types import CaptionResult

    class _Engine:
        """Engine stand-in returning a fixed caption."""

        def caption(self, pil):
            """Return a fixed result."""
            return CaptionResult(final_caption="ok")

    monkeypatch.setattr(compat._EnginePool, "get", lambda self, model_id: _Engine())
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    b64 = base64.encodebytes(buf.getvalue()).decode()  # wraps at 76 chars with "\n"
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body("data:image/png;base64," + b64))
    assert resp.status_code == 200, resp.text


def test_allowed_backends_pin_limits_v1_models() -> None:
    """With a backend pin, /v1 neither lists nor accepts a model whose backend is not allowed."""
    app = FastAPI()
    app.include_router(compat.create_openai_router(allowed_backends={"hybrid"}), prefix="/v1")
    client = TestClient(app)
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["argus-hybrid"]
    resp = client.post("/v1/chat/completions", json=_body("data:image/png;base64,AAAA"))
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "model_not_found"


def _png_data_uri(color: str = "red", size: tuple[int, int] = (4, 4)) -> str:
    """A small PNG as a base64 data URI."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _image_part(url: str) -> dict:
    """An OpenAI image_url content part."""
    return {"type": "image_url", "image_url": {"url": url}}


def _recording_engine(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Route every model to an engine stand-in that records the image size it captioned."""
    from argus_lens.types import CaptionResult

    seen: dict = {}

    class _Engine:
        """Engine stand-in recording what it was asked to caption."""

        def caption(self, pil):
            """Record the image size and return a fixed result."""
            seen["size"] = pil.size
            return CaptionResult(final_caption="ok")

    monkeypatch.setattr(compat._EnginePool, "get", lambda self, model_id: _Engine())
    return seen


def _client() -> TestClient:
    """A TestClient over just the /v1 router."""
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    return TestClient(app)


def test_image_comes_from_the_last_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later multimodal assistant/system turn is never captioned in place of the user's image."""
    seen = _recording_engine(monkeypatch)
    body = {
        "model": "argus-wd14",
        "messages": [
            {"role": "system", "content": [_image_part(_png_data_uri("blue", (2, 2)))]},
            {"role": "user", "content": [{"type": "text", "text": "describe"}, _image_part(_png_data_uri())]},
            {"role": "assistant", "content": [_image_part(_png_data_uri("green", (8, 8)))]},
        ],
    }
    resp = _client().post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    assert seen["size"] == (4, 4)


def test_image_only_in_a_non_user_message_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image carried only by an assistant turn is not the user's image: no_image."""
    seen = _recording_engine(monkeypatch)
    body = {
        "model": "argus-wd14",
        "messages": [
            {"role": "user", "content": "what is this?"},
            {"role": "assistant", "content": [_image_part(_png_data_uri())]},
        ],
    }
    resp = _client().post("/v1/chat/completions", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["code"] == "no_image"
    assert seen == {}


def test_image_decode_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base64 and PIL decode happen in a worker thread, not on the event loop."""
    import asyncio

    _recording_engine(monkeypatch)
    on_loop: list[str] = []
    real_open, real_b64 = compat.Image.open, compat.base64.b64decode

    def _on_loop() -> bool:
        """True when called from a thread running an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _open(*a, **k):
        """Record whether Image.open ran on the loop."""
        if _on_loop():
            on_loop.append("Image.open")
        return real_open(*a, **k)

    def _b64(*a, **k):
        """Record whether b64decode ran on the loop."""
        if _on_loop():
            on_loop.append("b64decode")
        return real_b64(*a, **k)

    monkeypatch.setattr(compat.Image, "open", _open)
    monkeypatch.setattr(compat.base64, "b64decode", _b64)
    resp = _client().post("/v1/chat/completions", json=_body(_png_data_uri()))
    assert resp.status_code == 200, resp.text
    assert on_loop == []


def test_backend_error_detail_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend failure is a 503 backend_unavailable without the exception text."""

    class _Engine:
        """Engine stand-in failing with internal details."""

        def caption(self, pil):
            """Fail like a backend leaking a model path."""
            raise RuntimeError("CUDA error: out of memory at /srv/models/secret-model/model.onnx")

    monkeypatch.setattr(compat._EnginePool, "get", lambda self, model_id: _Engine())
    resp = _client().post("/v1/chat/completions", json=_body(_png_data_uri()))
    assert resp.status_code == 503
    err = resp.json()["detail"]["error"]
    assert err["code"] == "backend_unavailable" and err["type"] == "server_error"
    assert "/srv/models" not in err["message"] and "CUDA" not in err["message"]


def test_engine_construction_error_detail_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure building the engine (e.g. a bad model_dir) is a generic 503 too."""

    def _boom(self, model_id):
        """Fail like a missing model directory."""
        raise FileNotFoundError("/srv/models/wd14/model.onnx")

    monkeypatch.setattr(compat._EnginePool, "get", _boom)
    resp = _client().post("/v1/chat/completions", json=_body(_png_data_uri()))
    assert resp.status_code == 503
    assert "/srv/models" not in resp.json()["detail"]["error"]["message"]


def test_undecodable_image_is_a_generic_400() -> None:
    """Valid base64 that isn't an image is a 400 invalid_image without PIL's message."""
    url = "data:image/png;base64," + base64.b64encode(b"<html>not an image</html>").decode()
    resp = _client().post("/v1/chat/completions", json=_body(url))
    assert resp.status_code == 400
    err = resp.json()["detail"]["error"]
    assert err["code"] == "invalid_image"
    assert err["message"] == "Invalid image data"
