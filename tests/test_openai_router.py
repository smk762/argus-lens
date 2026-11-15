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


def _body(url: str) -> dict:
    """A Frigate-shaped chat completion request carrying *url* as the image."""
    return {
        "model": "argus-wd14",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
    }


@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/", "https://intranet.local/x.jpg"])
def test_remote_image_url_is_refused_without_fetching(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """An http(s) image_url is a 400 and no outbound request is made (SSRF)."""
    import httpx

    def _no_fetch(*_a, **_k):
        """Fail the test if the router tries to fetch anything."""
        raise AssertionError("router fetched a client-supplied URL")

    monkeypatch.setattr(httpx, "get", _no_fetch)
    monkeypatch.setattr(httpx.Client, "send", _no_fetch)
    app = FastAPI()
    app.include_router(compat.create_openai_router(), prefix="/v1")
    resp = TestClient(app).post("/v1/chat/completions", json=_body(url))
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "remote_url_not_supported"


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
