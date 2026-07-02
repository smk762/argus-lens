"""Tests for the generic OpenAI-compatible backend (Ollama / vLLM / LM Studio)."""

import json

import httpx
import pytest
from PIL import Image

from argus_lens.backends.openai_compat import OpenAICompatBackend
from argus_lens.engine import _resolve_backend


def _image() -> Image.Image:
    """Return a small solid-red RGB test image."""
    return Image.new("RGB", (16, 16), color=(255, 0, 0))


def _mock_client(backend: OpenAICompatBackend, handler, *, headers=None) -> dict:
    """Attach a MockTransport client to *backend* and return a capture dict."""
    captured: dict = {}

    def _wrapped(request: httpx.Request) -> httpx.Response:
        """Record the request URL, auth header, and JSON body, then delegate to *handler*."""
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return handler(request)

    backend._client = httpx.Client(
        base_url=backend._base_url,
        headers=headers or {},
        transport=httpx.MockTransport(_wrapped),
    )
    return captured


def _ok_handler(_request: httpx.Request) -> httpx.Response:
    """Return a successful chat-completions response with a padded caption."""
    return httpx.Response(200, json={"choices": [{"message": {"content": "  a red square  "}}]})


def test_defaults_to_ollama_localhost():
    """Defaults to the local Ollama URL and llava model, and is available without an API key."""
    b = OpenAICompatBackend()
    assert b.name == "openai-compat"
    assert b._base_url == "http://localhost:11434/v1"
    assert b.model_id == "llava"
    # No API key required for local servers.
    assert b.is_available() is True
    assert b.availability_reason() is None


def test_env_var_resolution(monkeypatch):
    """Reads base URL and model from ARGUS_OPENAI_COMPAT_* environment variables."""
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_BASE_URL", "http://env-host/v1")
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_MODEL", "qwen2-vl")
    b = OpenAICompatBackend()
    assert b._base_url == "http://env-host/v1"
    assert b.model_id == "qwen2-vl"


def test_constructor_overrides_env(monkeypatch):
    """Explicit constructor arguments take precedence over environment variables."""
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_BASE_URL", "http://env-host/v1")
    b = OpenAICompatBackend(base_url="http://explicit/v1", model_id="llama3.2-vision")
    assert b._base_url == "http://explicit/v1"
    assert b.model_id == "llama3.2-vision"


def test_resolves_via_engine_with_base_url():
    """The engine resolves "openai-compat" and forwards base_url and model_id."""
    b = _resolve_backend("openai-compat", base_url="http://host:8000/v1", model_id="m")
    assert isinstance(b, OpenAICompatBackend)
    assert b._base_url == "http://host:8000/v1"
    assert b.model_id == "m"


def test_caption_request_shape_and_parsing():
    """Sends a chat-completions request with text plus base64 JPEG parts and strips the reply."""
    b = OpenAICompatBackend(base_url="http://localhost:11434/v1", model_id="llava")
    captured = _mock_client(b, _ok_handler)

    out = b.caption_image(_image())

    assert out == "a red square"  # stripped
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["body"]["model"] == "llava"
    content = captured["body"]["messages"][1]["content"]
    assert [c["type"] for c in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_no_auth_header_without_key():
    """Omits the Authorization header when no API key is configured."""
    b = OpenAICompatBackend(base_url="http://localhost:11434/v1")
    captured = _mock_client(b, _ok_handler)
    b.caption_image(_image())
    assert captured["auth"] is None


def test_auth_header_with_key():
    """Sends a Bearer Authorization header when an API key is configured."""
    b = OpenAICompatBackend(base_url="http://x/v1", api_key="secret123")
    captured = _mock_client(b, _ok_handler, headers={"Authorization": "Bearer secret123"})
    b.caption_image(_image())
    assert captured["auth"] == "Bearer secret123"


def test_http_error_propagates():
    """Raises httpx.HTTPStatusError when the server responds with a 500."""
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        b.caption_image(_image())


def test_null_content_raises_clear_error():
    """A 200 with content:null (refusal / content filter) raises RuntimeError, not AttributeError."""
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    with pytest.raises(RuntimeError, match="no caption text"):
        b.caption_image(_image())


def test_empty_choices_raises_clear_error():
    """A 200 with an empty choices list raises RuntimeError, not IndexError."""
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(RuntimeError, match="response shape"):
        b.caption_image(_image())


def test_list_content_parts_are_joined():
    """Servers returning content as typed parts still yield the joined caption text."""
    parts = [{"type": "text", "text": "a red"}, {"type": "text", "text": " square"}]
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": parts}}]}))
    assert b.caption_image(_image()) == "a red square"
