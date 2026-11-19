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


def _mock_client(backend: OpenAICompatBackend, handler) -> dict:
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


def _intercept_transport(monkeypatch, handler) -> dict:
    """Route the client the backend builds itself (headers and all) through a MockTransport.

    Unlike :func:`_mock_client`, nothing about the request headers comes from the test:
    whatever ``Authorization`` arrives was set by the backend's own ``load()``.
    """
    captured: dict = {}
    real_client = httpx.Client

    def _wrapped(request: httpx.Request) -> httpx.Response:
        """Record the auth header the backend sent, then delegate to *handler*."""
        captured["auth"] = request.headers.get("Authorization")
        return handler(request)

    def _client(*args, **kwargs):
        """Build the backend's client as asked, swapping only the transport."""
        return real_client(*args, transport=httpx.MockTransport(_wrapped), **kwargs)

    monkeypatch.setattr(httpx, "Client", _client)
    return captured


def test_no_auth_header_without_key(monkeypatch):
    """Omits the Authorization header when no API key is configured."""
    monkeypatch.delenv("ARGUS_OPENAI_COMPAT_API_KEY", raising=False)
    captured = _intercept_transport(monkeypatch, _ok_handler)
    b = OpenAICompatBackend(base_url="http://localhost:11434/v1")
    assert b.caption_image(_image()) == "a red square"
    assert captured["auth"] is None


def test_auth_header_with_key(monkeypatch):
    """Sends a Bearer Authorization header built by the backend from its configured API key."""
    monkeypatch.delenv("ARGUS_OPENAI_COMPAT_API_KEY", raising=False)
    captured = _intercept_transport(monkeypatch, _ok_handler)
    b = OpenAICompatBackend(base_url="http://x/v1", api_key="secret123")
    assert b.caption_image(_image()) == "a red square"
    assert captured["auth"] == "Bearer secret123"


def test_auth_header_from_env_key(monkeypatch):
    """The key also resolves from ARGUS_OPENAI_COMPAT_API_KEY."""
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_API_KEY", "envkey")
    captured = _intercept_transport(monkeypatch, _ok_handler)
    OpenAICompatBackend(base_url="http://x/v1").caption_image(_image())
    assert captured["auth"] == "Bearer envkey"


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


def test_list_content_skips_null_text_parts():
    """A non-text part carrying ``"text": null`` is skipped, not a TypeError from join."""
    parts = [{"type": "text", "text": "a red square"}, {"type": "image", "text": None}, {"type": "refusal"}]
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": parts}}]}))
    assert b.caption_image(_image()) == "a red square"


@pytest.mark.parametrize("message", [None, "oops", ["a"]])
def test_null_or_non_dict_message_raises_clear_error(message):
    """``"message": null`` (or another non-object) raises RuntimeError, not AttributeError."""
    b = OpenAICompatBackend(base_url="http://x/v1")
    _mock_client(b, lambda _r: httpx.Response(200, json={"choices": [{"message": message}]}))
    with pytest.raises(RuntimeError, match="response shape"):
        b.caption_image(_image())


def test_large_images_are_downscaled_before_sending(monkeypatch):
    """A 3600px photo is sent at the default 1024px long edge (Ollama context overflow)."""
    import base64
    import io

    monkeypatch.delenv("ARGUS_OPENAI_COMPAT_MAX_SIDE", raising=False)
    backend = OpenAICompatBackend()
    captured = _mock_client(backend, _ok_handler)
    backend.caption_image(Image.new("RGB", (3600, 2400)))
    url = captured["body"]["messages"][1]["content"][1]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.size == (1024, 683)

    full = OpenAICompatBackend(max_image_side=0)
    captured = _mock_client(full, _ok_handler)
    full.caption_image(Image.new("RGB", (1200, 800)))
    url = captured["body"]["messages"][1]["content"][1]["image_url"]["url"]
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (1200, 800)


def test_reasoning_exhaustion_gets_actionable_error():
    """Empty content + reasoning + finish=length names the cause and the fix."""
    backend = OpenAICompatBackend()

    def handler(_request):
        """Reply like a thinking model that ran out of tokens."""
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning": "Got it, let's"}, "finish_reason": "length"}]},
        )

    _mock_client(backend, handler)
    with pytest.raises(RuntimeError, match="instruct"):
        backend.caption_image(_image())
