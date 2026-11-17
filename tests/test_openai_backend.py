"""OpenAI (official SDK) backend response handling, with the SDK client faked (no network, no ``openai``)."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image
from structlog.testing import capture_logs

from argus_lens.backends.openai import OpenAIBackend


def _backend(content: str, finish_reason: str) -> OpenAIBackend:
    """An OpenAIBackend whose client returns one choice with *content* / *finish_reason*."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    response = SimpleNamespace(choices=[choice])
    backend = OpenAIBackend(api_key="k", max_tokens=300)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    return backend


def test_truncated_caption_is_flagged() -> None:
    """finish_reason == "length" (cut off at max_tokens) is logged, not passed off silently as complete."""
    with capture_logs() as logs:
        caption = _backend("A man stands by the", "length").caption_image(Image.new("RGB", (4, 4)))
    assert caption == "A man stands by the"
    assert [e for e in logs if e["event"] == "openai.caption_truncated" and e["log_level"] == "warning"]


def test_complete_caption_is_not_flagged() -> None:
    """A normal stop logs no truncation warning."""
    with capture_logs() as logs:
        caption = _backend("  A man stands by the sea.  ", "stop").caption_image(Image.new("RGB", (4, 4)))
    assert caption == "A man stands by the sea."
    assert not [e for e in logs if e["event"] == "openai.caption_truncated"]
