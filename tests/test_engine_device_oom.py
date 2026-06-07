"""Tests for device passthrough (#10) and OOM-retry wiring (#9)."""

from unittest.mock import patch

from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.engine import ArgusLens


class _RecordingBackend(CaptionBackend):
    """Prose backend that records the device passed to caption_image."""

    name = "recording"
    style = "photo"

    def __init__(self) -> None:
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image, device: str = "auto") -> str:
        self.seen_device = device
        return "a test caption"

    def unload(self) -> None:
        pass


class _FlakyBackend(CaptionBackend):
    """Raises a CUDA OOM once, then succeeds (no device kwarg)."""

    name = "flaky"
    style = "photo"

    def __init__(self) -> None:
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return "recovered caption"

    def unload(self) -> None:
        pass


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), (1, 2, 3))


def test_device_is_forwarded_to_backend():
    backend = _RecordingBackend()
    ArgusLens(backend=backend, device="cpu").caption(_img())
    assert backend.seen_device == "cpu"


def test_oom_is_retried_then_succeeds():
    backend = _FlakyBackend()
    with patch("argus_lens.retry.time.sleep"):
        result = ArgusLens(backend=backend).caption(_img())
    assert backend.calls == 2
    assert "recovered caption" in result.raw_prose
