"""Tests for device passthrough (#10, #21) and OOM-retry wiring (#9)."""

from unittest.mock import patch

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.engine import ArgusLens
from argus_lens.retry import OOMDeadlineExceededError


class _RecordingBackend(CaptionBackend):
    """Prose backend that records the device passed to load() (#21)."""

    name = "recording"
    style = "photo"

    def __init__(self) -> None:
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        self.seen_device = device

    def caption_image(self, image: Image.Image) -> str:
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


class _CountingLoadBackend(CaptionBackend):
    """Counts load() invocations to verify lazy single-load behaviour."""

    name = "counting"
    style = "photo"

    def __init__(self) -> None:
        self.load_calls = 0
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        self.load_calls += 1
        self.seen_device = device

    def caption_image(self, image: Image.Image) -> str:
        return "a test caption"

    def unload(self) -> None:
        pass


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), (1, 2, 3))


def test_device_is_forwarded_to_backend():
    backend = _RecordingBackend()
    ArgusLens(backend=backend, device="cpu").caption(_img())
    assert backend.seen_device == "cpu"


def test_backend_loaded_once_across_multiple_images():
    backend = _CountingLoadBackend()
    engine = ArgusLens(backend=backend, device="cpu")
    engine.caption(_img())
    engine.caption(_img())
    # load(device) is configured once, lazily — not per image.
    assert backend.load_calls == 1
    assert backend.seen_device == "cpu"


class _AlwaysOOMBackend(CaptionBackend):
    """Always raises CUDA OOM."""

    name = "always-oom"
    style = "photo"

    def __init__(self) -> None:
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        self.calls += 1
        raise RuntimeError("CUDA out of memory")

    def unload(self) -> None:
        pass


class _ValueErrorBackend(CaptionBackend):
    """Raises a non-OOM error that must propagate without retry."""

    name = "boom"
    style = "photo"

    def __init__(self) -> None:
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        self.calls += 1
        raise ValueError("not an OOM")

    def unload(self) -> None:
        pass


def test_oom_is_retried_then_succeeds():
    backend = _FlakyBackend()
    with patch("argus_lens.retry.time.sleep"):
        result = ArgusLens(backend=backend).caption(_img())
    assert backend.calls == 2
    assert "recovered caption" in result.raw_prose


def test_persistent_oom_raises_deadline_error():
    backend = _AlwaysOOMBackend()
    with patch("argus_lens.retry.time.sleep"), pytest.raises(OOMDeadlineExceededError):
        # tiny budget so the loop exhausts quickly without real waiting
        ArgusLens(backend=backend, oom_retry_max_wait_s=0.0).caption(_img())
    assert backend.calls >= 1


def test_non_oom_error_propagates_without_retry():
    backend = _ValueErrorBackend()
    with pytest.raises(ValueError):
        ArgusLens(backend=backend).caption(_img())
    # no retry loop for non-OOM errors
    assert backend.calls == 1


def test_hybrid_forwards_device_to_both_subbackends_via_load():
    tag = _RecordingBackend()
    prose = _RecordingBackend()
    pipeline = HybridPipeline(tag_backend=tag, prose_backend=prose)

    ArgusLens(backend=pipeline, device="cpu").caption(_img())

    # HybridPipeline.load(device) forwards the engine device to both stages.
    assert tag.seen_device == "cpu"
    assert prose.seen_device == "cpu"
