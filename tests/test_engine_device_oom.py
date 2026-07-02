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
        """Initialise the last-seen device to None."""
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        """Record the device passed in."""
        self.seen_device = device

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a test caption"

    def unload(self) -> None:
        """No-op."""
        pass


class _FlakyBackend(CaptionBackend):
    """Raises a CUDA OOM once, then succeeds (no device kwarg)."""

    name = "flaky"
    style = "photo"

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        """No-op."""
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Raise a CUDA OOM on the first call, then return a caption."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return "recovered caption"

    def unload(self) -> None:
        """No-op."""
        pass


class _CountingLoadBackend(CaptionBackend):
    """Counts load() invocations to verify lazy single-load behaviour."""

    name = "counting"
    style = "photo"

    def __init__(self) -> None:
        """Initialise load counter and last-seen device."""
        self.load_calls = 0
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        """Count the call and record the device."""
        self.load_calls += 1
        self.seen_device = device

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a test caption"

    def unload(self) -> None:
        """No-op."""
        pass


def _img() -> Image.Image:
    """Return a tiny 8x8 RGB test image."""
    return Image.new("RGB", (8, 8), (1, 2, 3))


def test_device_is_forwarded_to_backend():
    """The engine's device argument is forwarded to the backend's load()."""
    backend = _RecordingBackend()
    ArgusLens(backend=backend, device="cpu").caption(_img())
    assert backend.seen_device == "cpu"


def test_backend_loaded_once_across_multiple_images():
    """The backend is loaded lazily once, not once per captioned image."""
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
        """Initialise the call counter."""
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        """No-op."""
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Always raise a CUDA OOM error."""
        self.calls += 1
        raise RuntimeError("CUDA out of memory")

    def unload(self) -> None:
        """No-op."""
        pass


class _ValueErrorBackend(CaptionBackend):
    """Raises a non-OOM error that must propagate without retry."""

    name = "boom"
    style = "photo"

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        """No-op."""
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Always raise a non-OOM ValueError."""
        self.calls += 1
        raise ValueError("not an OOM")

    def unload(self) -> None:
        """No-op."""
        pass


def test_oom_is_retried_then_succeeds():
    """A CUDA OOM is retried and the caption from the successful retry is returned."""
    backend = _FlakyBackend()
    with patch("argus_lens.retry.time.sleep"):
        result = ArgusLens(backend=backend).caption(_img())
    assert backend.calls == 2
    assert "recovered caption" in result.raw_prose


def test_persistent_oom_raises_deadline_error():
    """Persistent OOMs raise OOMDeadlineExceededError once the retry budget is exhausted."""
    backend = _AlwaysOOMBackend()
    with patch("argus_lens.retry.time.sleep"), pytest.raises(OOMDeadlineExceededError):
        # tiny budget so the loop exhausts quickly without real waiting
        ArgusLens(backend=backend, oom_retry_max_wait_s=0.0).caption(_img())
    assert backend.calls >= 1


def test_non_oom_error_propagates_without_retry():
    """Non-OOM errors propagate immediately without entering the retry loop."""
    backend = _ValueErrorBackend()
    with pytest.raises(ValueError):
        ArgusLens(backend=backend).caption(_img())
    # no retry loop for non-OOM errors
    assert backend.calls == 1


def test_hybrid_forwards_device_to_both_subbackends_via_load():
    """HybridPipeline.load forwards the engine device to both the tag and prose backends."""
    tag = _RecordingBackend()
    prose = _RecordingBackend()
    pipeline = HybridPipeline(tag_backend=tag, prose_backend=prose)

    ArgusLens(backend=pipeline, device="cpu").caption(_img())

    # HybridPipeline.load(device) forwards the engine device to both stages.
    assert tag.seen_device == "cpu"
    assert prose.seen_device == "cpu"
