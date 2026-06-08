"""Tests for device passthrough (#10) and OOM-retry wiring (#9)."""

from unittest.mock import patch

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.engine import ArgusLens
from argus_lens.retry import OOMDeadlineExceededError


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


class _NoDeviceTagBackend(CaptionBackend):
    """Tag backend without a device kwarg (like wd14)."""

    name = "tags"
    style = "anime"

    def __init__(self) -> None:
        self.calls = 0

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        self.calls += 1
        return "tag1, tag2"

    def unload(self) -> None:
        pass


def test_hybrid_forwards_device_to_device_aware_subbackend():
    tag = _NoDeviceTagBackend()
    prose = _RecordingBackend()
    pipeline = HybridPipeline(tag_backend=tag, prose_backend=prose)

    ArgusLens(backend=pipeline, device="cpu").caption(_img())

    # device-aware prose backend receives the explicit engine device ...
    assert prose.seen_device == "cpu"
    # ... and the no-device tag backend is still called without error.
    assert tag.calls == 1
