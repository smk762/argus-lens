"""Tests for WD14 v3 tag parsing and preprocessing (#23).

These exercise tag filtering and image preprocessing with a fake ONNX
session, so they do not require the real model download. ``numpy`` is an
optional (``wd14``) extra, so the tests skip when it is unavailable.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from PIL import Image

from argus_lens.backends.wd14 import WD14Backend


class _FakeInput:
    """Fake ONNX input descriptor exposing a fixed shape and name."""

    shape = [None, 4, 4, 3]
    name = "input"


class _FakeSession:
    """Minimal stand-in for an ONNX InferenceSession."""

    def __init__(self, probs: list[float]) -> None:
        """Store the probabilities to return from run()."""
        import numpy as np

        self._probs = np.asarray([probs], dtype=np.float32)

    def get_inputs(self) -> list[_FakeInput]:
        """Return the single fake input descriptor."""
        return [_FakeInput()]

    def run(self, outputs: Any, feed: Any) -> list[Any]:
        """Return the canned probability array, ignoring the feed."""
        return [self._probs]


class _FakeRegistry:
    """Yields a prebuilt payload, ignoring the loader."""

    def __init__(self, payload: Any) -> None:
        """Store the payload to yield from acquire()."""
        self._payload = payload

    @contextlib.contextmanager
    def acquire(self, key: str, loader: Any):  # noqa: ANN201
        """Yield the prebuilt payload without invoking the loader."""
        yield self._payload


def test_excludes_rating_tags_and_applies_threshold():
    """Drops category-9 rating tags and tags scoring below the threshold from the caption."""
    pytest.importorskip("numpy")
    tags = [
        ("general", 9),
        ("explicit", 9),
        ("1girl", 0),
        ("solo", 0),
        ("cat", 4),
        ("blurry", 0),
    ]
    probs = [0.99, 0.95, 0.9, 0.5, 0.8, 0.1]
    backend = WD14Backend(registry=_FakeRegistry((_FakeSession(probs), tags, "input")), threshold=0.35)

    out = backend.caption_image(Image.new("RGB", (8, 8), (255, 0, 0)))

    # category 9 (ratings) excluded; "blurry" below threshold excluded.
    assert out == "1girl, solo, cat"


def test_preprocess_is_bgr_float32_square():
    """_preprocess yields a float32 NHWC batch with channels flipped RGB to BGR."""
    np = pytest.importorskip("numpy")
    arr = WD14Backend._preprocess(Image.new("RGB", (8, 8), (255, 0, 0)), 8)
    assert arr.shape == (1, 8, 8, 3)
    assert arr.dtype == np.float32
    # red (R=255,G=0,B=0) -> BGR -> [0,0,255]
    assert list(arr[0, 4, 4]) == [0.0, 0.0, 255.0]


def test_preprocess_pads_non_square_with_white():
    """_preprocess pads non-square images to a white square before resizing."""
    pytest.importorskip("numpy")
    # 8x2 wide image -> padded to 8x8 square; top rows become white.
    arr = WD14Backend._preprocess(Image.new("RGB", (8, 2), (255, 0, 0)), 8)
    assert list(arr[0, 0, 4]) == [255.0, 255.0, 255.0]  # white pad (BGR)
    assert list(arr[0, 3, 4]) == [0.0, 0.0, 255.0]  # red content (BGR)
