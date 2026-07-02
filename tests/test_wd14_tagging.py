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


def test_cache_key_includes_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backends pointing at different model dirs must never share a cached session."""
    monkeypatch.delenv("WD14_MODEL_DIR", raising=False)
    a = WD14Backend(model_dir="/models/base")
    b = WD14Backend(model_dir="/models/finetuned")
    default = WD14Backend()
    assert a._cache_key("cpu") != b._cache_key("cpu")
    assert a._cache_key("cpu") != default._cache_key("cpu")
    assert default._cache_key("cpu") == "wd14:cpu:default"
    monkeypatch.setenv("WD14_MODEL_DIR", "/env/dir")
    assert WD14Backend()._cache_key("cpu") == "wd14:cpu:/env/dir"


def test_ensure_model_refreshes_stale_tags_csv(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the model file is missing, a leftover CSV from a prior model is re-downloaded with it."""
    from argus_lens.backends import wd14 as wd14_mod

    def _fake_download(dest_dir: Any) -> None:
        """Mimic _download_model's skip-if-exists behavior without the network."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in (wd14_mod._MODEL_FILENAME, wd14_mod._TAGS_FILENAME):
            dest = dest_dir / name
            if not dest.exists():
                dest.write_text("v3-fresh")

    monkeypatch.setattr(wd14_mod, "_download_model", _fake_download)
    stale_csv = tmp_path / wd14_mod._TAGS_FILENAME
    stale_csv.write_text("v2-stale")

    backend = WD14Backend(model_dir=tmp_path)
    model_path = backend._ensure_model()

    assert model_path == tmp_path / wd14_mod._MODEL_FILENAME
    assert stale_csv.read_text() == "v3-fresh"  # stale pair was replaced, not kept
