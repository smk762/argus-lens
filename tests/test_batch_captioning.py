"""caption_batch decodes lazily; /caption/batch keeps uploads that share a filename apart."""

from __future__ import annotations

import io

import pytest
from PIL import Image

import argus_lens.engine as engine_mod
from argus_lens.backends.base import CaptionBackend
from argus_lens.engine import ArgusLens


class _StubBackend(CaptionBackend):
    """CPU-only stub backend that returns a fixed caption without loading a model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        """No-op."""


def _png(color: tuple[int, int, int] = (120, 120, 120)) -> bytes:
    """Return PNG bytes for a small solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def test_caption_batch_decodes_one_image_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each image is decoded just before it is captioned, not the whole batch up front."""
    loads: list[int] = []
    real_load = engine_mod._load_image

    def _counting_load(source):
        """Record each decode, then decode as normal."""
        loads.append(1)
        return real_load(source)

    monkeypatch.setattr(engine_mod, "_load_image", _counting_load)
    seen: list[int] = []
    engine = ArgusLens(backend=_StubBackend())
    engine.caption_batch(
        [_png((1, 1, 1)), _png((2, 2, 2)), _png((3, 3, 3))],
        progress=lambda done, total, name, result: seen.append(len(loads)),
    )
    assert seen == [1, 2, 3]


def test_unique_names_suffixes_repeats() -> None:
    """Repeats get #N suffixes that never clash with a literal name in the input."""
    pytest.importorskip("fastapi")
    from argus_lens.server import _unique_names

    assert _unique_names(["a", "b", "a", "a"]) == ["a", "b", "a#2", "a#3"]
    assert _unique_names(["a", "a", "a#2"]) == ["a", "a#3", "a#2"]


def test_batch_endpoint_keeps_same_named_uploads_apart() -> None:
    """Two uploads with one filename (or none) come back as two results, not one."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus_lens.server import create_app

    client = TestClient(create_app(default_backend=_StubBackend()))
    resp = client.post(
        "/caption/batch",
        files=[
            ("files", ("cat.png", _png((1, 1, 1)), "image/png")),
            ("files", ("cat.png", _png((2, 2, 2)), "image/png")),
        ],
    )
    assert resp.status_code == 200, resp.text
    assert sorted(resp.json()["results"]) == ["cat.png", "cat.png#2"]
