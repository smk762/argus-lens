"""HybridPipeline lifecycle: a failure in one stage must not leave the other stage's model resident."""

from __future__ import annotations

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline


class _Stage(CaptionBackend):
    """Sub-backend stand-in that tracks whether its 'model' is loaded and can fail on demand."""

    def __init__(self, name: str, *, fail_load: bool = False, fail_unload: bool = False) -> None:
        """Configure which lifecycle call raises."""
        self.name = name
        self.loaded = False
        self._fail_load = fail_load
        self._fail_unload = fail_unload

    def load(self, device: str = "auto") -> None:
        """Mark loaded, or raise as a CUDA OOM / download error would."""
        if self._fail_load:
            raise RuntimeError(f"{self.name} load failed")
        self.loaded = True

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return self.name

    def unload(self) -> None:
        """Mark unloaded, or raise."""
        if self._fail_unload:
            raise RuntimeError(f"{self.name} unload failed")
        self.loaded = False


def test_prose_load_failure_unloads_the_tag_stage() -> None:
    """If the prose model fails to load, the already-loaded tag model is released before the error propagates."""
    tag, prose = _Stage("tag"), _Stage("prose", fail_load=True)
    with pytest.raises(RuntimeError, match="prose load failed"):
        HybridPipeline(tag, prose).load("cpu")
    assert tag.loaded is False


def test_tag_unload_failure_still_unloads_prose() -> None:
    """A raising tag unload doesn't skip the prose unload (and the error still surfaces)."""
    tag, prose = _Stage("tag", fail_unload=True), _Stage("prose")
    pipeline = HybridPipeline(tag, prose)
    pipeline.load("cpu")
    with pytest.raises(RuntimeError, match="tag unload failed"):
        pipeline.unload()
    assert prose.loaded is False
