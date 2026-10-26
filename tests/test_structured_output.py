"""annotate_image() on every backend (#57) and the Region contract (#58)."""

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.backends.output import BackendOutput, Tag, region_to_mwg_area, region_to_normalized
from argus_lens.engine import _BACKEND_REGISTRY, _register_backends
from argus_lens.provenance import build_provenance


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), "white")


class _Fixed(CaptionBackend):
    """Returns a fixed raw string; `emits_tags` set per instance."""

    name = "fixed"

    def __init__(self, raw: str, emits_tags: bool = False) -> None:
        """Store the canned output and output kind."""
        self.raw = raw
        self.emits_tags = emits_tags

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image: Image.Image) -> str:
        """Return the canned string."""
        return self.raw

    def unload(self) -> None:
        """No-op."""


# -- annotate_image default ------------------------------------------------------------


def test_default_annotate_treats_prose_backends_as_prose():
    """A non-tagger's string becomes BackendOutput.prose (Florence-2, BLIP-2, cloud VLMs)."""
    out = _Fixed("  A dog on a sofa, looking at the camera.  ").annotate_image(_img())
    assert out == BackendOutput(prose="A dog on a sofa, looking at the camera.")


def test_default_annotate_splits_tagger_output_into_unscored_tags():
    """A tagger's (emits_tags) comma string becomes unscored Tags carrying their source."""
    out = _Fixed("beach, sunset, , ocean ", emits_tags=True).annotate_image(_img())
    assert out.tags == [Tag("beach", source="fixed"), Tag("sunset", source="fixed"), Tag("ocean", source="fixed")]
    assert out.prose == ""
    assert _Fixed("", emits_tags=True).annotate_image(_img()) == BackendOutput()


def test_every_registered_backend_has_annotate_image():
    """Callers can rely on annotate_image existing on every backend, not just RAM++."""
    _register_backends()
    for name, cls in _BACKEND_REGISTRY.items():
        assert callable(getattr(cls, "annotate_image", None)), name


# -- WD14 keeps its scores -----------------------------------------------------------------


def test_wd14_annotate_keeps_scores_and_matches_caption_image():
    """WD14 overrides the default so its per-tag probabilities survive; caption_image is unchanged."""
    pytest.importorskip("numpy")
    from argus_lens.backends.wd14 import WD14Backend
    from tests.test_wd14_tagging import _FakeRegistry, _FakeSession

    tags = [("general", 9), ("1girl", 0), ("cat", 4), ("blurry", 0)]
    probs = [0.99, 0.9, 0.81234567, 0.1]
    backend = WD14Backend(registry=_FakeRegistry((_FakeSession(probs), tags, "input")), threshold=0.35)

    out = backend.annotate_image(_img())
    assert [(t.label, t.score, t.source) for t in out.tags] == [("1girl", 0.9, "wd14"), ("cat", 0.8123, "wd14")]
    assert backend.caption_image(_img()) == out.tag_string() == "1girl, cat"


# -- hybrid -----------------------------------------------------------------------------------


def test_hybrid_annotate_keeps_tag_scores_and_adds_prose():
    """Hybrid returns the tag stage's (scored) tags plus the prose stage's prose."""

    class _Scored(_Fixed):
        def annotate_image(self, image):
            """Return scored tags, as RAM++ / WD14 do."""
            return BackendOutput(tags=[Tag("beach", score=0.97, source="ram")])

    hybrid = HybridPipeline(tag_backend=_Scored("", emits_tags=True), prose_backend=_Fixed("A sunny beach."))
    out = hybrid.annotate_image(_img())
    assert out.tags == [Tag("beach", score=0.97, source="ram")]
    assert out.prose == "A sunny beach."
    assert out.raw == {"tag_backend": "fixed", "prose_backend": "fixed"}


# -- Region contract ------------------------------------------------------------------------


def test_region_is_float_pixel_xyxy_and_serialises_unrounded():
    """Regions keep detector floats (no rounding) through provenance."""
    out = BackendOutput(tags=[Tag("dog", score=0.9, source="florence2", region=(12.5, 30.25, 200.75, 180.0))])
    assert build_provenance(out, backend_name="florence2")["tags"][0]["region"] == [12.5, 30.25, 200.75, 180.0]


def test_region_to_normalized_scales_and_clamps():
    """Pixel boxes scale to 0–1 by image size; boxes spilling past the frame are clamped."""
    assert region_to_normalized((100.0, 50.0, 300.0, 150.0), 400, 200) == (0.25, 0.25, 0.75, 0.75)
    assert region_to_normalized((-10.0, -5.0, 450.0, 220.0), 400, 200) == (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        region_to_normalized((0.0, 0.0, 1.0, 1.0), 0, 100)


def test_region_to_mwg_area_is_centre_based():
    """The XMP/MWG form is (centre x, centre y, width, height), normalised."""
    cx, cy, w, h = region_to_mwg_area((100.0, 50.0, 300.0, 150.0), 400, 200)
    assert (cx, cy, w, h) == (0.5, 0.5, 0.5, 0.5)
