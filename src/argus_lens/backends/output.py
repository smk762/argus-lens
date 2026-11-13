"""Structured backend output types.

Foundation for the metadata-first / vision-tagging direction: backends should
preserve per-tag confidence, region (bounding box), and provenance instead of
flattening everything to a single string.

See issue #1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A bounding box as (x1, y1, x2, y2): top-left and bottom-right corners, in absolute pixels of
# the *source* image (origin top-left), as floats (#58). This is what Florence-2 ``<OD>`` and most
# detectors emit, so a backend stores it unrounded. Normalised coordinates are derived at the
# boundary that needs them, where the image size is known: see :func:`region_to_normalized` and
# :func:`region_to_mwg_area` (the centre-based form XMP ``mwg-rs`` face regions use).
Region = tuple[float, float, float, float]


def region_to_normalized(region: Region, width: float, height: float) -> Region:
    """Scale a pixel ``(x1, y1, x2, y2)`` box to 0–1 by the image size, clamped to the frame."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    x1, y1, x2, y2 = region
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return (clamp(x1 / width), clamp(y1 / height), clamp(x2 / width), clamp(y2 / height))


def region_to_mwg_area(region: Region, width: float, height: float) -> tuple[float, float, float, float]:
    """Return ``(cx, cy, w, h)`` normalised to 0–1: the MWG region area XMP sidecars store.

    ``cx``/``cy`` are the box *centre* (MWG's convention, used by ``argus_cortex.xmp.XmpRegion``).
    """
    x1, y1, x2, y2 = region_to_normalized(region, width, height)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


@dataclass(frozen=True)
class Tag:
    """A single tag/label produced by a backend.

    Attributes:
        label: The tag text (e.g. ``"mountain"``).
        score: Confidence in ``[0, 1]`` if the model provides one, else ``None``.
        source: Name of the backend that produced this tag (provenance).
        region: Optional bounding box for grounded tags (e.g. Florence-2 ``<OD>``).
    """

    label: str
    score: float | None = None
    source: str = ""
    region: Region | None = None

    def passes(self, min_score: float | None) -> bool:
        """Return True if this tag clears *min_score* (unscored tags always pass; ``None`` passes all)."""
        return min_score is None or self.score is None or self.score >= min_score


@dataclass
class BackendOutput:
    """Structured result of a single backend inference.

    ``caption_image() -> str`` remains the backward-compatible shim; new code
    should prefer ``annotate_image() -> BackendOutput`` so confidence, regions,
    and provenance survive into the assembly pipeline.
    """

    tags: list[Tag] = field(default_factory=list)
    prose: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def tag_string(self, *, min_score: float | None = None) -> str:
        """Flatten tags into the legacy comma-separated string.

        Args:
            min_score: If set, drop tags whose ``score`` is below this threshold.
                Tags without a score are always kept.
        """
        labels = [tag.label.strip() for tag in self.tags if tag.passes(min_score)]
        return ", ".join(label for label in labels if label)

    def to_caption_string(self, *, min_score: float | None = None) -> str:
        """Best-effort single string: prose if present, else flattened tags."""
        return self.prose if self.prose.strip() else self.tag_string(min_score=min_score)
