"""Structured backend output types.

Foundation for the metadata-first / vision-tagging direction: backends should
preserve per-tag confidence, region (bounding box), and provenance instead of
flattening everything to a single string.

See issue #1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (x1, y1, x2, y2) in pixel coordinates.
Region = tuple[int, int, int, int]


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


@dataclass
class BackendOutput:
    """Structured result of a single backend inference.

    ``caption_image() -> str`` remains the backward-compatible shim; new code
    should prefer ``annotate_image() -> BackendOutput`` so confidence, regions,
    and provenance survive into the assembly pipeline.
    """

    tags: list[Tag] = field(default_factory=list)
    prose: str = ""
    raw: dict = field(default_factory=dict)

    def tag_string(self, *, min_score: float | None = None) -> str:
        """Flatten tags into the legacy comma-separated string.

        Args:
            min_score: If set, drop tags whose ``score`` is below this threshold.
                Tags without a score are always kept.
        """
        labels = [tag.label for tag in self.tags if min_score is None or tag.score is None or tag.score >= min_score]
        return ", ".join(labels)

    def to_caption_string(self, *, min_score: float | None = None) -> str:
        """Best-effort single string: prose if present, else flattened tags."""
        return self.prose or self.tag_string(min_score=min_score)
