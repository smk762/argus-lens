"""Hybrid pipeline — combines a tag backend + a prose backend with redundancy filtering."""

from __future__ import annotations

from PIL import Image

from argus_lens.assembly.filtering import filter_redundant_clauses
from argus_lens.backends.base import CaptionBackend


class HybridPipeline(CaptionBackend):
    """Two-stage hybrid captioner: tag backend + prose backend.

    Stage 1 runs the *tag_backend* (e.g. WD14) to produce structured tags.
    Stage 2 runs the *prose_backend* (e.g. Florence-2, OpenAI) to produce
    a prose description.  Clauses in the prose that duplicate tag content
    are automatically removed before merging.

    This works with any combination of local and cloud backends.
    """

    name = "hybrid"
    style = "photo"
    requires_gpu = False

    def __init__(
        self,
        tag_backend: CaptionBackend,
        prose_backend: CaptionBackend,
    ) -> None:
        """Wire up the two sub-backends; GPU is required if either sub-backend needs it."""
        self._tag = tag_backend
        self._prose = prose_backend
        self.requires_gpu = tag_backend.requires_gpu or prose_backend.requires_gpu

    def load(self, device: str = "auto") -> None:
        """Forward the device to both sub-backends."""
        self._tag.load(device)
        self._prose.load(device)

    def caption_image(self, image: Image.Image) -> str:
        """Run both stages and merge tags with redundancy-filtered prose."""
        tags = self._tag.caption_image(image)
        prose = self._prose.caption_image(image)
        filtered = filter_redundant_clauses(prose, tags) if prose and tags else prose
        parts = [p for p in [tags, filtered] if p]
        return ", ".join(parts)

    def caption_image_split(self, image: Image.Image) -> tuple[str, str]:
        """Return ``(tags, prose)`` separately for structured assembly.

        Device placement is configured once via :meth:`load`, which forwards
        the device to both sub-backends; inference itself is device-free.
        """
        tags = self._tag.caption_image(image)
        prose = self._prose.caption_image(image)
        return tags, prose

    def unload(self) -> None:
        """Unload both sub-backends."""
        self._tag.unload()
        self._prose.unload()

    def is_available(self) -> bool:
        """Return True only if both sub-backends are available."""
        return self._tag.is_available() and self._prose.is_available()

    def availability_reason(self) -> str | None:
        """Combine the sub-backend reasons, labelling which stage is blocked."""
        tag_reason = self._tag.availability_reason()
        prose_reason = self._prose.availability_reason()
        if tag_reason and prose_reason:
            return f"Tag backend: {tag_reason}; Prose backend: {prose_reason}"
        return tag_reason or prose_reason
