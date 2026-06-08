"""Hybrid pipeline — combines a tag backend + a prose backend with redundancy filtering."""

from __future__ import annotations

import inspect

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
        self._tag = tag_backend
        self._prose = prose_backend
        self.requires_gpu = tag_backend.requires_gpu or prose_backend.requires_gpu

    def load(self, device: str = "auto") -> None:
        self._tag.load(device)
        self._prose.load(device)

    @staticmethod
    def _caption(backend: CaptionBackend, image: Image.Image, device: str) -> str:
        """Call ``backend.caption_image``, forwarding ``device`` when accepted.

        Some local backends (florence2, blip2) take a ``device`` kwarg on
        ``caption_image``; others (wd14, cloud) do not. We sniff the signature
        so an explicit engine device is honoured by the prose/tag sub-backend.
        See #21 for the eventual ``load(device)``-based cleanup.
        """
        try:
            accepts_device = "device" in inspect.signature(backend.caption_image).parameters
        except (TypeError, ValueError):
            accepts_device = False
        if accepts_device:
            return backend.caption_image(image, device=device)  # type: ignore[call-arg]
        return backend.caption_image(image)

    def caption_image(self, image: Image.Image, device: str = "auto") -> str:
        tags = self._caption(self._tag, image, device)
        prose = self._caption(self._prose, image, device)
        filtered = filter_redundant_clauses(prose, tags) if prose and tags else prose
        parts = [p for p in [tags, filtered] if p]
        return ", ".join(parts)

    def caption_image_split(self, image: Image.Image, device: str = "auto") -> tuple[str, str]:
        """Return ``(tags, prose)`` separately for structured assembly."""
        tags = self._caption(self._tag, image, device)
        prose = self._caption(self._prose, image, device)
        return tags, prose

    def unload(self) -> None:
        self._tag.unload()
        self._prose.unload()

    def is_available(self) -> bool:
        return self._tag.is_available() and self._prose.is_available()

    def availability_reason(self) -> str | None:
        tag_reason = self._tag.availability_reason()
        prose_reason = self._prose.availability_reason()
        if tag_reason and prose_reason:
            return f"Tag backend: {tag_reason}; Prose backend: {prose_reason}"
        return tag_reason or prose_reason
