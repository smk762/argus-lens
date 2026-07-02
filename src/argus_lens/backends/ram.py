"""RAM++ (Recognize Anything) photo-domain tagging backend (issue #3).

WD14 is anime-biased; RAM / RAM++ is the open-vocabulary photo-domain
equivalent and is intended to become the default tagger for the photo-library
(DAM) vertical. This backend emits scored tags via :class:`BackendOutput`.

Status: scaffold. Model loading and inference are not yet implemented.
"""

from __future__ import annotations

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.backends.output import BackendOutput, Tag

DEFAULT_MODEL_ID = "xinyu1205/recognize-anything-plus-model"


class RamBackend(LocalBackend):
    """Open-vocabulary photo-domain tagger backed by RAM++."""

    name = "ram"
    style = "photo"
    requires_gpu = True

    def __init__(self, model_id: str | None = None, threshold: float = 0.35) -> None:
        """Configure the model id and the score threshold applied when flattening tags."""
        self._model_id = model_id or DEFAULT_MODEL_ID
        self._threshold = threshold
        self._model = None

    def is_available(self) -> bool:
        """Return False; the backend is a scaffold pending implementation (#3)."""
        # Scaffold: model loading/inference are not implemented yet, so the
        # backend must not advertise itself as usable once it is registered.
        return False

    def availability_reason(self) -> str | None:
        """Explain that the RAM++ backend is not yet implemented."""
        return "RAM++ backend not yet implemented (#3)"

    def load(self, device: str = "auto") -> None:
        """Raise NotImplementedError; model loading is pending (#3)."""
        raise NotImplementedError("RAM++ model loading is not yet implemented (#3)")

    def annotate_image(self, image: Image.Image) -> BackendOutput:
        """Run RAM++ and return scored tags. To be implemented (#3)."""
        raise NotImplementedError("RAM++ inference is not yet implemented (#3)")

    def caption_image(self, image: Image.Image) -> str:
        """Backward-compatible shim: flatten scored tags to a string."""
        return self.annotate_image(image).tag_string(min_score=self._threshold)

    def unload(self) -> None:
        """Drop the model reference."""
        self._model = None

    def _build_output(self, labels_with_scores: list[tuple[str, float]]) -> BackendOutput:
        """Helper to construct a BackendOutput from (label, score) pairs."""
        tags = [Tag(label=label, score=score, source=self.name) for label, score in labels_with_scores]
        return BackendOutput(tags=tags)
