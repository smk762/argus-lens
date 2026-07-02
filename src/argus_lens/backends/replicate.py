"""Replicate backend — pay-per-run hosted model inference."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from argus_lens.backends.base import CloudBackend


class ReplicateBackend(CloudBackend):
    """Replicate API backend.

    Runs any model with an image-to-text prediction endpoint on Replicate.
    """

    name = "replicate"
    style = "photo"
    env_var = "REPLICATE_API_TOKEN"
    estimated_cost_per_image = 0.003

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store credentials; defer importing the replicate SDK to :meth:`load`."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._replicate: Any = None

    @property
    def model_id(self) -> str:
        """Replicate model to run, defaulting to ``lucataco/florence-2-large``."""
        return self._model_id or "lucataco/florence-2-large"

    def load(self, device: str = "auto") -> None:
        """Import the replicate SDK and export the API token to the environment."""
        try:
            import replicate
        except ImportError as exc:
            raise ImportError("pip install argus-lens[replicate]") from exc
        import os

        os.environ.setdefault("REPLICATE_API_TOKEN", self.resolve_api_key())
        self._replicate = replicate

    def caption_image(self, image: Image.Image) -> str:
        """Run a Replicate prediction on the JPEG-encoded image and return its text output."""
        if self._replicate is None:
            self.load()

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        buf.seek(0)

        output = self._replicate.run(
            self.model_id,
            input={
                "image": buf,
                "prompt": self.system_prompt,
            },
        )

        if isinstance(output, list):
            return " ".join(str(o) for o in output).strip()
        return str(output).strip()

    def unload(self) -> None:
        """Drop the SDK module reference."""
        self._replicate = None
