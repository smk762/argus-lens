"""Replicate backend — pay-per-run hosted model inference."""

from __future__ import annotations

import io
from collections.abc import Iterable
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
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """Replicate model to run, defaulting to ``lucataco/florence-2-large``."""
        return self._model_id or "lucataco/florence-2-large"

    def load(self, device: str = "auto") -> None:
        """Import the replicate SDK and build a client authorised with this backend's key.

        The client is per instance, so an explicit ``api_key`` wins over ``REPLICATE_API_TOKEN``
        and the token is never exported to the process environment.
        """
        try:
            import replicate
        except ImportError as exc:
            raise ImportError("pip install argus-lens[replicate]") from exc
        self._client = replicate.Client(api_token=self.resolve_api_key())

    def caption_image(self, image: Image.Image) -> str:
        """Run a Replicate prediction on the JPEG-encoded image and return its text output."""
        if self._client is None:
            self.load()

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        buf.seek(0)

        output = self._client.run(
            self.model_id,
            input={
                "image": buf,
                "prompt": self.system_prompt,
            },
        )

        return _output_text(output)

    def unload(self) -> None:
        """Drop the SDK client."""
        self._client = None


def _output_text(output: Any) -> str:
    """Flatten a Replicate prediction output into caption text.

    Language models stream tokens (a list or iterator) that carry their own spacing, so
    they are joined with ``""``. A dict output yields its text field (a dict without one is
    an error, not a caption of its repr); anything else is stringified.
    """
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        for field in ("text", "caption", "output"):
            if isinstance(output.get(field), str):
                return output[field].strip()
        raise ValueError(f"replicate output has no text field (keys: {sorted(output)})")
    if isinstance(output, Iterable):
        return "".join(str(o) for o in output).strip()
    return str(output).strip()
