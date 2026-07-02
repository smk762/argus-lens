"""HuggingFace Inference API backend — many models, free tier available."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from argus_lens.backends.base import CloudBackend


class HFInferenceBackend(CloudBackend):
    """HuggingFace Inference API backend.

    Supports any model deployed on the HF Inference API, including
    BLIP-2, Florence-2, and community-hosted VLMs.
    """

    name = "hf-inference"
    style = "photo"
    env_var = "HF_TOKEN"
    estimated_cost_per_image = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store credentials and defer HTTP client creation to :meth:`load`."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """HF model to query, defaulting to ``Salesforce/blip2-opt-2.7b``."""
        return self._model_id or "Salesforce/blip2-opt-2.7b"

    def load(self, device: str = "auto") -> None:
        """Create an authenticated httpx client for the HF Inference API."""
        import httpx

        self._client = httpx.Client(
            base_url="https://api-inference.huggingface.co",
            headers={"Authorization": f"Bearer {self.resolve_api_key()}"},
            timeout=120.0,
        )

    def caption_image(self, image: Image.Image) -> str:
        """POST the image as JPEG to the model endpoint and return the generated text."""
        if self._client is None:
            self.load()

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        response = self._client.post(
            f"/models/{self.model_id}",
            content=img_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return item.get("generated_text", "").strip()
            return str(item).strip()
        if isinstance(data, dict):
            return data.get("generated_text", "").strip()
        return str(data).strip()

    def unload(self) -> None:
        """Close the HTTP client and drop the reference."""
        if self._client is not None:
            self._client.close()
            self._client = None
