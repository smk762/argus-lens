"""NVIDIA NIM backend — enterprise vision-language models."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from argus_lens.backends.base import CloudBackend


class NVIDIANIMBackend(CloudBackend):
    """NVIDIA NIM API backend.

    Supports Kosmos-2, Llama 3.2 Vision, Gemma 3, and other models
    available through the NVIDIA NIM inference microservice.
    """

    name = "nvidia-nim"
    style = "photo"
    env_var = "NVIDIA_API_KEY"
    estimated_cost_per_image = 0.002

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        **kwargs: Any,
    ) -> None:
        """Store credentials and endpoint; defer HTTP client creation to :meth:`load`."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._base_url = base_url
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """NIM model to query, defaulting to ``microsoft/kosmos-2``."""
        return self._model_id or "microsoft/kosmos-2"

    def load(self, device: str = "auto") -> None:
        """Create an authenticated httpx client for the NVIDIA NIM endpoint."""
        import httpx

        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.resolve_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def caption_image(self, image: Image.Image) -> str:
        """Send the image as a base64 data URI to NIM chat completions and return the reply."""
        if self._client is None:
            self.load()

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 300,
        }

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def unload(self) -> None:
        """Close the HTTP client and drop the reference."""
        if self._client is not None:
            self._client.close()
            self._client = None
