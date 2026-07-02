"""OpenAI backend — GPT-4o / GPT-4o-mini vision captioning."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from argus_lens.backends.base import CloudBackend


class OpenAIBackend(CloudBackend):
    """OpenAI vision API backend (GPT-4o, GPT-4o-mini)."""

    name = "openai"
    style = "photo"
    env_var = "OPENAI_API_KEY"
    estimated_cost_per_image = 0.005

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 300,
        **kwargs: Any,
    ) -> None:
        """Store credentials and token budget; defer SDK client creation to :meth:`load`."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._max_tokens = max_tokens
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """OpenAI model to query, defaulting to ``gpt-4o``."""
        return self._model_id or "gpt-4o"

    def load(self, device: str = "auto") -> None:
        """Create the official OpenAI SDK client, raising if the package is missing."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install argus-lens[openai]") from exc
        self._client = OpenAI(api_key=self.resolve_api_key())

    def caption_image(self, image: Image.Image) -> str:
        """Send the image as a high-detail base64 data URI to chat completions and return the reply."""
        if self._client is None:
            self.load()

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                        },
                    ],
                },
            ],
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content.strip()

    def unload(self) -> None:
        """Drop the SDK client reference."""
        self._client = None

    def is_available(self) -> bool:
        """Return True if the openai package is installed and an API key is configured."""
        try:
            __import__("openai")
        except ImportError:
            return False
        return super().is_available()

    def availability_reason(self) -> str | None:
        """Report a missing openai package or missing API key, or None if usable."""
        try:
            __import__("openai")
        except ImportError:
            return "Missing package: openai (pip install argus-lens[openai])"
        return super().availability_reason()
