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
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._max_tokens = max_tokens
        self._client: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id or "gpt-4o"

    def load(self, device: str = "auto") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install argus-lens[openai]") from exc
        self._client = OpenAI(api_key=self.resolve_api_key())

    def caption_image(self, image: Image.Image) -> str:
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
        self._client = None

    def is_available(self) -> bool:
        try:
            __import__("openai")
        except ImportError:
            return False
        return super().is_available()

    def availability_reason(self) -> str | None:
        try:
            __import__("openai")
        except ImportError:
            return "Missing package: openai (pip install argus-lens[openai])"
        return super().availability_reason()
