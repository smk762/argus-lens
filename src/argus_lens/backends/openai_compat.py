"""Generic OpenAI-compatible backend — point at any /v1 chat-completions server.

Works with any service that speaks the OpenAI ``/chat/completions`` wire format
and accepts ``image_url`` data URIs, including:

* **Ollama** (``http://localhost:11434/v1``) with a vision model (``llava``,
  ``llama3.2-vision``, ``qwen2-vl``, ...)
* **vLLM**, **LM Studio**, **LocalAI**, **llama.cpp** server
* Hosted OpenAI-compatible proxies / gateways

Unlike the ``openai`` backend (official SDK) or ``nvidia-nim`` (fixed NVIDIA
endpoint), this backend has **no required API key** — local servers usually
don't need one — and the endpoint is fully configurable.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from argus_lens.backends.base import CloudBackend

_DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama default
_DEFAULT_MODEL = "llava"


class OpenAICompatBackend(CloudBackend):
    """Caption via any OpenAI-compatible ``/chat/completions`` endpoint.

    The endpoint and model are resolved from (in order) constructor arguments,
    environment variables, then defaults:

    * ``base_url`` / ``ARGUS_OPENAI_COMPAT_BASE_URL`` (default Ollama localhost)
    * ``model_id`` / ``ARGUS_OPENAI_COMPAT_MODEL`` (default ``llava``)
    * ``api_key`` / ``ARGUS_OPENAI_COMPAT_API_KEY`` (optional)
    """

    name = "openai-compat"
    style = "photo"
    env_var = "ARGUS_OPENAI_COMPAT_API_KEY"
    estimated_cost_per_image = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 300,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        """Resolve the endpoint from argument, env var, or the Ollama localhost default."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._base_url = base_url or os.environ.get("ARGUS_OPENAI_COMPAT_BASE_URL") or _DEFAULT_BASE_URL
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """Model to query: constructor, ``ARGUS_OPENAI_COMPAT_MODEL``, or ``llava``."""
        return self._model_id or os.environ.get("ARGUS_OPENAI_COMPAT_MODEL") or _DEFAULT_MODEL

    def _optional_api_key(self) -> str | None:
        """Like ``resolve_api_key`` but returns ``None`` instead of raising.

        Local servers (Ollama, vLLM, LM Studio) typically need no credentials.
        """
        if self._api_key:
            return self._api_key
        return os.environ.get(self.env_var) or None

    def load(self, device: str = "auto") -> None:
        """Create an httpx client for the endpoint, attaching a bearer token only if one is set."""
        import httpx

        headers = {"Content-Type": "application/json"}
        key = self._optional_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        self._client = httpx.Client(base_url=self._base_url, headers=headers, timeout=self._timeout)

    def caption_image(self, image: Image.Image) -> str:
        """Send the image as a base64 data URI to ``/chat/completions`` and return the reply."""
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
            "max_tokens": self._max_tokens,
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

    def is_available(self) -> bool:
        """Always configurable — an endpoint is always resolved (defaults to Ollama).

        Whether the server is actually reachable is only known at call time.
        """
        return True

    def availability_reason(self) -> str | None:
        """Return None; this backend is always considered configurable."""
        return None
