"""Generic OpenAI-compatible backend — point at any /v1 chat-completions server.

Works with any service that speaks the OpenAI ``/chat/completions`` wire format
and accepts ``image_url`` data URIs, including:

* **Ollama** (``http://localhost:11434/v1``) with a vision model
  (``qwen3-vl:8b``, ``gemma3``, ``llama3.2-vision``, ``llava``, ...)
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

# Named user prompts. ``prompt`` may be one of these names or literal text.
# "straightforward", "descriptive" and "booru" follow JoyCaption Beta One's
# trained prompt modes but work with any instruction-tuned VLM.
PROMPT_PRESETS: dict[str, str] = {
    "detailed": "Describe this image in detail.",
    "straightforward": (
        "Write a straightforward caption for this image. Begin with the main subject and medium. "
        "Mention pivotal elements (people, objects, scenery) using confident, definite language. "
        "Focus on concrete details like color, shape, texture, and spatial relationships. "
        "Show how elements interact. Omit mood and speculative wording. "
        "Never mention what's absent, resolution, or unobservable details. "
        'Keep the description concise, without starting with "This image is" or similar phrasing.'
    ),
    "descriptive": "Write a long descriptive caption for this image in a formal tone.",
    "booru": "Write a list of Booru-like tags for this image.",
    "count": (
        "How many people are visible in this image, including partially visible people and "
        "reflections? Answer with a single integer and nothing else."
    ),
}
_DEFAULT_PROMPT = "detailed"


def _extract_content(data: Any) -> str:
    """Pull the assistant text out of a chat-completions response body.

    Tolerates the shape variance of OpenAI-compatible servers: missing or
    empty ``choices``, ``null`` content (refusals / content filters), and
    list-form content parts. Raises a clear RuntimeError instead of an opaque
    AttributeError/IndexError/KeyError when no caption text is present.
    """
    try:
        content = data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat-completions response shape: {exc!r}") from exc
    if isinstance(content, list):
        # Some vision servers return content as a list of typed parts.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat-completions response contained no caption text (empty or null content)")
    return content.strip()


class OpenAICompatBackend(CloudBackend):
    """Caption via any OpenAI-compatible ``/chat/completions`` endpoint.

    The endpoint and model are resolved from (in order) constructor arguments,
    environment variables, then defaults:

    * ``base_url`` / ``ARGUS_OPENAI_COMPAT_BASE_URL`` (default Ollama localhost)
    * ``model_id`` / ``ARGUS_OPENAI_COMPAT_MODEL`` (default ``llava``)
    * ``api_key`` / ``ARGUS_OPENAI_COMPAT_API_KEY`` (optional)
    * ``prompt`` / ``ARGUS_OPENAI_COMPAT_PROMPT`` — a ``PROMPT_PRESETS`` name
      or literal user-prompt text (default ``detailed``)
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
        prompt: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Resolve the endpoint from argument, env var, or the Ollama localhost default."""
        super().__init__(api_key=api_key, model_id=model_id, system_prompt=system_prompt, **kwargs)
        self._base_url = base_url or os.environ.get("ARGUS_OPENAI_COMPAT_BASE_URL") or _DEFAULT_BASE_URL
        self._prompt = prompt or os.environ.get("ARGUS_OPENAI_COMPAT_PROMPT") or _DEFAULT_PROMPT
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: Any = None

    @property
    def model_id(self) -> str:
        """Model to query: constructor, ``ARGUS_OPENAI_COMPAT_MODEL``, or ``llava``."""
        return self._model_id or os.environ.get("ARGUS_OPENAI_COMPAT_MODEL") or _DEFAULT_MODEL

    @property
    def user_prompt(self) -> str:
        """The user-turn text sent with each image (preset name resolved)."""
        return PROMPT_PRESETS.get(self._prompt, self._prompt)

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
                        {"type": "text", "text": self.user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return _extract_content(response.json())

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
