"""Caption rewriter over any OpenAI-compatible chat endpoint.

Works with Ollama, vLLM, LM Studio, llama.cpp server, or a hosted gateway.
A text-only instruct model is enough (it reads the tags and prose); pass
``send_image=True`` to the :class:`~argus_lens.refine.Refiner` with a vision
model to let it look at the pixels too. Requests use ``temperature=0`` and a
fixed ``seed`` so runs are reproducible on servers that honour them.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

import httpx

from argus_lens.backends.openai_compat import _DEFAULT_MAX_SIDE, _extract_content, prepare_image

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.refine.refiner import RefineRequest

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "qwen3:8b"

SYSTEM_PROMPT = (
    "You turn machine-generated image annotations into clean captions for diffusion-model training "
    "and dataset search. For ONE image you receive: tags from an image tagger, a description from a "
    "captioning model, the tags grouped by category, a person count, and draft captions assembled "
    "mechanically from those inputs (the drafts can be choppy or repetitive). Rewrite each requested "
    "caption so it follows its rules and reads naturally.\n"
    "- Use only facts present in the inputs (or visible in the image, if one is attached). Never invent.\n"
    "- When the tags and the description disagree, trust the tags.\n"
    "- Begin every caption with the trigger word, if one is given, followed by a comma.\n"
    '- No markdown, no hedging ("appears to be", "possibly"), no filler ("The image shows").\n'
    'Reply with ONLY a JSON object mapping each requested caption name to its text, e.g. {"training": "..."}.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_rewrite(content: str) -> dict[str, str]:
    """Parse the rewriter's JSON reply, tolerating code fences and chatter around it."""
    match = _JSON_OBJECT.search(content)
    if not match:
        raise ValueError("rewriter reply contained no JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("rewriter reply was not a JSON object")
    return {str(k): v.strip() for k, v in data.items() if isinstance(v, str)}


def build_user_message(request: RefineRequest) -> str:
    """Render a :class:`RefineRequest` as the user turn."""
    payload: dict[str, Any] = {
        "trigger_word": request.trigger_word or None,
        "tags": request.tags,
        "description": request.prose,
        "tags_by_category": request.buckets.get("tags", {}),
        "person_count": request.subjects.get("count"),
        "drafts": request.drafts,
        "captions_to_write": request.rules,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if request.feedback:
        lines = [f"- {name}: {'; '.join(problems)}" for name, problems in request.feedback.items()]
        text += "\n\nYour previous attempt was rejected. Fix these problems:\n" + "\n".join(lines)
    return text


class OpenAICompatRewriter:
    """:class:`~argus_lens.refine.CaptionRewriter` backed by ``/chat/completions``.

    Endpoint, model and key resolve from arguments, then
    ``ARGUS_REFINE_BASE_URL`` / ``ARGUS_REFINE_MODEL`` / ``ARGUS_REFINE_API_KEY``,
    then defaults (Ollama on localhost, ``qwen3:8b``).

    *reasoning_effort* (``ARGUS_REFINE_REASONING_EFFORT``, default ``"none"``)
    turns off "thinking" on hybrid reasoning models such as Qwen3: left on,
    they spend the token budget reasoning and return empty content. A server
    that rejects the parameter (HTTP 400) is retried once without it, and it
    is not sent again. Pass ``""`` to never send it.
    """

    name = "openai-compat"

    def __init__(
        self,
        base_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_tokens: int = 1200,
        timeout: float = 120.0,
        reasoning_effort: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Resolve endpoint settings; the HTTP client is created lazily."""
        self.base_url = (base_url or os.environ.get("ARGUS_REFINE_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.model_id = model_id or os.environ.get("ARGUS_REFINE_MODEL") or _DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ARGUS_REFINE_API_KEY") or None
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        if reasoning_effort is None:
            reasoning_effort = os.environ.get("ARGUS_REFINE_REASONING_EFFORT", "none")
        self.reasoning_effort = reasoning_effort or None
        self._client = client

    def _http(self) -> httpx.Client:
        """Return (creating on first use) the HTTP client."""
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._client

    def rewrite(self, request: RefineRequest, image: Image.Image | None = None) -> dict[str, str]:
        """POST one chat completion and parse the JSON caption map."""
        user: Any = build_user_message(request)
        if image is not None:
            from argus_lens.reconcile.verifiers.openai_compat import encode_data_url

            image = prepare_image(image, _DEFAULT_MAX_SIDE)
            user = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": encode_data_url(image)}}]
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = self._http().post("/chat/completions", json=payload)
        if response.status_code == 400 and "reasoning_effort" in payload:
            # Server doesn't know the parameter: drop it for this and later calls.
            self.reasoning_effort = None
            del payload["reasoning_effort"]
            response = self._http().post("/chat/completions", json=payload)
        response.raise_for_status()
        return parse_rewrite(_extract_content(response.json()))

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
