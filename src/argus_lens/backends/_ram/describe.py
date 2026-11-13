"""Write the 51 visual descriptions a RAM++ tag needs, with any OpenAI-compatible LLM (#62).

Mirrors upstream's ``generate_tag_des_llm.py`` so pack embeddings match the description
distribution RAM++ was trained against: the template ``"a photo of a {tag}."`` plus 5
prompts × 10 samples (temperature 0.99, ≤77 tokens). Upstream calls ``gpt-3.5-turbo``
with ``n=10``; here each sample is its own request, because many local servers (Ollama,
llama.cpp) ignore ``n``. The endpoint defaults to the same ``ARGUS_REFINE_*`` settings
lens's caption refinement uses, so one text-LLM configuration serves both.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from argus_lens.backends._ram.vocab import DESCRIPTIONS_PER_TAG

UPSTREAM_PROMPTS: tuple[str, ...] = (
    "Describe concisely what a(n) {tag} looks like:",
    "How can you identify a(n) {tag} concisely?",
    "What does a(n) {tag} look like concisely?",
    "What are the identifying characteristics of a(n) {tag}:",
    "Please provide a concise description of the visual characteristics of {tag}:",
)
SAMPLES_PER_PROMPT = 10


def template(tag: str) -> str:
    """The fixed first description upstream gives every tag."""
    return f"a photo of a {tag}."


def fit_descriptions(tag: str, descriptions: Sequence[str], count: int = DESCRIPTIONS_PER_TAG) -> list[str]:
    """Return exactly *count* descriptions for *tag*.

    A full list (``count`` or more, as :meth:`DescriptionWriter.describe` and upstream's JSON
    produce) is used as given, so a supplied set embeds exactly as it would upstream. A short
    list gets the template prepended (unless it already has one, in any case) and is cycled to
    length: RAM++ needs a fixed number per tag, and repeating a description only re-weights it
    in the per-image softmax.
    """
    body = [d.strip() for d in descriptions if d and d.strip()]
    if len(body) >= count:
        return body[:count]
    if not any(d.lower() == template(tag).lower() for d in body):
        body = [template(tag), *body]
    filled = list(body)
    while len(filled) < count:
        filled.append(body[len(filled) % len(body)])
    return filled[:count]


class DescriptionWriter:
    """Generate visual descriptions for tags over an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        prompts: Sequence[str] = UPSTREAM_PROMPTS,
        samples_per_prompt: int = SAMPLES_PER_PROMPT,
        temperature: float = 0.99,
        max_tokens: int = 77,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Resolve the endpoint (defaults: ``ARGUS_REFINE_BASE_URL`` / ``_MODEL`` / ``_API_KEY``)."""
        from argus_lens.refine.openai_compat import _DEFAULT_BASE_URL, _DEFAULT_MODEL

        self.base_url = (base_url or os.environ.get("ARGUS_REFINE_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.model_id = model_id or os.environ.get("ARGUS_REFINE_MODEL") or _DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ARGUS_REFINE_API_KEY") or None
        self.prompts = tuple(prompts)
        self.samples_per_prompt = samples_per_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = client

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._client

    def _sample(self, prompt: str) -> str:
        from argus_lens.backends.openai_compat import _extract_content

        payload: dict[str, Any] = {
            "model": self.model_id,
            # Upstream sends the prompt as an "assistant" turn; a user turn is what chat servers expect.
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._http().post("/chat/completions", json=payload)
        response.raise_for_status()
        return _extract_content(response.json()).strip()

    def describe(self, tag: str) -> list[str]:
        """Return the tag's descriptions: the template plus prompts × samples (51 by default)."""
        out = [template(tag)]
        for prompt in self.prompts:
            text = prompt.format(tag=tag)
            out.extend(s for s in (self._sample(text) for _ in range(self.samples_per_prompt)) if s)
        return out

    def describe_all(
        self, tags: Sequence[str], on_progress: Callable[[int, int, str], None] | None = None
    ) -> dict[str, list[str]]:
        """Describe every tag, reporting ``(done, total, tag)`` after each."""
        result: dict[str, list[str]] = {}
        for i, tag in enumerate(tags, 1):
            result[tag] = self.describe(tag)
            if on_progress:
                on_progress(i, len(tags), tag)
        return result

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
