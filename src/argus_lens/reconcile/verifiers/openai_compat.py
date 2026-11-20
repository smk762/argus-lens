"""VQA verifier over any OpenAI-compatible vision endpoint (Ollama, vLLM, ...).

Routes a pointed question ("what colour is the dress?") to a served vision
model and parses the one-word answer. Needs only ``httpx`` (a core dep), so it
runs without torch and reuses infrastructure you already host.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import httpx

from argus_lens.reconcile.questions import answer_budget, build_question, parse_answer
from argus_lens.reconcile.types import Verdict

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def encode_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a ``data:image/png;base64,...`` URL."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chat_completion(
    base_url: str,
    model_id: str,
    image_url: str,
    prompt: str,
    *,
    api_key: str | None = None,
    max_tokens: int = 10,
    timeout: float = 30.0,
) -> str:
    """POST one image + prompt to ``{base_url}/chat/completions``; return the answer text.

    *image_url* is a ``data:`` URL from :func:`encode_data_url`, so a caller asking
    several questions about one image encodes it once. Greedy (``temperature=0``).
    A server that returns no, empty or null content yields ``""``.
    """
    payload = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    choices = resp.json().get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


class OpenAICompatVQAVerifier:
    """Adjudicate disputes by asking a served OpenAI-compatible vision model."""

    name = "openai-compat"
    reads_text = True  # a "transcribe the text" question answers text claims

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model_id: str = "llama3.2-vision",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout

    def _ask(self, image: Image.Image, question: str, max_tokens: int = 10) -> str:
        """POST a single-image chat completion and return the raw answer text."""
        return chat_completion(
            self.base_url,
            self.model_id,
            encode_data_url(image),
            question,
            api_key=self.api_key,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ask the model and map its answer onto the palette/pose vocabulary."""
        answer = self._ask(image, build_question(dispute), answer_budget(dispute.kind))
        return Verdict(subject=dispute.subject, value=parse_answer(answer, dispute.kind), source=self.name)
