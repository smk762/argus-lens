"""OpenAI-compatible /v1 API shim for Frigate GenAI integration.

Exposes two endpoints:

    GET  /v1/models                 — list available argus-* model IDs
    POST /v1/chat/completions       — vision captioning in OpenAI request/response format

Configure Frigate's config.yml to use this server:

    genai:
      enabled: true
      provider: openai
      base_url: http://<morpheus-ip>:8080/v1
      model: argus-hybrid     # or argus-florence2 / argus-wd14
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError("OpenAI compat requires: pip install argus-lens[server]") from exc

from PIL import Image

from argus_lens.engine import ArgusLens
from argus_lens.types import CaptionResult

# ── Model ID registry ──────────────────────────────────────────────────────────
# Keys are the strings Frigate sends in the "model" field.
# Values are the argus-lens backend strings passed to ArgusLens().

_MODEL_MAP: dict[str, str] = {
    "argus-wd14": "wd14",
    "argus-florence2": "florence2",
    "argus-hybrid": "hybrid",
}

# ── Content assembly ───────────────────────────────────────────────────────────
# Order matters: set the scene first, then activity, then visual identifiers.
# "identity" is intentionally omitted — surveillance context / privacy.

_BUCKET_ORDER = ("setting", "action", "wardrobe", "pose_composition", "lighting")


def _assemble_content(result: CaptionResult) -> str:
    """Flatten structured caption_variants into a single surveillance-safe string."""
    parts: list[str] = []
    for key in _BUCKET_ORDER:
        val = result.caption_variants.get(key, "").strip()
        if val:
            if not val.endswith((".", "!", "?")):
                val += "."
            parts.append(val)
    # Fall back to final_caption when buckets are empty (e.g. pure WD14 run)
    return " ".join(parts) if parts else result.final_caption


# ── Image extraction ───────────────────────────────────────────────────────────


def _extract_image_bytes(content: list[dict[str, Any]]) -> bytes:
    """Return raw image bytes from an OpenAI-format message content list.

    Handles:
      • ``data:image/*;base64,<b64>``  — Frigate always sends this form
      • ``http://`` / ``https://``     — fallback for direct URL usage
    """
    for item in content:
        if item.get("type") != "image_url":
            continue
        url: str = (item.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            # Strip the MIME header — everything after the comma is base64
            _header, _, b64 = url.partition(",")
            return base64.b64decode(b64)
        if url.startswith(("http://", "https://")):
            import httpx

            resp = httpx.get(url, timeout=15)
            resp.raise_for_status()
            return resp.content
    return b""


# ── Lazy engine pool ───────────────────────────────────────────────────────────


class _EnginePool:
    """Thread-safe per-model-ID engine cache.  Engines are initialised on first use."""

    def __init__(self, **engine_kwargs: Any) -> None:
        """Store kwargs forwarded to every lazily constructed ``ArgusLens`` engine."""
        self._kwargs = engine_kwargs
        self._engines: dict[str, ArgusLens] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str) -> ArgusLens:
        """Return the cached engine for *model_id*, constructing it on first use."""
        if model_id in self._engines:
            return self._engines[model_id]
        with self._lock:
            if model_id not in self._engines:
                backend = _MODEL_MAP[model_id]
                self._engines[model_id] = ArgusLens(backend=backend, **self._kwargs)
        return self._engines[model_id]


# ── Pydantic request model ─────────────────────────────────────────────────────


class _ImageURL(BaseModel):
    """OpenAI ``image_url`` payload: a data URI or http(s) URL."""

    url: str
    detail: str = "auto"


class _ContentPart(BaseModel):
    """One element of a multimodal message content list (text or image_url)."""

    type: str
    image_url: _ImageURL | None = None
    text: str | None = None


class _Message(BaseModel):
    """Chat message whose content is either a plain string or multimodal parts."""

    role: str
    content: list[_ContentPart] | str


class _ChatCompletionRequest(BaseModel):
    """Request body for ``POST /v1/chat/completions`` (OpenAI chat format)."""

    model: str = "argus-hybrid"
    messages: list[_Message]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


# ── Router factory ─────────────────────────────────────────────────────────────


def create_openai_router(**engine_kwargs: Any) -> APIRouter:
    """Return an APIRouter with the OpenAI-compatible /v1 endpoints.

    Mount at ``/v1`` in the parent FastAPI app:

        app.include_router(create_openai_router(), prefix="/v1")

    Any ``engine_kwargs`` (e.g. ``model_dir``, ``florence_model_id``) are
    forwarded to every ArgusLens engine created by this router.
    """
    router = APIRouter(tags=["openai-compat"])
    pool = _EnginePool(**engine_kwargs)

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        """Return registered argus-* model IDs in OpenAI list format."""
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": 0,
                    "owned_by": "argus-lens",
                }
                for mid in _MODEL_MAP
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(req: _ChatCompletionRequest) -> dict[str, Any]:
        """Caption a security camera snapshot and return an OpenAI-shaped response.

        Frigate sends the snapshot as a base64 data URI inside the ``image_url``
        content part of the last user message.  The response ``content`` field
        contains a plain-English description assembled from the structured
        caption buckets (setting → action → wardrobe → pose).
        """
        if req.stream:
            _openai_error(400, "Streaming is not supported", "invalid_request_error", "stream_not_supported")

        if req.model not in _MODEL_MAP:
            _openai_error(
                404,
                f"Model '{req.model}' not found. Available: {list(_MODEL_MAP)}",
                "invalid_request_error",
                "model_not_found",
            )

        # Gather content parts from the last user message (Frigate sends one message)
        raw_parts: list[dict[str, Any]] = []
        for msg in req.messages:
            if isinstance(msg.content, list):
                raw_parts = [part.model_dump() for part in msg.content]

        img_bytes = _extract_image_bytes(raw_parts)
        if not img_bytes:
            _openai_error(422, "No image found in message content", "invalid_request_error", "no_image")

        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            _openai_error(400, f"Invalid image data: {exc}", "invalid_request_error", "invalid_image")

        try:
            engine = pool.get(req.model)
            result: CaptionResult = await asyncio.to_thread(engine.caption, pil)
        except Exception as exc:
            _openai_error(503, f"Caption backend error: {exc}", "server_error", "backend_unavailable")

        content = _assemble_content(result)
        ts = int(time.time())
        return {
            "id": f"argus-cmpl-{ts}",
            "object": "chat.completion",
            "created": ts,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return router


# ── Error helper ───────────────────────────────────────────────────────────────


def _openai_error(
    status: int,
    message: str,
    error_type: str,
    code: str,
    param: str | None = None,
) -> None:
    """Raise an HTTPException whose detail matches the OpenAI error envelope."""
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }
    if param:
        body["error"]["param"] = param
    raise HTTPException(status_code=status, detail=body)
