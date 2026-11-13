"""CLIP ViT-B/16 text embeddings for RAM++ vocabulary packs (#62).

RAM++'s tag embeddings are OpenAI CLIP ViT-B/16 *text* embeddings, L2-normalised per
description (``ram/utils/openset_utils.py::build_openset_llm_label_embedding`` upstream).
This uses the Hugging Face conversion of the same weights through ``transformers``
(in the ``[torch]`` extra) instead of upstream's ``git+https://github.com/openai/CLIP``
dependency. Long descriptions are truncated to CLIP's 77-token context, as upstream does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from argus_lens.backends._ram.packs import CLIP_MODEL


class ClipTextEmbedder:
    """Embed text with CLIP ViT-B/16's text tower + projection, normalised to unit length."""

    def __init__(self, model_id: str = CLIP_MODEL, device: str = "auto", batch_size: int = 64) -> None:
        """Remember the model and device; weights load on first :meth:`embed`."""
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self._model = CLIPTextModelWithProjection.from_pretrained(self.model_id).to(self.device).eval()

    def embed(self, texts: Sequence[str]) -> Any:
        """Return a float32 ``(len(texts), 512)`` tensor of unit-length text embeddings (on CPU)."""
        import torch

        if self._model is None:
            self._load()
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                tokens = self._tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
                embeds = self._model(**tokens.to(self.device)).text_embeds.float()
                out.append((embeds / embeds.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out) if out else torch.empty(0, 512)
