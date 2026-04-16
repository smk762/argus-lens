"""BLIP-2 backend — natural language captions via Salesforce model."""

from __future__ import annotations

from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.registry import ModelRegistry, get_default_registry


class BLIP2Backend(LocalBackend):
    """Salesforce BLIP-2 (blip2-opt-2.7b) natural language captioner."""

    name = "blip2"
    style = "photo"
    requires_gpu = True

    def __init__(
        self,
        model_id: str = "Salesforce/blip2-opt-2.7b",
        registry: ModelRegistry | None = None,
    ) -> None:
        self._model_id = model_id
        self._registry = registry or get_default_registry()

    def _cache_key(self, device: str) -> str:
        return f"blip2:{self._model_id}:{device}"

    def _loader(self, device: str) -> tuple[Any, Any, str]:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = Blip2Processor.from_pretrained(self._model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self._model_id, torch_dtype=dtype,
        ).to(device)
        model.eval()
        return processor, model, device

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image, device: str = "auto") -> str:
        import torch

        resolved = self.resolve_device(device)
        cache_key = self._cache_key(resolved)

        with self._registry.acquire(cache_key, lambda: self._loader(resolved)) as (processor, model, dev):
            dtype = getattr(model, "dtype", None)
            pil = image.convert("RGB")
            inputs = processor(images=pil, return_tensors="pt")
            prepared: dict[str, Any] = {}
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    t = v.to(dev)
                    if dtype is not None and getattr(t, "is_floating_point", lambda: False)():
                        t = t.to(dtype=dtype)
                    prepared[k] = t
                else:
                    prepared[k] = v

            with torch.no_grad():
                out = model.generate(**prepared, max_new_tokens=120)
            return processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self) -> None:
        pass

    def is_available(self) -> bool:
        try:
            __import__("torch")
            __import__("transformers")
        except ImportError:
            return False
        return True

    def availability_reason(self) -> str | None:
        try:
            __import__("torch")
        except ImportError:
            return "Missing package: torch (pip install argus-lens[torch])"
        try:
            __import__("transformers")
        except ImportError:
            return "Missing package: transformers (pip install argus-lens[torch])"
        return None
