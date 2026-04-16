"""Florence-2 backend — structured captions via Microsoft model."""

from __future__ import annotations

from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.registry import ModelRegistry, get_default_registry


class Florence2Backend(LocalBackend):
    """Microsoft Florence-2 captioner (base or large)."""

    name = "florence2"
    style = "photo"
    requires_gpu = True

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        task: str = "<MORE_DETAILED_CAPTION>",
        registry: ModelRegistry | None = None,
    ) -> None:
        self._model_id = model_id
        self._task = task
        self._registry = registry or get_default_registry()

    def _cache_key(self, device: str) -> str:
        return f"florence2:{self._model_id}:{device}"

    def _loader(self, device: str) -> tuple[Any, Any, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = torch.float16 if device == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_id, trust_remote_code=True, torch_dtype=dtype,
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
            inputs = processor(text=self._task, images=pil, return_tensors="pt")
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
                gen_ids = model.generate(**prepared, max_new_tokens=256, do_sample=False)
            gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                gen_text, task=self._task, image_size=(pil.width, pil.height),
            )
            return parsed.get(self._task, "").strip()

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
