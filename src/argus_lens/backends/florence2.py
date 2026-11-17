"""Florence-2 backend — structured captions via Microsoft model."""

from __future__ import annotations

import itertools
import os
from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.registry import ModelRegistry, get_default_registry

_NATIVE_MODEL_ID = "florence-community/Florence-2-base"
_LEGACY_MODEL_ID = "microsoft/Florence-2-base"
# Tasks whose post-processed result is plain text. The rest (<OD>, <DENSE_REGION_CAPTION>,
# <OCR_WITH_REGION>, ...) return boxes/regions, which is not a caption.
CAPTION_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>")
# The legacy remote-code weights break on transformers >= 4.52.
_LEGACY_MAX_TRANSFORMERS = (4, 52)


def _version_tuple(version: str) -> tuple[int, int]:
    """``(major, minor)`` of a version string such as ``4.51.3`` or ``5.0.0.dev0`` (unparseable -> (0, 0))."""
    parts = []
    for piece in version.split(".")[:2]:
        digits = "".join(itertools.takewhile(str.isdigit, piece))
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


class Florence2Backend(LocalBackend):
    """Microsoft Florence-2 captioner (base or large).

    By default uses the native ``florence-community/Florence-2-base`` weights
    which are natively supported in transformers (no ``trust_remote_code``).

    Set ``trust_remote_code=True`` (or ``HF_TRUST_REMOTE_CODE=true``) to use
    the legacy ``microsoft/Florence-2-base`` weights which ship custom model
    code and require ``transformers<4.52``.
    """

    name = "florence2"
    style = "photo"
    requires_gpu = True

    def __init__(
        self,
        model_id: str | None = None,
        task: str = "<MORE_DETAILED_CAPTION>",
        trust_remote_code: bool | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        """Pick native or legacy weights based on *trust_remote_code* and set the caption task.

        *task* must be one of :data:`CAPTION_TASKS`; region/detection tasks don't yield a caption.
        """
        if task not in CAPTION_TASKS:
            raise ValueError(f"Florence-2 task {task!r} does not produce a caption; use one of {CAPTION_TASKS}")
        if trust_remote_code is None:
            trust_remote_code = os.environ.get(
                "HF_TRUST_REMOTE_CODE",
                "0",
            ).lower() in ("1", "true", "yes")
        self._trust_remote_code = trust_remote_code

        if model_id is None:
            model_id = _LEGACY_MODEL_ID if trust_remote_code else _NATIVE_MODEL_ID
        self._model_id = model_id
        self._task = task
        self._registry = registry or get_default_registry()

    def _cache_key(self, device: str) -> str:
        """Return the registry cache key for this model/device pair."""
        return f"florence2:{self._model_id}:{device}"

    def _loader(self, device: str) -> tuple[Any, Any, str]:
        """Load the Florence-2 processor and model onto *device* (fp16 on CUDA)."""
        import torch
        from transformers import AutoProcessor

        dtype = torch.float16 if device == "cuda" else torch.float32

        load_kwargs: dict[str, Any] = {}
        if self._trust_remote_code:
            from transformers import AutoModelForCausalLM

            model_cls = AutoModelForCausalLM
            load_kwargs["trust_remote_code"] = True
        else:
            from transformers import Florence2ForConditionalGeneration

            model_cls = Florence2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            self._model_id,
            **({k: v for k, v in load_kwargs.items() if k == "trust_remote_code"}),
        )
        model = model_cls.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            **load_kwargs,
        ).to(device)
        model.eval()
        return processor, model, device

    def caption_image(self, image: Image.Image, device: str | None = None) -> str:
        """Run the configured Florence-2 task and return its post-processed caption text."""
        import torch

        # Canonical device placement flows through ``load(device)`` (#21), so
        # the engine calls this device-free. ``device`` is retained as an
        # optional override for backwards compatibility with direct callers
        # (pre-0.3 API); when omitted, the device remembered by ``load`` is used.
        resolved = self.resolve_device(device)
        cache_key = self._cache_key(resolved)

        with self._acquire(cache_key, lambda: self._loader(resolved)) as (processor, model, dev):
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
                gen_text,
                task=self._task,
                image_size=(pil.width, pil.height),
            )
            return parsed.get(self._task, "").strip()

    def prefetch(self) -> None:
        """Fetch the Florence-2 weights into the HF cache, outside the GPU lease (#64)."""
        from argus_lens.backends._download import prefetch_hf_snapshot

        prefetch_hf_snapshot(self._model_id)

    def is_available(self) -> bool:
        """Return True if torch and a transformers suited to the selected weights are importable."""
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        """Name the missing package, or the transformers mismatch for the selected weights, or None if usable."""
        try:
            __import__("torch")
        except ImportError:
            return "Missing package: torch (pip install argus-lens[torch])"
        try:
            transformers = __import__("transformers")
        except ImportError:
            return "Missing package: transformers (pip install argus-lens[torch])"
        version = str(getattr(transformers, "__version__", "0"))
        if self._trust_remote_code:
            if _version_tuple(version) >= _LEGACY_MAX_TRANSFORMERS:
                return (
                    f"legacy Florence-2 weights ({self._model_id}, trust_remote_code) need transformers<4.52; "
                    f"found {version}. Use the native weights or pin transformers<4.52"
                )
            return None
        try:
            native = getattr(transformers, "Florence2ForConditionalGeneration", None)
        except Exception:  # noqa: BLE001 - transformers' lazy module can fail importing the class
            native = None
        if native is None:
            return (
                f"transformers {version} has no native Florence-2 (Florence2ForConditionalGeneration); "
                "upgrade transformers or set HF_TRUST_REMOTE_CODE=true with transformers<4.52"
            )
        return None
