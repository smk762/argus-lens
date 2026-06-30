"""ArgusLens — the main entry point for structured image captioning."""

from __future__ import annotations

import hashlib
import io
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from argus_lens.assembly.composer import compose_caption_result
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.retry import clear_gpu_cache, run_with_oom_retry
from argus_lens.types import (
    CaptionResult,
    CategoryConfig,
    resolve_target_profile,
)

logger = structlog.get_logger()

# Backend string -> constructor lookup
_BACKEND_REGISTRY: dict[str, type] = {}


def _register_backends() -> None:
    """Lazily populate the backend registry from built-in modules."""
    if _BACKEND_REGISTRY:
        return
    from argus_lens.backends.blip2 import BLIP2Backend
    from argus_lens.backends.florence2 import Florence2Backend
    from argus_lens.backends.hf_inference import HFInferenceBackend
    from argus_lens.backends.nvidia_nim import NVIDIANIMBackend
    from argus_lens.backends.openai import OpenAIBackend
    from argus_lens.backends.replicate import ReplicateBackend
    from argus_lens.backends.wd14 import WD14Backend

    _BACKEND_REGISTRY.update(
        {
            "wd14": WD14Backend,
            "blip2": BLIP2Backend,
            "florence2": Florence2Backend,
            "openai": OpenAIBackend,
            "hf-inference": HFInferenceBackend,
            "replicate": ReplicateBackend,
            "nvidia-nim": NVIDIANIMBackend,
        }
    )


def _resolve_backend(
    backend: str | CaptionBackend,
    **kwargs: Any,
) -> CaptionBackend:
    """Resolve a backend string or instance into a ``CaptionBackend``."""
    if isinstance(backend, CaptionBackend):
        return backend

    _register_backends()

    # Hybrid shorthand: "hybrid:wd14+openai"
    if backend.startswith("hybrid:"):
        parts = backend[7:].split("+", 1)
        if len(parts) != 2:
            raise ValueError(f"Hybrid backend format: 'hybrid:tag+prose' (got {backend!r})")
        tag_name, prose_name = parts
        tag = _resolve_backend(tag_name.strip(), **kwargs)
        prose = _resolve_backend(prose_name.strip(), **kwargs)
        return HybridPipeline(tag_backend=tag, prose_backend=prose)

    # Default hybrid = wd14 + florence2
    if backend == "hybrid":
        from argus_lens.backends.florence2 import Florence2Backend
        from argus_lens.backends.wd14 import WD14Backend

        wd14_kwargs = {}
        florence_kwargs = {}
        if "model_dir" in kwargs:
            wd14_kwargs["model_dir"] = kwargs["model_dir"]
        if "florence_model_id" in kwargs:
            florence_kwargs["model_id"] = kwargs["florence_model_id"]
        return HybridPipeline(
            tag_backend=WD14Backend(**wd14_kwargs),
            prose_backend=Florence2Backend(**florence_kwargs),
        )

    if backend not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()) + ["hybrid"])
        raise ValueError(f"Unknown backend {backend!r}. Choose from: {available}")

    cls = _BACKEND_REGISTRY[backend]
    ctor_kwargs: dict[str, Any] = {}
    for key in ("api_key", "model_id", "system_prompt", "model_dir", "florence_model_id", "threshold", "base_url"):
        if key in kwargs:
            mapped = key
            if key == "florence_model_id":
                mapped = "model_id"
            ctor_kwargs[mapped] = kwargs[key]
    return cls(**ctor_kwargs)


def _image_hash(data: bytes) -> str:
    """Short SHA-256 fingerprint for deduplication."""
    return hashlib.sha256(data).hexdigest()[:20]


def _load_image(source: str | Path | bytes | Image.Image) -> tuple[str, Image.Image]:
    """Load an image from various source types.

    Accepts PIL images, raw bytes, file paths, or ``http(s)://`` URLs.
    Returns ``(name, pil_image)``.
    """
    if isinstance(source, Image.Image):
        return "image", source.convert("RGB")
    if isinstance(source, bytes):
        return "bytes", Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        import httpx

        resp = httpx.get(source, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        name = source.rsplit("/", 1)[-1].split("?")[0] or "image"
        return name, Image.open(io.BytesIO(resp.content)).convert("RGB")
    path = Path(source)
    if path.exists():
        return path.name, Image.open(path).convert("RGB")
    raise FileNotFoundError(f"Image not found: {source}")


class ArgusLens:
    """Structured image captioning engine.

    Wraps one or more backends with the assembly pipeline to produce
    structured, variant-aware captions for training and generation.

    Examples::

        engine = ArgusLens(backend="hybrid")
        result = engine.caption("photo.jpg", trigger_word="sks_person")
        print(result.final_caption)
        print(result.caption_variants["training"])

        # Cloud backend, no GPU needed
        engine = ArgusLens(backend="openai", api_key="sk-...")
        result = engine.caption("photo.jpg")
    """

    def __init__(
        self,
        backend: str | CaptionBackend = "hybrid",
        *,
        device: str = "auto",
        categories: tuple[CategoryConfig, ...] | None = None,
        oom_retry_max_wait_s: float = 180.0,
        oom_retry_interval_s: float = 5.0,
        **kwargs: Any,
    ) -> None:
        self._backend = _resolve_backend(backend, **kwargs)
        self._device = device
        self._categories = categories
        self._oom_retry_max_wait_s = oom_retry_max_wait_s
        self._oom_retry_interval_s = oom_retry_interval_s
        self._loaded = False
        self._load_lock = threading.Lock()
        self._kwargs = kwargs

    @property
    def backend(self) -> CaptionBackend:
        return self._backend

    def _ensure_loaded(self) -> None:
        """Configure the backend device once, lazily, before first inference.

        Device placement flows through ``load(device)`` (#21): the backend
        records the engine's configured device and uses it for subsequent
        (lazy) model loads. ``caption_image`` itself stays device-free.

        Thread-safe: a single engine may be shared across request threads
        (e.g. the server's per-model engine pool), so the check-and-set is
        guarded to call ``load()`` exactly once. Double-checked locking keeps
        the common (already-loaded) path lock-free.
        """
        if self._loaded:
            return
        with self._load_lock:
            if not self._loaded:
                self._backend.load(self._device)
                self._loaded = True

    def _infer(self, pil: Image.Image) -> tuple[str, str]:
        """Run backend inference, returning ``(tags, prose)``.

        Retries on CUDA OOM with cache cleanup (#9). The OOM wait budget is
        bounded by ``oom_retry_max_wait_s``; set it to ``0`` to fail fast.
        """
        self._ensure_loaded()

        def _call() -> tuple[str, str]:
            if isinstance(self._backend, HybridPipeline):
                return self._backend.caption_image_split(pil)
            raw = self._backend.caption_image(pil)
            if self._backend.style == "anime" or self._backend.name == "wd14":
                return raw, ""
            return "", raw

        def _on_oom(exc: Exception, attempt: int) -> None:
            logger.warning("backend_oom", backend=self._backend.name, attempt=attempt, error=str(exc))

        def _on_retry(wait_s: float, attempt: int) -> None:
            logger.info("backend_oom_retry", backend=self._backend.name, attempt=attempt, wait_s=round(wait_s, 1))

        tags, prose = run_with_oom_retry(  # type: ignore[misc]
            _call,
            max_wait_s=self._oom_retry_max_wait_s,
            interval_s=self._oom_retry_interval_s,
            cleanup_fn=clear_gpu_cache,
            on_oom=_on_oom,
            on_retry=_on_retry,
        )
        return tags, prose

    def caption(
        self,
        image: str | Path | bytes | Image.Image,
        *,
        trigger_word: str = "",
        target_style: str = "photo",
        target_category: str = "identity",
        target_backend: str | None = "sdxl",
        checkpoint: str | None = None,
        token_budget_override: int | None = None,
        prose_enrichment: bool = True,
    ) -> CaptionResult:
        """Caption a single image.

        Accepts file paths, raw bytes, PIL Images, or ``http(s)://`` URLs.

        When *prose_enrichment* is enabled (default), novel scene tokens
        extracted from prose output (e.g. Florence-2) are appended to the
        training variant at lowest priority.
        """
        name, pil = _load_image(image)
        profile = resolve_target_profile(
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            token_budget_override=token_budget_override,
            categories=self._categories,
        )

        tags, prose = self._infer(pil)

        return compose_caption_result(
            trigger_word=trigger_word,
            tags=tags,
            prose=prose,
            target_profile=profile,
            prose_enrichment=prose_enrichment,
            categories=self._categories,
            backend_name=self._backend.name,
        )

    def caption_batch(
        self,
        images: list[str | Path | bytes | Image.Image],
        *,
        trigger_word: str = "",
        target_style: str = "photo",
        target_category: str = "identity",
        target_backend: str | None = "sdxl",
        checkpoint: str | None = None,
        token_budget_override: int | None = None,
        progress: Any | None = None,
    ) -> dict[str, CaptionResult]:
        """Caption multiple images, returning ``{name: CaptionResult}``.

        Identical images are deduplicated by hash.
        """
        profile = resolve_target_profile(
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            token_budget_override=token_budget_override,
            categories=self._categories,
        )

        loaded = [_load_image(img) for img in images]
        total = len(loaded)
        results: dict[str, CaptionResult] = {}
        caption_cache: dict[str, tuple[str, str]] = {}

        for idx, (name, pil) in enumerate(loaded):
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            h = _image_hash(buf.getvalue())

            if h not in caption_cache:
                caption_cache[h] = self._infer(pil)

            cached_tags, cached_prose = caption_cache[h]
            results[name] = compose_caption_result(
                trigger_word=trigger_word,
                tags=cached_tags,
                prose=cached_prose,
                target_profile=profile,
                image_index=idx,
                categories=self._categories,
                backend_name=self._backend.name,
            )

            if progress is not None:
                progress(idx + 1, total, name, results[name])

        return results

    def caption_stream(
        self,
        images: list[str | Path | bytes | Image.Image],
        *,
        trigger_word: str = "",
        target_style: str = "photo",
        target_category: str = "identity",
        target_backend: str | None = "sdxl",
        checkpoint: str | None = None,
        token_budget_override: int | None = None,
    ) -> Generator[tuple[str, CaptionResult], None, None]:
        """Yield ``(name, CaptionResult)`` as each image is processed."""
        profile = resolve_target_profile(
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            token_budget_override=token_budget_override,
            categories=self._categories,
        )

        for idx, source in enumerate(images):
            name, pil = _load_image(source)
            tags, prose = self._infer(pil)

            result = compose_caption_result(
                trigger_word=trigger_word,
                tags=tags,
                prose=prose,
                target_profile=profile,
                image_index=idx,
                categories=self._categories,
                backend_name=self._backend.name,
            )
            yield name, result

    def caption_directory(
        self,
        path: str | Path,
        *,
        glob: str = "*.{png,jpg,jpeg,webp}",
        trigger_word: str = "",
        target_style: str = "photo",
        target_category: str = "identity",
        target_backend: str | None = "sdxl",
        checkpoint: str | None = None,
        token_budget_override: int | None = None,
        output_format: str = "txt",
        overwrite: bool = False,
        progress: Any | None = None,
    ) -> dict[str, CaptionResult]:
        """Caption all images in a directory and export results.

        Supported output formats: ``"txt"`` (sidecar files), ``"json"``,
        ``"jsonl"``, ``"csv"``.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        image_paths: list[Path] = []
        for pattern in glob.replace("{", "").replace("}", "").split(","):
            pattern = pattern.strip()
            if pattern:
                image_paths.extend(directory.glob(f"*.{pattern}" if not pattern.startswith("*") else pattern))

        image_paths = sorted(set(image_paths))

        if not overwrite and output_format == "txt":
            image_paths = [p for p in image_paths if not p.with_suffix(".txt").exists()]

        if not image_paths:
            return {}

        results = self.caption_batch(
            images=image_paths,
            trigger_word=trigger_word,
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            token_budget_override=token_budget_override,
            progress=progress,
        )

        from argus_lens.exporters import export_results

        export_results(results, directory, output_format)
        return results

    def available_backends(self) -> dict[str, dict[str, Any]]:
        """Return info about all registered backends and their availability."""
        _register_backends()
        result: dict[str, dict[str, Any]] = {}
        for name, cls in _BACKEND_REGISTRY.items():
            try:
                instance = cls()
                result[name] = {
                    "name": name,
                    "kind": instance.kind.value,
                    "style": instance.style,
                    "requires_gpu": instance.requires_gpu,
                    "available": instance.is_available(),
                    "reason": instance.availability_reason(),
                }
            except Exception as exc:
                result[name] = {
                    "name": name,
                    "available": False,
                    "reason": str(exc),
                }
        result["hybrid"] = {
            "name": "hybrid",
            "kind": "composite",
            "style": "photo",
            "requires_gpu": False,
            "available": True,
            "reason": None,
        }
        return result
