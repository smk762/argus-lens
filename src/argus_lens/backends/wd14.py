"""WD14 tagger backend — anime-style booru tags via ONNX Runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.registry import ModelRegistry, get_default_registry

logger = structlog.get_logger()

_HF_REPO = "SmilingWolf/wd-vit-tagger-v2"
_HF_BASE_URL = f"https://huggingface.co/{_HF_REPO}/resolve/main"
_MODEL_FILENAME = "wd14-vit-v2.onnx"
_TAGS_FILENAME = "selected_tags.csv"
_REMOTE_FILES = {
    _MODEL_FILENAME: f"{_HF_BASE_URL}/model.onnx",
    _TAGS_FILENAME: f"{_HF_BASE_URL}/selected_tags.csv",
}
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wd14_tagger"


def _download_model(dest_dir: Path) -> None:
    """Download WD14 model files from HuggingFace if not present."""
    import httpx

    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _REMOTE_FILES.items():
        dest = dest_dir / filename
        if dest.exists():
            continue

        logger.info("wd14.download", file=filename, dest=str(dest))
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        logger.debug("wd14.download.progress", file=filename, pct=pct)
        tmp.rename(dest)
        logger.info("wd14.download.done", file=filename, size_mb=round(dest.stat().st_size / 1024 / 1024, 1))


class WD14Backend(LocalBackend):
    """WD14 ViT-v2 tagger producing comma-separated booru tags.

    Model files (``wd14-vit-v2.onnx`` + ``selected_tags.csv``) are
    auto-downloaded from HuggingFace on first use if not found locally.

    Search order for model directory:
    1. *model_dir* constructor argument
    2. ``WD14_MODEL_DIR`` environment variable
    3. ``~/.cache/wd14_tagger/`` (auto-downloads here if needed)
    """

    name = "wd14"
    style = "anime"
    requires_gpu = False

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float = 0.35,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else None
        self.threshold = threshold
        self._registry = registry or get_default_registry()

    def _search_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        if self._model_dir:
            dirs.append(self._model_dir)
        env_dir = os.environ.get("WD14_MODEL_DIR")
        if env_dir:
            dirs.append(Path(env_dir))
        dirs.append(_DEFAULT_CACHE_DIR)
        return dirs

    def _find_model(self) -> Path | None:
        for d in self._search_dirs():
            if (d / _MODEL_FILENAME).exists():
                return d / _MODEL_FILENAME
        return None

    def _ensure_model(self) -> Path:
        """Find model on disk, downloading to the default cache dir if missing."""
        model_path = self._find_model()
        if model_path is not None:
            return model_path

        target_dir = self._model_dir or _DEFAULT_CACHE_DIR
        _download_model(target_dir)

        model_path = target_dir / _MODEL_FILENAME
        if not model_path.exists():
            raise RuntimeError(f"WD14 download failed: {model_path} not found after download")
        return model_path

    def is_available(self) -> bool:
        try:
            __import__("onnxruntime")
            __import__("numpy")
        except ImportError:
            return False
        return True

    def availability_reason(self) -> str | None:
        try:
            __import__("onnxruntime")
        except ImportError:
            return "Missing package: onnxruntime (pip install argus-lens[wd14])"
        try:
            __import__("numpy")
        except ImportError:
            return "Missing package: numpy"
        return None

    @staticmethod
    def _select_providers(device: str) -> list[str]:
        """Choose ONNX Runtime execution providers for a device intent.

        ONNX Runtime uses providers rather than torch devices. An explicit CPU
        request is honoured; otherwise CUDA is preferred when the runtime
        actually exposes it. Deliberately torch-free so the ``[wd14-gpu]``
        install (onnxruntime-gpu, no torch) still selects CUDA.
        """
        import onnxruntime as ort

        if device.startswith("cpu"):
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _loader(self, providers: list[str]) -> tuple[Any, list[str], str]:
        import csv

        import onnxruntime as ort

        model_path = self._ensure_model()
        tags_path = model_path.parent / _TAGS_FILENAME
        if not tags_path.exists():
            raise RuntimeError(f"WD14 tags CSV not found at {tags_path}")

        with tags_path.open(newline="", encoding="utf-8") as fh:
            tag_names = [row["name"] for row in csv.DictReader(fh)]

        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        return session, tag_names, input_name

    def caption_image(self, image: Image.Image) -> str:
        import numpy as np

        providers = self._select_providers(self._device)
        # Key the cache by the effective provider so device intents that resolve
        # to the same provider (e.g. "auto" and "cuda" on a CUDA box) share one
        # cached session instead of duplicating it.
        cache_key = f"wd14:{providers[0]}"
        with self._registry.acquire(cache_key, lambda: self._loader(providers)) as (session, tag_names, input_name):
            img = image.convert("RGB").resize((448, 448), Image.LANCZOS)
            img_np = np.expand_dims(np.array(img, dtype=np.float32)[:, :, ::-1], 0)
            probs = session.run(None, {input_name: img_np})[0][0]
            tags = ", ".join(
                tag_names[idx]
                for idx, prob in enumerate(probs)
                if prob > self.threshold and not tag_names[idx].startswith("rating:")
            )
            return tags

    def unload(self) -> None:
        pass
