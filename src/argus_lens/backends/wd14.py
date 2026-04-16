"""WD14 tagger backend — anime-style booru tags via ONNX Runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.registry import ModelRegistry, get_default_registry


class WD14Backend(LocalBackend):
    """WD14 ViT-v2 tagger producing comma-separated booru tags.

    Model files required:
    * ``wd14-vit-v2.onnx``
    * ``selected_tags.csv``

    Search order for model directory:
    1. *model_dir* constructor argument
    2. ``WD14_MODEL_DIR`` environment variable
    3. ``~/.cache/wd14_tagger/``
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
        dirs.append(Path.home() / ".cache" / "wd14_tagger")
        return dirs

    def _find_model(self) -> Path | None:
        for d in self._search_dirs():
            if (d / "wd14-vit-v2.onnx").exists():
                return d / "wd14-vit-v2.onnx"
        return None

    def is_available(self) -> bool:
        try:
            __import__("onnxruntime")
            __import__("numpy")
        except ImportError:
            return False
        return self._find_model() is not None

    def availability_reason(self) -> str | None:
        try:
            __import__("onnxruntime")
        except ImportError:
            return "Missing package: onnxruntime (pip install argus-lens[wd14])"
        try:
            __import__("numpy")
        except ImportError:
            return "Missing package: numpy"
        if self._find_model() is None:
            return "WD14 model files not found (wd14-vit-v2.onnx + selected_tags.csv)"
        return None

    def _loader(self) -> tuple[Any, list[str], str]:
        import csv

        import onnxruntime as ort

        model_path = self._find_model()
        if model_path is None:
            raise RuntimeError("WD14 ONNX model not found")
        tags_path = model_path.parent / "selected_tags.csv"
        if not tags_path.exists():
            raise RuntimeError(f"WD14 tags CSV not found at {tags_path}")

        with tags_path.open(newline="", encoding="utf-8") as fh:
            tag_names = [row["name"] for row in csv.DictReader(fh)]

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        return session, tag_names, input_name

    def load(self, device: str = "auto") -> None:
        pass

    def caption_image(self, image: Image.Image) -> str:
        import numpy as np

        with self._registry.acquire("wd14", self._loader) as (session, tag_names, input_name):
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
