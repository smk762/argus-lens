"""WD14 tagger backend — anime-style booru tags via ONNX Runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.backends.output import BackendOutput, Tag
from argus_lens.registry import ModelRegistry, get_default_registry

_HF_REPO = "SmilingWolf/wd-vit-tagger-v3"
_MODEL_FILENAME = "wd-vit-tagger-v3.onnx"
_TAGS_FILENAME = "selected_tags.csv"
# Other SmilingWolf v3 taggers share the model.onnx + selected_tags.csv layout
# and can be selected with ``model_repo`` / ``WD14_MODEL_REPO``. EVA02-Large
# is the most accurate of the family (and the slowest).
WD14_V3_REPOS: tuple[str, ...] = (
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-eva02-large-tagger-v3",
)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wd14_tagger"

# In selected_tags.csv, rating tags (general/sensitive/questionable/explicit)
# are marked with category 9. They are excluded from training captions.
_RATING_CATEGORY = 9
# WD-ViT-v3 expects 448px square input; read from the model when possible.
_DEFAULT_IMAGE_SIZE = 448


def _model_filename(repo: str) -> str:
    """Local ONNX filename for *repo* (``wd-vit-tagger-v3.onnx`` for the default)."""
    return f"{repo.rstrip('/').rsplit('/', 1)[-1]}.onnx"


def _download_model(dest_dir: Path, repo: str = _HF_REPO) -> None:
    """Download WD14 model files for *repo* from HuggingFace if not present."""
    from argus_lens.backends._download import stream_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{repo}/resolve/main"
    remote_files = {
        _model_filename(repo): f"{base_url}/model.onnx",
        _TAGS_FILENAME: f"{base_url}/selected_tags.csv",
    }

    for filename, url in remote_files.items():
        dest = dest_dir / filename
        if not dest.exists():
            stream_download(url, dest, event="wd14.download")


class WD14Backend(LocalBackend):
    """WD14 v3 tagger producing comma-separated booru tags.

    Model files (``<repo-name>.onnx`` + ``selected_tags.csv``) are
    auto-downloaded from HuggingFace on first use if not found locally.
    *model_repo* (or ``WD14_MODEL_REPO``) picks the tagger; the default is
    ``SmilingWolf/wd-vit-tagger-v3`` (see ``WD14_V3_REPOS``).

    Search order for model directory:
    1. *model_dir* constructor argument
    2. ``WD14_MODEL_DIR`` environment variable
    3. ``~/.cache/wd14_tagger/`` for the default model, or
       ``~/.cache/wd14_tagger/<repo-name>/`` for any other (auto-downloads
       here if needed; separate dirs keep each model's tag CSV paired)
    """

    name = "wd14"
    style = "anime"
    emits_tags = True
    requires_gpu = False

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float = 0.35,
        registry: ModelRegistry | None = None,
        model_repo: str | None = None,
    ) -> None:
        """Configure the model directory, tag confidence threshold, model repo, and model registry."""
        self._model_dir = Path(model_dir) if model_dir else None
        self._repo = model_repo or os.environ.get("WD14_MODEL_REPO") or _HF_REPO
        self._model_filename = _model_filename(self._repo)
        self.threshold = threshold
        self._registry = registry or get_default_registry()

    def _search_dirs(self) -> list[Path]:
        """Return candidate model directories in search-priority order."""
        dirs: list[Path] = []
        if self._model_dir:
            dirs.append(self._model_dir)
        env_dir = os.environ.get("WD14_MODEL_DIR")
        if env_dir:
            dirs.append(Path(env_dir))
        dirs.append(self._default_cache_dir())
        return dirs

    def _default_cache_dir(self) -> Path:
        """Auto-download location: the shared cache root, or a per-repo subdir."""
        if self._repo == _HF_REPO:
            return _DEFAULT_CACHE_DIR
        return _DEFAULT_CACHE_DIR / self._model_filename.removesuffix(".onnx")

    def _find_model(self) -> Path | None:
        """Return the first existing model file among the search dirs, or None."""
        for d in self._search_dirs():
            if (d / self._model_filename).exists():
                return d / self._model_filename
        return None

    def _ensure_model(self) -> Path:
        """Find model on disk, downloading to the default cache dir if missing."""
        model_path = self._find_model()
        if model_path is not None:
            return model_path

        target_dir = self._model_dir or self._default_cache_dir()
        # The model and tag CSV must stay a matched pair: when the model is
        # missing (fresh install or a version bump renamed it), drop any CSV
        # left over from a previous model so both are re-downloaded together.
        stale_csv = target_dir / _TAGS_FILENAME
        if stale_csv.exists():
            stale_csv.unlink()
        _download_model(target_dir, self._repo)

        model_path = target_dir / self._model_filename
        if not model_path.exists():
            raise RuntimeError(f"WD14 download failed: {model_path} not found after download")
        return model_path

    def is_available(self) -> bool:
        """Return True if onnxruntime and numpy are importable."""
        try:
            __import__("onnxruntime")
            __import__("numpy")
        except ImportError:
            return False
        return True

    def availability_reason(self) -> str | None:
        """Name the missing package (onnxruntime or numpy), or None if usable."""
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

    @staticmethod
    def _device_key(device: str) -> str:
        """Coarse cache key (``cpu`` / ``gpu``) for a device intent.

        Derived from the device string alone — no onnxruntime import — so the
        inference entrypoint stays import-light. GPU-targeting intents
        (``auto`` / ``cuda``) collapse to one cached session.
        """
        return "cpu" if device.startswith("cpu") else "gpu"

    def _cache_key(self, device: str) -> str:
        """Registry key for the cached session: device intent + model source.

        Provider-coarse and import-light (the actual ONNX provider selection
        happens lazily inside ``_loader``), but includes the configured model
        directory so two backends pointing at different models never share
        one cached session.
        """
        model_key = str(self._model_dir) if self._model_dir else os.environ.get("WD14_MODEL_DIR") or "default"
        if self._repo != _HF_REPO:
            model_key = f"{model_key}:{self._repo}"
        return f"wd14:{self._device_key(device)}:{model_key}"

    def _loader(self, device: str) -> tuple[Any, list[tuple[str, int]], str]:
        """Create the ONNX session and load the tag vocabulary from the CSV."""
        import csv

        import onnxruntime as ort

        model_path = self._ensure_model()
        tags_path = model_path.parent / _TAGS_FILENAME
        if not tags_path.exists():
            raise RuntimeError(f"WD14 tags CSV not found at {tags_path}")

        with tags_path.open(newline="", encoding="utf-8") as fh:
            tags = [(row["name"], int(row["category"])) for row in csv.DictReader(fh)]

        session = ort.InferenceSession(str(model_path), providers=self._select_providers(device))
        input_name = session.get_inputs()[0].name
        return session, tags, input_name

    @staticmethod
    def _input_size(session: Any) -> int:
        """Read the square input size (H/W) from the ONNX session."""
        shape = session.get_inputs()[0].shape  # NHWC, e.g. [batch, 448, 448, 3]
        for dim in shape[1:3]:
            if isinstance(dim, int) and dim > 0:
                return dim
        return _DEFAULT_IMAGE_SIZE

    @staticmethod
    def _preprocess(image: Image.Image, size: int) -> Any:
        """Pad to square (white), resize, and convert to BGR float32 [0,255].

        Matches the SmilingWolf WD-v3 preprocessing.
        """
        import numpy as np

        img = image.convert("RGB")
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        canvas = canvas.resize((size, size), Image.BICUBIC)
        arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(np.expand_dims(arr, 0))

    def annotate_image(self, image: Image.Image) -> BackendOutput:
        """Return the tags above the threshold with their scores, excluding rating tags (#57)."""
        tags = [Tag(label=name, score=round(score, 4), source=self.name) for name, score in self._scored_tags(image)]
        return BackendOutput(tags=tags)

    def caption_image(self, image: Image.Image) -> str:
        """Return comma-separated booru tags above the threshold, excluding rating tags."""
        return ", ".join(name for name, _score in self._scored_tags(image))

    def _scored_tags(self, image: Image.Image) -> list[tuple[str, float]]:
        """Run the tagger: ``(tag, probability)`` above the threshold, rating tags excluded, in vocab order."""
        device = self._device
        cache_key = self._cache_key(device)
        with self._acquire(cache_key, lambda: self._loader(device)) as (session, tags, input_name):
            img_np = self._preprocess(image, self._input_size(session))
            probs = session.run(None, {input_name: img_np})[0][0]
            if len(probs) != len(tags):
                raise RuntimeError(
                    f"WD14 model output ({len(probs)}) does not match tag count ({len(tags)}); "
                    "the model and selected_tags.csv are out of sync"
                )
            return [
                (name, float(prob))
                for (name, category), prob in zip(tags, probs, strict=True)
                if prob > self.threshold and category != _RATING_CATEGORY
            ]

    def prefetch(self) -> None:
        """Download the ONNX model + tag CSV if missing, outside the GPU lease (#64)."""
        self._ensure_model()
