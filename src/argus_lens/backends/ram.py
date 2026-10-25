"""RAM++ (Recognize Anything Plus) photo-domain tagging backend (#3, #60).

WD14 is a Danbooru tagger: the right caption for booru/anime checkpoints, a
poor index for a photo library. RAM++ is the open-vocabulary photo-domain
equivalent: 4585 everyday tags ("beach", "sunset", "wedding dress",
"skateboarder") with calibrated per-tag thresholds. It emits scored tags via
:class:`BackendOutput` and, like WD14, feeds the assembly pipeline's *tag* side
(standalone, or as the tag stage of ``hybrid:ram+<prose>``).

The model runs from a vendored, inference-only port of the upstream code
(:mod:`argus_lens.backends._ram`) that needs only torch: no timm,
transformers, torchvision or fairscale. Weights
(``ram_plus_swin_large_14m.pth``, ~3 GB) are auto-downloaded from
``xinyu1205/recognize-anything-plus-model`` on first use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.backends.output import BackendOutput, Tag
from argus_lens.registry import ModelRegistry, get_default_registry

logger = structlog.get_logger()

DEFAULT_MODEL_ID = "xinyu1205/recognize-anything-plus-model"
CHECKPOINT_FILENAME = "ram_plus_swin_large_14m.pth"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ram_plus"


def _download_checkpoint(dest: Path, repo: str) -> None:
    """Stream the RAM++ checkpoint from HuggingFace to *dest* (atomic via a ``.part`` file)."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{CHECKPOINT_FILENAME}"
    logger.info("ram.download", file=CHECKPOINT_FILENAME, dest=str(dest))
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=4 * 1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    logger.debug("ram.download.progress", pct=downloaded * 100 // total)
    tmp.rename(dest)
    logger.info("ram.download.done", size_mb=round(dest.stat().st_size / 1024 / 1024, 1))


class RamBackend(LocalBackend):
    """Open-vocabulary photo-domain tagger backed by RAM++ (Swin-L, 384px).

    Tags are kept when their probability clears RAM++'s calibrated per-tag
    threshold (mostly 0.65, tuned upstream per class between 0.45 and 1.0). Pass *threshold* to apply
    one uniform cut-off instead: lower keeps more tags, higher fewer.

    Search order for the checkpoint:
    1. *model_dir* constructor argument
    2. ``RAM_MODEL_DIR`` environment variable
    3. ``~/.cache/ram_plus/`` (auto-downloads here if needed)
    """

    name = "ram"
    style = "photo"
    emits_tags = True
    requires_gpu = True

    def __init__(
        self,
        model_id: str | None = None,
        threshold: float | None = None,
        model_dir: str | Path | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        """Configure the HF repo, an optional uniform score threshold, the model directory, and the registry."""
        self._model_id = model_id or DEFAULT_MODEL_ID
        self._threshold = threshold
        self._model_dir = Path(model_dir) if model_dir else None
        self._registry = registry or get_default_registry()

    # -- availability -----------------------------------------------------

    def is_available(self) -> bool:
        """Return True if torch is importable (weights download on first use)."""
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        """Name the missing package, or None if usable."""
        try:
            __import__("torch")
        except ImportError:
            return "Missing package: torch (pip install argus-lens[torch])"
        return None

    # -- weights ----------------------------------------------------------

    def _search_dirs(self) -> list[Path]:
        """Candidate checkpoint directories in priority order."""
        dirs = [self._model_dir] if self._model_dir else []
        if env_dir := os.environ.get("RAM_MODEL_DIR"):
            dirs.append(Path(env_dir))
        dirs.append(_DEFAULT_CACHE_DIR)
        return dirs

    def _ensure_checkpoint(self) -> Path:
        """Find the checkpoint on disk, downloading it to the first search dir if missing."""
        for d in self._search_dirs():
            if (d / CHECKPOINT_FILENAME).exists():
                return d / CHECKPOINT_FILENAME
        dest = self._search_dirs()[0] / CHECKPOINT_FILENAME
        _download_checkpoint(dest, self._model_id)
        return dest

    # -- model lifecycle --------------------------------------------------

    def _cache_key(self, device: str) -> str:
        """Registry key: device + checkpoint source, so differently-configured backends never share a model."""
        source = str(self._model_dir or os.environ.get("RAM_MODEL_DIR") or self._model_id)
        return f"ram:{device}:{source}"

    def _loader(self, device: str) -> tuple[Any, list[str], list[float], Any]:
        """Build the RAM++ module, load the checkpoint, and move it to *device* (fp16 on CUDA)."""
        import torch

        from argus_lens.backends._ram.model import RamPlus
        from argus_lens.backends._ram.vocab import load_tag_vocabulary, prepare_state_dict, unexpectedly_missing

        tags, thresholds = load_tag_vocabulary()
        model = RamPlus(num_tags=len(tags))
        # weights_only: the checkpoint is a plain tensor dict, so never unpickle arbitrary objects.
        checkpoint = torch.load(self._ensure_checkpoint(), map_location="cpu", weights_only=True)
        result = model.load_state_dict(prepare_state_dict(checkpoint), strict=False)
        if missing := unexpectedly_missing(result.missing_keys):
            raise RuntimeError(f"RAM++ checkpoint is missing {len(missing)} weights (e.g. {missing[:3]})")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = model.to(device=device, dtype=dtype).eval()
        # Round thresholds through fp32 so the comparison matches upstream's tensor `>` exactly.
        return model, tags, torch.tensor(thresholds).tolist(), dtype

    def unload(self) -> None:
        """Do nothing; model lifetime is managed by the shared registry."""
        pass

    # -- inference --------------------------------------------------------

    def annotate_image(self, image: Image.Image) -> BackendOutput:
        """Run RAM++ and return the tags that clear their threshold, highest score first."""
        from argus_lens.backends._ram.model import preprocess
        from argus_lens.backends._ram.vocab import select_tags

        device = self.resolve_device()
        with self._registry.acquire(self._cache_key(device), lambda: self._loader(device)) as loaded:
            model, tags, thresholds, dtype = loaded
            probs = model(preprocess(image, device, dtype))[0].float().cpu().tolist()
        return self._build_output([(tags[i], p) for i, p in select_tags(probs, thresholds, self._threshold)])

    def caption_image(self, image: Image.Image) -> str:
        """Backward-compatible shim: the kept tags as a comma-separated string."""
        return self.annotate_image(image).tag_string()

    def _build_output(self, labels_with_scores: list[tuple[str, float]]) -> BackendOutput:
        """Construct a BackendOutput from (label, score) pairs."""
        tags = [Tag(label=label, score=round(score, 4), source=self.name) for label, score in labels_with_scores]
        return BackendOutput(tags=tags)
