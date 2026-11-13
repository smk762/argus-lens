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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from argus_lens.backends.base import LocalBackend
from argus_lens.backends.output import BackendOutput, Tag
from argus_lens.registry import ModelRegistry, get_default_registry

DEFAULT_MODEL_ID = "xinyu1205/recognize-anything-plus-model"
CHECKPOINT_FILENAME = "ram_plus_swin_large_14m.pth"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ram_plus"


def _download_checkpoint(dest: Path, repo: str) -> None:
    """Stream the RAM++ checkpoint from HuggingFace *repo* to *dest*, atomically."""
    from argus_lens.backends._download import stream_download

    url = f"https://huggingface.co/{repo}/resolve/main/{CHECKPOINT_FILENAME}"
    stream_download(url, dest, event="ram.download", chunk_size=4 * 1024 * 1024)


class RamBackend(LocalBackend):
    """Open-vocabulary photo-domain tagger backed by RAM++ (Swin-L, 384px).

    Tags are kept when their probability clears RAM++'s calibrated per-tag
    threshold (mostly 0.65, tuned upstream per class between 0.45 and 1.0). Pass *threshold* to apply
    one uniform cut-off instead: lower keeps more tags, higher fewer.

    *repo_id* is the HuggingFace repo the checkpoint downloads from. It is not
    called ``model_id`` because the engine forwards ``model_id`` to every stage
    of a hybrid, where it names the prose model (``hybrid:ram+openai-compat
    --model-id qwen3-vl:8b-instruct``).

    Search order for the checkpoint:
    1. *model_dir* constructor argument
    2. ``RAM_MODEL_DIR`` environment variable
    3. ``~/.cache/ram_plus/`` for the default repo, ``~/.cache/ram_plus/<org>--<name>/``
       for any other

    When none has it, it is downloaded into the first of these.

    *vocab_packs* (or ``RAM_VOCAB_PACKS``, ``os.pathsep``-separated) adds custom tags from
    vocabulary packs built with ``argus-lens ram-vocab build`` (#62). Pack tags extend the
    built-in 4585; a pack tag that duplicates a built-in one replaces it, and a tag in two
    packs is an error. ``builtin_vocab=False`` tags with the packs alone (upstream's
    open-set mode). Pack tags carry ``source="ram:<pack name>"``.
    """

    name = "ram"
    style = "photo"
    emits_tags = True
    requires_gpu = True

    def __init__(
        self,
        repo_id: str | None = None,
        threshold: float | None = None,
        model_dir: str | Path | None = None,
        registry: ModelRegistry | None = None,
        vocab_packs: Sequence[str | Path] | None = None,
        builtin_vocab: bool = True,
    ) -> None:
        """Configure the HF repo, score threshold, model directory, registry, and vocabulary packs."""
        self._repo_id = repo_id or DEFAULT_MODEL_ID
        self._threshold = threshold
        self._model_dir = Path(model_dir) if model_dir else None
        self._registry = registry or get_default_registry()
        if vocab_packs is None:
            vocab_packs = [p for p in os.environ.get("RAM_VOCAB_PACKS", "").split(os.pathsep) if p.strip()]
        self._vocab_packs = [Path(p).expanduser().resolve() for p in vocab_packs]
        self._builtin_vocab = builtin_vocab

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
        dirs.append(self._default_cache_dir())
        return dirs

    def _default_cache_dir(self) -> Path:
        """Auto-download location: the cache root for the default repo, a per-repo subdir otherwise."""
        if self._repo_id == DEFAULT_MODEL_ID:
            return _DEFAULT_CACHE_DIR
        return _DEFAULT_CACHE_DIR / self._repo_id.strip("/").replace("/", "--")

    def _ensure_checkpoint(self) -> Path:
        """Find the checkpoint on disk, downloading it to the first search dir if missing."""
        dirs = self._search_dirs()
        for d in dirs:
            if (d / CHECKPOINT_FILENAME).exists():
                return d / CHECKPOINT_FILENAME
        dest = dirs[0] / CHECKPOINT_FILENAME
        _download_checkpoint(dest, self._repo_id)
        return dest

    # -- model lifecycle --------------------------------------------------

    def _cache_key(self, device: str) -> str:
        """Registry key: device + checkpoint source, so differently-configured backends never share a model."""
        source = str(self._model_dir or os.environ.get("RAM_MODEL_DIR") or self._repo_id)
        vocab = ("" if self._builtin_vocab else "packs-only") + "".join(f"+{p}" for p in self._vocab_packs)
        return f"ram:{device}:{source}{':' + vocab if vocab else ''}"

    def _loader(self, device: str) -> tuple[Any, list[str], list[float], Any, list[str]]:
        """Build the RAM++ module, load the checkpoint, and move it to *device* (fp16 on CUDA)."""
        import torch

        from argus_lens.backends._ram.model import RamPlus
        from argus_lens.backends._ram.vocab import load_tag_vocabulary, prepare_state_dict, unexpectedly_missing

        builtin_tags, builtin_thresholds = load_tag_vocabulary()
        model = RamPlus(num_tags=len(builtin_tags))
        # weights_only: the checkpoint is a plain tensor dict, so never unpickle arbitrary objects.
        # mmap: page the ~3 GB file in as load_state_dict copies it, rather than holding a full
        # in-memory copy next to the model's own fp32 weights.
        checkpoint = torch.load(self._ensure_checkpoint(), map_location="cpu", weights_only=True, mmap=True)
        result = model.load_state_dict(prepare_state_dict(checkpoint), strict=False)
        del checkpoint
        if missing := unexpectedly_missing(result.missing_keys):
            raise RuntimeError(f"RAM++ checkpoint is missing {len(missing)} weights (e.g. {missing[:3]})")
        tags, thresholds, sources = self._apply_vocab_packs(model, builtin_tags, builtin_thresholds)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = model.to(device=device, dtype=dtype).eval()
        # Round thresholds through fp32 so the comparison matches upstream's tensor `>` exactly.
        return model, tags, torch.tensor(thresholds).tolist(), dtype, sources

    def _apply_vocab_packs(
        self, model: Any, builtin_tags: list[str], builtin_thresholds: list[float]
    ) -> tuple[list[str], list[float], list[str]]:
        """Swap the model's label embeddings for built-in (kept) + pack tags; return tags, thresholds, sources."""
        if not self._vocab_packs and self._builtin_vocab:
            return builtin_tags, builtin_thresholds, [""] * len(builtin_tags)
        import torch

        from argus_lens.backends._ram.packs import PackManifest, load_pack_embeddings, merge_vocabularies
        from argus_lens.backends._ram.vocab import DESCRIPTIONS_PER_TAG

        manifests = [PackManifest.read(p) for p in self._vocab_packs]
        merged = merge_vocabularies(builtin_tags, builtin_thresholds, manifests, include_builtin=self._builtin_vocab)
        builtin = model.label_embed.data.view(len(builtin_tags), DESCRIPTIONS_PER_TAG, -1)
        parts = [builtin[merged.builtin_keep].reshape(-1, builtin.shape[-1])]
        parts += [load_pack_embeddings(p, m).float() for p, m in zip(self._vocab_packs, manifests, strict=True)]
        model.label_embed = torch.nn.Parameter(torch.cat(parts), requires_grad=False)
        model.num_tags = len(merged.tags)
        return merged.tags, merged.thresholds, merged.sources

    def prefetch(self) -> None:
        """Download the RAM++ checkpoint if missing, outside the GPU lease (#64)."""
        self._ensure_checkpoint()

    # -- inference --------------------------------------------------------

    def annotate_image(self, image: Image.Image) -> BackendOutput:
        """Run RAM++ and return the tags that clear their threshold, highest score first."""
        from argus_lens.backends._ram.model import preprocess
        from argus_lens.backends._ram.vocab import select_tags

        device = self.resolve_device()
        with self._acquire(self._cache_key(device), lambda: self._loader(device)) as loaded:
            model, tags, thresholds, dtype, sources = loaded
            probs = model(preprocess(image, device, dtype))[0].float().cpu().tolist()
        kept = select_tags(probs, thresholds, self._threshold)
        return BackendOutput(
            tags=[Tag(label=tags[i], score=round(p, 4), source=self._source(sources[i])) for i, p in kept]
        )

    def _source(self, pack: str) -> str:
        """Provenance for a tag: ``ram`` for built-in tags, ``ram:<pack>`` for pack tags."""
        return f"{self.name}:{pack}" if pack else self.name

    def caption_image(self, image: Image.Image) -> str:
        """Backward-compatible shim: the kept tags as a comma-separated string."""
        return self.annotate_image(image).tag_string()

    def _build_output(self, labels_with_scores: list[tuple[str, float]]) -> BackendOutput:
        """Construct a BackendOutput from (label, score) pairs."""
        tags = [Tag(label=label, score=round(score, 4), source=self.name) for label, score in labels_with_scores]
        return BackendOutput(tags=tags)
