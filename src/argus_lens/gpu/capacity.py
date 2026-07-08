"""VRAM capacity helpers (#37/#38).

Free-VRAM query (delegates to the existing CUDA probe) plus a rough per-backend
footprint estimate, used to size a capacity lease and to warn before a load
that likely won't fit.
"""

from __future__ import annotations

from argus_lens.retry import cuda_free_vram_mb

# Rough resident footprints in MB (fp16), from the model VRAM survey. Cloud
# backends hold no local VRAM. Unknown local backends use DEFAULT_FOOTPRINT_MB.
_FOOTPRINT_MB: dict[str, int] = {
    "wd14": 1500,
    "florence2": 2500,
    "blip2": 8000,
    "ram": 2000,
}
# Backends that hold no VRAM *in the argus process*. Note: openai-compat may
# point at a local model server (Ollama) that does use the shared GPU — argus's
# lease can't gate that; rely on a resident-aware coordinator (gothmog) there.
_CLOUD_BACKENDS: frozenset[str] = frozenset({"openai", "openai-compat", "hf-inference", "replicate", "nvidia-nim"})
DEFAULT_FOOTPRINT_MB = 4000  # e.g. the default hybrid = wd14 (1500) + florence2 (2500)


def free_vram_mb() -> int | None:
    """Free CUDA VRAM in MB, or ``None`` when CUDA/torch is unavailable."""
    return cuda_free_vram_mb()


def estimate_footprint_mb(backend_name: str) -> int:
    """Estimate the VRAM a backend occupies, in MB.

    Returns ``0`` for cloud backends (no local VRAM → no lease needed) and
    ``DEFAULT_FOOTPRINT_MB`` for unrecognised local backends (e.g. hybrid).
    """
    key = (backend_name or "").lower().strip()
    if key in _CLOUD_BACKENDS:
        return 0
    return _FOOTPRINT_MB.get(key, DEFAULT_FOOTPRINT_MB)
