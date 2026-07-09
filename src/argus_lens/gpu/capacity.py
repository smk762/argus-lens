"""VRAM capacity helpers (#37/#38).

Free-VRAM query (delegates to the existing CUDA probe) plus a rough per-backend
footprint estimate, used to size a capacity lease and to warn before a load
that likely won't fit.
"""

from __future__ import annotations

import os

import structlog

from argus_lens.retry import cuda_free_vram_mb

logger = structlog.get_logger()

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


def resolve_min_vram_mb(backend_name: str) -> int:
    """Resolve the VRAM a lease should request for *backend_name*, in MB.

    An ``ARGUS_GPU_MIN_VRAM_MB`` env override wins over the per-backend estimate
    (for models whose real footprint differs from the table, e.g. Florence-2-large
    or a non-default hybrid). A non-integer/negative override is ignored.
    """
    override = os.environ.get("ARGUS_GPU_MIN_VRAM_MB")
    if override:
        try:
            value = int(override)
            if value >= 0:
                return value
            logger.warning("invalid_min_vram_override", value=override)
        except ValueError:
            logger.warning("invalid_min_vram_override", value=override)
    return estimate_footprint_mb(backend_name)
