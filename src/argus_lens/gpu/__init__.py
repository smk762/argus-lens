"""GPU lifecycle + coordination (#37/#38).

``capacity`` exposes free-VRAM and per-backend footprint estimates; ``coordinator``
provides the pluggable capacity-lease (``none``/``lease``/``gothmog``). The model
lifecycle levers (unload, idle eviction) live on ``ArgusLens`` itself.
"""

from __future__ import annotations

from argus_lens.gpu.capacity import (
    DEFAULT_FOOTPRINT_MB,
    estimate_footprint_mb,
    free_vram_mb,
    resolve_min_vram_mb,
)
from argus_lens.gpu.coordinator import (
    GothmogCoordinator,
    GpuCoordinator,
    GpuLeaseTimeout,
    LocalLeaseCoordinator,
    NullCoordinator,
    build_coordinator,
    coordinator_from_env,
)

__all__ = [
    "DEFAULT_FOOTPRINT_MB",
    "GothmogCoordinator",
    "GpuCoordinator",
    "GpuLeaseTimeout",
    "LocalLeaseCoordinator",
    "NullCoordinator",
    "build_coordinator",
    "coordinator_from_env",
    "estimate_footprint_mb",
    "free_vram_mb",
    "resolve_min_vram_mb",
]
