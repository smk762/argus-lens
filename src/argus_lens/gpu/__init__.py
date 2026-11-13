"""GPU lifecycle + coordination (#37/#38).

``capacity`` exposes free-VRAM and per-backend footprint estimates (each stage
owns its own). The coordinator is the suite's shared one, :mod:`argus_cortex.gpu`:
``none`` or ``lease`` (a host-wide ``flock`` taken by every argus service, then a
wait for free VRAM), selected by ``ARGUS_GPU_COORDINATOR``. The model lifecycle
levers (unload, idle eviction) live on ``ArgusLens`` itself.
"""

from __future__ import annotations

from argus_cortex.gpu import (
    GpuCoordinator,
    GpuLeaseTimeout,
    LeaseCoordinator,
    NullCoordinator,
    build_coordinator,
    coordinator_from_env,
)

from argus_lens.gpu.capacity import (
    DEFAULT_FOOTPRINT_MB,
    estimate_footprint_mb,
    free_vram_mb,
    resolve_min_vram_mb,
)

__all__ = [
    "DEFAULT_FOOTPRINT_MB",
    "GpuCoordinator",
    "GpuLeaseTimeout",
    "LeaseCoordinator",
    "NullCoordinator",
    "build_coordinator",
    "coordinator_from_env",
    "estimate_footprint_mb",
    "free_vram_mb",
    "resolve_min_vram_mb",
]
