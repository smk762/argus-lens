"""The ``X-Argus-Priority`` lens sends with a sibling call made while it captions an image.

Lens ranks its own GPU work by lease kind (``gpu-tenant.yaml``): one image someone is waiting on is ``interactive``,
a multi-image job is ``batch``. A sibling it calls mid-caption (argus-puppet for the pose check) gets the same class,
so a batch run's pose calls yield to someone's single image there too. ``ArgusLens._infer`` sets
:data:`sibling_priority` for the duration of each caption; outside one it reads ``interactive``.
"""

from __future__ import annotations

from contextvars import ContextVar

# Lease kind -> the priority gpu-tenant.yaml gives it (tests/test_gpu_tenant.py keeps the two in step).
PRIORITY_BY_LEASE_KIND: dict[str, str] = {"caption": "interactive", "caption-batch": "batch"}
DEFAULT_PRIORITY = "interactive"

sibling_priority: ContextVar[str] = ContextVar("lens_sibling_priority", default=DEFAULT_PRIORITY)


def priority_for_lease_kind(kind: str) -> str:
    """The sibling priority for lens work of lease *kind*; an unknown kind is ``interactive``."""
    return PRIORITY_BY_LEASE_KIND.get(kind, DEFAULT_PRIORITY)
