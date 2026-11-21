"""The argus-iris broker wiring (#118): the committed tenant manifest and the lease kinds."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from argus_cortex.gpu_wire import load_tenant_manifest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.engine import LEASE_KIND_BATCH, LEASE_KIND_CAPTION, ArgusLens
from argus_lens.gpu import coordinator_from_env

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


class _GpuBackend(CaptionBackend):
    """A local backend stub: takes a lease, loads nothing."""

    name = "florence2"
    requires_gpu = True

    def load(self, device: str = "auto") -> None:  # noqa: D102
        pass

    def caption_image(self, image: Image.Image) -> str:  # noqa: D102
        return "a caption"

    def unload(self) -> None:  # noqa: D102
        pass


class _Recording:
    """A coordinator that records the kind each lease asks for."""

    name = "rec"

    def __init__(self) -> None:
        self.kinds: list[str | None] = []

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int, kind: str | None = None, **_):
        self.kinds.append(kind)
        yield


def _img(shade: int = 0) -> Image.Image:
    return Image.new("RGB", (4, 4), (shade, shade, shade))


def test_manifest_validates_and_declares_every_lease_kind() -> None:
    """gpu-tenant.yaml is a valid TenantManifest and declares each kind the engine leases under."""
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == "argus-lens"
    for kind in (LEASE_KIND_CAPTION, LEASE_KIND_BATCH):
        assert manifest.kind_for(kind) is not None, kind
    assert manifest.kind_for(LEASE_KIND_CAPTION).priority == "interactive"
    batch = manifest.kind_for(LEASE_KIND_BATCH)
    assert batch.priority == "batch" and batch.preemptible
    (resident,) = manifest.residents
    assert resident.adapter == "argus-http"
    assert resident.unload_path == "/admin/unload" and resident.admin_auth


def test_sibling_priority_matches_each_lease_kind() -> None:
    """The X-Argus-Priority lens sends a sibling mid-caption is the manifest's priority for that work's kind."""
    from argus_lens.priority import PRIORITY_BY_LEASE_KIND, priority_for_lease_kind

    manifest = load_tenant_manifest(MANIFEST)
    assert set(PRIORITY_BY_LEASE_KIND) == {LEASE_KIND_CAPTION, LEASE_KIND_BATCH}
    for kind in (LEASE_KIND_CAPTION, LEASE_KIND_BATCH):
        assert priority_for_lease_kind(kind) == manifest.kind_for(kind).priority
    assert priority_for_lease_kind("something-new") == "interactive"


def test_single_caption_leases_as_interactive_kind() -> None:
    """caption() takes an interactive lease unless the caller says it is looping."""
    coord = _Recording()
    eng = ArgusLens(backend=_GpuBackend(), coordinator=coord)
    eng.caption(_img())
    eng.caption(_img(), lease_kind=LEASE_KIND_BATCH)
    assert coord.kinds == [LEASE_KIND_CAPTION, LEASE_KIND_BATCH]


def test_batch_and_stream_lease_as_batch_kind() -> None:
    """Multi-image entry points lease under the preemptible batch kind, once per image."""
    coord = _Recording()
    eng = ArgusLens(backend=_GpuBackend(), coordinator=coord)
    eng.caption_batch([_img(1), _img(2)])
    list(eng.caption_stream([_img(3)]))
    assert coord.kinds == [LEASE_KIND_BATCH] * 3


def test_broker_coordinator_from_env(monkeypatch) -> None:
    """ARGUS_GPU_COORDINATOR=broker builds cortex's broker client, tenant read from the manifest."""
    pytest.importorskip("httpx")
    from argus_cortex.gpu_broker import BrokerCoordinator

    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "broker")
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://broker.invalid:8120")
    monkeypatch.setenv("ARGUS_GPU_TENANT_MANIFEST", str(MANIFEST))
    monkeypatch.delenv("ARGUS_GPU_TENANT", raising=False)
    coord = coordinator_from_env()
    assert isinstance(coord, BrokerCoordinator)
    assert coord.tenant == "argus-lens"
    assert coord.manifest is not None and coord.manifest.tenant == "argus-lens"
