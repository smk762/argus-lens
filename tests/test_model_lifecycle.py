"""Model lifecycle: explicit registry eviction on unload (#63) and prefetch outside the GPU lease (#64)."""

import contextlib
import sys
import types

import pytest
from PIL import Image

from argus_lens.backends import _download
from argus_lens.backends.base import CaptionBackend, LocalBackend
from argus_lens.backends.blip2 import BLIP2Backend
from argus_lens.backends.florence2 import Florence2Backend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.backends.ram import RamBackend
from argus_lens.backends.wd14 import WD14Backend
from argus_lens.engine import ArgusLens
from argus_lens.registry import ModelRegistry


def _img() -> Image.Image:
    return Image.new("RGB", (16, 16), "gray")


class _RegistryTagger(LocalBackend):
    """A local backend that caches its 'model' in a registry, like wd14/florence2/ram."""

    name = "florence2"  # a known footprint (2500 MB), so the engine takes a lease
    emits_tags = True

    def __init__(self, registry: ModelRegistry, events: list | None = None) -> None:
        """Use *registry* and record prefetch/inference order in *events*."""
        self._registry = registry
        self.events = events if events is not None else []
        self.prefetches = 0

    def prefetch(self) -> None:
        """Record the prefetch."""
        self.prefetches += 1
        self.events.append("prefetch")

    def caption_image(self, image: Image.Image) -> str:
        """Acquire the cached model per device, then 'infer'."""
        with self._acquire(f"stub:{self.resolve_device()}", lambda: object()):
            self.events.append("infer")
            return "beach, sunset"


# -- registry.evict ------------------------------------------------------------------


def test_registry_evict_frees_idle_and_refuses_in_use():
    """evict() frees an unreferenced model, refuses one in use, and ignores unknown keys."""
    registry = ModelRegistry(idle_seconds=3600)
    with registry.acquire("a", lambda: "model-a"):
        assert registry.evict("a") is False  # in use: never pulled out from under a caption
        assert "a" in registry.status()
    assert registry.evict("a") is True
    assert registry.status() == {}
    assert registry.evict("a") is False
    assert registry.evict("never-loaded") is False


# -- LocalBackend.unload --------------------------------------------------------------


def test_unload_evicts_only_this_backends_keys():
    """unload() evicts the keys this instance acquired, and leaves other models cached."""
    registry = ModelRegistry(idle_seconds=3600)
    backend = _RegistryTagger(registry)
    backend.load("cpu")
    backend.caption_image(_img())
    with registry.acquire("someone-else", lambda: object()):
        pass
    assert set(registry.status()) == {"stub:cpu", "someone-else"}

    backend.unload()
    assert set(registry.status()) == {"someone-else"}


def test_unload_keeps_a_model_in_use_and_retries_later():
    """A model still referenced survives unload() and is evicted by the next one."""
    registry = ModelRegistry(idle_seconds=3600)
    backend = _RegistryTagger(registry)
    backend.load("cpu")
    backend.caption_image(_img())
    with registry.acquire("stub:cpu", lambda: pytest.fail("cached, must not reload")):
        backend.unload()
        assert "stub:cpu" in registry.status()
    backend.unload()
    assert registry.status() == {}


@pytest.mark.parametrize("cls", [WD14Backend, Florence2Backend, BLIP2Backend, RamBackend])
def test_registry_backends_no_longer_noop_unload(cls):
    """The built-in registry-backed backends inherit the evicting unload instead of a no-op (#63)."""
    assert cls.unload is LocalBackend.unload


def test_engine_unload_frees_the_registry_model():
    """engine.unload() (what /admin/unload and the idle reaper call) actually empties the registry."""
    registry = ModelRegistry(idle_seconds=3600)
    engine = ArgusLens(backend=_RegistryTagger(registry), device="cpu", coordinator=_NoLease())
    engine.caption(_img())
    assert registry.status()
    assert engine.unload() is True
    assert registry.status() == {}
    engine.caption(_img())  # reloads lazily
    assert registry.status()


# -- prefetch outside the lease ---------------------------------------------------------


class _NoLease:
    name = "none"

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int):
        yield


def test_prefetch_runs_before_the_lease_and_only_once():
    """Weights are fetched before the GPU lease is entered, once per engine (#64)."""
    events: list[str] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            events.append("lease")
            try:
                yield
            finally:
                events.append("release")

    backend = _RegistryTagger(ModelRegistry(idle_seconds=3600), events)
    engine = ArgusLens(backend=backend, device="cpu", coordinator=_Recording())
    engine.caption(_img())
    engine.caption(_img())
    assert events == ["prefetch", "lease", "infer", "release", "lease", "infer", "release"]
    assert backend.prefetches == 1


def test_failed_prefetch_takes_no_lease_and_retries():
    """A download failure surfaces without ever taking the lease, and the next caption retries it."""
    leases: list[str] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            leases.append(caller)
            yield

    class _Flaky(_RegistryTagger):
        def prefetch(self) -> None:
            """Fail the first download."""
            super().prefetch()
            if self.prefetches == 1:
                raise ConnectionError("hub unreachable")

    backend = _Flaky(ModelRegistry(idle_seconds=3600))
    engine = ArgusLens(backend=backend, device="cpu", coordinator=_Recording())
    with pytest.raises(ConnectionError):
        engine.caption(_img())
    assert leases == []
    engine.caption(_img())
    assert backend.prefetches == 2 and leases == ["florence2"]


def test_hybrid_prefetches_both_stages():
    """HybridPipeline forwards prefetch to its tag and prose backends."""
    tag, prose = _RegistryTagger(ModelRegistry()), _RegistryTagger(ModelRegistry())
    HybridPipeline(tag_backend=tag, prose_backend=prose).prefetch()
    assert tag.prefetches == prose.prefetches == 1


def test_default_prefetch_is_a_noop():
    """Backends without weights to fetch (cloud APIs, custom backends) inherit a harmless no-op."""

    class _Cloud(CaptionBackend):
        def load(self, device: str = "auto") -> None:
            """No-op."""

        def caption_image(self, image: Image.Image) -> str:
            """Return nothing."""
            return ""

        def unload(self) -> None:
            """No-op."""

    assert _Cloud().prefetch() is None


def test_builtin_prefetch_targets(monkeypatch, tmp_path):
    """wd14/ram fetch their files; florence2/blip2 snapshot their HF repo."""
    calls: list = []
    monkeypatch.setattr(WD14Backend, "_ensure_model", lambda self: calls.append("wd14"))
    monkeypatch.setattr(RamBackend, "_ensure_checkpoint", lambda self: calls.append("ram"))
    monkeypatch.setattr(_download, "prefetch_hf_snapshot", lambda repo: calls.append(repo))
    WD14Backend().prefetch()
    RamBackend().prefetch()
    Florence2Backend(model_id="org/florence").prefetch()
    BLIP2Backend(model_id="org/blip2").prefetch()
    assert calls == ["wd14", "ram", "org/florence", "org/blip2"]


def test_prefetch_hf_snapshot_skips_duplicate_weight_formats(monkeypatch):
    """The HF snapshot skips .bin-style duplicates transformers won't load when safetensors exist."""
    seen = {}
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda repo_id, **kw: seen.update(repo=repo_id, **kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    _download.prefetch_hf_snapshot("microsoft/Florence-2-base")
    assert seen["repo"] == "microsoft/Florence-2-base"
    assert "*.bin" in seen["ignore_patterns"] and "*.safetensors" not in seen["ignore_patterns"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # not installed: silently skipped
    _download.prefetch_hf_snapshot("any/repo")
