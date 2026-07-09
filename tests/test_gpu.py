"""Tests for GPU lifecycle (#37) and the pluggable coordinator (#38)."""

from __future__ import annotations

import contextlib

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.engine import ArgusLens
from argus_lens.gpu import (
    DEFAULT_FOOTPRINT_MB,
    GothmogCoordinator,
    GpuLeaseTimeout,
    LocalLeaseCoordinator,
    NullCoordinator,
    build_coordinator,
    coordinator_from_env,
    estimate_footprint_mb,
    free_vram_mb,
    resolve_min_vram_mb,
)


class _LifecycleBackend(CaptionBackend):
    """Local backend stub that counts load/unload calls (no model)."""

    name = "florence2"  # a local backend → footprint 2500 → lease used
    requires_gpu = True

    def __init__(self) -> None:
        self.loads = 0
        self.unloads = 0

    def load(self, device: str = "auto") -> None:  # noqa: D102
        self.loads += 1

    def caption_image(self, image: Image.Image) -> str:  # noqa: D102
        return "a caption"

    def unload(self) -> None:  # noqa: D102
        self.unloads += 1


def _img() -> Image.Image:
    return Image.new("RGB", (4, 4))


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #


def test_estimate_footprint() -> None:
    """Per-backend footprints; cloud = 0; unknown/hybrid = default."""
    assert estimate_footprint_mb("wd14") == 1500
    assert estimate_footprint_mb("florence2") == 2500
    assert estimate_footprint_mb("openai") == 0
    assert estimate_footprint_mb("hybrid") == DEFAULT_FOOTPRINT_MB
    assert estimate_footprint_mb("") == DEFAULT_FOOTPRINT_MB


def test_free_vram_none_or_int() -> None:
    """Free-VRAM probe is None without CUDA, else a non-negative int."""
    v = free_vram_mb()
    assert v is None or (isinstance(v, int) and v >= 0)


def test_resolve_min_vram_mb_env_override(monkeypatch) -> None:
    """ARGUS_GPU_MIN_VRAM_MB overrides the per-backend estimate; junk is ignored."""
    monkeypatch.delenv("ARGUS_GPU_MIN_VRAM_MB", raising=False)
    assert resolve_min_vram_mb("florence2") == 2500  # falls back to the estimate
    monkeypatch.setenv("ARGUS_GPU_MIN_VRAM_MB", "12000")
    assert resolve_min_vram_mb("florence2") == 12000
    monkeypatch.setenv("ARGUS_GPU_MIN_VRAM_MB", "not-a-number")
    assert resolve_min_vram_mb("florence2") == 2500  # invalid → estimate


# --------------------------------------------------------------------------- #
# Coordinators
# --------------------------------------------------------------------------- #


def test_null_coordinator_is_passthrough() -> None:
    """The default coordinator grants immediately and does nothing."""
    entered = False
    with NullCoordinator().lease(caller="t", min_vram_mb=1000):
        entered = True
    assert entered


def test_local_lease_acquires_when_free(tmp_path) -> None:
    """A free lock is acquired and released around the body."""
    coord = LocalLeaseCoordinator(lock_path=str(tmp_path / "g.lock"), timeout_s=1.0)
    with coord.lease(caller="t", min_vram_mb=0):
        pass  # acquired without raising
    # A second lease succeeds because the first released.
    with coord.lease(caller="t", min_vram_mb=0):
        pass


def test_local_lease_times_out_when_held(tmp_path) -> None:
    """When the lock is already held, lease() times out."""
    import fcntl
    import os

    lock = str(tmp_path / "g.lock")
    coord = LocalLeaseCoordinator(lock_path=lock, timeout_s=0.3, poll_s=0.05)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(TimeoutError), coord.lease(caller="t", min_vram_mb=1000):
            pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_gothmog_coordinator_acquire_release(monkeypatch) -> None:
    """The gothmog adapter acquires a token, runs, then releases it."""
    import httpx

    calls: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"token_id": "tok-1", "vram_free_mb": 1000}

    monkeypatch.setattr(httpx, "post", lambda url, json, headers, timeout: calls.update(acquire=(url, json)) or _Resp())
    monkeypatch.setattr(httpx, "delete", lambda url, headers, timeout: calls.update(release=url))

    with GothmogCoordinator(base_url="http://gm:8030").lease(caller="argus-lens", min_vram_mb=2500):
        pass

    assert calls["acquire"][0].endswith("/v1/gpu/capacity/acquire")
    assert calls["acquire"][1] == {"caller": "argus-lens", "min_vram_mb": 2500, "timeout_s": 300.0}
    assert str(calls["release"]).endswith("/v1/gpu/capacity/tokens/tok-1")


def test_build_coordinator_and_from_env(monkeypatch) -> None:
    """Factory builds each kind and validates; env selects the default + timeout."""
    assert build_coordinator("none").name == "none"
    assert build_coordinator("lease").name == "lease"
    assert build_coordinator("gothmog", base_url="http://x").name == "gothmog"
    with pytest.raises(ValueError, match="GOTHMOG_URL"):
        build_coordinator("gothmog")
    with pytest.raises(ValueError, match="Unknown GPU coordinator"):
        build_coordinator("bogus")

    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    monkeypatch.delenv("ARGUS_GPU_LEASE_TIMEOUT_S", raising=False)
    assert coordinator_from_env().name == "none"
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "lease")
    monkeypatch.setenv("ARGUS_GPU_LEASE_TIMEOUT_S", "42")
    coord = coordinator_from_env()
    assert coord.name == "lease"
    assert coord.timeout_s == 42.0  # env timeout threaded through


def test_lease_timeout_is_gpu_lease_timeout(tmp_path) -> None:
    """A held lock raises GpuLeaseTimeout (a TimeoutError subclass callers can catch)."""
    import fcntl
    import os

    lock = str(tmp_path / "g.lock")
    coord = LocalLeaseCoordinator(lock_path=lock, timeout_s=0.2, poll_s=0.05)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(GpuLeaseTimeout), coord.lease(caller="t", min_vram_mb=1000):
            pass
        assert issubclass(GpuLeaseTimeout, TimeoutError)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #


def test_engine_unload_and_reload() -> None:
    """unload() releases the backend; the next caption reloads it."""
    b = _LifecycleBackend()
    eng = ArgusLens(backend=b)
    eng.caption(_img())
    assert b.loads == 1 and eng._loaded
    eng.unload()
    assert b.unloads == 1 and not eng._loaded
    eng.caption(_img())
    assert b.loads == 2


def test_unload_if_idle() -> None:
    """unload_if_idle unloads only after the TTL and only when previously used."""
    b = _LifecycleBackend()
    eng = ArgusLens(backend=b)
    assert eng.unload_if_idle(0) is False  # never used → nothing to unload
    eng.caption(_img())
    assert eng.unload_if_idle(10_000) is False  # not idle yet
    assert eng.unload_if_idle(0) is True  # idle >= 0 → unloads
    assert not eng._loaded


def test_vram_status_shape(monkeypatch) -> None:
    """vram_status reports backend, residency, coordinator, free VRAM, idle time."""
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    eng = ArgusLens(backend=_LifecycleBackend())
    s = eng.vram_status()
    assert s["backend"] == "florence2"
    assert s["loaded"] is False
    assert s["coordinator"] == "none"
    assert s["idle_s"] is None
    assert "free_vram_mb" in s  # key must survive (regression guard)


def test_reaper_does_not_unload_during_inference() -> None:
    """An in-flight caption is not unloaded by unload()/unload_if_idle (#42 F1)."""
    import threading

    started = threading.Event()
    release = threading.Event()

    class _Blocking(_LifecycleBackend):
        def caption_image(self, image: Image.Image) -> str:
            started.set()
            release.wait(2.0)
            return "a caption"

    b = _Blocking()
    eng = ArgusLens(backend=b)
    t = threading.Thread(target=lambda: eng.caption(_img()))
    t.start()
    assert started.wait(2.0)
    # Mid-inference: idleness measured from the *start* boundary is 0, and the
    # engine is busy → neither the reaper nor an explicit unload may free it.
    assert eng.unload_if_idle(0) is False
    assert eng.unload() is False
    assert b.unloads == 0
    release.set()
    t.join(2.0)
    assert eng.unload_if_idle(0) is True  # now idle → unloads


def test_cloud_hybrid_bypasses_lease() -> None:
    """A hybrid with requires_gpu=False takes no lease despite a non-zero name footprint (#42 F6)."""

    class _CloudHybrid(_LifecycleBackend):
        name = "hybrid"  # footprint 4000, but...
        requires_gpu = False  # ...no local GPU

    events: list[str] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            events.append("leased")
            yield

    ArgusLens(backend=_CloudHybrid(), coordinator=_Recording()).caption(_img())
    assert events == []


def test_two_local_lease_instances_serialize(tmp_path) -> None:
    """Two coordinators on the same lock file mutually exclude."""
    lock = str(tmp_path / "g.lock")
    held = LocalLeaseCoordinator(lock_path=lock, timeout_s=1.0)
    other = LocalLeaseCoordinator(lock_path=lock, timeout_s=0.3, poll_s=0.05)
    with held.lease(caller="a", min_vram_mb=0), pytest.raises(TimeoutError), other.lease(caller="b", min_vram_mb=0):
        pass


def test_coordinator_lease_wraps_inference() -> None:
    """A local backend's inference runs inside the coordinator lease, sized by footprint."""
    events: list[tuple] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            events.append(("enter", caller, min_vram_mb))
            try:
                yield
            finally:
                events.append(("exit",))

    eng = ArgusLens(backend=_LifecycleBackend(), coordinator=_Recording())
    eng.caption(_img())
    assert events[0] == ("enter", "florence2", 2500)
    assert events[-1] == ("exit",)


def test_engine_honors_min_vram_override(monkeypatch) -> None:
    """ARGUS_GPU_MIN_VRAM_MB flows into the lease request (#38)."""
    monkeypatch.setenv("ARGUS_GPU_MIN_VRAM_MB", "9000")
    requested: list[int] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            requested.append(min_vram_mb)
            yield

    ArgusLens(backend=_LifecycleBackend(), coordinator=_Recording()).caption(_img())
    assert requested == [9000]


def test_cloud_backend_bypasses_lease() -> None:
    """A zero-footprint (cloud) backend does not take the lease."""

    class _Cloud(_LifecycleBackend):
        name = "openai"
        requires_gpu = False

    events: list[str] = []

    class _Recording:
        name = "rec"

        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):
            events.append("leased")
            yield

    eng = ArgusLens(backend=_Cloud(), coordinator=_Recording())
    eng.caption(_img())
    assert events == []


def test_idle_reaper_starts_and_close_stops() -> None:
    """An idle_unload_s engine starts a reaper thread that close() stops."""
    eng = ArgusLens(backend=_LifecycleBackend(), idle_unload_s=0.05)
    assert eng._reaper_thread is not None and eng._reaper_thread.is_alive()
    eng.close()
    eng._reaper_thread.join(timeout=2)
    assert not eng._reaper_thread.is_alive()
