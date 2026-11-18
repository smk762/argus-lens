"""ModelRegistry refcounting under clear()/reload/concurrent loads, and freeing robustness.

Torch-free: models are fakes with ``.parameters``/``.cpu()``; threads are sequenced with events.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest
from structlog.testing import capture_logs

from argus_lens.registry import ModelRegistry

_TIMEOUT = 5


class _FakeModel:
    """Stand-in for a torch module: records whether it was moved to the CPU."""

    def __init__(self, name: str = "m", fail_cpu: bool = False) -> None:
        """Name the fake; *fail_cpu* makes ``.cpu()`` raise like a quantised model."""
        self.name = name
        self.fail_cpu = fail_cpu
        self.freed = False

    def parameters(self):
        """Present the torch-module shape so the registry tries a CPU move."""
        return iter(())

    def cpu(self):
        """Mark freed, or raise when configured to."""
        if self.fail_cpu:
            raise RuntimeError(".cpu() is not supported for this model")
        self.freed = True
        return self


def test_clear_defers_free_of_model_in_use():
    """clear() never frees a model a holder is using; the last release frees it."""
    registry = ModelRegistry(idle_seconds=3600)
    idle, busy = _FakeModel("idle"), _FakeModel("busy")
    with registry.acquire("idle", lambda: idle):
        pass
    with registry.acquire("busy", lambda: busy) as model:
        assert registry.clear() == ["busy"]
        assert idle.freed is True
        assert model.freed is False
        assert set(registry.status()) == {"busy"}
    assert busy.freed is True
    assert registry.status() == {}


def test_clear_then_acquire_shares_the_held_model_and_keeps_refcount():
    """A post-clear acquire reuses the held model (no duplicate load); releases count against it."""
    registry = ModelRegistry(idle_seconds=3600)
    first = _FakeModel("first")
    loads: list[str] = []

    def loader():
        loads.append("load")
        return first

    with registry.acquire("k", loader):
        registry.clear()
        with registry.acquire("k", loader) as again:
            assert again is first
            assert registry.status()["k"]["refs"] == 2
        # the inner release must not free the model the outer holder still uses
        assert first.freed is False
        assert registry.status()["k"]["refs"] == 1
    assert loads == ["load"]
    assert first.freed is True


def test_release_after_clear_and_reload_does_not_touch_new_entry():
    """A last release of a cleared entry, after its last holder let go, leaves a later reload alone."""
    registry = ModelRegistry(idle_seconds=3600)
    old, new = _FakeModel("old"), _FakeModel("new")
    ctx = registry.acquire("k", lambda: old)
    ctx.__enter__()
    registry.clear()  # deferred: old is still held
    ctx.__exit__(None, None, None)  # last release frees old and drops it from the cache
    assert old.freed is True
    with registry.acquire("k", lambda: new) as model:
        assert model is new
        assert registry.status()["k"]["refs"] == 1
        assert registry.evict("k") is False  # still in use: refcount intact
    assert registry.evict("k") is True


def test_stale_holder_release_does_not_touch_a_reloaded_entry():
    """A holder of a replaced entry releases *its* entry, by identity, never the reloaded one under the same key.

    The public API keeps a held entry cached (clear() defers it), so the cache is swapped
    directly here to stand in for any path that drops a held entry: the stale release must
    neither decrement nor evict the new model.
    """
    registry = ModelRegistry(idle_seconds=3600)
    old, new = _FakeModel("old"), _FakeModel("new")
    old_ctx = registry.acquire("k", lambda: old)
    assert old_ctx.__enter__() is old
    registry.clear()  # deferred: old is held, so it is freed at its last release
    with registry._lock:
        del registry._models["k"]  # old is no longer the cached entry, but its holder lives on
    with registry.acquire("k", lambda: new) as model:
        assert model is new
        assert registry.status()["k"]["refs"] == 1
        old_ctx.__exit__(None, None, None)  # the stale holder releases while new is held
        assert old.freed is True
        assert new.freed is False
        assert registry.status()["k"]["refs"] == 1
        assert registry.status()["k"]["release_pending"] is False
        assert registry.evict("k") is False  # new is still in use
    assert registry.evict("k") is True
    assert new.freed is True


def test_clear_during_load_does_not_start_a_second_load():
    """clear() while a loader runs keeps the in-progress load, so a concurrent acquire waits on it."""
    registry = ModelRegistry(idle_seconds=3600)
    started, proceed, second_waiting, first_acquired, second_done, hold_first = (threading.Event() for _ in range(6))
    calls: list[_FakeModel] = []
    seen: dict[str, _FakeModel] = {}

    def slow_loader():
        obj = _FakeModel(f"load{len(calls)}")
        calls.append(obj)
        started.set()
        assert proceed.wait(_TIMEOUT)
        return obj

    def first():
        with registry.acquire("k", slow_loader) as m:
            seen["first"] = m
            first_acquired.set()
            assert hold_first.wait(_TIMEOUT)

    def second():
        with registry.acquire("k", slow_loader) as m:
            seen["second"] = m
        second_done.set()

    t1 = threading.Thread(target=first)
    t1.start()
    assert started.wait(_TIMEOUT)
    assert registry.clear() == ["k"]  # still loading: deferred, not dropped
    # Observe t2 blocking on the in-progress load, so it can't slip in after the load finished.
    load = registry._loading["k"]
    real_wait = load.event.wait

    def observed_wait(timeout=None):
        second_waiting.set()
        return real_wait(timeout)

    load.event.wait = observed_wait
    t2 = threading.Thread(target=second)
    t2.start()
    assert second_waiting.wait(_TIMEOUT)
    assert not second_done.is_set()
    proceed.set()
    assert first_acquired.wait(_TIMEOUT)
    assert second_done.wait(_TIMEOUT)
    assert len(calls) == 1
    assert seen["second"] is seen["first"]
    assert calls[0].freed is False  # the first holder still has it
    hold_first.set()
    t1.join(_TIMEOUT)
    t2.join(_TIMEOUT)
    # the clear() carried over to the loaded entry: freed at its last release, nothing leaked
    assert calls[0].freed is True
    assert registry.status() == {}


def test_failed_load_lets_the_next_acquire_retry():
    """A loader that raises leaves no in-progress marker behind."""
    registry = ModelRegistry(idle_seconds=3600)

    def boom():
        raise RuntimeError("no weights")

    with pytest.raises(RuntimeError), registry.acquire("k", boom):
        pass
    with registry.acquire("k", lambda: "ok") as model:
        assert model == "ok"


def test_free_empties_cuda_cache_and_logs_when_cpu_move_fails(monkeypatch):
    """A model whose .cpu() raises is logged, the other parts are still moved, and empty_cache() runs."""
    emptied: list[bool] = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: emptied.append(True))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    registry = ModelRegistry(idle_seconds=3600)
    stuck, ok = _FakeModel("stuck", fail_cpu=True), _FakeModel("ok")
    with registry.acquire("k", lambda: (stuck, ok)):
        pass
    with capture_logs() as logs:
        assert registry.evict("k") is True
    assert ok.freed is True
    assert emptied == [True]
    assert any(e["event"] == "model_cpu_move_failed" for e in logs)
