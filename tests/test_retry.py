"""OOM retry backoff and GPU cache clearing (retry.py)."""

from __future__ import annotations

import sys
import types
import weakref

import pytest

from argus_lens import retry
from argus_lens.retry import OOMDeadlineExceededError, run_with_oom_retry


def test_zero_interval_survives_many_oom_attempts() -> None:
    """interval_s=0 spins fast; >1024 attempts must not overflow the backoff and hide the OOM."""
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] <= 1100:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert run_with_oom_retry(fn, max_wait_s=600.0, interval_s=0.0) == "ok"
    assert calls["n"] == 1101


def test_clear_gpu_cache_collects_before_emptying(monkeypatch) -> None:
    """gc.collect() runs first, so cycle-held tensors are freed before empty_cache() returns blocks."""
    order: list[str] = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: order.append("empty_cache"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(retry.gc, "collect", lambda: order.append("collect"))
    retry.clear_gpu_cache()
    assert order == ["collect", "empty_cache"]


def test_cleanup_runs_after_the_failed_frames_are_released() -> None:
    """The OOM's traceback holds fn's locals (the tensors that did not fit); it must be gone before cleanup,
    the callbacks and the backoff sleep, or none of them can free the memory that ran out."""

    class _Tensor:
        pass

    alive_at: dict[str, bool] = {}
    refs: list[weakref.ref] = []

    def fn() -> str:
        if not refs:
            tensor = _Tensor()
            refs.append(weakref.ref(tensor))
            raise RuntimeError("CUDA out of memory")
        return "ok"

    def note(stage: str) -> None:
        alive_at[stage] = refs[0]() is not None

    result = run_with_oom_retry(
        fn,
        max_wait_s=5.0,
        interval_s=0.001,
        cleanup_fn=lambda: note("cleanup"),
        on_oom=lambda exc, attempt: note("on_oom"),
        on_retry=lambda wait_s, attempt: note("on_retry"),
    )
    assert result == "ok"
    assert alive_at == {"cleanup": False, "on_oom": False, "on_retry": False}


def test_deadline_error_keeps_the_oom_without_its_frames() -> None:
    """The deadline error still carries the last OOM, as its cause too, but not the frames it failed in."""

    def fn() -> str:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(OOMDeadlineExceededError) as info:
        run_with_oom_retry(fn, max_wait_s=0.0, interval_s=0.0)
    assert str(info.value.last_error) == "CUDA out of memory"
    assert info.value.__cause__ is info.value.last_error
    assert info.value.last_error.__traceback__ is None
    assert info.value.attempts == 1
