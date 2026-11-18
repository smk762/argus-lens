"""OOM retry backoff and GPU cache clearing (retry.py)."""

from __future__ import annotations

import sys
import types

from argus_lens import retry
from argus_lens.retry import run_with_oom_retry


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
