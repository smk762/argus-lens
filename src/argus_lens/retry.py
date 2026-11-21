"""OOM retry engine and VRAM monitoring utilities."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

OOM_ERROR_SUBSTRINGS = (
    "out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cuda error: out of memory",
)


# 2**60 seconds is already far past any deadline; keeps the backoff a finite float.
_MAX_BACKOFF_EXPONENT = 60


class OOMDeadlineExceededError(RuntimeError):
    """Raised when OOM retries exceed the configured wait budget."""

    def __init__(self, *, attempts: int, max_wait_s: float, last_error: Exception):
        """Record retry statistics and keep the last OOM exception for inspection."""
        super().__init__(f"OOM wait deadline exceeded after {attempts} attempts in {max_wait_s:.1f}s")
        self.attempts = attempts
        self.max_wait_s = max_wait_s
        self.last_error = last_error


def is_oom_error(exc: Exception) -> bool:
    """Detect CUDA OOM and OOM-like backend errors."""
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    text = str(exc).lower()
    return any(token in text for token in OOM_ERROR_SUBSTRINGS)


def run_with_oom_retry(
    fn: Callable[[], object],
    *,
    max_wait_s: float = 180.0,
    interval_s: float = 5.0,
    cleanup_fn: Callable[[], None] | None = None,
    on_oom: Callable[[Exception, int], None] | None = None,
    on_retry: Callable[[float, int], None] | None = None,
) -> object:
    """Run *fn* and retry on OOM errors until *max_wait_s* is exceeded.

    Between retries the function calls *cleanup_fn* (e.g. clear model
    cache, empty CUDA cache) and waits with exponential backoff.

    The cleanup, the callbacks and the wait all run after the ``except`` block has
    exited and with the error's traceback dropped. A live traceback holds the failed
    frames, and their locals (the tensors that did not fit) stay allocated for as
    long as it does, so nothing the cleanup frees would be the memory that ran out.
    The ``last_error`` of :class:`OOMDeadlineExceededError` therefore has no traceback.
    """
    attempts = 0
    deadline = time.monotonic() + max(0.0, max_wait_s)
    interval_s = max(0.0, interval_s)

    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            # Keep the error, not its frames; everything else waits until the block has exited.
            oom = exc.with_traceback(None)

        attempts += 1
        if cleanup_fn is not None:
            cleanup_fn()
        if on_oom is not None:
            on_oom(oom, attempts)

        now = time.monotonic()
        if now >= deadline:
            raise OOMDeadlineExceededError(
                attempts=attempts,
                max_wait_s=max_wait_s,
                last_error=oom,
            ) from oom

        # Cap the exponent: past ~1000 attempts (interval_s=0 spins fast) 2**n is
        # too large for a float and would raise OverflowError over the real OOM.
        backoff = interval_s * 2.0 ** min(attempts - 1, _MAX_BACKOFF_EXPONENT)
        wait_s = min(backoff, max(0.0, deadline - now))
        del oom
        if wait_s > 0:
            if on_retry is not None:
                on_retry(wait_s, attempts)
            time.sleep(wait_s)


def cuda_free_vram_mb() -> int | None:
    """Return free CUDA VRAM in MB, or None if CUDA is unavailable.

    Uses the driver-level ``mem_get_info`` (free across *all* processes), not
    ``memory_reserved`` (this process's torch allocator only) — so a co-resident
    tenant's usage is reflected, which is the whole point on a shared GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(0)
        return max(0, int(free_bytes) // (1024 * 1024))
    except Exception:
        return None


def resolve_device() -> str:
    """Return ``"cuda"`` if available, else ``"cpu"``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def clear_gpu_cache() -> None:
    """Run garbage collection, then empty the CUDA cache.

    Collect first: tensors kept alive only by reference cycles (e.g. an OOM
    traceback's frames) are freed by ``gc.collect()``, and only then can
    ``empty_cache()`` hand their blocks back to the driver.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
