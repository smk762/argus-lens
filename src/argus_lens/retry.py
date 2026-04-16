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


class OOMDeadlineExceededError(RuntimeError):
    """Raised when OOM retries exceed the configured wait budget."""

    def __init__(self, *, attempts: int, max_wait_s: float, last_error: Exception):
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

            attempts += 1
            if cleanup_fn is not None:
                cleanup_fn()
            if on_oom is not None:
                on_oom(exc, attempts)

            now = time.monotonic()
            if now >= deadline:
                raise OOMDeadlineExceededError(
                    attempts=attempts,
                    max_wait_s=max_wait_s,
                    last_error=exc,
                ) from exc

            wait_s = min(interval_s * (2 ** (attempts - 1)), max(0.0, deadline - now))
            if wait_s > 0:
                if on_retry is not None:
                    on_retry(wait_s, attempts)
                time.sleep(wait_s)


def cuda_free_vram_mb() -> int | None:
    """Return free CUDA VRAM in MB, or None if CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        reserved = int(torch.cuda.memory_reserved(0))
        return max(0, (total - reserved) // (1024 * 1024))
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
    """Empty CUDA cache and run garbage collection."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
