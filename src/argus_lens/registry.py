"""Process-level model cache with TTL eviction and reference counting.

Models are loaded lazily on first ``acquire()`` and kept alive while at
least one reference is held.  A background thread evicts models that have
been idle for longer than *idle_seconds*.
"""

from __future__ import annotations

import contextlib
import gc
import os
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

_DEFAULT_IDLE_SECONDS = float(os.environ.get("ARGUS_MODEL_IDLE_SECONDS", "300"))


def _move_to_cpu(obj: Any) -> None:
    """Move *obj* (or each part of a tuple) off the GPU; log, never raise, when a move fails.

    Some models refuse ``.cpu()`` (bitsandbytes-quantised, accelerate-dispatched with offload
    hooks); that is logged so an un-freed model is visible, and the caller still empties the cache.
    """
    if isinstance(obj, tuple):
        for part in obj:
            _move_to_cpu(part)
        return
    if not (hasattr(obj, "parameters") and hasattr(obj, "cpu")):
        return
    try:
        obj.cpu()
    except Exception as exc:
        logger.warning("model_cpu_move_failed", model_type=type(obj).__name__, error=str(exc))


def _release_cuda_cache() -> None:
    """Collect garbage and return cached CUDA blocks to the driver; a no-op without torch."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("cuda_empty_cache_failed", error=str(exc))


def _free_entry(entry: _Entry) -> None:
    """Release memory held by an entry that has already been removed from the cache.

    The entry's own reference is dropped before collecting, so the model can actually be
    reclaimed even when the CPU move fails; the cache is emptied whatever the move did.
    """
    obj, entry.obj = entry.obj, None
    try:
        _move_to_cpu(obj)
    finally:
        del obj
        _release_cuda_cache()


@dataclass(eq=False)
class _Entry:
    """A cached model with its own refcount; holders release the entry they acquired, by identity."""

    obj: Any
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    # Set by clear() on an entry in use: it is evicted and freed when its last holder releases it.
    release_pending: bool = False


@dataclass(eq=False)
class _Load:
    """An in-progress load: waiters block on *event*; *release_pending* carries a clear() over to the entry."""

    event: threading.Event = field(default_factory=threading.Event)
    release_pending: bool = False


class ModelRegistry:
    """Thread-safe model cache with TTL eviction and reference counting.

    Usage::

        registry = ModelRegistry(idle_seconds=300)

        with registry.acquire("wd14", loader_fn) as model:
            result = model.run(image)
        # reference released; model stays cached until idle timeout

    Concurrent ``acquire()`` calls for the same key wait for the first
    load to finish rather than loading the model multiple times.

    A model with live references is never freed: not by the idle sweeper,
    :meth:`evict`, or :meth:`clear` (which defers it to its last release).
    """

    def __init__(self, idle_seconds: float | None = None) -> None:
        """Initialise an empty cache; *idle_seconds* defaults to ``ARGUS_MODEL_IDLE_SECONDS``."""
        self._models: dict[str, _Entry] = {}
        self._loading: dict[str, _Load] = {}
        self._lock = threading.Lock()
        self._idle_seconds = idle_seconds if idle_seconds is not None else _DEFAULT_IDLE_SECONDS
        self._evict_thread: threading.Thread | None = None

    def _ensure_evict_thread(self) -> None:
        """Start the daemon eviction thread if it is not already running."""
        if self._evict_thread and self._evict_thread.is_alive():
            return
        t = threading.Thread(target=self._evict_loop, daemon=True, name="argus-model-evict")
        t.start()
        self._evict_thread = t

    def _evict_loop(self) -> None:
        """Sweep for idle models every 30 seconds, forever."""
        while True:
            time.sleep(30)
            self._evict_idle()

    def _evict_idle(self) -> None:
        """Evict and free models with no live references that exceeded the idle timeout."""
        now = time.monotonic()
        with self._lock:
            to_evict = [k for k, e in self._models.items() if e.refs == 0 and now - e.last_used > self._idle_seconds]
            evicted = {k: self._models.pop(k) for k in to_evict}

        for k, entry in evicted.items():
            logger.info("model_evicted", key=k, idle_seconds=self._idle_seconds)
            _free_entry(entry)

    def _release(self, key: str, entry: _Entry) -> None:
        """Drop one reference to *entry*; free it if a clear() was waiting on this last release."""
        with self._lock:
            entry.refs = max(0, entry.refs - 1)
            entry.last_used = time.monotonic()
            if entry.refs > 0 or not entry.release_pending:
                return
            if self._models.get(key) is entry:
                del self._models[key]
        logger.info("model_cleared", key=key, deferred=True)
        _free_entry(entry)

    @contextlib.contextmanager
    def acquire(self, key: str, loader: Callable[[], Any]) -> Generator[Any, None, None]:
        """Context manager that provides the model for *key*.

        If the model is not cached it is loaded by calling *loader()*.
        Concurrent calls with the same key block until the first load
        finishes.
        """
        entry: _Entry | None = None

        while entry is None:
            wait_load: _Load | None = None
            my_load: _Load | None = None
            with self._lock:
                if key in self._models:
                    entry = self._models[key]
                    entry.refs += 1
                    entry.last_used = time.monotonic()
                elif key in self._loading:
                    wait_load = self._loading[key]
                else:
                    my_load = self._loading[key] = _Load()

            if wait_load is not None:
                wait_load.event.wait()
            elif my_load is not None:
                try:
                    logger.info("model_loading", key=key)
                    obj = loader()
                    logger.info("model_loaded", key=key)
                except BaseException:
                    with self._lock:
                        if self._loading.get(key) is my_load:
                            del self._loading[key]
                    my_load.event.set()
                    raise
                with self._lock:
                    entry = _Entry(obj, refs=1, release_pending=my_load.release_pending)
                    self._models[key] = entry
                    if self._loading.get(key) is my_load:
                        del self._loading[key]
                    self._ensure_evict_thread()
                my_load.event.set()
        try:
            yield entry.obj
        finally:
            self._release(key, entry)

    def evict(self, key: str) -> bool:
        """Evict and free *key* now if nothing holds it; return whether it was freed (#63).

        A key with live references (an inference in flight) is left alone and
        reported as not freed, so a caller can't pull a model out from under a
        running caption. An unknown key is simply not freed.
        """
        with self._lock:
            entry = self._models.get(key)
            if entry is None or entry.refs > 0:
                return False
            del self._models[key]
        logger.info("model_evicted", key=key, reason="explicit")
        _free_entry(entry)
        return True

    def status(self) -> dict[str, dict[str, Any]]:
        """Return current cache state for diagnostics."""
        with self._lock:
            now = time.monotonic()
            return {
                k: {"refs": e.refs, "idle_seconds": round(now - e.last_used, 1), "release_pending": e.release_pending}
                for k, e in self._models.items()
            }

    def clear(self) -> list[str]:
        """Evict every model; return the keys whose free was deferred because they are in use.

        Unreferenced models are freed now. A model still held (or still loading) stays
        cached, so concurrent callers share it instead of loading a second copy, and is
        freed when its last holder releases it.
        """
        with self._lock:
            evicted = {k: e for k, e in self._models.items() if e.refs == 0}
            for k in evicted:
                del self._models[k]
            deferred = [k for k in self._models]
            for e in self._models.values():
                e.release_pending = True
            for k, load in self._loading.items():
                load.release_pending = True
                deferred.append(k)
        for key, entry in evicted.items():
            logger.info("model_cleared", key=key)
            _free_entry(entry)
        if deferred:
            logger.info("model_clear_deferred", keys=sorted(deferred))
        return deferred


# Shared singleton used by built-in backends.
_default_registry = ModelRegistry()


def get_default_registry() -> ModelRegistry:
    """Return the process-level shared registry."""
    return _default_registry
