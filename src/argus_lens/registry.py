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
from typing import Any

import structlog

logger = structlog.get_logger()

_DEFAULT_IDLE_SECONDS = float(os.environ.get("ARGUS_MODEL_IDLE_SECONDS", "300"))


def _free_model(obj: Any) -> None:
    """Release memory held by a model object."""
    if isinstance(obj, tuple):
        for part in obj:
            _free_model(part)
        return
    try:
        import torch

        if hasattr(obj, "parameters"):
            obj.cpu()
        del obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class ModelRegistry:
    """Thread-safe model cache with TTL eviction and reference counting.

    Usage::

        registry = ModelRegistry(idle_seconds=300)

        with registry.acquire("wd14", loader_fn) as model:
            result = model.run(image)
        # reference released; model stays cached until idle timeout

    Concurrent ``acquire()`` calls for the same key wait for the first
    load to finish rather than loading the model multiple times.
    """

    def __init__(self, idle_seconds: float | None = None) -> None:
        self._models: dict[str, Any] = {}
        self._refs: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self._loading: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._idle_seconds = idle_seconds if idle_seconds is not None else _DEFAULT_IDLE_SECONDS
        self._evict_thread: threading.Thread | None = None

    def _ensure_evict_thread(self) -> None:
        if self._evict_thread and self._evict_thread.is_alive():
            return
        t = threading.Thread(target=self._evict_loop, daemon=True, name="argus-model-evict")
        t.start()
        self._evict_thread = t

    def _evict_loop(self) -> None:
        while True:
            time.sleep(30)
            self._evict_idle()

    def _evict_idle(self) -> None:
        now = time.monotonic()
        with self._lock:
            to_evict = [
                k for k in list(self._models)
                if self._refs.get(k, 0) == 0
                and now - self._last_used.get(k, 0) > self._idle_seconds
            ]
            evicted = {k: self._models.pop(k) for k in to_evict}
            for k in to_evict:
                self._refs.pop(k, None)
                self._last_used.pop(k, None)

        for k, obj in evicted.items():
            logger.info("model_evicted", key=k, idle_seconds=self._idle_seconds)
            _free_model(obj)

    @contextlib.contextmanager
    def acquire(self, key: str, loader: Callable[[], Any]) -> Generator[Any, None, None]:
        """Context manager that provides the model for *key*.

        If the model is not cached it is loaded by calling *loader()*.
        Concurrent calls with the same key block until the first load
        finishes.
        """
        model: Any = None
        load_event: threading.Event | None = None

        while model is None:
            wait_event: threading.Event | None = None
            with self._lock:
                if key in self._models:
                    self._refs[key] = self._refs.get(key, 0) + 1
                    self._last_used[key] = time.monotonic()
                    model = self._models[key]
                elif key in self._loading:
                    wait_event = self._loading[key]
                else:
                    load_event = threading.Event()
                    self._loading[key] = load_event

            if wait_event is not None:
                wait_event.wait()
            elif model is None and load_event is not None:
                try:
                    logger.info("model_loading", key=key)
                    obj = loader()
                    logger.info("model_loaded", key=key)
                    with self._lock:
                        self._models[key] = obj
                        self._refs[key] = 1
                        self._last_used[key] = time.monotonic()
                        self._loading.pop(key, None)
                        self._ensure_evict_thread()
                    load_event.set()
                    model = obj
                except Exception:
                    with self._lock:
                        self._loading.pop(key, None)
                    load_event.set()
                    raise
        try:
            yield model
        finally:
            with self._lock:
                self._refs[key] = max(0, self._refs.get(key, 1) - 1)
                self._last_used[key] = time.monotonic()

    def status(self) -> dict[str, dict[str, Any]]:
        """Return current cache state for diagnostics."""
        with self._lock:
            now = time.monotonic()
            return {
                k: {
                    "refs": self._refs.get(k, 0),
                    "idle_seconds": round(now - self._last_used.get(k, now), 1),
                }
                for k in self._models
            }

    def clear(self) -> None:
        """Evict all models immediately."""
        with self._lock:
            evicted = self._models
            self._models = {}
            self._refs = {}
            self._last_used = {}
            self._loading = {}
        for key, obj in evicted.items():
            logger.info("model_cleared", key=key)
            _free_model(obj)
        gc.collect()


# Shared singleton used by built-in backends.
_default_registry = ModelRegistry()


def get_default_registry() -> ModelRegistry:
    """Return the process-level shared registry."""
    return _default_registry
