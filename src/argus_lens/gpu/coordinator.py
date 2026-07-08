"""Pluggable GPU coordination (#38).

A ``GpuCoordinator`` gates heavy inference so argus-lens is a good tenant on a
GPU it shares with other processes (e.g. an SDXL generator). Three
implementations, selected via ``.env``:

* ``none`` — no-op (default); inference is unchanged.
* ``lease`` — a self-contained cross-process file lock (one heavy job at a
  time); the portable "standard shape" for hosts without a broker.
* ``gothmog`` — an adapter to gothmog's ``/v1/gpu`` capacity API.

The coordinator is only ever a *gate*; the model lifecycle levers (unload,
capacity probe) live on the engine and in ``gpu.capacity``.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()


@runtime_checkable
class GpuCoordinator(Protocol):
    """Gates a block of GPU work behind a capacity lease."""

    name: str

    def lease(self, *, caller: str, min_vram_mb: int) -> contextlib.AbstractContextManager[None]:
        """Return a context manager that holds a GPU slot for its body."""
        ...


class NullCoordinator:
    """No-op coordinator: leasing is a pass-through. The default."""

    name = "none"

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int) -> Iterator[None]:
        """Grant immediately; do nothing."""
        yield


class LocalLeaseCoordinator:
    """One-heavy-job-at-a-time via an advisory file lock (POSIX ``flock``).

    Cross-process on a single host with no extra services. Blocks (with polling)
    until the lock is free or *timeout_s* elapses, then raises ``TimeoutError``.
    """

    name = "lease"

    def __init__(
        self,
        lock_path: str | None = None,
        timeout_s: float = 300.0,
        poll_s: float = 0.5,
    ) -> None:
        self.lock_path = lock_path or os.environ.get("ARGUS_GPU_LEASE_PATH") or _default_lock_path()
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int) -> Iterator[None]:
        """Acquire the exclusive file lock for the body, releasing on exit."""
        try:
            import fcntl  # noqa: PLC0415 - POSIX only; imported lazily
        except ImportError as exc:  # pragma: no cover - non-POSIX host
            raise RuntimeError("the 'lease' coordinator needs POSIX fcntl; use 'gothmog' or 'none'") from exc

        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + max(0.0, self.timeout_s)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"GPU lease not acquired within {self.timeout_s:.0f}s ({self.lock_path})"
                        ) from None
                    time.sleep(self.poll_s)
            logger.debug("gpu_lease_acquired", caller=caller, min_vram_mb=min_vram_mb, lock=self.lock_path)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class GothmogCoordinator:
    """Adapter to gothmog's ``/v1/gpu`` capacity API (acquire → run → release)."""

    name = "gothmog"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        acquire_timeout_s: float = 300.0,
        http_timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.acquire_timeout_s = acquire_timeout_s
        self.http_timeout_s = http_timeout_s

    def _headers(self) -> dict[str, str]:
        """Auth header when an API key is configured."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _acquire(self, caller: str, min_vram_mb: int) -> str | None:
        """POST /v1/gpu/capacity/acquire → token id (or None if the body lacks one)."""
        import httpx  # noqa: PLC0415

        # The broker may long-poll for up to acquire_timeout_s; the HTTP read
        # timeout must exceed that or the client abandons a request the broker
        # would still grant (dropping the caption and leaking the granted token).
        resp = httpx.post(
            f"{self.base_url}/v1/gpu/capacity/acquire",
            json={"caller": caller, "min_vram_mb": min_vram_mb, "timeout_s": self.acquire_timeout_s},
            headers=self._headers(),
            timeout=self.acquire_timeout_s + self.http_timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        token = body.get("token_id") or body.get("id")
        if token is None:
            logger.warning("gothmog_acquire_no_token", caller=caller, body_keys=sorted(body))
        return token

    def _release(self, token_id: str) -> None:
        """DELETE /v1/gpu/capacity/tokens/{token_id}."""
        import httpx  # noqa: PLC0415

        with contextlib.suppress(Exception):
            httpx.delete(
                f"{self.base_url}/v1/gpu/capacity/tokens/{token_id}",
                headers=self._headers(),
                timeout=self.http_timeout_s,
            )

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int) -> Iterator[None]:
        """Acquire a gothmog capacity token for the body, releasing on exit."""
        token = self._acquire(caller, min_vram_mb)
        logger.debug("gpu_lease_acquired", coordinator="gothmog", caller=caller, token=token)
        try:
            yield
        finally:
            if token:
                self._release(token)


def _default_lock_path() -> str:
    """Default lock file shared across argus-lens processes on this host."""
    import tempfile

    return os.path.join(tempfile.gettempdir(), "argus-lens-gpu.lock")


def build_coordinator(
    name: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    lock_path: str | None = None,
) -> GpuCoordinator:
    """Construct a coordinator by name. ``gothmog`` requires *base_url*."""
    key = (name or "none").strip().lower()
    if key in ("none", ""):
        return NullCoordinator()
    if key == "lease":
        return LocalLeaseCoordinator(lock_path=lock_path)
    if key == "gothmog":
        url = base_url or os.environ.get("GOTHMOG_URL")
        if not url:
            raise ValueError("the 'gothmog' coordinator needs GOTHMOG_URL (or base_url=)")
        return GothmogCoordinator(base_url=url, api_key=api_key)
    raise ValueError(f"Unknown GPU coordinator {name!r}. Choose from: none, lease, gothmog")


def coordinator_from_env() -> GpuCoordinator:
    """Build the coordinator from the environment (defaults to ``none``).

    Reads ``ARGUS_GPU_COORDINATOR`` (``none``/``lease``/``gothmog``),
    ``GOTHMOG_URL``, ``GOTHMOG_API_KEY``, ``ARGUS_GPU_LEASE_PATH``.
    """
    return build_coordinator(
        os.environ.get("ARGUS_GPU_COORDINATOR", "none"),
        base_url=os.environ.get("GOTHMOG_URL"),
        api_key=os.environ.get("GOTHMOG_API_KEY"),
        lock_path=os.environ.get("ARGUS_GPU_LEASE_PATH"),
    )
