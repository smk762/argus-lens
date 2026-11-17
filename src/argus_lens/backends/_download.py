"""Shared streaming download for local-backend model files (WD14, RAM++)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()


def _umask() -> int:
    """The process umask, read without changing it where the OS allows (Linux ``/proc``)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of *directory* so a rename into it survives a crash (no-op where unsupported)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def stream_download(url: str, dest: Path, *, event: str, chunk_size: int = 1024 * 1024) -> None:
    """Stream *url* to *dest*, atomically.

    The body goes to a uniquely named temp file next to *dest*, which is renamed
    into place only once complete, so a concurrent download (another process, or
    another engine in this one) never interleaves with it, and a failed one never
    leaves a truncated file under the real name. The temp file is removed on any
    failure. *event* prefixes the structlog events (``<event>``, ``<event>.progress``,
    ``<event>.done``).
    """
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(event, file=dest.name, dest=str(dest))
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh, httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            for chunk in resp.iter_bytes(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    logger.debug(f"{event}.progress", file=dest.name, pct=downloaded * 100 // total)
            # Durable before it becomes visible: a crash after the rename must not leave
            # an empty/partial file under the real name (callers treat existence as done).
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates 0600; give the file the mode a plain open() would (shared caches).
        os.chmod(tmp, 0o666 & ~_umask())
        tmp.replace(dest)
        _fsync_dir(dest.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info(f"{event}.done", file=dest.name, size_mb=round(dest.stat().st_size / 1024 / 1024, 1))


# Weight formats transformers skips when a repo also ships safetensors (it prefers them), plus
# docs. Fetching them too would double the download (e.g. microsoft/Florence-2-base ships both).
_HF_SKIP_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "*.md"]


def prefetch_hf_snapshot(repo_id: str) -> None:
    """Download a Hugging Face model repo into the shared HF cache, without loading it (#64).

    ``from_pretrained`` later finds the files locally. Repos that ship *only* a
    skipped format (``.bin`` with no safetensors) still work: ``from_pretrained``
    fetches the remainder at load time, i.e. the old behaviour. A missing
    ``huggingface_hub`` (it comes with ``transformers``) is not an error here.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return
    logger.info("hf.prefetch", repo=repo_id)
    snapshot_download(repo_id, ignore_patterns=_HF_SKIP_PATTERNS)
