"""Shared streaming download for local-backend model files (WD14, RAM++)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger()


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
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info(f"{event}.done", file=dest.name, size_mb=round(dest.stat().st_size / 1024 / 1024, 1))
