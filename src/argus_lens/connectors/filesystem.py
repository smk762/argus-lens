"""Filesystem source connector (issue #6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from argus_lens.connectors.base import AssetRef

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


class FilesystemSource:
    """Iterates image files under a directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        for path in sorted(self._root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield AssetRef(id=str(path.relative_to(self._root)), path=str(path))

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path")
        return Image.open(ref.path).convert("RGB")
