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
        """Store the root directory to scan for images."""
        self._root = Path(root)

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        """Yield a ref for every image file under the root, in sorted path order.

        *since* is accepted for protocol compatibility but ignored — the
        filesystem source has no change tracking.
        """
        for path in sorted(self._root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield AssetRef(id=str(path.relative_to(self._root)), path=str(path))

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        """Load the referenced file from disk as an RGB PIL image."""
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path")
        # Use a context manager so the source file handle is closed; convert()
        # returns a fully-loaded copy that outlives the closed file.
        with Image.open(ref.path) as img:
            return img.convert("RGB")
