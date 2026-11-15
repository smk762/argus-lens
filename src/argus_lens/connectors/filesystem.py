"""Filesystem source connector (issue #6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from argus_lens.connectors.base import AssetRef

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


class FilesystemSource:
    """Iterates image files under a directory.

    Reads are confined to the root: a symlink whose target lies outside it is
    not listed, and :meth:`fetch_image` refuses a ref whose path resolves
    outside it (``..`` parts, absolute paths, escaping symlinks).
    """

    def __init__(self, root: str | Path) -> None:
        """Store the root directory to scan for images."""
        self._root = Path(root)

    def _within_root(self, path: Path) -> bool:
        """True when *path*, with symlinks resolved, lies under the root."""
        return path.resolve().is_relative_to(self._root.resolve())

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        """Yield a ref for every image file under the root, in sorted path order.

        *since* is accepted for protocol compatibility but ignored — the
        filesystem source has no change tracking. Files reached through a
        symlink pointing outside the root are skipped.
        """
        for path in sorted(self._root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and self._within_root(path):
                yield AssetRef(id=str(path.relative_to(self._root)), path=str(path))

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        """Load the referenced file from disk as an RGB PIL image.

        Raises ``ValueError`` when the ref has no local path or its path
        resolves outside the root.
        """
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path")
        path = Path(ref.path)
        if not self._within_root(path):
            raise ValueError(f"AssetRef {ref.id!r} path is outside the source root")
        # Use a context manager so the source file handle is closed; convert()
        # returns a fully-loaded copy that outlives the closed file.
        with Image.open(path) as img:
            return img.convert("RGB")
