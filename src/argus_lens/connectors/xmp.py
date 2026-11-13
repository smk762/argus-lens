"""XMP sidecar sink (issue #6).

Writes ``dc:subject`` (keywords) and ``dc:description`` (caption) into a
``.xmp`` sidecar. This is the zero-coupling integration surface: Immich,
Lightroom, and digiKam all ingest XMP sidecars, so the same output drops into
any of them.

The XMP read/merge/write itself lives in :mod:`argus_cortex.xmp` — the suite's
one implementation, shared with argus-curator (face regions, content rating) —
so writes **merge** into an existing sidecar: lens's two properties are
replaced (latest wins), everything another tool or stage stored is preserved.
"""

from __future__ import annotations

from pathlib import Path

from argus_cortex.xmp import XmpData, render_xmp, sidecar_path, write_xmp

from argus_lens.connectors.base import AssetRef


class XmpSink:
    """Writes keywords/description to a ``<image>.xmp`` sidecar."""

    @staticmethod
    def sidecar_path(ref: AssetRef) -> Path:
        """Return the ``<image>.xmp`` sidecar path for *ref* (e.g. ``cat.jpg.xmp``).

        Raises:
            ValueError: if *ref* has no local path.
        """
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path for an XMP sidecar")
        return sidecar_path(ref.path)

    def write(
        self,
        ref: AssetRef,
        *,
        keywords: list[str],
        description: str = "",
        overwrite: bool = False,
    ) -> None:
        """Write keywords/description into the ``<image>.xmp`` sidecar next to *ref*.

        An existing sidecar is merged: ``dc:subject``/``dc:description`` are
        replaced and every other property (ratings, GPS, develop settings, face
        regions, …) is kept; an empty *keywords*/*description* leaves that
        property as it was. ``overwrite=True`` instead discards the existing
        sidecar and writes a fresh one holding only these two properties.

        Raises:
            ValueError: if *ref* has no local path.
            argus_cortex.xmp.XmpError: if merging and the existing sidecar can't
                be parsed (it is left untouched).
        """
        self.sidecar_path(ref)  # path-less refs fail before any I/O
        # No keywords / no caption means lens has nothing to say about that
        # property, so leave it (None) rather than clear it ([]/"" removes it).
        data = XmpData(keywords=keywords or None, description=description or None)
        write_xmp(ref.path, data, replace=overwrite)

    def render(self, *, keywords: list[str], description: str = "") -> str:
        """Render a fresh XMP document as a string (pure; useful for tests)."""
        return render_xmp(XmpData(keywords=keywords, description=description))
