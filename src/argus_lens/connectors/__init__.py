"""External-system connectors (issue #6).

Source/Sink abstractions that let the same engine drop into any external system
(filesystem, Immich, Lightroom, ...). ``XmpSink`` is the zero-coupling default
that also works for Lightroom/digiKam.
"""

from argus_lens.connectors.base import AssetRef, Sink, Source
from argus_lens.connectors.filesystem import FilesystemSource
from argus_lens.connectors.xmp import XmpSink

__all__ = ["AssetRef", "Sink", "Source", "FilesystemSource", "XmpSink"]
