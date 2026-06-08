"""Source/Sink connector protocols (issue #6)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class AssetRef:
    """A reference to an external asset.

    Attributes:
        id: Stable identifier within the source system.
        path: Local filesystem path, if applicable.
        uri: Remote URI/URL, if applicable.
    """

    id: str
    path: str | None = None
    uri: str | None = None


@runtime_checkable
class Source(Protocol):
    """Lists and fetches assets from an external system."""

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        """Yield asset refs, optionally only those changed since a cursor."""
        ...

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        """Load the image for *ref*."""
        ...


@runtime_checkable
class Sink(Protocol):
    """Writes tagging results back to an external system."""

    def write(self, ref: AssetRef, *, keywords: list[str], description: str = "") -> None:
        """Persist keywords/description for *ref*."""
        ...
