"""Immich connector (issue #7).

Immich is strong at CLIP search + faces but weak at descriptive keywords /
captions -- the gap Argus Lens fills. Immich has no in-process ML plugin hook,
so this runs as a companion service: pull assets via the API, tag them, and
push keywords/description back (or fall back to ``XmpSink``).

Status: scaffold. Request-building helpers are implemented and tested; the
paged listing and write-back calls are stubbed pending implementation.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from urllib.parse import quote

import httpx
from PIL import Image

from argus_lens.connectors.base import AssetRef

DEFAULT_TIMEOUT = 30.0


class _ImmichClient:
    """Shared base: holds connection details and auth header."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Store server URL (trailing slash stripped), API key, and request timeout."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        """Auth headers. *accept* is overridable since binary endpoints (e.g.
        ``/original``) should not request a JSON response."""
        return {"x-api-key": self._api_key, "Accept": accept}

    def _url(self, path: str) -> str:
        """Join *path* onto the server base URL."""
        return f"{self._base_url}/{path.lstrip('/')}"

    def _asset_path(self, asset_id: str, suffix: str = "") -> str:
        """Build an ``/api/assets/<id>`` path with the id percent-encoded."""
        return f"/api/assets/{quote(asset_id, safe='')}{suffix}"


class ImmichSource(_ImmichClient):
    """Lists and fetches assets from an Immich server."""

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        """Page through Immich assets — stub, not yet implemented (#7)."""
        # TODO(#7): page through POST /api/search/metadata (use `since` -> updatedAfter).
        raise NotImplementedError("Immich asset listing is not yet implemented (#7)")

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        """Download the asset's original file from Immich and decode it as RGB."""
        resp = httpx.get(
            self._url(self._asset_path(ref.id, "/original")),
            headers=self._headers(accept="*/*"),
            timeout=self._timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        with Image.open(io.BytesIO(resp.content)) as img:
            return img.convert("RGB")


class ImmichSink(_ImmichClient):
    """Writes keywords/description back to Immich."""

    def write(self, ref: AssetRef, *, keywords: list[str], description: str = "") -> None:
        """Push keywords and description to an Immich asset — stub, not yet implemented (#7)."""
        # TODO(#7): PUT /api/assets/{id} description; upsert tags via /api/tags + assign.
        raise NotImplementedError("Immich write-back is not yet implemented (#7)")
