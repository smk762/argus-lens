"""Immich connector (issue #7).

Immich is strong at CLIP search + faces but weak at descriptive keywords /
captions -- the gap Argus Lens fills. Immich has no in-process ML plugin hook,
so this runs as a companion service: pull assets via the API, tag them, and
push keywords/description back (or fall back to ``XmpSink``).

Status: scaffold. Request-building helpers are implemented and tested; the
paged listing and write-back calls are stubbed pending implementation.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
from PIL import Image

from argus_lens.connectors.base import AssetRef

DEFAULT_TIMEOUT = 30.0


class _ImmichClient:
    """Shared base: holds connection details and auth header."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "Accept": "application/json"}

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"


class ImmichSource(_ImmichClient):
    """Lists and fetches assets from an Immich server."""

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        # TODO(#7): page through POST /api/search/metadata (use `since` -> updatedAfter).
        raise NotImplementedError("Immich asset listing is not yet implemented (#7)")

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        resp = httpx.get(
            self._url(f"/api/assets/{ref.id}/original"),
            headers={"x-api-key": self._api_key},
            timeout=self._timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        import io

        return Image.open(io.BytesIO(resp.content)).convert("RGB")


class ImmichSink(_ImmichClient):
    """Writes keywords/description back to Immich."""

    def write(self, ref: AssetRef, *, keywords: list[str], description: str = "") -> None:
        # TODO(#7): PUT /api/assets/{id} description; upsert tags via /api/tags + assign.
        raise NotImplementedError("Immich write-back is not yet implemented (#7)")
