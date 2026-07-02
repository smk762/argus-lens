"""Immich connector (issues #7, #29).

Immich is strong at CLIP search + faces but weak at descriptive keywords /
captions -- the gap Argus Lens fills. Immich has no in-process ML plugin hook,
so this runs as a companion service: pull assets via the API, tag them, and
push keywords/description back (or fall back to ``XmpSink``).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image

from argus_lens.connectors.base import AssetRef

DEFAULT_TIMEOUT = 30.0
DEFAULT_PAGE_SIZE = 250  # Immich's server-side default for /api/search/metadata


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

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        """Store connection details plus the page size used by :meth:`list_assets`."""
        super().__init__(base_url, api_key, timeout=timeout)
        self._page_size = page_size

    def list_assets(self, since: str | None = None) -> Iterator[AssetRef]:
        """Yield every image asset, paging through ``POST /api/search/metadata``.

        Args:
            since: ISO 8601 timestamp; when given, only assets updated after it
                are listed (maps to Immich's ``updatedAfter``), which makes
                repeated runs an incremental change feed.
        """
        page: int | None = 1
        while page is not None:
            payload: dict[str, Any] = {"page": page, "size": self._page_size, "type": "IMAGE"}
            if since is not None:
                payload["updatedAfter"] = since
            resp = httpx.post(
                self._url("/api/search/metadata"),
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            assets = resp.json()["assets"]
            for item in assets["items"]:
                # `originalPath` is a path on the *Immich server*, not this
                # machine, so it goes in `uri` context via the API URL instead
                # of `path` (which sinks like XmpSink treat as local).
                yield AssetRef(id=item["id"], uri=self._url(self._asset_path(item["id"], "/original")))
            next_page = assets.get("nextPage")
            page = int(next_page) if next_page else None

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
        """Push *keywords* and *description* to the Immich asset *ref*.

        Keywords are upserted via ``PUT /api/tags`` (idempotent; existing tags
        are reused) and attached with ``PUT /api/tags/assets``. The description
        is set via ``PUT /api/assets/{id}``.

        Empty values are skipped rather than written, so an empty *description*
        never clobbers one already set in Immich.
        """
        if description:
            resp = httpx.put(
                self._url(self._asset_path(ref.id)),
                headers=self._headers(),
                json={"description": description},
                timeout=self._timeout,
            )
            resp.raise_for_status()

        if keywords:
            resp = httpx.put(
                self._url("/api/tags"),
                headers=self._headers(),
                json={"tags": keywords},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            tag_ids = [tag["id"] for tag in resp.json()]

            resp = httpx.put(
                self._url("/api/tags/assets"),
                headers=self._headers(),
                json={"tagIds": tag_ids, "assetIds": [ref.id]},
                timeout=self._timeout,
            )
            resp.raise_for_status()
