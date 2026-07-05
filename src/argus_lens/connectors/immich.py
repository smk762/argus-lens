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
    """Shared base: holds connection details, auth header, and a pooled HTTP client."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Store server URL (trailing slash stripped), API key, and request timeout."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def _http(self) -> httpx.Client:
        """Lazily-created pooled client shared by every request this instance makes.

        One-shot ``httpx.get``/``httpx.put`` calls open a fresh TCP+TLS
        connection each time; the per-asset loops (album pulls, batch
        write-back) would pay that handshake once per asset.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        """Close the pooled HTTP connection (safe to call repeatedly or when unused)."""
        if self._client is not None:
            self._client.close()
            self._client = None

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
            resp = self._http.post(
                self._url("/api/search/metadata"),
                headers=self._headers(),
                json=payload,
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

    def list_albums(self) -> list[dict[str, Any]]:
        """List albums via ``GET /api/albums``.

        Returns one ``{"id", "name", "asset_count"}`` dict per album, mapping
        Immich's ``albumName``/``assetCount`` fields to snake_case.
        """
        resp = self._http.get(self._url("/api/albums"), headers=self._headers())
        resp.raise_for_status()
        return [
            {
                "id": album["id"],
                "name": album.get("albumName", ""),
                "asset_count": int(album.get("assetCount") or 0),
            }
            for album in resp.json()
        ]

    def list_album_assets(self, album_id: str) -> list[dict[str, Any]]:
        """List an album's image assets via ``GET /api/albums/{id}``.

        Returns one ``{"id", "name"}`` dict per image asset (``name`` is
        Immich's ``originalFileName``); non-image assets (videos) are skipped.
        """
        resp = self._http.get(
            self._url(f"/api/albums/{quote(album_id, safe='')}"),
            headers=self._headers(),
        )
        resp.raise_for_status()
        return [
            {"id": asset["id"], "name": asset.get("originalFileName") or asset["id"]}
            for asset in resp.json().get("assets", [])
            if asset.get("type", "IMAGE") == "IMAGE"
        ]

    def fetch_original(self, ref: AssetRef) -> bytes:
        """Download the asset's original file bytes from Immich (no decoding)."""
        resp = self._http.get(
            self._url(self._asset_path(ref.id, "/original")),
            headers=self._headers(accept="*/*"),
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.content

    def fetch_image(self, ref: AssetRef) -> Image.Image:
        """Download the asset's original file from Immich and decode it as RGB."""
        data = self.fetch_original(ref)
        with Image.open(io.BytesIO(data)) as img:
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
            resp = self._http.put(
                self._url(self._asset_path(ref.id)),
                headers=self._headers(),
                json={"description": description},
            )
            resp.raise_for_status()

        if keywords:
            resp = self._http.put(
                self._url("/api/tags"),
                headers=self._headers(),
                json={"tags": keywords},
            )
            resp.raise_for_status()
            tag_ids = [tag["id"] for tag in resp.json()]

            resp = self._http.put(
                self._url("/api/tags/assets"),
                headers=self._headers(),
                json={"tagIds": tag_ids, "assetIds": [ref.id]},
            )
            resp.raise_for_status()
