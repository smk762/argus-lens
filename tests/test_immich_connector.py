"""Tests for the Immich connector (issues #7, #29)."""

import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from argus_lens.connectors import AssetRef, ImmichSink, ImmichSource, Sink, Source


def _png_bytes(size=(5, 5), color=(0, 0, 255)) -> bytes:
    """Return PNG-encoded bytes for a solid-color test image."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _search_page(ids: list[str], next_page: str | None) -> dict:
    """Build a fake ``/api/search/metadata`` response page for *ids*."""
    return {
        "albums": {"total": 0, "count": 0, "items": [], "facets": []},
        "assets": {
            "total": len(ids),
            "count": len(ids),
            "items": [{"id": i, "originalPath": f"/on/server/{i}.jpg"} for i in ids],
            "facets": [],
            "nextPage": next_page,
        },
    }


class _FakeResponse:
    """Fake httpx response carrying canned bytes/JSON and a status code."""

    def __init__(self, content: bytes = b"", json_data: dict | list | None = None, status_code: int = 200) -> None:
        """Store the response body, optional JSON payload, and status code."""
        self.content = content
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        """Return the canned JSON payload."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Raise httpx.HTTPStatusError for 4xx/5xx status codes, like the real client."""
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://immich.local"),
                response=httpx.Response(self.status_code),
            )


def test_immich_implements_protocols():
    """ImmichSource and ImmichSink satisfy the Source and Sink protocols."""
    assert isinstance(ImmichSource("http://immich.local", "key"), Source)
    assert isinstance(ImmichSink("http://immich.local", "key"), Sink)


def test_url_and_headers():
    """Builds URLs from the base host and sends the API key with an overridable Accept header."""
    src = ImmichSource("http://immich.local/", "secret")
    assert src._url("/api/server-info/ping") == "http://immich.local/api/server-info/ping"
    headers = src._headers()
    assert headers["x-api-key"] == "secret"
    assert headers["Accept"] == "application/json"
    # binary endpoints override Accept so the server doesn't return JSON
    assert src._headers(accept="*/*")["Accept"] == "*/*"


# --- ImmichSource.list_assets ---


def test_list_assets_pages_until_next_page_is_null():
    """Follows the nextPage token across search pages and yields refs with API URIs."""
    pages = [
        _FakeResponse(json_data=_search_page(["a", "b"], next_page="2")),
        _FakeResponse(json_data=_search_page(["c"], next_page=None)),
    ]
    with patch("httpx.post", side_effect=pages) as mock_post:
        refs = list(ImmichSource("http://immich.local", "key", page_size=2).list_assets())

    assert [r.id for r in refs] == ["a", "b", "c"]
    # refs carry the API download URL, never a server-side filesystem path
    assert refs[0].uri == "http://immich.local/api/assets/a/original"
    assert refs[0].path is None

    assert mock_post.call_count == 2
    first, second = mock_post.call_args_list
    assert first[0][0] == "http://immich.local/api/search/metadata"
    assert first.kwargs["json"] == {"page": 1, "size": 2, "type": "IMAGE"}
    assert second.kwargs["json"] == {"page": 2, "size": 2, "type": "IMAGE"}
    assert first.kwargs["headers"]["x-api-key"] == "key"


def test_list_assets_since_maps_to_updated_after():
    """Maps the `since` cursor to Immich's updatedAfter filter for incremental sync."""
    with patch("httpx.post", return_value=_FakeResponse(json_data=_search_page([], next_page=None))) as mock_post:
        list(ImmichSource("http://immich.local", "key").list_assets(since="2026-07-01T00:00:00Z"))

    assert mock_post.call_args.kwargs["json"]["updatedAfter"] == "2026-07-01T00:00:00Z"


def test_list_assets_is_lazy_and_raises_on_http_error():
    """Makes no request until consumed, then propagates HTTP errors from the search API."""
    with patch("httpx.post", return_value=_FakeResponse(status_code=401)) as mock_post:
        it = ImmichSource("http://immich.local", "key").list_assets()
        mock_post.assert_not_called()  # generator: no request until consumed
        with pytest.raises(httpx.HTTPStatusError):
            next(it)


# --- ImmichSource.list_albums / list_album_assets ---


def test_list_albums_maps_immich_fields():
    """Maps Immich's albumName/assetCount to snake_case with safe fallbacks."""
    albums_json = [
        {"id": "al1", "albumName": "Trip", "assetCount": 3},
        {"id": "al2"},  # missing name/count fall back to "" / 0
    ]
    with patch("httpx.get", return_value=_FakeResponse(json_data=albums_json)) as mock_get:
        albums = ImmichSource("http://immich.local", "key").list_albums()

    assert albums == [
        {"id": "al1", "name": "Trip", "asset_count": 3},
        {"id": "al2", "name": "", "asset_count": 0},
    ]
    assert mock_get.call_args[0][0] == "http://immich.local/api/albums"
    assert mock_get.call_args.kwargs["headers"]["x-api-key"] == "key"


def test_list_albums_raises_on_http_error():
    """Propagates httpx.HTTPStatusError when the album listing fails."""
    with (
        patch("httpx.get", return_value=_FakeResponse(status_code=401)),
        pytest.raises(httpx.HTTPStatusError),
    ):
        ImmichSource("http://immich.local", "key").list_albums()


def test_list_album_assets_keeps_images_and_encodes_album_id():
    """Keeps image assets only, falls back to the id for missing filenames, and encodes the album id."""
    album_json = {
        "assets": [
            {"id": "a", "originalFileName": "a.jpg", "type": "IMAGE"},
            {"id": "v", "originalFileName": "v.mp4", "type": "VIDEO"},
            {"id": "b"},  # missing type is treated as IMAGE; name falls back to id
        ]
    }
    with patch("httpx.get", return_value=_FakeResponse(json_data=album_json)) as mock_get:
        assets = ImmichSource("http://immich.local", "key").list_album_assets("al/1")

    assert assets == [{"id": "a", "name": "a.jpg"}, {"id": "b", "name": "b"}]
    assert mock_get.call_args[0][0] == "http://immich.local/api/albums/al%2F1"


# --- ImmichSource.fetch_original ---


def test_fetch_original_returns_raw_bytes():
    """Returns the undecoded original bytes with a binary Accept header."""
    with patch("httpx.get", return_value=_FakeResponse(b"raw-bytes")) as mock_get:
        data = ImmichSource("http://immich.local", "key").fetch_original(AssetRef(id="abc"))

    assert data == b"raw-bytes"
    assert mock_get.call_args[0][0] == "http://immich.local/api/assets/abc/original"
    assert mock_get.call_args.kwargs["headers"]["Accept"] == "*/*"


# --- ImmichSource.fetch_image ---


def test_fetch_image_hits_original_endpoint():
    """Fetches the original-asset endpoint with a binary Accept header and decodes the image."""
    with patch("httpx.get", return_value=_FakeResponse(_png_bytes())) as mock_get:
        img = ImmichSource("http://immich.local", "key").fetch_image(AssetRef(id="abc123"))

    assert img.size == (5, 5)
    assert mock_get.call_args[0][0] == "http://immich.local/api/assets/abc123/original"
    # binary fetch must not request a JSON response
    assert mock_get.call_args.kwargs["headers"]["Accept"] == "*/*"


def test_fetch_image_percent_encodes_asset_id():
    """Percent-encodes asset IDs when building the original-asset URL."""
    with patch("httpx.get", return_value=_FakeResponse(_png_bytes())) as mock_get:
        ImmichSource("http://immich.local", "key").fetch_image(AssetRef(id="x?a=1/../y"))

    # The id is encoded into a single safe path segment (no raw ? or / leaking).
    assert mock_get.call_args[0][0] == "http://immich.local/api/assets/x%3Fa%3D1%2F..%2Fy/original"


def test_fetch_image_raises_on_http_error():
    """Propagates httpx.HTTPStatusError when the asset fetch returns an error status."""
    with (
        patch("httpx.get", return_value=_FakeResponse(status_code=404)),
        pytest.raises(httpx.HTTPStatusError),
    ):
        ImmichSource("http://immich.local", "key").fetch_image(AssetRef(id="missing"))


# --- ImmichSink.write ---


def test_write_sets_description_and_upserts_and_assigns_tags():
    """Sets the description, upserts keywords as tags, and bulk-assigns them to the asset."""
    responses = [
        _FakeResponse(json_data={"id": "x"}),  # PUT /api/assets/x
        _FakeResponse(json_data=[{"id": "t1", "name": "beach"}, {"id": "t2", "name": "sunset"}]),  # PUT /api/tags
        _FakeResponse(json_data={"count": 1}),  # PUT /api/tags/assets
    ]
    with patch("httpx.put", side_effect=responses) as mock_put:
        ImmichSink("http://immich.local", "key").write(
            AssetRef(id="x"), keywords=["beach", "sunset"], description="A sunset at the beach"
        )

    urls = [c[0][0] for c in mock_put.call_args_list]
    assert urls == [
        "http://immich.local/api/assets/x",
        "http://immich.local/api/tags",
        "http://immich.local/api/tags/assets",
    ]
    desc_call, upsert_call, assign_call = mock_put.call_args_list
    assert desc_call.kwargs["json"] == {"description": "A sunset at the beach"}
    assert upsert_call.kwargs["json"] == {"tags": ["beach", "sunset"]}
    assert assign_call.kwargs["json"] == {"tagIds": ["t1", "t2"], "assetIds": ["x"]}
    assert desc_call.kwargs["headers"]["x-api-key"] == "key"


def test_write_percent_encodes_asset_id_in_description_update():
    """Percent-encodes asset IDs when building the asset-update URL."""
    with patch("httpx.put", return_value=_FakeResponse(json_data={})) as mock_put:
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x/../y"), keywords=[], description="d")

    assert mock_put.call_args[0][0] == "http://immich.local/api/assets/x%2F..%2Fy"


def test_write_skips_empty_description_and_empty_keywords():
    """Skips API calls for empty values so blanks never clobber existing Immich metadata."""
    with patch("httpx.put") as mock_put:
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x"), keywords=[], description="")

    mock_put.assert_not_called()

    with patch(
        "httpx.put",
        side_effect=[_FakeResponse(json_data=[{"id": "t1", "name": "a"}]), _FakeResponse(json_data={"count": 1})],
    ) as mock_put:
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x"), keywords=["a"])

    # no description -> no PUT /api/assets, straight to tag upsert + assign
    assert [c[0][0] for c in mock_put.call_args_list] == [
        "http://immich.local/api/tags",
        "http://immich.local/api/tags/assets",
    ]


def test_write_raises_on_http_error_and_stops():
    """Stops after the first failing call and propagates the HTTP error."""
    with (
        patch("httpx.put", return_value=_FakeResponse(status_code=403)) as mock_put,
        pytest.raises(httpx.HTTPStatusError),
    ):
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x"), keywords=["a"], description="d")

    mock_put.assert_called_once()  # failed on description update; no tag calls followed
