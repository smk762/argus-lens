"""Tests for the Immich connector scaffold (issue #7)."""

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


class _FakeResponse:
    """Fake httpx response carrying canned bytes and a status code."""

    def __init__(self, content: bytes, status_code: int = 200) -> None:
        """Store the response body and status code."""
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise httpx.HTTPStatusError for 4xx/5xx status codes, like the real client."""
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://immich.local"),
                response=httpx.Response(self.status_code),
            )


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
        patch("httpx.get", return_value=_FakeResponse(b"", status_code=404)),
        pytest.raises(httpx.HTTPStatusError),
    ):
        ImmichSource("http://immich.local", "key").fetch_image(AssetRef(id="missing"))


def test_listing_and_write_are_stubbed():
    """list_assets and Sink.write raise NotImplementedError while still scaffolded."""
    with pytest.raises(NotImplementedError):
        next(ImmichSource("http://immich.local", "key").list_assets())
    with pytest.raises(NotImplementedError):
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x"), keywords=["a"])
