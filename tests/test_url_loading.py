"""Tests for URL-based image loading in _load_image."""

import hashlib
import io
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from argus_lens.engine import _load_image


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Return PNG-encoded bytes for a solid-red test image."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    """Fake httpx response returning canned PNG bytes."""

    status_code = 200
    content = _make_test_png()

    def raise_for_status(self) -> None:
        """No-op; the response is a success."""
        pass


class _FakeResponseBadStatus:
    """Fake httpx response simulating a 404 error."""

    status_code = 404

    def raise_for_status(self) -> None:
        """Raise httpx.HTTPStatusError as the real client would for a 404."""
        import httpx

        raise httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("GET", "https://example.com/missing.jpg"),
            response=self,  # type: ignore[arg-type]
        )


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_from_url(mock_get):
    """Downloads an http(s) URL with redirects and a timeout, naming the image after the file."""
    asset, pil = _load_image("https://example.com/photos/sunset.jpg")

    mock_get.assert_called_once_with(
        "https://example.com/photos/sunset.jpg",
        follow_redirects=True,
        timeout=30.0,
    )
    assert asset.name == "sunset.jpg"
    # The original-bytes sha256 and source URL are captured for replay keying.
    assert asset.uri == "https://example.com/photos/sunset.jpg"
    assert asset.sha256 == hashlib.sha256(_FakeResponse.content).hexdigest()
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    assert pil.size == (64, 64)


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_url_with_query_params(mock_get):
    """Derives the image name from the URL path, ignoring query parameters."""
    asset, pil = _load_image("https://cdn.example.com/image.png?token=abc&size=lg")
    assert asset.name == "image.png"


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_url_trailing_slash(mock_get):
    """Falls back to the name "image" when the URL path has no filename."""
    asset, _ = _load_image("https://example.com/")
    assert asset.name == "image"


@patch("httpx.get", return_value=_FakeResponseBadStatus())
def test_load_image_url_http_error(mock_get):
    """Propagates httpx.HTTPStatusError when the URL fetch fails."""
    with pytest.raises(httpx.HTTPStatusError):
        _load_image("https://example.com/missing.jpg")


def test_load_image_file_not_found():
    """Raises FileNotFoundError for a nonexistent local path."""
    with pytest.raises(FileNotFoundError):
        _load_image("/nonexistent/path/to/image.jpg")


def test_load_image_from_bytes():
    """Decodes raw image bytes, names the result "bytes", and hashes the bytes."""
    data = _make_test_png()
    asset, pil = _load_image(data)
    assert asset.name == "bytes"
    assert asset.sha256 == hashlib.sha256(data).hexdigest()
    assert pil.size == (64, 64)


def test_load_image_from_pil():
    """Accepts a PIL image directly; it carries no original-bytes sha256."""
    img = Image.new("RGB", (32, 32))
    asset, pil = _load_image(img)
    assert asset.name == "image"
    assert asset.sha256 is None
    assert pil.size == (32, 32)
