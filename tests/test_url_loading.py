"""Tests for URL-based image loading in _load_image."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from argus_lens.engine import _load_image


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    status_code = 200
    content = _make_test_png()

    def raise_for_status(self) -> None:
        pass


class _FakeResponseBadStatus:
    status_code = 404

    def raise_for_status(self) -> None:
        import httpx

        raise httpx.HTTPStatusError(
            "Not Found",
            request=httpx.Request("GET", "https://example.com/missing.jpg"),
            response=self,  # type: ignore[arg-type]
        )


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_from_url(mock_get):
    name, pil = _load_image("https://example.com/photos/sunset.jpg")

    mock_get.assert_called_once_with(
        "https://example.com/photos/sunset.jpg",
        follow_redirects=True,
        timeout=30.0,
    )
    assert name == "sunset.jpg"
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    assert pil.size == (64, 64)


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_url_with_query_params(mock_get):
    name, pil = _load_image("https://cdn.example.com/image.png?token=abc&size=lg")
    assert name == "image.png"


@patch("httpx.get", return_value=_FakeResponse())
def test_load_image_url_trailing_slash(mock_get):
    name, _ = _load_image("https://example.com/")
    assert name == "image"


@patch("httpx.get", return_value=_FakeResponseBadStatus())
def test_load_image_url_http_error(mock_get):
    with pytest.raises(Exception):
        _load_image("https://example.com/missing.jpg")


def test_load_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        _load_image("/nonexistent/path/to/image.jpg")


def test_load_image_from_bytes():
    data = _make_test_png()
    name, pil = _load_image(data)
    assert name == "bytes"
    assert pil.size == (64, 64)


def test_load_image_from_pil():
    img = Image.new("RGB", (32, 32))
    name, pil = _load_image(img)
    assert name == "image"
    assert pil.size == (32, 32)
