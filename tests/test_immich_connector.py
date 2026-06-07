"""Tests for the Immich connector scaffold (issue #7)."""

from unittest.mock import patch

import pytest
from PIL import Image

from argus_lens.connectors.base import AssetRef, Sink, Source
from argus_lens.connectors.immich import ImmichSink, ImmichSource


def test_immich_implements_protocols():
    assert isinstance(ImmichSource("http://immich.local", "key"), Source)
    assert isinstance(ImmichSink("http://immich.local", "key"), Sink)


def test_url_and_headers():
    src = ImmichSource("http://immich.local/", "secret")
    assert src._url("/api/server-info/ping") == "http://immich.local/api/server-info/ping"
    assert src._headers()["x-api-key"] == "secret"


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass


def test_fetch_image_hits_original_endpoint():
    import io

    buf = io.BytesIO()
    Image.new("RGB", (5, 5), (0, 0, 255)).save(buf, format="PNG")

    with patch("httpx.get", return_value=_FakeResponse(buf.getvalue())) as mock_get:
        img = ImmichSource("http://immich.local", "key").fetch_image(AssetRef(id="abc123"))

    assert img.size == (5, 5)
    assert mock_get.call_args[0][0] == "http://immich.local/api/assets/abc123/original"


def test_listing_and_write_are_stubbed():
    with pytest.raises(NotImplementedError):
        next(ImmichSource("http://immich.local", "key").list_assets())
    with pytest.raises(NotImplementedError):
        ImmichSink("http://immich.local", "key").write(AssetRef(id="x"), keywords=["a"])
