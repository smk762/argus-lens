"""Tests for URL-based image loading in _load_image."""

import functools
import hashlib
import io

import httpx
import pytest
from argus_cortex import net as cortex_net
from argus_cortex.net import FetchFailed, UrlRefused
from PIL import Image

from argus_lens import url_fetch
from argus_lens.engine import _load_image


def _make_test_png(width: int = 64, height: int = 64) -> bytes:
    """Return PNG-encoded bytes for a solid-red test image."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_PNG = _make_test_png()


@pytest.fixture
def net(monkeypatch):
    """Stub DNS and HTTP: ``net.dns`` maps host -> IPs, ``net.routes`` maps URL -> handler.

    Wired in through ``argus_cortex.net.fetch``'s own injection points
    (``resolve=`` and ``transport=``), so the real policy runs end to end.
    Every request that reaches the transport is recorded in ``net.seen`` as
    ``(connected_url, host_header, sni)``, so a test can assert no connection
    was made, or that it went to the checked IP.
    """

    class Net:
        dns: dict[str, list[str]] = {}
        routes: dict[str, httpx.Response] = {}
        seen: list[tuple[str, str, str | None]] = []

    state = Net()
    state.dns, state.routes, state.seen = {}, {}, []
    state.down = set()

    def fake_resolve(host, port):
        if host not in state.dns:
            raise OSError(f"cannot resolve host {host!r}")
        return state.dns[host]

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.headers["host"]
        if request.url.host in state.down:
            raise httpx.ConnectError("Network is unreachable", request=request)
        state.seen.append((str(request.url), host, request.extensions.get("sni_hostname")))
        scheme = request.url.scheme
        key = f"{scheme}://{host}{request.url.raw_path.decode()}"
        if key not in state.routes:
            return httpx.Response(404)
        return state.routes[key]

    monkeypatch.setattr(
        cortex_net,
        "fetch",
        functools.partial(cortex_net.fetch, resolve=fake_resolve, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.delenv("LENS_URL_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("LENS_URL_FETCH", raising=False)
    return state


def test_load_image_from_url(net):
    """Downloads a public http(s) URL, connecting to the checked IP with the original Host/SNI."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/photos/sunset.jpg"] = httpx.Response(200, content=_PNG)
    asset, pil = _load_image("https://example.com/photos/sunset.jpg")

    assert net.seen == [("https://93.184.215.14/photos/sunset.jpg", "example.com", "example.com")]
    assert asset.name == "sunset.jpg"
    # The original-bytes sha256 and source URL are captured for replay keying.
    assert asset.uri == "https://example.com/photos/sunset.jpg"
    assert asset.sha256 == hashlib.sha256(_PNG).hexdigest()
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    assert pil.size == (64, 64)


def test_load_image_url_with_query_params(net):
    """Derives the image name from the URL path, ignoring query parameters."""
    net.dns["cdn.example.com"] = ["93.184.215.14"]
    net.routes["https://cdn.example.com/image.png?token=abc&size=lg"] = httpx.Response(200, content=_PNG)
    asset, pil = _load_image("https://cdn.example.com/image.png?token=abc&size=lg")
    assert asset.name == "image.png"


def test_load_image_url_trailing_slash(net):
    """Falls back to the name "image" when the URL path has no filename."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/"] = httpx.Response(200, content=_PNG)
    asset, _ = _load_image("https://example.com/")
    assert asset.name == "image"


def test_load_image_url_http_error(net):
    """Raises FetchFailed (carrying the status) when the URL answers with an error."""
    net.dns["example.com"] = ["93.184.215.14"]
    with pytest.raises(FetchFailed) as excinfo:
        _load_image("https://example.com/missing.jpg")
    assert excinfo.value.status == 404


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.200.10",
        "172.16.0.5",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
    ],
)
def test_internal_address_refused_without_connecting(net, ip):
    """A host resolving to a non-public address is refused before any connection (#110)."""
    net.dns["internal.example"] = [ip]
    with pytest.raises(UrlRefused):
        _load_image("http://internal.example/latest/meta-data")
    assert net.seen == []


def test_ip_literal_refused(net):
    """An internal IP literal is refused the same way as a hostname."""
    net.dns["169.254.169.254"] = ["169.254.169.254"]
    with pytest.raises(UrlRefused):
        _load_image("http://169.254.169.254/latest/meta-data/")
    assert net.seen == []


def test_mixed_resolution_refused(net):
    """One internal address among public ones is enough to refuse (no rebinding lottery)."""
    net.dns["sneaky.example"] = ["93.184.215.14", "127.0.0.1"]
    with pytest.raises(UrlRefused):
        _load_image("https://sneaky.example/a.png")
    assert net.seen == []


def test_redirect_to_internal_refused(net):
    """A public URL that redirects to an internal address is refused at the hop."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.dns["metadata.internal"] = ["169.254.169.254"]
    net.routes["https://example.com/a.png"] = httpx.Response(
        302, headers={"location": "http://metadata.internal/latest/"}
    )
    with pytest.raises(UrlRefused):
        _load_image("https://example.com/a.png")
    assert [s[1] for s in net.seen] == ["example.com"]


def test_public_redirect_followed(net):
    """Redirects between public hosts are followed, each hop pinned to its own IP."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.dns["cdn.example.net"] = ["93.184.215.15"]
    net.routes["https://example.com/a.png"] = httpx.Response(301, headers={"location": "https://cdn.example.net/b.png"})
    net.routes["https://cdn.example.net/b.png"] = httpx.Response(200, content=_PNG)
    _, pil = _load_image("https://example.com/a.png")
    assert pil.size == (64, 64)
    assert net.seen[-1] == ("https://93.184.215.15/b.png", "cdn.example.net", "cdn.example.net")


def test_redirect_loop_capped(net):
    """More than MAX_REDIRECTS hops is refused."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/a.png"] = httpx.Response(302, headers={"location": "/a.png"})
    with pytest.raises(UrlRefused, match="redirects"):
        _load_image("https://example.com/a.png")


def test_allowed_host_bypasses_private_check(net, monkeypatch):
    """A host named in LENS_URL_ALLOWED_HOSTS may resolve to a LAN address."""
    monkeypatch.setenv("LENS_URL_ALLOWED_HOSTS", "immich.lan, other.lan")
    net.dns["immich.lan"] = ["192.168.200.20"]
    net.routes["http://immich.lan:2283/thumb.png"] = httpx.Response(200, content=_PNG)
    _, pil = _load_image("http://immich.lan:2283/thumb.png")
    assert pil.size == (64, 64)
    assert net.seen == [("http://192.168.200.20:2283/thumb.png", "immich.lan:2283", "immich.lan")]


def test_allowed_host_with_port_only_allows_that_port(net, monkeypatch):
    """A ``host:port`` entry in LENS_URL_ALLOWED_HOSTS allows that port and no other."""
    monkeypatch.setenv("LENS_URL_ALLOWED_HOSTS", "immich.lan:2283")
    net.dns["immich.lan"] = ["192.168.200.20"]
    net.routes["http://immich.lan:2283/thumb.png"] = httpx.Response(200, content=_PNG)
    _, pil = _load_image("http://immich.lan:2283/thumb.png")
    assert pil.size == (64, 64)
    with pytest.raises(UrlRefused):
        _load_image("http://immich.lan:8080/thumb.png")
    assert [s[0] for s in net.seen] == ["http://192.168.200.20:2283/thumb.png"]


def test_url_fetch_off(net, monkeypatch):
    """LENS_URL_FETCH=off refuses every URL."""
    monkeypatch.setenv("LENS_URL_FETCH", "off")
    net.dns["example.com"] = ["93.184.215.14"]
    with pytest.raises(UrlRefused, match="disabled"):
        _load_image("https://example.com/a.png")
    assert net.seen == []


def test_oversized_body_refused(net, monkeypatch):
    """A body over url_fetch.MAX_BYTES (passed to cortex as max_bytes) is refused rather than buffered."""
    monkeypatch.setattr(url_fetch, "MAX_BYTES", 100)
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/big.png"] = httpx.Response(200, content=b"x" * 1000)
    with pytest.raises(UrlRefused, match="larger"):
        _load_image("https://example.com/big.png")


def test_credentials_in_url_refused(net):
    """userinfo in the URL is refused."""
    net.dns["example.com"] = ["93.184.215.14"]
    with pytest.raises(UrlRefused):
        _load_image("https://user:pw@example.com/a.png")


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


def test_falls_back_to_next_checked_address(net):
    """An unreachable first address (no IPv6 route) falls back to the next checked one."""
    net.dns["example.com"] = ["2606:2800:21f:cb07:6820:80da:af6b:8b2c", "93.184.215.14"]
    net.down.add("2606:2800:21f:cb07:6820:80da:af6b:8b2c")
    net.routes["https://example.com/a.png"] = httpx.Response(200, content=_PNG)
    _, pil = _load_image("https://example.com/a.png")
    assert pil.size == (64, 64)
    assert net.seen == [("https://93.184.215.14/a.png", "example.com", "example.com")]


def test_uppercase_scheme_is_fetched_as_a_url(net):
    """``HTTPS://`` is a URL under the SSRF policy, not a server-side path."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/a.png"] = httpx.Response(200, content=_PNG)
    asset, _ = _load_image("HTTPS://example.com/a.png")
    assert asset.uri == "HTTPS://example.com/a.png"
    assert net.seen and net.seen[0][1] == "example.com"


def test_total_deadline_caps_a_slow_body(net, monkeypatch):
    """A fetch past TOTAL_TIMEOUT_S (cortex's deadline) is refused (per-read timeouts alone never fire)."""
    net.dns["example.com"] = ["93.184.215.14"]
    net.routes["https://example.com/slow.png"] = httpx.Response(200, content=_PNG)
    monkeypatch.setattr(url_fetch, "TOTAL_TIMEOUT_S", -1.0)
    with pytest.raises(UrlRefused, match="longer than"):
        url_fetch.fetch_url("https://example.com/slow.png")
