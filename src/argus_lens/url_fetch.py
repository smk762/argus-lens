"""SSRF-safe image download for ``http(s)://`` sources (#110).

``POST /caption/url`` hands a client-supplied URL to the engine, so a plain
``httpx.get`` would let anyone who can reach the port make lens GET internal
targets (cloud metadata, localhost admin ports, LAN services) and read the
result back as a caption. :func:`fetch_url` closes that:

* the host is resolved once and every address must be public (globally
  routable); private, loopback, link-local, CGNAT, multicast and reserved
  ranges are refused unless the host is named in ``LENS_URL_ALLOWED_HOSTS``;
* the connection goes to the IP that was checked (with the original
  ``Host`` header and TLS SNI), so DNS rebinding can't swap in another address
  between the check and the connect;
* redirects are followed by hand, at most :data:`MAX_REDIRECTS`, and each hop
  is checked the same way;
* the body is capped at :data:`MAX_BYTES`.

``LENS_URL_FETCH=off`` turns URL fetching off entirely.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urljoin, urlsplit

MAX_REDIRECTS = 5
MAX_BYTES = 50 * 1024 * 1024
TIMEOUT_S = 30.0
_ALLOWED_PORTS = {"http": 80, "https": 443}


class URLRefused(ValueError):
    """The URL (or a redirect hop) is not one lens will fetch."""


def _allowed_hosts() -> set[str]:
    raw = os.environ.get("LENS_URL_ALLOWED_HOSTS", "")
    return {h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip()}


def fetching_enabled() -> bool:
    """False when ``LENS_URL_FETCH`` turns URL fetching off."""
    return os.environ.get("LENS_URL_FETCH", "on").strip().lower() not in {"0", "off", "false", "no"}


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise URLRefused(f"cannot resolve host {host!r}") from exc
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def check_url(url: str) -> tuple[str, list[str], int]:
    """Validate *url*; return ``(host, checked_ips, port)`` or raise :class:`URLRefused`."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_PORTS:
        raise URLRefused(f"only http(s) URLs are fetched, got {scheme or 'no'} scheme")
    if parts.username or parts.password:
        raise URLRefused("URLs with credentials are not fetched")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise URLRefused("URL has no host")
    try:
        port = parts.port or _ALLOWED_PORTS[scheme]
    except ValueError as exc:
        raise URLRefused("URL has an invalid port") from exc

    addresses = _resolve(host, port)
    if not addresses:
        raise URLRefused(f"cannot resolve host {host!r}")
    if host not in _allowed_hosts():
        for addr in addresses:
            if not _is_public(ipaddress.ip_address(addr.split("%", 1)[0])):
                raise URLRefused(
                    f"host {host!r} resolves to a non-public address; name it in LENS_URL_ALLOWED_HOSTS to allow it"
                )
    return host, addresses, port


def _pinned_url(url: str, ip: str, port: int) -> str:
    parts = urlsplit(url)
    ip_literal = f"[{ip}]" if ":" in ip else ip
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return f"{parts.scheme.lower()}://{ip_literal}:{port}{path}"


def fetch_url(url: str) -> bytes:
    """Download *url* under the SSRF policy above and return the body bytes.

    Raises :class:`URLRefused` for a refused URL or hop, and
    ``httpx.HTTPStatusError`` for a non-2xx final response.
    """
    import httpx

    if not fetching_enabled():
        raise URLRefused("URL fetching is disabled (LENS_URL_FETCH=off)")

    current = url
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=False, trust_env=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            host, ips, port = check_url(current)
            default_port = _ALLOWED_PORTS[urlsplit(current).scheme.lower()]
            host_header = host if port == default_port else f"{host}:{port}"
            resp = None
            # Try each checked address in order, as a normal client would
            # (e.g. IPv6 first, falling back to IPv4 when there's no v6 route).
            for ip in ips:
                request = client.build_request(
                    "GET",
                    _pinned_url(current, ip, port),
                    headers={"Host": host_header},
                    extensions={"sni_hostname": host},
                )
                try:
                    resp = client.send(request, stream=True)
                    break
                except httpx.ConnectError:
                    continue
                except httpx.TransportError as exc:
                    raise URLRefused(f"could not fetch {host!r}") from exc
            if resp is None:
                raise URLRefused(f"could not connect to {host!r}")
            try:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise URLRefused("redirect without a Location header")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    resp.read()
                    resp.request = httpx.Request("GET", current)
                    resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                    raise URLRefused(f"image is larger than {MAX_BYTES} bytes")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        raise URLRefused(f"image is larger than {MAX_BYTES} bytes")
                return bytes(body)
            finally:
                resp.close()
    raise URLRefused(f"more than {MAX_REDIRECTS} redirects")
