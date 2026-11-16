"""Image download for ``http(s)://`` sources, through ``argus_cortex.net`` (#110).

``POST /caption/url`` hands a client-supplied URL to the engine, so the fetch
goes through cortex's shared outbound-fetch helper (:func:`argus_cortex.net.fetch`):
every resolved address must be public unless the host is named in
``LENS_URL_ALLOWED_HOSTS``, the connection is pinned to the checked address,
redirects are re-checked (at most :data:`MAX_REDIRECTS`), the body is capped at
:data:`MAX_BYTES`, and the whole fetch at :data:`TOTAL_TIMEOUT_S`. See
argus-cortex's ``docs/paved-roads.md``.

This module only supplies lens's settings:

* ``LENS_URL_ALLOWED_HOSTS``: comma-separated ``host`` or ``host:port`` entries
  that may resolve to a private/LAN address.
* ``LENS_URL_FETCH=off`` turns URL fetching off entirely.

Refusals raise :class:`argus_cortex.net.UrlRefused`; a 4xx/5xx answer raises
:class:`argus_cortex.net.FetchFailed`.
"""

from __future__ import annotations

import os

from argus_cortex import net

MAX_REDIRECTS = 5
MAX_BYTES = 50 * 1024 * 1024
TIMEOUT_S = 30.0
TOTAL_TIMEOUT_S = 120.0


def allowed_hosts() -> list[str]:
    """The ``LENS_URL_ALLOWED_HOSTS`` entries (``host`` or ``host:port``), lower-cased."""
    raw = os.environ.get("LENS_URL_ALLOWED_HOSTS", "")
    return [h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip()]


def fetching_enabled() -> bool:
    """False when ``LENS_URL_FETCH`` turns URL fetching off."""
    return os.environ.get("LENS_URL_FETCH", "on").strip().lower() not in {"0", "off", "false", "no"}


def fetch_url(url: str) -> bytes:
    """Download *url* under cortex's outbound-fetch policy and return the body bytes.

    Raises :class:`argus_cortex.net.UrlRefused` for a refused URL or hop (or when
    ``LENS_URL_FETCH=off``), and :class:`argus_cortex.net.FetchFailed` for a
    non-2xx final response.
    """
    if not fetching_enabled():
        raise net.UrlRefused("URL fetching is disabled (LENS_URL_FETCH=off)")
    reply = net.fetch(
        url,
        allow_hosts=allowed_hosts(),
        timeout=TIMEOUT_S,
        max_bytes=MAX_BYTES,
        max_redirects=MAX_REDIRECTS,
        deadline=TOTAL_TIMEOUT_S,
    )
    return reply.raise_for_status().body
