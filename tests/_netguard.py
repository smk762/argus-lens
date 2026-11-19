"""Test helper: fail on any real outbound connection, sync or async.

Patching ``httpx.Client.send`` is the wrong seam for SSRF tests: depending on
the Starlette version ``TestClient`` *is* an ``httpx.Client`` (so the test's own
request trips the guard), and an ``httpx.AsyncClient`` fetch is not covered at
all. Instead this blocks the network transports (``TestClient`` uses its own
in-process transport, so it is unaffected) and the socket layer below them, so
urllib/requests/raw-socket fetches are caught too.
"""

from __future__ import annotations

import socket

import httpx
import pytest


def block_outbound(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every outbound connection attempt fail the test; return the attempts made.

    The returned list records each attempt (it should stay empty), so a caller
    can also assert on it in case the code under test swallows the failure.
    """
    attempts: list[str] = []

    def _refuse(what: str) -> None:
        """Record *what* and fail the test."""
        attempts.append(what)
        pytest.fail(f"outbound connection attempted: {what}")

    def _sync(self, request: httpx.Request) -> httpx.Response:
        """Stand-in for ``HTTPTransport.handle_request``."""
        _refuse(f"httpx sync {request.url}")
        raise AssertionError  # pragma: no cover - pytest.fail raises

    async def _async(self, request: httpx.Request) -> httpx.Response:
        """Stand-in for ``AsyncHTTPTransport.handle_async_request``."""
        _refuse(f"httpx async {request.url}")
        raise AssertionError  # pragma: no cover

    def _connect(self, address, *_a, **_k):
        """Stand-in for ``socket.socket.connect`` / ``connect_ex``."""
        _refuse(f"socket connect {address!r}")

    def _create_connection(address, *_a, **_k):
        """Stand-in for ``socket.create_connection``."""
        _refuse(f"socket.create_connection {address!r}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _async)
    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect)
    monkeypatch.setattr(socket, "create_connection", _create_connection)
    return attempts
