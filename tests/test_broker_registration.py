"""`argus-lens serve` registers with the argus-iris broker at startup, and only when a broker URL is set."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import argus_lens.gpu  # noqa: E402
from argus_lens.server import create_app  # noqa: E402

from .test_meta_endpoints import _StubBackend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def broker(monkeypatch):
    """Route every broker coordinator lens builds to an in-memory broker; return the requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200 if request.method == "POST" else 204, json={})

    real = argus_lens.gpu.coordinator_from_env

    def with_transport():
        coord = real()
        if hasattr(coord, "_async_transport"):
            coord._async_transport = httpx.MockTransport(handler)
            coord._transport = httpx.MockTransport(handler)
        return coord

    monkeypatch.setattr(argus_lens.gpu, "coordinator_from_env", with_transport)
    monkeypatch.chdir(ROOT)  # ./gpu-tenant.yaml, as /app/gpu-tenant.yaml is in the image
    return seen


def _wait_for(seen: list, n: int) -> None:
    deadline = time.monotonic() + 5
    while len(seen) < n and time.monotonic() < deadline:
        time.sleep(0.01)


def test_startup_registers_and_shutdown_signs_off(broker, monkeypatch) -> None:
    """With the URL set, startup posts the manifest with the admin token; shutdown deletes the tenant."""
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://broker.invalid:8120")
    monkeypatch.setenv("ARGUS_GPU_BROKER_TOKEN", "caller-secret")
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "admin-secret")
    with TestClient(create_app(default_backend=_StubBackend())):
        _wait_for(broker, 1)
        assert broker, "no registration at startup"
        post = broker[0]
        assert (post.method, post.url.path) == ("POST", "/tenants")
        assert post.headers["authorization"] == "Bearer caller-secret"
        assert post.headers["x-argus-admin-token"] == "admin-secret"
        body = httpx.Response(200, content=post.content).json()
        assert body["tenant"] == "argus-lens"
        assert [r["name"] for r in body["residents"]] == ["lens-captioner"]
    assert (broker[-1].method, broker[-1].url.path) == ("DELETE", "/tenants/argus-lens")


def test_no_broker_url_registers_nothing(broker) -> None:
    """Without ARGUS_GPU_BROKER_URL the coordinator is local and startup talks to no broker."""
    app = create_app(default_backend=_StubBackend())
    with TestClient(app) as client:
        assert client.get("/health").json()["gpu"]["coordinator"] == "none"
        time.sleep(0.05)
    assert broker == []
