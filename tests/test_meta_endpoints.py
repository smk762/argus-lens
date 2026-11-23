"""Tests for GET /health and GET /profiles (service metadata endpoints)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from argus_lens import __version__  # noqa: E402
from argus_lens.assembly.profiles import available_profiles  # noqa: E402
from argus_lens.backends.base import CaptionBackend  # noqa: E402
from argus_lens.server import create_app  # noqa: E402
from argus_lens.types import (  # noqa: E402
    BACKEND_TOKEN_BUDGETS,
    CAPTION_TARGET_STYLES,
    get_category_names,
)


class _StubBackend(CaptionBackend):
    """Caption backend stub that returns a fixed caption without any model."""

    name = "stub"
    requires_gpu = False

    def load(self, device: str = "auto") -> None:
        """No-op load."""
        pass

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption for any image."""
        return "a person, plain studio background, soft lighting"

    def unload(self) -> None:
        """No-op unload."""
        pass


def test_health_reports_service_version_and_source_root(tmp_path: Path, monkeypatch) -> None:
    """GET /health mirrors argus-curator's shape, including the resolved source root."""
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "argus-lens"
    assert body["version"] == __version__
    assert body["source_root"] == str(tmp_path.resolve())
    # GPU residency block (#37): backend name + loaded flag + coordinator.
    assert body["gpu"]["backend"] == "stub"
    assert body["gpu"]["loaded"] is False
    assert body["gpu"]["coordinator"] == "none"


def test_health_source_root_is_null_when_unset(monkeypatch) -> None:
    """GET /health reports source_root as null when no root is configured."""
    monkeypatch.delenv("LENS_SOURCE_PATH", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend()))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["source_root"] is None


def test_admin_unload_frees_the_model(monkeypatch) -> None:
    """POST /admin/unload actually unloads a loaded model (#37)."""
    import io

    monkeypatch.delenv("LENS_SOURCE_PATH", raising=False)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend()))

    # Caption once to load the model, then confirm it's resident.
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    assert client.post("/caption", files={"file": ("x.png", buf, "image/png")}).status_code == 200
    assert client.get("/health").json()["gpu"]["loaded"] is True

    body = client.post("/admin/unload").json()
    assert body["unloaded"] is True
    assert body["gpu"]["loaded"] is False


def test_health_and_unload_follow_the_resident_contract(monkeypatch) -> None:
    """argus-iris reads /health as cortex's ResidentHealth and /admin/unload as its ResidentActionReply: off that
    shape, iris logged lens as off-contract and could not see a caption running."""
    import io

    from argus_cortex.gpu_wire import ResidentActionReply, ResidentHealth

    monkeypatch.delenv("LENS_SOURCE_PATH", raising=False)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    app = create_app(default_backend=_StubBackend())
    client = TestClient(app)

    cold = ResidentHealth.model_validate(client.get("/health").json())
    assert (cold.model_state, cold.busy, cold.idle_s) == ("unloaded", False, None)

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    assert client.post("/caption", files={"file": ("x.png", buf, "image/png")}).status_code == 200
    warm = ResidentHealth.model_validate(client.get("/health").json())
    assert warm.model_state == "gpu" and warm.busy is False and warm.idle_s is not None

    reply = ResidentActionReply.model_validate(client.post("/admin/unload").json())
    assert (reply.model_state, reply.unloaded) == ("unloaded", True)
    again = ResidentActionReply.model_validate(client.post("/admin/unload").json())
    assert (again.model_state, again.unloaded) == ("unloaded", False)


def test_unload_while_captioning_is_refused_busy(monkeypatch) -> None:
    """A caption in flight: /health says busy and /admin/unload answers 409 without unloading anything."""
    import io
    import threading

    from argus_cortex.gpu_wire import ResidentBusyReply, ResidentHealth

    started, release = threading.Event(), threading.Event()

    class _Slow(_StubBackend):
        def caption_image(self, image: Image.Image) -> str:
            started.set()
            release.wait(10)
            return super().caption_image(image)

    monkeypatch.delenv("LENS_SOURCE_PATH", raising=False)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    client = TestClient(create_app(default_backend=_Slow()))
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    caption = threading.Thread(
        target=lambda: client.post("/caption", files={"file": ("x.png", buf, "image/png")}), daemon=True
    )
    caption.start()
    try:
        assert started.wait(10)
        assert ResidentHealth.model_validate(client.get("/health").json()).busy is True
        refused = client.post("/admin/unload")
        assert refused.status_code == 409
        assert ResidentBusyReply.model_validate(refused.json()).active == 1
    finally:
        release.set()
        caption.join(10)
    assert ResidentHealth.model_validate(client.get("/health").json()).model_state == "gpu"  # nothing was unloaded


def test_admin_unload_requires_token_when_configured(monkeypatch) -> None:
    """With ARGUS_ADMIN_TOKEN set, /admin/unload rejects unauthenticated calls (#42)."""
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "secret")
    client = TestClient(create_app(default_backend=_StubBackend()))
    missing = client.post("/admin/unload")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert client.post("/admin/unload", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/admin/unload", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert client.post("/admin/unload", headers={"Authorization": "Bearer wrong!"}).status_code == 401
    assert client.post("/admin/unload", headers={"Authorization": "secret"}).status_code == 401


def test_admin_unload_stays_open_when_token_unset(monkeypatch) -> None:
    """With ARGUS_ADMIN_TOKEN unset, /admin/unload stays open (open_when_unset=True, as before)."""
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend()))
    assert client.post("/admin/unload").status_code == 200


def test_profiles_exposes_taxonomy_from_sources_of_truth() -> None:
    """GET /profiles derives its lists from the registry/types, not literals."""
    client = TestClient(create_app(default_backend=_StubBackend()))
    body = client.get("/profiles").json()
    assert body["assembly_profiles"] == list(available_profiles())
    # non-vacuous: the trunk pipeline is registered, so the list is never empty
    assert "lora_training" in body["assembly_profiles"]
    assert body["target_styles"] == list(CAPTION_TARGET_STYLES)
    assert body["target_categories"] == list(get_category_names())
    assert body["target_backends"] == list(BACKEND_TOKEN_BUDGETS)
    assert body["token_budgets"] == BACKEND_TOKEN_BUDGETS
    assert "identity" in body["target_categories"]
    assert body["token_budgets"]["sdxl"] == 60


def test_server_env_enables_refiner_and_idle_unload(monkeypatch) -> None:
    """ARGUS_REFINE_MODEL and ARGUS_IDLE_UNLOAD_S configure the served engine."""
    monkeypatch.delenv("ARGUS_REFINE", raising=False)
    monkeypatch.setenv("ARGUS_REFINE_MODEL", "qwen3:8b")
    monkeypatch.setenv("ARGUS_IDLE_UNLOAD_S", "120")
    captured = {}
    import argus_lens.server as server_mod

    real = server_mod.ArgusLens

    def _spy(**kwargs):
        """Record the engine kwargs, without starting a reaper thread."""
        captured.update(kwargs)
        return real(backend=kwargs["backend"], refiner=kwargs["refiner"])

    monkeypatch.setattr(server_mod, "ArgusLens", _spy)
    create_app(default_backend=_StubBackend())
    assert captured["refiner"].rewriter.model_id == "qwen3:8b"
    assert captured["idle_unload_s"] == 120.0


# --- cortex's ServiceHealth on /health (argus-cortex#212) ---------------------------------------------------------

# Every field /health returned before it served ServiceHealth; UIs, argus-iris and argus-lightbox read them.
_LEGACY_HEALTH_FIELDS = {
    "status",
    "service",
    "version",
    "backend",
    "allowed_backends",
    "source_root",
    "read_only",
    "immich",
    "immich_write_back",
    "gpu",
    "model_state",
    "busy",
    "idle_s",
}


def test_health_is_service_health_and_keeps_every_earlier_field(tmp_path: Path, monkeypatch) -> None:
    """One body is cortex's ServiceHealth, its ResidentHealth, and everything lens's /health carried before."""
    from argus_cortex.gpu_wire import ResidentHealth
    from argus_cortex.service import ServiceHealth

    monkeypatch.delenv("IMMICH_URL", raising=False)
    monkeypatch.delenv("IMMICH_API_KEY", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path), read_only=True))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= _LEGACY_HEALTH_FIELDS
    assert set(body["gpu"]) == {"backend", "loaded", "coordinator", "free_vram_mb", "idle_s", "resident"}

    health = ServiceHealth.model_validate(body)
    assert (health.service, health.version, health.status, health.read_only) == ("argus-lens", __version__, "ok", True)
    assert health.capabilities == {"caption": True, "immich": False, "immich_write_back": False}
    assert health.capability_names == ("caption",)
    assert health.checks == {}
    resident = ResidentHealth.model_validate(body)
    assert (resident.model_state, resident.busy) == ("unloaded", False)
    assert (body["backend"], body["allowed_backends"], body["source_root"]) == ("stub", None, str(tmp_path.resolve()))


def test_health_answers_200_whatever_the_model_state() -> None:
    """argus-iris raises on a non-200 from a resident's /health, so an unloaded model is still a 200 (no readiness
    gate), with or without ``?checks=0``; the route is left out of the OpenAPI document."""
    client = TestClient(create_app(default_backend=_StubBackend()))
    for path in ("/health", "/health?checks=0"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["model_state"] == "unloaded"
        assert response.json()["status"] == "ok"
    assert "/health" not in client.get("/openapi.json").json()["paths"]


def test_health_capabilities_follow_immich_and_read_only(monkeypatch) -> None:
    """The capability map says the same as the top-level ``immich`` / ``immich_write_back`` booleans."""
    monkeypatch.setenv("IMMICH_URL", "http://immich.invalid:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "made-up")
    body = TestClient(create_app(default_backend=_StubBackend())).get("/health").json()
    assert body["capabilities"] == {"caption": True, "immich": True, "immich_write_back": True}
    assert (body["immich"], body["immich_write_back"]) == (True, True)
    body = TestClient(create_app(default_backend=_StubBackend(), read_only=True)).get("/health").json()
    assert body["capabilities"] == {"caption": True, "immich": True, "immich_write_back": False}
