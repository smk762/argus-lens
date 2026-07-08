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


def test_admin_unload_requires_token_when_configured(monkeypatch) -> None:
    """With ARGUS_ADMIN_TOKEN set, /admin/unload rejects unauthenticated calls (#42)."""
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "secret")
    client = TestClient(create_app(default_backend=_StubBackend()))
    assert client.post("/admin/unload").status_code == 401
    assert client.post("/admin/unload", headers={"Authorization": "Bearer secret"}).status_code == 200


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
