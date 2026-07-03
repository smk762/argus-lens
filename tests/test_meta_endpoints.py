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


def test_health_reports_service_version_and_source_root(tmp_path: Path) -> None:
    """GET /health mirrors argus-curator's shape, including the resolved source root."""
    client = TestClient(create_app(default_backend=_StubBackend(), source_root=str(tmp_path)))
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "service": "argus-lens",
        "version": __version__,
        "source_root": str(tmp_path.resolve()),
    }


def test_health_source_root_is_null_when_unset(monkeypatch) -> None:
    """GET /health reports source_root as null when no root is configured."""
    monkeypatch.delenv("LENS_SOURCE_PATH", raising=False)
    client = TestClient(create_app(default_backend=_StubBackend()))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["source_root"] is None


def test_profiles_exposes_taxonomy_from_sources_of_truth() -> None:
    """GET /profiles derives its lists from the registry/types, not literals."""
    client = TestClient(create_app(default_backend=_StubBackend()))
    body = client.get("/profiles").json()
    assert body["assembly_profiles"] == list(available_profiles())
    assert body["target_styles"] == list(CAPTION_TARGET_STYLES)
    assert body["target_categories"] == list(get_category_names())
    assert body["target_backends"] == list(BACKEND_TOKEN_BUDGETS)
    assert body["token_budgets"] == BACKEND_TOKEN_BUDGETS
    assert "identity" in body["target_categories"]
    assert body["token_budgets"]["sdxl"] == 60
