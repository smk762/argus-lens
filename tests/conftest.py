"""Shared fixtures for Argus Lens tests."""

import pytest


@pytest.fixture(autouse=True)
def _refine_off(monkeypatch):
    """Refinement is on by default in serve/CLI; keep tests off the network unless they opt in."""
    monkeypatch.setenv("ARGUS_REFINE", "off")


@pytest.fixture(autouse=True)
def _ocr_unset(monkeypatch):
    """A developer's LENS_OCR_URL / LENS_TEXT_VERIFIER must not reroute text claims in tests."""
    monkeypatch.delenv("LENS_OCR_URL", raising=False)
    monkeypatch.delenv("LENS_TEXT_VERIFIER", raising=False)


@pytest.fixture(autouse=True)
def _pose_unset(monkeypatch):
    """A developer's LENS_POSE_URL must not route posture to a live argus-puppet in tests."""
    monkeypatch.delenv("LENS_POSE_URL", raising=False)
    monkeypatch.delenv("LENS_POSE_TIMEOUT_S", raising=False)
