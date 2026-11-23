"""Shared fixtures for Argus Lens tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_suite_file(monkeypatch):
    """A developer's suite file (``~/.config/argus/suite.env``) and suite-wide peer URLs never reach a test."""
    monkeypatch.setenv("ARGUS_ENV_FILE", "off")
    for name in ("ARGUS_OCR_URL", "ARGUS_PUPPET_URL", "ARGUS_PEER", "ARGUS_BUILD"):
        monkeypatch.delenv(name, raising=False)


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


@pytest.fixture(autouse=True)
def _gpu_broker_unset(monkeypatch):
    """A developer's ARGUS_GPU_BROKER_URL alone selects the broker (cortex 0.15): keep tests off a live iris."""
    for name in ("ARGUS_GPU_COORDINATOR", "ARGUS_GPU_BROKER_URL", "ARGUS_GPU_TENANT_MANIFEST", "ARGUS_GPU_TENANT"):
        monkeypatch.delenv(name, raising=False)
