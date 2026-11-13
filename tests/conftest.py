"""Shared fixtures for Argus Lens tests."""

import pytest


@pytest.fixture(autouse=True)
def _refine_off(monkeypatch):
    """Refinement is on by default in serve/CLI; keep tests off the network unless they opt in."""
    monkeypatch.setenv("ARGUS_REFINE", "off")
