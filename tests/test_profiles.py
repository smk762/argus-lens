"""Tests for the assembly profile registry (issue #4)."""

import pytest

from argus_lens.assembly import profiles
from argus_lens.types import CaptionResult


class _DummyProfile:
    name = "dummy"

    def assemble(self, *, tags: str = "", prose: str = "", **kwargs) -> CaptionResult:
        return CaptionResult(final_caption=prose or tags)


def test_register_and_get_profile():
    profile = profiles.register_profile(_DummyProfile())
    try:
        assert profiles.get_profile("dummy") is profile
        assert "dummy" in profiles.available_profiles()
        assert isinstance(profile, profiles.AssemblyProfile)
    finally:
        profiles._REGISTRY.pop("dummy", None)


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        profiles.get_profile("does-not-exist")
