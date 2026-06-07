"""Tests for the assembly profile registry (issue #4)."""

import pytest

from argus_lens.assembly import profiles
from argus_lens.types import CaptionResult


class _DummyProfile:
    name = "dummy"

    def assemble(self, *, tags: str = "", prose: str = "", **kwargs) -> CaptionResult:
        return CaptionResult(final_caption=prose or tags)


@pytest.fixture
def _clean_registry():
    """Snapshot and restore the module registry around each test."""
    snapshot = dict(profiles._REGISTRY)
    try:
        yield
    finally:
        profiles._REGISTRY.clear()
        profiles._REGISTRY.update(snapshot)


def test_register_and_get_profile(_clean_registry):
    profile = profiles.register_profile(_DummyProfile())
    assert profiles.get_profile("dummy") is profile
    assert "dummy" in profiles.available_profiles()
    assert isinstance(profile, profiles.AssemblyProfile)


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        profiles.get_profile("does-not-exist")


def test_register_rejects_non_callable_assemble(_clean_registry):
    class Bad:
        name = "bad"
        assemble = 123

    with pytest.raises(TypeError):
        profiles.register_profile(Bad())
    assert "bad" not in profiles.available_profiles()


def test_register_rejects_blank_name(_clean_registry):
    class Blank:
        name = "   "

        def assemble(self, *, tags="", prose="", **kwargs):
            return CaptionResult(final_caption="")

    with pytest.raises(ValueError):
        profiles.register_profile(Blank())


def test_register_rejects_duplicate_name(_clean_registry):
    profiles.register_profile(_DummyProfile())
    with pytest.raises(ValueError):
        profiles.register_profile(_DummyProfile())  # different object, same name


def test_reregistering_same_object_is_noop(_clean_registry):
    profile = _DummyProfile()
    profiles.register_profile(profile)
    profiles.register_profile(profile)  # same object -> no error
    assert profiles.available_profiles().count("dummy") == 1


def test_available_profiles_sorted(_clean_registry):
    class _Named(_DummyProfile):
        def __init__(self, name):
            self.name = name

    profiles.register_profile(_Named("zebra"))
    profiles.register_profile(_Named("alpha"))
    names = profiles.available_profiles()
    assert list(names) == sorted(names)
    assert {"alpha", "zebra"} <= set(names)
