"""Tests for the assembly profile registry (issue #4)."""

import pytest

from argus_lens.assembly import profiles
from argus_lens.types import CaptionResult


class _DummyProfile:
    """Minimal valid assembly profile for registry tests."""

    name = "dummy"

    def assemble(self, *, tags: str = "", prose: str = "", **kwargs) -> CaptionResult:
        """Return prose as the caption, falling back to tags."""
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
    """Registered profiles are retrievable by name and listed as available."""
    profile = profiles.register_profile(_DummyProfile())
    assert profiles.get_profile("dummy") is profile
    assert "dummy" in profiles.available_profiles()
    assert isinstance(profile, profiles.AssemblyProfile)


def test_unknown_profile_raises():
    """Looking up an unregistered profile name raises KeyError."""
    with pytest.raises(KeyError):
        profiles.get_profile("does-not-exist")


def test_register_rejects_non_callable_assemble(_clean_registry):
    """Rejects profiles whose assemble attribute is not callable, without registering them."""

    class Bad:
        """Profile with a non-callable assemble attribute."""

        name = "bad"
        assemble = 123

    with pytest.raises(TypeError):
        profiles.register_profile(Bad())
    assert "bad" not in profiles.available_profiles()


def test_register_rejects_blank_name(_clean_registry):
    """Rejects profiles whose name is blank or whitespace-only."""

    class Blank:
        """Profile with a whitespace-only name."""

        name = "   "

        def assemble(self, *, tags="", prose="", **kwargs):
            """Return an empty caption."""
            return CaptionResult(final_caption="")

    with pytest.raises(ValueError):
        profiles.register_profile(Blank())


def test_register_rejects_duplicate_name(_clean_registry):
    """Registering a different object under an already-taken name raises ValueError."""
    profiles.register_profile(_DummyProfile())
    with pytest.raises(ValueError):
        profiles.register_profile(_DummyProfile())  # different object, same name


def test_reregistering_same_object_is_noop(_clean_registry):
    """Re-registering the same profile object is a no-op rather than an error."""
    profile = _DummyProfile()
    profiles.register_profile(profile)
    profiles.register_profile(profile)  # same object -> no error
    assert profiles.available_profiles().count("dummy") == 1


def test_available_profiles_sorted(_clean_registry):
    """available_profiles returns names in sorted order."""

    class _Named(_DummyProfile):
        """Dummy profile with a configurable name."""

        def __init__(self, name):
            """Set the profile name."""
            self.name = name

    profiles.register_profile(_Named("zebra"))
    profiles.register_profile(_Named("alpha"))
    names = profiles.available_profiles()
    assert list(names) == sorted(names)
    assert {"alpha", "zebra"} <= set(names)
