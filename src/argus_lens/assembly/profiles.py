"""Pluggable assembly intent profiles (issue #4).

The assembly pipeline is currently hard-specialised for diffusion training
(identity suppression, CLIP/T5 token budgets, omission cycles). This module
introduces an ``AssemblyProfile`` protocol so the same normalised model output
can be assembled for different downstream intents.

The existing logic should move into a ``lora_training`` profile; new verticals
(``dam_keywording``, ``alt_text``, ``search_index``, ``surveillance``) plug in
alongside it without forking the trunk.

Status: scaffold (protocol + registry). Concrete profiles to follow.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from argus_lens.types import CaptionResult


@runtime_checkable
class AssemblyProfile(Protocol):
    """Assembles normalised model output for one downstream intent."""

    name: str

    def assemble(self, *, tags: str = "", prose: str = "", **kwargs) -> CaptionResult:
        """Produce a :class:`CaptionResult` for this profile's intent."""
        ...


_REGISTRY: dict[str, AssemblyProfile] = {}


def register_profile(profile: AssemblyProfile) -> AssemblyProfile:
    """Register an assembly profile under its ``name``."""
    _REGISTRY[profile.name] = profile
    return profile


def get_profile(name: str) -> AssemblyProfile:
    """Look up a registered profile by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown assembly profile {name!r}. Registered: {sorted(_REGISTRY)}") from None


def available_profiles() -> tuple[str, ...]:
    """Return the names of all registered profiles."""
    return tuple(sorted(_REGISTRY))
