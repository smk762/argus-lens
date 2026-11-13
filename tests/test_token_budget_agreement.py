"""Packing and the whole-text token estimate must agree (#80)."""

from __future__ import annotations

import pytest

from argus_lens.assembly.prose_training import assemble_prose_training_variant
from argus_lens.assembly.token_budget import caption_tokens, estimate_tokens, try_add_fragment

_TAGS = [f"tag{i}" for i in range(30)]


@pytest.mark.parametrize("backend", ["sdxl", "flux"])
@pytest.mark.parametrize("trigger", ["", "ohwx"])
def test_packed_caption_fits_whole_text_estimate(backend: str, trigger: str) -> None:
    """30 one-word tags packed into 60 tokens measure at most 60 as finished text."""
    kept: list[str] = []
    for tag in _TAGS:
        try_add_fragment(tag, kept, max_segments=100, token_budget=60, backend=backend, trigger_word=trigger)
    text = f"{trigger}, {', '.join(kept)}" if trigger else ", ".join(kept)
    assert estimate_tokens(text, backend) <= 60
    assert caption_tokens(kept, backend, trigger_word=trigger) == estimate_tokens(text, backend)
    # Packing is tight: the next tag would have overflowed the whole-text estimate.
    assert len(kept) < len(_TAGS)
    assert caption_tokens([*kept, _TAGS[len(kept)]], backend, trigger_word=trigger) > 60


def test_issue_80_example_no_longer_overruns() -> None:
    """The issue's case: packing used to fit 30 tags at 59 tokens that measured 68."""
    assert estimate_tokens(", ".join(_TAGS), "sdxl") > 60  # all 30 really are over budget
    kept: list[str] = []
    for tag in _TAGS:
        try_add_fragment(tag, kept, max_segments=100, token_budget=60, backend="sdxl")
    assert caption_tokens(kept, "sdxl") <= 60


def test_segment_cap_still_applies() -> None:
    """max_segments limits fragments regardless of budget."""
    kept: list[str] = []
    for tag in _TAGS:
        try_add_fragment(tag, kept, max_segments=3, token_budget=1000)
    assert kept == _TAGS[:3]


def test_prose_training_caption_fits_whole_text_estimate() -> None:
    """Sentence packing measures the joined sentences plus trigger, as the checks do."""
    prose = " ".join(f"A short sentence number {i} about the scene." for i in range(20))
    caption, _ = assemble_prose_training_variant("ohwx", prose, token_budget=200, target_backend="flux")
    assert caption
    assert estimate_tokens(caption, "flux") <= 200
