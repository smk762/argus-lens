"""``CaptionResult`` is the wire schema: endpoints ``asdict()`` it, and argus-studio and lightbox read the keys.

Renaming a field or a ``caption_variants`` key is a breaking change. If one of these
tests fails, the change needs a deliberate update here and a note in the changelog.
"""

from __future__ import annotations

import dataclasses

import pytest

from argus_lens.assembly.composer import compose_caption_result
from argus_lens.types import DEFAULT_CATEGORY_CONFIGS, CaptionResult, CaptionTargetProfile

CAPTION_RESULT_FIELDS = [
    "final_caption",
    "caption_variants",
    "selected_category",
    "removed_phrases",
    "compaction_notes",
    "raw_tags",
    "raw_prose",
    "backend_name",
    "metadata",
]

CATEGORY_KEYS = {"identity", "wardrobe", "pose_composition", "setting", "lighting", "action"}
VARIANT_KEYS = CATEGORY_KEYS | {"training", "training_prose", "zeroshot"}


def test_caption_result_fields_are_the_wire_schema() -> None:
    """Field names and their order match the published JSON."""
    assert [f.name for f in dataclasses.fields(CaptionResult)] == CAPTION_RESULT_FIELDS


def test_default_category_names_are_stable() -> None:
    """The category names double as ``caption_variants`` keys and as ``target_category`` values."""
    assert {c.name for c in DEFAULT_CATEGORY_CONFIGS} == CATEGORY_KEYS


@pytest.mark.parametrize(
    ("tags", "prose"),
    [
        ("1girl, red dress, standing, beach, sunset, smiling", "A woman in a red dress stands on a beach at sunset."),
        ("tree", ""),
        ("", ""),
    ],
)
def test_every_result_carries_every_variant_key(tags: str, prose: str) -> None:
    """Every variant key is present, even when the inputs leave it empty, so clients can index it."""
    result = compose_caption_result(trigger_word="ohwx", tags=tags, prose=prose, target_profile=CaptionTargetProfile())
    assert set(result.caption_variants) == VARIANT_KEYS
    assert list(dataclasses.asdict(result)) == CAPTION_RESULT_FIELDS
