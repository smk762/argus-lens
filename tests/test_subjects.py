"""Tests for subject (headcount) estimation."""

import pytest

from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.subjects import count_from_prose, count_from_tags, count_subjects, is_subject_count_tag
from argus_lens.types import resolve_target_profile


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["solo", "1girl"], 1),
        (["1girl", "1boy"], 2),
        (["2girls", "multiple_girls"], 2),
        (["6+girls"], 6),
        (["multiple_boys"], 2),
        (["solo_focus", "1girl"], 2),
        (["no_humans"], 0),
        (["kitchen"], None),
    ],
)
def test_count_from_tags(tags: list[str], expected: int | None) -> None:
    """Numbered tags sum across genders; ``multiple_*``/``solo_focus`` imply two or more."""
    assert count_from_tags(tags)[0] == expected


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        ("Two young women laugh on a street.", 2),
        ("A group of people stand in a park.", 3),
        ("A man and a woman are dancing.", 2),
        ("A couple of friends share a meal.", 2),
        ("A woman sits on a bench.", 1),
        ("An empty kitchen with a table.", None),
    ],
)
def test_count_from_prose(prose: str, expected: int | None) -> None:
    """Plural quantities beat pairs, pairs beat a single subject."""
    assert count_from_prose(prose)[0] == expected


def test_count_subjects_flags_multiple_when_sources_disagree() -> None:
    """Either source seeing 2+ people sets ``multiple``; disagreement is surfaced."""
    result = count_subjects(["1girl", "solo"], "Two women sit on a sofa.")
    assert result.multiple and result.agree is False and result.count == 1


@pytest.mark.parametrize("tag", ["solo", "2girls", "2 girls", "multiple_boys", "no_humans", "6+boys"])
def test_is_subject_count_tag(tag: str) -> None:
    """Headcount tags are recognised in underscore and space forms."""
    assert is_subject_count_tag(tag)


def test_headcount_tags_stay_out_of_buckets_but_reach_metadata() -> None:
    """``2girls`` is not an identity trait; it lands in ``metadata['subjects']`` instead."""
    result = compose_caption_result(
        trigger_word="",
        tags="2girls, multiple_girls, brown_hair, kitchen",
        prose="Two women cook in a kitchen.",
        target_profile=resolve_target_profile(),
    )
    assert "2girls" not in result.caption_variants["identity"]
    assert "2girls" not in result.metadata["buckets"]["tags"]["identity"]
    assert result.metadata["subjects"]["count"] == 2
    assert result.metadata["subjects"]["multiple"] is True
    assert "2girls" in result.caption_variants["zeroshot"]  # still useful for generation
