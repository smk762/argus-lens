"""Tests for structured backend output types (issue #1)."""

from argus_lens.backends.output import BackendOutput, Tag


def test_tag_defaults():
    tag = Tag(label="mountain")
    assert tag.score is None
    assert tag.source == ""
    assert tag.region is None


def test_tag_string_filters_by_score():
    out = BackendOutput(
        tags=[
            Tag("mountain", score=0.9, source="ram"),
            Tag("blurry", score=0.2, source="ram"),
            Tag("nature", score=None, source="ram"),
        ]
    )
    assert out.tag_string() == "mountain, blurry, nature"
    # Below-threshold scored tags are dropped; unscored tags are kept.
    assert out.tag_string(min_score=0.5) == "mountain, nature"


def test_to_caption_string_prefers_prose():
    out = BackendOutput(tags=[Tag("cat", score=0.8)], prose="a cat on a sofa")
    assert out.to_caption_string() == "a cat on a sofa"

    tags_only = BackendOutput(tags=[Tag("cat", score=0.8)])
    assert tags_only.to_caption_string() == "cat"
