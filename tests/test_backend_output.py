"""Tests for structured backend output types (issue #1)."""

from argus_lens.backends.output import BackendOutput, Tag


def test_tag_defaults():
    """Tag defaults to no score, empty source, and no region."""
    tag = Tag(label="mountain")
    assert tag.score is None
    assert tag.source == ""
    assert tag.region is None


def test_tag_string_filters_by_score():
    """tag_string(min_score=...) drops below-threshold scored tags but keeps unscored ones."""
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
    """to_caption_string returns prose when present, else the joined tags."""
    out = BackendOutput(tags=[Tag("cat", score=0.8)], prose="a cat on a sofa")
    assert out.to_caption_string() == "a cat on a sofa"

    tags_only = BackendOutput(tags=[Tag("cat", score=0.8)])
    assert tags_only.to_caption_string() == "cat"


def test_tag_string_threshold_is_inclusive():
    """A tag whose score equals min_score is kept, and min_score=0.0 is honoured (not treated as falsy)."""
    out = BackendOutput(tags=[Tag("edge", score=0.5), Tag("zero", score=0.0)])
    assert out.tag_string(min_score=0.5) == "edge"
    assert out.tag_string(min_score=0.0) == "edge, zero"


def test_tag_string_skips_blank_labels():
    """Empty or whitespace-only labels do not produce empty comma-separated entries."""
    out = BackendOutput(tags=[Tag("a"), Tag(""), Tag("  "), Tag("b")])
    assert out.tag_string() == "a, b"


def test_to_caption_string_ignores_blank_prose():
    """Whitespace-only prose falls back to the flattened tags."""
    out = BackendOutput(tags=[Tag("cat")], prose="  \n")
    assert out.to_caption_string() == "cat"
