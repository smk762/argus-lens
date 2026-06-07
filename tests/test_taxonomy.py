"""Tests for the taxonomy normalization layer (issue #5)."""

from argus_lens.taxonomy import DEFAULT_TAXONOMY, Taxonomy


def test_synonym_collapse():
    tax = Taxonomy(synonyms={"automobile": "car"})
    assert tax.canonical("Automobile") == "car"
    assert tax.canonical("bicycle") == "bicycle"


def test_hierarchy_expansion():
    assert DEFAULT_TAXONOMY.expand("mountain") == ["mountain", "landscape", "nature"]


def test_normalize_dedupes_and_preserves_order():
    out = DEFAULT_TAXONOMY.normalize(["automobile", "car", "mountain"])
    assert out == ["car", "mountain", "landscape", "nature"]


def test_normalize_without_expansion():
    out = DEFAULT_TAXONOMY.normalize(["mountain"], expand_hierarchy=False)
    assert out == ["mountain"]
