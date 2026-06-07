"""Tests for the taxonomy normalization layer (issue #5)."""

from dataclasses import FrozenInstanceError

import pytest

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


def test_blank_labels_are_dropped():
    assert DEFAULT_TAXONOMY.canonical("   ") == ""
    assert DEFAULT_TAXONOMY.expand("   ") == []
    assert DEFAULT_TAXONOMY.normalize(["  ", "", "mountain"]) == ["mountain", "landscape", "nature"]
    # blank labels are dropped even without hierarchy expansion
    assert DEFAULT_TAXONOMY.normalize(["  ", "car"], expand_hierarchy=False) == ["car"]


def test_synonym_then_hierarchy_compose():
    tax = Taxonomy(
        synonyms={"peak": "mountain"},
        parents={"mountain": ("landscape", "nature")},
    )
    # alias collapses to canonical, then the canonical's ancestors expand
    assert tax.normalize(["peak"]) == ["mountain", "landscape", "nature"]


def test_synonyms_are_resolved_single_level():
    # Documented contract: resolution is one lookup, values must be canonical.
    tax = Taxonomy(synonyms={"a": "b", "b": "c"})
    assert tax.canonical("a") == "b"


def test_default_taxonomy_is_immutable():
    # Frozen dataclass: cannot reassign the mapping attributes.
    with pytest.raises(FrozenInstanceError):
        DEFAULT_TAXONOMY.synonyms = {}  # type: ignore[misc]
    # Read-only mappings: cannot mutate contents in place.
    with pytest.raises(TypeError):
        DEFAULT_TAXONOMY.synonyms["hacked"] = "boom"  # type: ignore[index]
