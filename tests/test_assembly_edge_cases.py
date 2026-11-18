"""Edge cases in the assembly helpers (audit follow-ups)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from argus_lens.assembly.classifier import _content_words
from argus_lens.assembly.coherence import _split_trigger, check_caption
from argus_lens.assembly.filtering import filter_redundant_clauses_detailed
from argus_lens.assembly.prose_training import _clean_sentence
from argus_lens.assembly.subjects import count_from_tags, exact_tag_count, is_subject_count_tag
from argus_lens.assembly.variants import _variant_bucket_plan
from argus_lens.assembly.zeroshot import assemble_zeroshot_variant

# --- classifier ---


def test_content_words_filter_on_the_stripped_token() -> None:
    """Punctuation-only tokens and one-letter words left after stripping are not content words."""
    words = _content_words('She waits... "!!" x. (b, garden')
    assert "" not in words
    assert "x" not in words
    assert {"waits", "garden"} <= words
    assert _content_words("... !!") == frozenset()


# --- coherence ---


def test_split_trigger_needs_a_word_boundary() -> None:
    """A caption whose first word merely starts with the trigger is not split mid-word."""
    assert _split_trigger("artistic portrait of a woman", "art") == ("", "artistic portrait of a woman")
    assert _split_trigger("art, portrait of a woman", "art") == ("art", "portrait of a woman")
    assert _split_trigger("art portrait", "art") == ("art", "portrait")
    assert _split_trigger("art", "art") == ("art", "")


def test_vocabulary_guard_accepts_es_and_ies_plurals() -> None:
    """``glasses``/``dresses``/``berries`` are grounded when their singular is in the vocabulary."""
    vocabulary = frozenset({"woman", "wearing", "glass", "dress", "berry", "holding"})
    text = "woman wearing glasses, dresses, holding berries"
    assert check_caption(text, "list", allowed_vocabulary=vocabulary) == []


# --- filtering ---


def test_clause_without_content_words_is_accounted_for() -> None:
    """Every clause lands in kept or removed, even one made only of stopwords."""
    kept, removed = filter_redundant_clauses_detailed("She is. A woman relaxes in a park.", "blonde hair")
    assert len(kept) + len(removed) == 2
    assert "She is" in removed
    assert any("park" in c for c in kept)


# --- prose training ---


def test_clean_sentence_never_reports_an_empty_removal() -> None:
    """A sentence reduced to nothing records its excised parts, not a blank entry."""
    cleaned, removed = _clean_sentence("She has green eyes.", None)
    assert cleaned == ""
    assert "" not in removed
    assert removed


# --- subjects ---


def test_tag_normalisation_handles_separator_runs() -> None:
    """``2girls``, ``2 girls``, ``2  girls`` and ``2-girls`` are all headcount tags meaning two."""
    for tag in ("2girls", "2 girls", "2  girls", "2-girls", "2_girls"):
        assert is_subject_count_tag(tag), tag
        assert count_from_tags([tag])[0] == 2, tag


def test_duplicate_headcount_tags_count_once() -> None:
    """A headcount tag repeated (in any spelling) is not summed twice."""
    count, evidence = count_from_tags(["1girl", "1girl", "1 girl"])
    assert count == 1
    assert evidence == ["1girl"]
    assert count_from_tags(["1girl", "1boy", "1boy"])[0] == 2
    assert exact_tag_count(["1girl", "1_girl"]) == 1


# --- tag reconcile ---


def test_monochrome_fold_is_independent_of_hash_seed() -> None:
    """The monochrome fold follows tag order, so every interpreter run gives the same result."""
    code = (
        "import json;from argus_lens.assembly.tag_reconcile import reconcile_tags;"
        "r=reconcile_tags(['black','1girl','white','grey','gray'],monochrome=True);"
        "print(json.dumps([r.tags,r.dropped,r.added]))"
    )
    outputs = set()
    for seed in range(8):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1, outputs
    tags, dropped, _ = json.loads(outputs.pop())
    assert tags == ["monochrome", "1girl"]
    assert dropped == ["black", "white", "grey", "gray"]


# --- variants ---


def test_variant_caps_are_monotonic_in_priority() -> None:
    """A less important category never gets more slots than a more important one, and none gets 0."""
    plan = dict(_variant_bucket_plan("identity"))
    assert plan["identity"] is None
    assert all(cap is not None and cap >= 1 for name, cap in plan.items() if name != "identity")
    assert plan["lighting"] >= plan["setting"]  # zeroshot_priority 4 vs 5


# --- zeroshot ---


def test_zeroshot_reports_unconfigured_buckets_as_removed() -> None:
    """Fragments in a bucket with no category config are reported, not silently lost."""
    buckets = {"identity": ["blonde hair"], "mood": ["melancholic", "quiet"]}
    caption, removed = assemble_zeroshot_variant("sks", buckets, "photo")
    assert "melancholic" not in caption
    assert "melancholic" in removed and "quiet" in removed
