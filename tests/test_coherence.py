"""Tests for caption coherence: post-check, repair, and the assembled corpus."""

import json
from pathlib import Path

import pytest

from argus_lens.assembly.classifier import category_scores, classify_fragment, classify_fragment_detailed
from argus_lens.assembly.coherence import (
    check_caption,
    clean_prose,
    repair_caption,
    repair_fragment,
    split_sentences,
)
from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.filtering import extract_prose_tokens
from argus_lens.assembly.prose_training import assemble_prose_training_variant
from argus_lens.types import resolve_target_profile

CORPUS = json.loads((Path(__file__).parent / "fixtures" / "caption_corpus.json").read_text())


def _compose(case: dict, target_backend: str = "sdxl"):
    """Compose a corpus case with trigger ``sks`` for *target_backend*."""
    return compose_caption_result(
        trigger_word="sks",
        tags=case["tags"],
        prose=case["prose"],
        target_profile=resolve_target_profile(target_backend=target_backend),
    )


# --- corpus regression: every variant of every case passes the post-check ---


@pytest.mark.parametrize("target_backend", ["sdxl", "flux"])
@pytest.mark.parametrize("case", CORPUS, ids=[c["id"] for c in CORPUS])
def test_corpus_variants_pass_post_check(case: dict, target_backend: str) -> None:
    """No assembled variant contains fragments, markdown, hedging, or a missing trigger."""
    result = _compose(case, target_backend)
    for name, caption in result.caption_variants.items():
        kind = "prose" if name == "training_prose" and case["prose"] else "list"
        assert check_caption(caption, kind, trigger_word="sks") == [], (name, caption)
    assert result.final_caption == result.caption_variants[result.selected_category]


@pytest.mark.parametrize("case", CORPUS, ids=[c["id"] for c in CORPUS])
def test_corpus_training_prose_suppresses_identity(case: dict) -> None:
    """The Flux training caption never mentions hair or eyes."""
    caption = _compose(case, "flux").caption_variants["training_prose"].lower()
    assert " hair" not in caption and " eyes" not in caption, caption


@pytest.mark.parametrize("case", CORPUS, ids=[c["id"] for c in CORPUS])
def test_corpus_training_has_no_identity_tags(case: dict) -> None:
    """The SDXL training caption keeps identity out, including via prose enrichment."""
    caption = _compose(case).caption_variants["training"]
    assert "_hair" not in caption and "_eyes" not in caption, caption


def test_corpus_category_variants_lead_with_their_category() -> None:
    """A category caption opens with a fragment from that category when one exists."""
    result = _compose(next(c for c in CORPUS if c["id"] == "florence_park"))
    assert result.caption_variants["setting"].startswith("sks, bench")
    assert result.caption_variants["wardrobe"].startswith("sks, denim_jacket")


# --- classifier word boundaries ---


@pytest.mark.parametrize(
    ("fragment", "not_category"),
    [("chair", "identity"), ("that is all", "wardrobe"), ("the image", "identity"), ("street", "wardrobe")],
)
def test_hints_match_whole_words(fragment: str, not_category: str) -> None:
    """``hair`` must not fire in ``chair``, ``hat`` in ``that``, ``age`` in ``image``, ``tee`` in ``street``."""
    assert category_scores(fragment)[not_category] == 0


def test_underscored_tags_and_plurals_match() -> None:
    """``brown_hair`` hits ``hair``; ``trees`` hits ``tree``."""
    assert classify_fragment("brown_hair") == "identity"
    assert classify_fragment("trees") == "setting"


def test_unmatched_fragment_reports_fallback() -> None:
    """Fragments no hint matches fall back to identity with ``matched=False``."""
    assert classify_fragment_detailed("xyzzy") == ("identity", False)
    assert classify_fragment_detailed("red_dress") == ("wardrobe", True)


# --- clean_prose ---


def test_clean_prose_strips_markdown_and_labels() -> None:
    """Bold, bullets, and ``Label:`` prefixes are removed; bullet lines become sentences."""
    text, notes = clean_prose("**Scene:** A woman dances.\n- She wears a red gown\n- A chandelier hangs above")
    assert text == "A woman dances. She wears a red gown. A chandelier hangs above."
    assert "markdown" in notes


def test_clean_prose_removes_hedging() -> None:
    """Hedges become plain statements."""
    text, _ = clean_prose("She seems to be wearing what appears to be a yellow sundress, possibly cotton.")
    assert text == "She is wearing a yellow sundress, cotton."


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        ("In this image, a man is dancing.", "A man is dancing."),
        ("The image shows a woman in a park.", "A woman in a park."),
        ("Photograph of a woman standing.", "A woman standing."),
        ("A cat sleeps. This photo depicts a dog.", "A cat sleeps. A dog."),
    ],
)
def test_clean_prose_strips_openers(prose: str, expected: str) -> None:
    """Captioner openers are removed and the sentence re-capitalised."""
    assert clean_prose(prose)[0] == expected


# --- repair ---


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ("and wearing a hat", "wearing a hat"),
        ("holding a", "holding"),
        ("she is wearing a green robe", "wearing a green robe"),
        ("she wears a red dress", "wearing a red dress"),
        ("she has red hair and is smiling", "she has red hair and is smiling"),
        ("the the kitchen", "the kitchen"),
        ("She", ""),
        ("and the", ""),
    ],
)
def test_repair_fragment(fragment: str, expected: str) -> None:
    """List fragments lose dangling words and pronoun lead-ins, but never their subject."""
    assert repair_fragment(fragment) == expected


def test_repair_list_dedupes_and_absorbs_generic_tags() -> None:
    """``red_dress``/``red dress`` collapse, and ``dress`` is absorbed by ``red dress``."""
    fixed, _ = repair_caption("sks, red_dress, dress, red dress, kitchen", "list", trigger_word="sks")
    assert fixed == "sks, red_dress, kitchen"


def test_repair_list_keeps_tag_mentioned_in_prose_clause() -> None:
    """A long prose clause mentioning ``kitchen`` does not swallow the ``kitchen`` tag."""
    fixed, _ = repair_caption("kitchen, a woman cooking in a bright kitchen", "list")
    assert fixed == "kitchen, a woman cooking in a bright kitchen"


def test_repair_prose_merges_subjectless_and_drops_orphans() -> None:
    """Participle-led sentences join the previous one; orphans are dropped."""
    fixed, notes = repair_caption("A woman sits on a bench. Smiling at the camera. She.", "prose")
    assert fixed == "A woman sits on a bench, smiling at the camera."
    assert "merged subjectless sentence" in notes


def test_repair_prose_resolves_leading_pronoun() -> None:
    """A caption opening on ``She`` (antecedent removed) names the subject."""
    fixed, _ = repair_caption("sks, She is wearing a red dress.", "prose", trigger_word="sks")
    assert fixed == "sks, The woman is wearing a red dress."


def test_repair_prose_keeps_inner_connectives() -> None:
    """``, and`` / ``, while`` inside a sentence are grammar, not dangling words."""
    text = "The man wears a tuxedo, while the woman wears a gown, and they dance."
    assert repair_caption(text, "prose")[0] == text


# --- check_caption ---


def test_check_flags_missing_trigger_and_fragments() -> None:
    """Missing trigger and subjectless fragments are reported."""
    problems = check_caption("long blonde hair and is smiling, kitchen", "list", trigger_word="sks")
    assert any("trigger" in p for p in problems)
    assert any("missing its subject" in p for p in problems)


def test_check_flags_identity_budget_and_grounding() -> None:
    """Forbidden terms, token budget, and ungrounded vocabulary are enforced."""
    problems = check_caption(
        "a woman with brown hair rides a purple dragon over volcanoes",
        "list",
        forbidden_words=frozenset({"hair"}),
        token_budget=5,
        allowed_vocabulary=frozenset({"woman", "brown", "hair"}),
    )
    assert any("suppressed terms: hair" in p for p in problems)
    assert any("token budget" in p for p in problems)
    assert any("not grounded" in p for p in problems)


@pytest.mark.parametrize(
    "caption",
    [
        # #86: accepted identity rewrites from qwen3:4b with no commas at all.
        "man beard",
        "man beard officer soldier",
        "army soldier horse officer man",
        "woman geisha kimono",
        "sks, sword shield helmet armor horse",
    ],
)
def test_check_flags_space_separated_lists(caption: str) -> None:
    """Nouns run together without commas are a list missing its separators (#86)."""
    problems = check_caption(caption, "list", trigger_word="sks" if caption.startswith("sks") else "")
    assert any("separate items with commas" in p for p in problems), problems


def test_check_flags_dangling_participle() -> None:
    """A fragment ending on a transitive participle is dangling (#86)."""
    problems = check_caption("black dress, white dress, woman wearing", "list")
    assert any("dangling fragment" in p and "wearing" in p for p in problems), problems


@pytest.mark.parametrize(
    "caption",
    [
        "man, beard, officer, soldier",
        "man, white hair, black beard",
        "long sleeve white shirt",
        "man in a dark suit, woman holding a cup",
        "woman smiling, red dress",
        "sks, man bun, holding",
        "army officer, police officer",
    ],
)
def test_check_accepts_well_formed_lists(caption: str) -> None:
    """Comma-separated phrases, grammatical clauses and person-noun compounds pass."""
    assert check_caption(caption, "list", trigger_word="sks" if caption.startswith("sks") else "") == []


def test_check_passes_clean_prose() -> None:
    """A well-formed sentence caption passes."""
    assert check_caption("sks, A woman reads a book on a sofa.", "prose", trigger_word="sks") == []


# --- prose training variant ---


def test_prose_training_excises_coordinated_identity_phrases() -> None:
    """``with freckles and bright green eyes`` goes as a whole, leaving a clean sentence."""
    caption, removed = assemble_prose_training_variant(
        "", "A close-up portrait of a woman with freckles and bright green eyes.", token_budget=200
    )
    assert caption == "A close-up portrait of a woman."
    assert any("freckles" in r for r in removed)


def test_prose_training_rejoins_after_parenthetical_removal() -> None:
    """``A woman, in her twenties, walking`` reads ``A woman walking`` once age is removed."""
    caption, _ = assemble_prose_training_variant(
        "", "A woman, in her twenties, walking along a beach.", token_budget=200
    )
    assert caption == "A woman walking along a beach."


def test_prose_training_respects_budget() -> None:
    """Sentences beyond the token budget are dropped whole."""
    prose = "A woman stands in a kitchen. " * 20
    caption, removed = assemble_prose_training_variant("sks", prose, token_budget=20, target_backend="flux")
    assert caption.count(".") < 20 and removed


# #88: raw qwen3-vl prose from the lightbox tape whose training_prose was cut mid-phrase.
@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        (
            "A black-and-white portrait of a woman with light-colored, wavy hair pulled back from her face.",
            "A black-and-white portrait of a woman.",
        ),
        (
            "A black and white photographic portrait of an elderly man with a full, wavy white hair and a "
            "prominent, thick mustache.",
            "A black and white photographic portrait of an elderly man.",
        ),
        (
            "A portrait of a man in a military uniform. He has a thick, well-groomed beard and curly hair, "
            "with a serious expression.",
            "A portrait of a man in a military uniform.",
        ),
        (
            "His voluminous, wild white hair cascades over his forehead and clings to his ears, and he wears "
            "a textured, buttoned coat.",
            "He wears a textured, buttoned coat.",
        ),
        (
            "A black and white portrait captures a man with a voluminous, curly white hairstyle and a thick "
            "mustache, looking to the side.",
            "A black and white portrait captures a man, looking to the side.",
        ),
        (
            "A studio portrait of Marie Curie, a woman with dark, tightly curled hair pulled back from her face.",
            "A studio portrait of Marie Curie, a woman.",
        ),
        (
            "A woodblock print of a woman in a kimono. Her hair is styled in an intricate updo, and her face "
            "is rendered with subtle, calm features.",
            "A woodblock print of a woman in a kimono.",
        ),
        (
            "A formal portrait of a man with voluminous, light-colored hair and a prominent mustache, dressed "
            "in a dark suit and bow tie.",
            "A formal portrait of a man, dressed in a dark suit and bow tie.",
        ),
    ],
)
def test_prose_training_excises_whole_identity_phrase(prose: str, expected: str) -> None:
    """Coordinated adjectives go with their identity noun; no stub or orphan is left (#88)."""
    caption, _ = assemble_prose_training_variant("", prose, token_budget=200, target_backend="flux")
    assert caption == expected


def test_prose_training_keeps_short_sentence_without_identity() -> None:
    """Only identity removal makes a short remainder an orphan; filler removal does not."""
    caption, _ = assemble_prose_training_variant("", "The image shows a dog.", token_budget=200)
    assert caption == "A dog."


def test_prose_training_keeps_quoted_abbreviation_whole() -> None:
    """``"J.T. GUARDS."`` is neither split nor given a second full stop (#88)."""
    prose = 'He holds a small banner with the text "J.T. GUARDS." The photograph is set within a gold frame.'
    caption, _ = assemble_prose_training_variant("", prose, token_budget=200, target_backend="flux")
    assert caption == prose
    result = _compose({"tags": "man, banner", "prose": prose}, "flux")
    assert '"J.T. GUARDS." The photograph' in result.caption_variants["training_prose"]


def test_prose_training_caps_sentences_for_clip() -> None:
    """A CLIP target keeps two sentences; T5 keeps them all (#88)."""
    prose = "A woman stands in a park. She wears a red coat. A dog sits at her feet. Trees line the path."
    clip, removed = assemble_prose_training_variant("", prose, token_budget=200, target_backend="sdxl")
    assert clip == "A woman stands in a park. She wears a red coat."
    assert "A dog sits at her feet." in removed
    t5, _ = assemble_prose_training_variant("", prose, token_budget=200, target_backend="flux")
    assert t5 == prose


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            'He holds a banner with the text "J.T. GUARDS." The photograph is set.',
            ['He holds a banner with the text "J.T. GUARDS."', "The photograph is set."],
        ),
        ("Dr. Smith waves. She smiles!", ["Dr. Smith waves.", "She smiles!"]),
        ('It reads "Stop. Go." Then a car', ['It reads "Stop. Go."', "Then a car"]),
        ("One; two.\nThree", ["One; two.", "Three"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    """Initials, titles and quotations do not end a sentence; a closing quote can."""
    assert split_sentences(text) == expected


# --- enrichment ---


def test_enrichment_yields_compounds_not_loose_words() -> None:
    """Only modifier+noun compounds are lifted; verbs and pronoun-led nouns are not."""
    tokens = extract_prose_tokens(["she wears a cream knit sweater next to her lamp", "two chairs"], frozenset())
    assert tokens == ["knit_sweater"]


def test_split_sentences_ignores_an_unclosed_inch_mark() -> None:
    """A stray ``"`` (12" ruler) is not an open quotation swallowing the rest of the text."""
    assert split_sentences('A man holds a 12" ruler. He smiles. The room is dark.') == [
        'A man holds a 12" ruler.',
        "He smiles.",
        "The room is dark.",
    ]
