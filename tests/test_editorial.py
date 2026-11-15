"""Editorial prose (mood, praise, meta) never reaches a caption variant (#104)."""

from __future__ import annotations

import re

import pytest

from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.editorial import drop_editorial_sentences, is_editorial_sentence, is_mood_clause
from argus_lens.backends.openai_compat import _DEFAULT_PROMPT, PROMPT_PRESETS, OpenAICompatBackend
from argus_lens.types import CaptionTargetProfile


def test_default_prompt_is_factual(monkeypatch) -> None:
    """The default asks for concrete detail, forbids mood and meta openers, and bounds length.

    Pinned so an edit can't silently fall back to "Describe this image in detail.".
    """
    monkeypatch.delenv("ARGUS_OPENAI_COMPAT_PROMPT", raising=False)
    assert _DEFAULT_PROMPT == "straightforward"
    prompt = OpenAICompatBackend().user_prompt
    assert prompt == PROMPT_PRESETS["straightforward"]
    assert "in detail" not in prompt.lower()
    assert "concrete details" in prompt
    assert "Omit mood and speculative wording" in prompt
    assert 'without starting with "This image is"' in prompt
    assert "concise" in prompt


# Recorded editorial outputs in the shape qwen3-vl:2b gave under the "detailed" prompt (#104).
_EDITORIAL = [
    (
        "brown bear, river, water",
        "This is a photograph of a brown bear standing in a shallow river, its head angled down toward the water. "
        "Its fur is thick and shaggy. The overall scene is natural and peaceful, with the bear's gentle movement "
        "and the serene environment creating a tranquil and wildlife-focused image.",
    ),
    (
        "hill, pasture, grassy",
        "The image shows a green pasture with a curved hill in the middle ground. The overall atmosphere of the "
        "image is peaceful and idyllic, capturing the serene beauty of a countryside setting.",
    ),
    (
        "bus, curb, bus stop",
        "A blue and white bus is parked at a curb, captured in a vibrant and well-lit environment. "
        "It has a sleek design with rounded corners. This is a detailed description of the image.",
    ),
    (
        "woman, red dress, kitchen",
        "A woman in a red dress stands at a kitchen counter. The image evokes a sense of warmth and domestic "
        "comfort, inviting the viewer into a quiet moment.",
    ),
]
_BANNED = re.compile(
    r"this is a (?:\w+ )?photograph|the image shows|this is a detailed description|overall scene|"
    r"overall atmosphere|evokes|sense of|the viewer|creating a tranquil|serene beauty|captured in a vibrant",
    re.I,
)


@pytest.mark.parametrize(("tags", "prose"), _EDITORIAL)
@pytest.mark.parametrize("backend", ["sdxl", "flux"])
def test_no_variant_carries_editorial_prose(tags: str, prose: str, backend: str) -> None:
    """No caption variant contains a meta opener or a mood sentence/clause."""
    profile = CaptionTargetProfile(target_style="photo", target_backend=backend, target_category="identity")
    result = compose_caption_result(trigger_word="sks", tags=tags, prose=prose, target_profile=profile)
    for name, text in result.caption_variants.items():
        assert not _BANNED.search(text), f"{name}: {text}"
    assert result.raw_prose == prose  # the raw model output is still recorded verbatim


def test_concrete_content_survives() -> None:
    """The concrete sentences around the editorial ones still reach the variants."""
    profile = CaptionTargetProfile(target_style="photo", target_backend="flux", target_category="identity")
    tags, prose = _EDITORIAL[0]
    result = compose_caption_result(trigger_word="", tags=tags, prose=prose, target_profile=profile)
    assert "shallow river" in result.caption_variants["training_prose"]
    assert "thick and shaggy" in result.caption_variants["zeroshot"]
    assert any("overall scene" in p for p in result.removed_phrases)


@pytest.mark.parametrize(
    "sentence",
    [
        "The overall atmosphere is peaceful and idyllic.",
        "This is a detailed description of the image.",
        "The composition evokes a sense of nostalgia.",
        "The viewer is drawn into the quiet scene.",
        "The image conveys a feeling of calm.",
    ],
)
def test_editorial_sentences(sentence: str) -> None:
    """Mood/meta-only sentences are editorial."""
    assert is_editorial_sentence(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "A brown bear stands in a shallow river.",
        "Warm light from the left window evokes late afternoon.",  # names a direction: kept
        "The image is black and white.",
        "Overall, two red buses are parked at the curb.",
        "Fog fills the valley below the ridge.",
    ],
)
def test_concrete_sentences_kept(sentence: str) -> None:
    """Sentences with visual content are not editorial, even with a mood marker."""
    assert not is_editorial_sentence(sentence)


def test_mood_clause() -> None:
    """A participle clause carrying only mood is dropped; a concrete one is kept."""
    assert is_mood_clause("creating a peaceful atmosphere")
    assert is_mood_clause("adding to the charm of the scene")
    assert not is_mood_clause("highlighting the texture of the bark")
    assert not is_mood_clause("holding a cup")


def test_never_drops_all_prose() -> None:
    """If every sentence is editorial the prose is left alone rather than emptied."""
    prose = "The overall scene is serene and peaceful."
    assert drop_editorial_sentences(prose) == (prose, [])


def test_mood_subclause_stripped_from_concrete_sentence() -> None:
    """A trailing mood clause is cut; the subject clause stays."""
    text, dropped = drop_editorial_sentences("A bus at a curb, captured in a vibrant and well-lit environment.")
    assert text == "A bus at a curb."
    assert dropped == ["captured in a vibrant and well-lit environment"]
