"""Lighting/action bucketing and empty category variants (#89)."""

import pytest

from argus_lens.assembly.classifier import classify_fragment, split_lighting_clauses
from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.variants import assemble_variant
from argus_lens.refine import Refiner
from argus_lens.types import resolve_target_profile

PROFILE = resolve_target_profile(target_style="photo", target_backend="sdxl", target_category="identity")


@pytest.mark.parametrize(
    "clause",
    [
        # Prose from the lightbox tape that was routed to identity/wardrobe/setting.
        "the lighting is directional, illuminating the subject's face and clothing with soft highlights",
        "the lighting is even and highlights the texture of the fabric and the details of the uniform",
        "the lighting highlights the texture of his hair and the folds in his clothing",
        "the background is softly lit, with a smooth gradient of light, creating a subtle, diffused light effect",
        "softly lit studio with a plain backdrop",
    ],
)
def test_lighting_clauses_go_to_lighting(clause: str) -> None:
    """A clause led by, or mostly about, the light is lighting even with hair/clothing words in it."""
    assert classify_fragment(clause) == "lighting"


def test_light_colored_is_not_lighting() -> None:
    """``light-colored`` describes a colour, not the light."""
    assert classify_fragment("light-colored, wavy hair") == "identity"


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        (
            "he is dressed in a formal suit jacket, and the lighting highlights the folds in his clothing",
            ["he is dressed in a formal suit jacket", "the lighting highlights the folds in his clothing"],
        ),
        (
            "the subject stands against a dark background, with the lighting focused on her from the front",
            ["the subject stands against a dark background", "the lighting focused on her from the front"],
        ),
        ("a woman in a red dress", ["a woman in a red dress"]),
        (
            "a man with light-colored hair, the light-colored coat",
            ["a man with light-colored hair, the light-colored coat"],
        ),
    ],
)
def test_split_lighting_clauses(clause: str, expected: list[str]) -> None:
    """A sentence is split where it turns to the lighting, and only there."""
    assert split_lighting_clauses(clause) == expected


@pytest.mark.parametrize(
    "clause",
    ["his tongue extended at the camera", "a cavalryman on horseback", "a soldier saluting the flag"],
)
def test_posed_and_historic_actions(clause: str) -> None:
    """Action hints cover posed and historic subjects, not only everyday verbs."""
    assert classify_fragment(clause) == "action"


def test_writing_tag_is_not_an_action() -> None:
    """RAM++ ``writing`` (text on an object) stays out of the action bucket; prose ``writing a letter`` doesn't."""
    tagged = compose_caption_result(trigger_word="", tags="wall, plaque, writing", target_profile=PROFILE)
    assert tagged.metadata["buckets"]["tags"]["action"] == []
    assert "writing" in tagged.metadata["buckets"]["tags"]["unclassified"]
    described = compose_caption_result(trigger_word="", prose="A man is writing a letter.", target_profile=PROFILE)
    assert described.metadata["buckets"]["prose"]["action"] == ["a man is writing a letter"]


def test_lighting_prose_reaches_the_lighting_variant() -> None:
    """``474787222160a479``: the lighting sentence now fills the lighting variant (was wardrobe)."""
    result = compose_caption_result(
        trigger_word="",
        tags="man, portrait, uniform, beard, photo",
        prose=(
            "A formal portrait of a man in a military uniform. "
            "The lighting is even and highlights the texture of the fabric and the details of the uniform."
        ),
        target_profile=PROFILE,
    )
    assert result.caption_variants["lighting"].startswith("the lighting is even")
    assert result.metadata["buckets"]["prose"]["lighting"]


def test_empty_category_gives_empty_variant() -> None:
    """No lighting or action content: those variants are empty (trigger only), not padded copies."""
    result = compose_caption_result(
        trigger_word="sks", tags="woman, red dress, kitchen", prose="", target_profile=PROFILE
    )
    assert result.caption_variants["lighting"] == "sks"
    assert result.caption_variants["action"] == "sks"
    assert {"lighting", "action"} <= set(result.metadata["empty_categories"])
    assert result.caption_variants["wardrobe"].startswith("sks, red dress")


def test_padding_still_available_on_request() -> None:
    """``pad_empty=True`` keeps the old fill-from-other-buckets behaviour."""
    buckets = {"identity": ["woman"], "wardrobe": ["red dress"], "lighting": []}
    assert assemble_variant("", buckets, "lighting", "photo") == ("", [])
    padded, _ = assemble_variant("", buckets, "lighting", "photo", pad_empty=True)
    assert "red dress" in padded


def test_final_caption_never_empty() -> None:
    """With an empty selected category the sidecar caption falls back to the padded caption."""
    profile = resolve_target_profile(target_style="photo", target_backend="sdxl", target_category="lighting")
    result = compose_caption_result(trigger_word="sks", tags="woman, red dress", prose="", target_profile=profile)
    assert result.caption_variants["lighting"] == "sks"
    assert result.final_caption.startswith("sks, ") and "red dress" in result.final_caption
    assert any("No lighting content" in n for n in result.compaction_notes)


class _Rewriter:
    """Rewriter returning one fixed reply."""

    name = "scripted"
    model_id = "test-model"

    def __init__(self, reply):
        self.reply = reply

    def rewrite(self, request, image=None):
        """Return the reply."""
        return self.reply


def test_refined_empty_category_keeps_final_caption() -> None:
    """An accepted empty category rewrite never empties ``final_caption``."""
    profile = resolve_target_profile(target_style="photo", target_backend="sdxl", target_category="lighting")
    result = compose_caption_result(trigger_word="sks", tags="woman, red dress", prose="", target_profile=profile)
    refined = Refiner(_Rewriter({"lighting": ""}), variants=("lighting",)).refine(result, profile, trigger_word="sks")
    assert refined.caption_variants["lighting"] == "sks"
    assert refined.final_caption == result.final_caption
