"""Tests for tag reconciliation: person nouns (#85), facial hair (#84), monochrome (#83)."""

import random

import pytest
from PIL import Image

from argus_lens import ArgusLens
from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.tag_reconcile import is_monochrome, reconcile_tags
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.refine import Refiner
from argus_lens.refine.refiner import _ungrounded_hue_pairs
from argus_lens.types import resolve_target_profile


def _tags(text: str) -> list[str]:
    """Split a comma-separated tag string."""
    return [t.strip() for t in text.split(",") if t.strip()]


# --- monochrome detection ---


def _gradient(tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Image.Image:
    """A 64x64 luminance gradient multiplied by *tint* (per channel)."""
    img = Image.new("RGB", (64, 64))
    img.putdata([tuple(int(((x + y) * 2) * t) for t in tint) for y in range(64) for x in range(64)])
    return img


def test_greyscale_and_toned_prints_are_monochrome() -> None:
    """Grey, ``L``-mode and sepia-toned images count as monochrome."""
    assert is_monochrome(_gradient())
    assert is_monochrome(_gradient().convert("L"))
    assert is_monochrome(_gradient((1.0, 0.94, 0.82)))  # sepia-ish tone


def test_colour_images_are_not_monochrome() -> None:
    """Random colour, a saturated single colour, and a grey photo with a hand-coloured patch are colour."""
    rng = random.Random(0)
    noise = Image.new("RGB", (64, 64))
    noise.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 64)])
    assert not is_monochrome(noise)
    assert not is_monochrome(Image.new("RGB", (32, 32), (200, 20, 20)))
    hand_coloured = _gradient()
    hand_coloured.paste((180, 40, 40), (0, 0, 16, 16))  # 6 % of the frame
    assert not is_monochrome(hand_coloured)


# --- person nouns (#85) ---


def test_girl_dropped_beside_woman_when_prose_names_no_child() -> None:
    """``21cde71d47011ef1`` (Marie Curie, 1900): ``girl`` and ``woman`` both tagged, prose says a woman."""
    rec = reconcile_tags(
        _tags("portrait, black, photo, white, girl, woman, hair, pose"),
        "A black-and-white studio portrait of Marie Curie, a woman with dark, tightly curled hair.",
    )
    assert "girl" not in rec.tags and "woman" in rec.tags
    assert rec.dropped == ["girl"]


def test_girl_kept_when_prose_names_a_child() -> None:
    """A real child keeps ``girl``."""
    rec = reconcile_tags(_tags("woman, girl, park"), "A woman walks with a young girl through a park.")
    assert {"woman", "girl"} <= set(rec.tags)


def test_girl_without_adult_tag_follows_the_prose() -> None:
    """Only ``girl`` tagged, prose names a woman: the tag becomes ``woman``."""
    rec = reconcile_tags(_tags("girl, kimono"), "A woman in a kimono holds a fan.")
    assert rec.tags == ["woman", "kimono"]


def test_tags_trusted_without_prose() -> None:
    """With no prose only the both-tagged rule applies; beard and a lone girl stay."""
    rec = reconcile_tags(_tags("girl, beard"), "")
    assert rec.tags == ["girl", "beard"]


def test_man_dropped_when_prose_describes_only_a_woman() -> None:
    """``e2d0ebdf6206e466`` (Marie Curie c. 1920s) is tagged both ``man`` and ``woman``."""
    rec = reconcile_tags(
        _tags("portrait, man, hair, woman"), "A portrait of a woman. She wears a dark, collared jacket."
    )
    assert "man" not in rec.tags and "woman" in rec.tags


def test_man_and_woman_kept_for_a_couple() -> None:
    """Both sexes in the prose keep both nouns."""
    rec = reconcile_tags(_tags("man, woman"), "A man and a woman dance.")
    assert rec.tags == ["man", "woman"]


# --- facial hair (#84) ---


@pytest.mark.parametrize(
    "prose",
    [
        "A black and white photographic portrait of an elderly man with a full, wavy white hair and a "
        "prominent, thick mustache.",
        "A portrait of a man with a prominent moustache.",
    ],
)
def test_beard_becomes_mustache_when_prose_says_mustache(prose: str) -> None:
    """``95e49647950bb96b`` (Mark Twain): RAM++ says beard, the prose says mustache."""
    rec = reconcile_tags(_tags("man, portrait, white, suit, hair, beard, photo"), prose)
    assert "beard" not in rec.tags and "mustache" in rec.tags


def test_beard_kept_when_prose_confirms_it() -> None:
    """``474787222160a479`` (Brady studio) has a real beard."""
    rec = reconcile_tags(_tags("man, uniform, beard"), "He has a thick, well-groomed beard and curly hair.")
    assert "beard" in rec.tags


def test_beard_kept_when_prose_is_silent() -> None:
    """``d145da92b2a4756c`` (Einstein 1921): prose silent on facial hair, so the tag is trusted."""
    rec = reconcile_tags(_tags("man, suit, beard"), "A portrait of a man standing before a chalkboard.")
    assert "beard" in rec.tags
    assert "facial hair" not in rec.tags and "mustache" not in rec.tags
    assert rec.dropped == [] and rec.added == []


def test_beard_kept_when_prose_names_both() -> None:
    """A full beard and moustache: the beard is confirmed, nothing is replaced."""
    rec = reconcile_tags(_tags("man, beard"), "An old man with a long beard and a drooping mustache.")
    assert "beard" in rec.tags and "mustache" not in rec.tags


# --- monochrome (#83) ---


def test_monochrome_folds_bare_colour_tags() -> None:
    """``black``/``white`` on a B&W photo become one ``monochrome`` tag; compounds are untouched."""
    rec = reconcile_tags(_tags("man, white, black, white hair, suit"), "", monochrome=True)
    assert rec.tags == ["man", "monochrome", "white hair", "suit"]


def test_monochrome_added_and_colour_tags_kept_for_colour_images() -> None:
    """A monochrome image gains the tag; a colour image keeps ``black``/``white``."""
    assert reconcile_tags(["church"], "", monochrome=True).tags == ["church", "monochrome"]
    assert reconcile_tags(["black", "white"], "", monochrome=False).tags == ["black", "white"]


# --- through the composer ---

PROFILE = resolve_target_profile(target_style="photo", target_backend="sdxl", target_category="identity")


def test_composer_applies_reconciliation_everywhere() -> None:
    """``girl``, ``black`` and ``white`` leave every variant and the index buckets (#83, #85)."""
    result = compose_caption_result(
        trigger_word="",
        tags="portrait, black, photo, white, girl, woman, hair, pose",
        prose="A black-and-white studio portrait of a woman with dark, tightly curled hair.",
        target_profile=PROFILE,
        monochrome=True,
    )
    for name, caption in result.caption_variants.items():
        fragments = {f.strip() for f in caption.split(",")}
        assert not {"girl", "black", "white"} & fragments, (name, caption)
    tag_buckets = result.metadata["buckets"]["tags"]
    assert not any({"girl", "black", "white"} & set(frags) for frags in tag_buckets.values())
    assert "monochrome" in tag_buckets["lighting"]
    assert result.metadata["monochrome"] is True
    assert "girl" not in result.metadata["reconciled_tags"]
    assert result.raw_tags.startswith("portrait, black")  # raw output untouched
    assert any("Reconciled tags" in n for n in result.compaction_notes)


def test_composer_mustache_in_identity_not_training() -> None:
    """``mustache`` lands in identity and stays out of the training caption (#84)."""
    result = compose_caption_result(
        trigger_word="",
        tags="man, beard, suit, hair",
        prose="A portrait of a man with a thick mustache, wearing a suit.",
        target_profile=PROFILE,
    )
    assert "mustache" in result.caption_variants["identity"]
    assert "beard" not in result.caption_variants["identity"]
    assert "mustache" not in result.caption_variants["training"]


# --- refine respects the reconciliation ---


class _Rewriter:
    """Rewriter returning queued replies."""

    name = "scripted"
    model_id = "test-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def rewrite(self, request, image=None):
        """Pop the next reply."""
        self.requests.append(request)
        return self.replies.pop(0)


def _refine(tags: str, prose: str, reply: str, *, monochrome: bool = False):
    """Refine the identity variant once (no retry) with *reply*."""
    result = compose_caption_result(
        trigger_word="", tags=tags, prose=prose, target_profile=PROFILE, monochrome=monochrome
    )
    rewriter = _Rewriter({"identity": reply})
    refined = Refiner(rewriter, variants=("identity",), retries=0).refine(result, PROFILE)
    return result, refined, rewriter


def test_refine_cannot_reintroduce_dropped_girl() -> None:
    """``96d0973fb186d3c3``: refined identity ``woman, geisha, girl`` is rejected (#85)."""
    result, refined, rewriter = _refine(
        "woman, drawing, kimono, stand, girl, geisha", "A woman in a traditional kimono stands.", "woman, geisha, girl"
    )
    assert refined.caption_variants["identity"] == result.caption_variants["identity"]
    assert "suppressed terms: girl" in " ".join(refined.metadata["refine"]["rejected"]["identity"])
    assert "girl" not in rewriter.requests[0].tags


def test_refine_cannot_put_beard_back_on_a_mustache() -> None:
    """``95e49647950bb96b``: ``man, white hair, beard`` is rejected when the prose says mustache (#84)."""
    _, refined, _ = _refine("man, beard, hair", "An elderly man with white hair and a thick mustache.", "man, beard")
    assert "suppressed terms: beard" in " ".join(refined.metadata["refine"]["rejected"]["identity"])


@pytest.mark.parametrize(
    ("reply", "accepted"),
    [
        ("man, white hair, black beard", False),  # b22c2745aae326c7
        ("black dress, white dress, woman", False),  # 1a1e2c89ae3a1cba
        ("man, white hair, black and white photo", True),
    ],
)
def test_refine_rejects_invented_colours_on_monochrome(reply: str, accepted: bool) -> None:
    """On a B&W image a hue word must be attached to the same noun in the inputs (#83)."""
    _, refined, rewriter = _refine(
        "portrait, hair, man, white, black, photo, beard, woman, dress",
        "A black and white studio portrait of a man with wild white hair beside a woman in a dark dress.",
        reply,
        monochrome=True,
    )
    assert rewriter.requests[0].monochrome is True
    assert ("identity" in refined.metadata["refine"]["accepted"]) is accepted, refined.metadata["refine"]


def test_refine_accepts_a_hue_the_prose_states_with_modifiers_between() -> None:
    """``d7a0818312de1e9e``: prose "wild, white, curly hair" grounds ``white hair`` (was rejected)."""
    _, refined, _ = _refine(
        "hair, man, photo, smile, portrait, sweater",
        "A black and white portrait of Albert Einstein. He has a full head of wild, white, curly hair.",
        "man, white hair, sweater",
        monochrome=True,
    )
    assert "identity" in refined.metadata["refine"]["accepted"], refined.metadata["refine"]


@pytest.mark.parametrize(
    ("text", "tags", "prose", "ungrounded"),
    [
        ("man, white hair", "man, hair", "a man with wild, white, curly hair.", []),
        ("man, white hair", "man, white hair", "", []),
        ("man, black beard", "white, black, photo, beard", "", ["black beard"]),  # tags never span a comma
        ("man, black hair", "man", "a white shirt and black trousers. his hair is short.", ["black hair"]),
        ("man, white hair", "man", "a man in a white shirt and black hair band.", ["white hair"]),
        ("man, white", "man, white", "", []),
    ],
)
def test_ungrounded_hue_pairs(text: str, tags: str, prose: str, ungrounded: list[str]) -> None:
    """A hue+noun pair is grounded by a tag holding both, or a prose sentence with the noun just after the hue."""
    assert _ungrounded_hue_pairs(text, tags, prose) == ungrounded


# --- engine wiring ---


class _Stub(CaptionBackend):
    """Backend returning a fixed string."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image) -> str:
        """Return the fixed text."""
        return self._text

    def unload(self) -> None:
        """No-op."""


def test_engine_detects_monochrome_on_every_path(tmp_path) -> None:
    """caption, caption_batch and caption_stream pass the pixel check to the composer."""
    grey, colour = tmp_path / "grey.png", tmp_path / "colour.png"
    _gradient().save(grey)
    Image.new("RGB", (16, 16), (30, 160, 40)).save(colour)
    engine = ArgusLens(backend=HybridPipeline(_Stub("tags", "man, black, white"), _Stub("prose", "A man.")))
    assert engine.caption(grey).metadata["monochrome"] is True
    assert engine.caption(colour).metadata["monochrome"] is False
    batch = engine.caption_batch([grey, colour])
    assert batch["grey.png"].metadata["reconciled_tags"] == "man, monochrome"
    assert batch["colour.png"].metadata["reconciled_tags"] == "man, black, white"
    ((_, streamed),) = list(engine.caption_stream([grey]))
    assert streamed.metadata["monochrome"] is True


def test_monochrome_folds_case_variants_into_one_tag() -> None:
    """``Black`` and ``black`` both fold into a single ``monochrome``, not two."""
    assert reconcile_tags(["Black", "black", "white", "man"], "", monochrome=True).tags == ["monochrome", "man"]


def test_refine_allows_black_and_white_medium_after_folding_colour_tags() -> None:
    """Folded ``black``/``white`` tags don't forbid naming the medium ("black and white photo")."""
    _, refined, _ = _refine(
        "man, black, white, suit",
        "A monochrome portrait of a man in a suit.",
        "man, suit, black and white photo",
        monochrome=True,
    )
    assert "suppressed terms" not in " ".join(refined.metadata["refine"].get("rejected", {}).get("identity", []))
