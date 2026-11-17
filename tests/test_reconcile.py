"""Tests for the attribute reconciliation package (issue #36)."""

from __future__ import annotations

import pytest
from PIL import Image

from argus_lens.assembly.subjects import hedge_prose_count
from argus_lens.engine import ArgusLens
from argus_lens.reconcile import Reconciler, build_verifier, detect_claims, detect_disputes
from argus_lens.reconcile.affiliation import contradicted_side
from argus_lens.reconcile.color_sample import dominant_color_name, nearest_color_name
from argus_lens.reconcile.questions import answer_budget, build_question, parse_answer
from argus_lens.reconcile.rewrite import apply_fix
from argus_lens.reconcile.text_claims import detect_text_claims, is_confirmed
from argus_lens.reconcile.types import AttributeDispute, Verdict
from argus_lens.reconcile.verifiers.florence import FlorenceGroundingVerifier
from argus_lens.reconcile.verifiers.molmo import MolmoVerifier
from argus_lens.reconcile.verifiers.openai_compat import OpenAICompatVQAVerifier, encode_data_url
from argus_lens.reconcile.verifiers.tag_prior import TagPriorVerifier

# Reuse the eval test's stub-backend helpers (no models needed).
from tests.test_eval import _stub_engine, _write_image

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def test_detect_disputes_color_and_pose() -> None:
    """Disputes mirror the eval detector: a colour and a posture conflict."""
    disputes = detect_disputes("red_dress, standing", "A woman in a blue dress, sitting.")
    kinds = {d.kind for d in disputes}
    assert kinds == {"color", "pose"}
    color = next(d for d in disputes if d.kind == "color")
    assert color.subject == "dress"
    assert color.tags_say == ("red",)
    assert color.prose_says == ("blue",)


# --------------------------------------------------------------------------- #
# Rewrite
# --------------------------------------------------------------------------- #


def test_apply_color_fix_confined_to_subject_clause() -> None:
    """The wrong colour is fixed on the subject, a same-colour elsewhere is left alone."""
    prose = "a blue dress, and a blue sky"
    dispute = AttributeDispute("color", "dress", ("red",), ("blue",))
    fixed = apply_fix(prose, dispute, "red")
    assert fixed == "a red dress, and a blue sky"


def test_apply_pose_fix() -> None:
    """A posture term is replaced across the prose."""
    dispute = AttributeDispute("pose", "posture", ("standing",), ("sitting",))
    assert apply_fix("she is sitting on a bench", dispute, "standing") == "she is standing on a bench"


def test_apply_color_fix_recolors_nearest_noun_only() -> None:
    """Only the colour nearest the subject is recoloured; a sibling noun keeps its colour."""
    dispute = AttributeDispute("color", "dress", ("green",), ("red",))
    assert apply_fix("blue eyes in a red dress", dispute, "green") == "blue eyes in a green dress"


def test_apply_color_fix_preserves_capitalization() -> None:
    """A capitalised source colour keeps its case."""
    dispute = AttributeDispute("color", "dress", ("blue",), ("red",))
    assert apply_fix("Red dress on the chair.", dispute, "blue") == "Blue dress on the chair."


def test_apply_pose_fix_only_flagged_terms() -> None:
    """A pose term the detector did not flag is left untouched."""
    # Only "sitting" was flagged; a stray "standing" elsewhere must not be rewritten.
    dispute = AttributeDispute("pose", "posture", ("kneeling",), ("sitting",))
    assert apply_fix("a sitting figure by a standing lamp", dispute, "kneeling") == (
        "a kneeling figure by a standing lamp"
    )


# --------------------------------------------------------------------------- #
# Colour sampling
# --------------------------------------------------------------------------- #


def test_nearest_color_name() -> None:
    """Pure RGB → palette name."""
    assert nearest_color_name((250, 10, 10)) == "red"
    assert nearest_color_name((10, 20, 210)) == "blue"


def test_dominant_color_name_from_box() -> None:
    """A red left half, blue right half — the box selects the right colour."""
    img = Image.new("RGB", (20, 10), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 10, 10))  # left half red
    assert dominant_color_name(img, (0, 0, 10, 10)) == "red"
    assert dominant_color_name(img, (10, 0, 20, 10)) == "blue"


def test_dominant_color_name_clamps_out_of_bounds_box() -> None:
    """An out-of-bounds box is clamped, not sampled as black padding."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    assert dominant_color_name(img, (50, 50, 300, 300)) == "red"


def test_dominant_color_name_degenerate_box_abstains() -> None:
    """A zero-area / rounded-flat box returns None (abstain), not the whole image."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    assert dominant_color_name(img, (10, 10, 10, 10)) is None
    assert dominant_color_name(img, (10.4, 10, 10.5, 90)) is None  # x rounds to 10==10


# --------------------------------------------------------------------------- #
# Questions / answer parsing
# --------------------------------------------------------------------------- #


def test_build_and_parse_question() -> None:
    """Colour/pose questions are built, and answers map onto the vocabulary."""
    assert "colour of the dress" in build_question(AttributeDispute("color", "dress", (), ()))
    assert parse_answer("It is red.", "color") == "red"
    assert parse_answer("grey", "color") == "gray"  # canonicalised
    assert parse_answer("The person is sitting down.", "pose") == "sitting"
    assert parse_answer("I am not sure", "color") is None  # abstain


# --------------------------------------------------------------------------- #
# Verifiers
# --------------------------------------------------------------------------- #


def test_tag_prior_verifier_trusts_tags() -> None:
    """The model-free verifier resolves to the tag value."""
    v = TagPriorVerifier()
    verdict = v.verify(Image.new("RGB", (4, 4)), AttributeDispute("color", "dress", ("red",), ("blue",)))
    assert verdict.value == "red"
    assert verdict.source == "tag-prior"


def test_florence_verifier_samples_grounded_region() -> None:
    """With an injected grounding fn, the colour comes from the box's pixels."""
    img = Image.new("RGB", (20, 10), (255, 0, 0))
    v = FlorenceGroundingVerifier(ground_fn=lambda image, phrase: [(0, 0, 20, 10)])
    verdict = v.verify(img, AttributeDispute("color", "dress", ("red",), ("blue",)))
    assert verdict.value == "red"
    # Pose is not a grounding task → abstain.
    assert v.verify(img, AttributeDispute("pose", "posture", ("standing",), ("sitting",))).value is None


def test_molmo_verifier_parses_answer() -> None:
    """With an injected answer fn, the parsed value is returned."""
    v = MolmoVerifier(answer_fn=lambda image, q: "red")
    verdict = v.verify(Image.new("RGB", (4, 4)), AttributeDispute("color", "dress", ("red",), ("blue",)))
    assert verdict.value == "red"
    assert verdict.source == "molmo"


def test_openai_compat_verifier_parses_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mocked HTTP answer flows through parsing into a verdict."""
    v = OpenAICompatVQAVerifier(base_url="http://x/v1", model_id="m")
    monkeypatch.setattr(v, "_ask", lambda image, q, max_tokens=10: "The dress is red.")
    verdict = v.verify(Image.new("RGB", (4, 4)), AttributeDispute("color", "dress", ("red",), ("blue",)))
    assert verdict.value == "red"
    assert verdict.source == "openai-compat"


def test_encode_data_url_is_png() -> None:
    """Image encodes to a base64 PNG data URL."""
    assert encode_data_url(Image.new("RGB", (2, 2))).startswith("data:image/png;base64,")


def test_build_verifier_factory() -> None:
    """The factory builds each verifier and rejects unknown names."""
    assert isinstance(build_verifier("tag-prior"), TagPriorVerifier)
    assert isinstance(build_verifier("openai-compat", base_url="http://x/v1"), OpenAICompatVQAVerifier)
    with pytest.raises(ValueError, match="Unknown verifier"):
        build_verifier("nope")


# --------------------------------------------------------------------------- #
# Reconciler
# --------------------------------------------------------------------------- #


def test_reconciler_rewrites_disputed_color() -> None:
    """A tag-prior reconciler flips the prose colour to match the tags."""
    r = Reconciler(TagPriorVerifier())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a woman in a blue dress")
    assert "red dress" in outcome.prose
    assert "blue" not in outcome.prose
    assert len(outcome.changes) == 1
    assert outcome.changes[0].now == "red"


def test_reconciler_no_change_when_consistent() -> None:
    """No disputes → prose untouched, no changes recorded."""
    r = Reconciler(TagPriorVerifier())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a woman in a red dress")
    assert outcome.prose == "a woman in a red dress"
    assert outcome.changes == []


def test_reconciler_survives_verifier_error() -> None:
    """A raising verifier is caught; the prose is returned unchanged, errors counted."""

    class _Boom:
        name = "boom"

        def verify(self, image, dispute):
            raise RuntimeError("nope")

    r = Reconciler(_Boom())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a blue dress")
    assert outcome.prose == "a blue dress"
    assert outcome.changes == []
    assert outcome.errors == 1


def test_reconciler_survives_verifier_returning_none() -> None:
    """A verifier that returns None (not a Verdict) must not crash captioning."""

    class _NoneVerifier:
        name = "none"

        def verify(self, image, dispute):
            return None

    r = Reconciler(_NoneVerifier())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a blue dress")
    assert outcome.prose == "a blue dress"
    assert outcome.changes == []


def test_reconciler_no_phantom_change_on_synonym_verdict() -> None:
    """A verdict that is a spelling variant of the prose colour records no change."""

    class _Violet:
        name = "violet"

        def verify(self, image, dispute):
            return Verdict(subject=dispute.subject, value="violet", source=self.name)

    # tags say red, prose says violet (→ canonical purple); verifier answers "violet".
    r = Reconciler(_Violet())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a violet dress")
    assert outcome.prose == "a violet dress"  # violet canon == purple == verdict → no textual change
    assert outcome.changes == []


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #


def test_engine_applies_reconciliation(tmp_path) -> None:
    """An engine with a verifier fixes the prose before assembly; without one it doesn't."""
    img = _write_image(tmp_path / "x.png")

    plain = _stub_engine(tags="red_dress", prose="a woman in a blue dress")
    assert "blue dress" in plain.caption(img).raw_prose

    fixed = ArgusLens(
        backend=plain.backend,  # reuse the same stub hybrid pipeline
        verifier=build_verifier("tag-prior"),
    )
    result = fixed.caption(img)
    assert "red dress" in result.raw_prose
    assert "blue" not in result.raw_prose


# --------------------------------------------------------------------------- #
# Headcount disputes (#87)
# --------------------------------------------------------------------------- #


def test_detect_count_dispute() -> None:
    """An exact tag headcount vs the one headcount the prose states is a ``count`` dispute."""
    disputes = detect_disputes("1girl, 1boy, standing", "Three people stand in a kitchen.")
    assert disputes == [AttributeDispute(kind="count", subject="people", tags_say=("2",), prose_says=("3",))]


@pytest.mark.parametrize(
    ("tags", "prose"),
    [
        ("1girl, 1boy", "Two people stand in a kitchen."),  # agree
        ("multiple_girls", "Three women sit on a bench."),  # lower bound, not exact
        ("6+girls", "Nine women sit on a bench."),  # lower bound
        ("2girls, 3boys", "Two women and three men sit at a table."),  # several headcounts
        ("1girl, 1boy", "A group of people at a table."),  # no exact prose count
        ("man, smile", "Six soldiers stand in a row."),  # no headcount tags (RAM++)
    ],
)
def test_no_count_dispute(tags: str, prose: str) -> None:
    """Only two exact, comparable headcounts can disagree."""
    assert not [d for d in detect_disputes(tags, prose) if d.kind == "count"]


def test_tag_prior_fixes_count() -> None:
    """tag-prior rewrites the stated headcount to the tag count."""
    res = Reconciler(TagPriorVerifier()).reconcile(
        Image.new("RGB", (8, 8)), "2girls, 1boy", "Six soldiers stand in a row; the soldiers wear grey."
    )
    assert res.prose == "Three soldiers stand in a row; the soldiers wear grey."
    assert [(c.kind, c.was, c.now) for c in res.changes] == [("count", ("6",), "3")]


def test_count_rewrite_keeps_grammar() -> None:
    """Digits stay digits; a singular verdict leaves the plural phrase alone."""
    dispute = AttributeDispute(kind="count", subject="people", tags_say=("5",), prose_says=("6",))
    assert apply_fix("A photo of 6 men.", dispute, "5") == "A photo of 5 men."
    assert apply_fix("Six men.", dispute, "1") == "Six men."
    assert apply_fix("Six men.", dispute, "several") == "Six men."


def test_count_question_and_answer() -> None:
    """VQA verifiers ask for an integer and parse digits or number words."""
    dispute = AttributeDispute(kind="count", subject="people", tags_say=("2",), prose_says=("3",))
    assert "How many people" in build_question(dispute)
    assert parse_answer("5", "count") == "5"
    assert parse_answer("There are five.", "count") == "5"
    assert parse_answer("I can't tell", "count") is None


def test_florence_abstains_on_count() -> None:
    """Florence grounding only adjudicates colour."""
    dispute = AttributeDispute(kind="count", subject="people", tags_say=("2",), prose_says=("3",))
    verdict = FlorenceGroundingVerifier.__new__(FlorenceGroundingVerifier).verify(Image.new("RGB", (8, 8)), dispute)
    assert verdict.value is None


# --------------------------------------------------------------------------- #
# Unchecked headcount claims (#87): no tag count, so only a verifier can check
# --------------------------------------------------------------------------- #

_NORWAY = "A group of six soldiers in 1863 uniforms, standing on a hillside in Norway."


class _CountVerifier:
    """A VQA-style verifier stub that answers every count question with *answer*."""

    name = "stub-vqa"

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.asked: list[AttributeDispute] = []

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        self.asked.append(dispute)
        return Verdict(subject=dispute.subject, value=self.answer, source=self.name)


def test_detect_claim_when_tags_state_no_headcount() -> None:
    """RAM++ tags say nothing about headcount, so the prose's one count becomes a claim."""
    claims = detect_claims("soldier, uniform, man, hillside", _NORWAY)
    assert claims == [AttributeDispute(kind="count", subject="people", tags_say=(), prose_says=("6",))]
    assert claims[0].is_claim


@pytest.mark.parametrize(
    ("tags", "prose"),
    [
        ("2girls, 1boy", "Six soldiers stand in a row."),  # tags state a count: a dispute, not a claim
        ("soldier", "Two women and three men sit at a table."),  # several headcounts
        ("soldier", "A group of soldiers on a hillside."),  # no exact prose count
        ("soldier", "A soldier stands on a hillside."),  # singular
    ],
)
def test_no_claim(tags: str, prose: str) -> None:
    """Only a single exact prose headcount with no tag count is a claim."""
    assert detect_claims(tags, prose) == []


def test_claims_are_not_contradictions() -> None:
    """The eval contradiction metric (and detect_disputes) ignores claims."""
    assert detect_disputes("soldier, uniform", _NORWAY) == []


def test_verifier_disagreeing_with_claim_drops_the_count() -> None:
    """A VQA count that disagrees with an untagged prose count drops the number rather than swapping it."""
    verifier = _CountVerifier("5")
    res = Reconciler(verifier).reconcile(Image.new("RGB", (8, 8)), "soldier, uniform", _NORWAY)
    assert res.prose == "A group of soldiers in 1863 uniforms, standing on a hillside in Norway."
    assert [(c.kind, c.was, c.now, c.verified) for c in res.changes] == [("count", ("6",), "unstated", "5")]
    assert "How many people" in build_question(verifier.asked[0])


@pytest.mark.parametrize("answer", ["6", None])
def test_confirmed_or_abstained_claim_keeps_the_count(answer: str | None) -> None:
    """A verifier that agrees, or abstains, leaves the prose as written."""
    res = Reconciler(_CountVerifier(answer)).reconcile(Image.new("RGB", (8, 8)), "soldier", _NORWAY)
    assert res.prose == _NORWAY and res.changes == []


def test_tag_prior_abstains_on_claims() -> None:
    """tag-prior has no tag value to offer for a claim, so the prose is untouched."""
    res = Reconciler(TagPriorVerifier()).reconcile(Image.new("RGB", (8, 8)), "soldier", _NORWAY)
    assert res.prose == _NORWAY and res.changes == []


@pytest.mark.parametrize(
    ("prose", "hedged"),
    [
        ("A group of six soldiers on a hill.", "A group of soldiers on a hill."),
        ("Six soldiers stand; the six soldiers wear grey.", "Several soldiers stand; the soldiers wear grey."),
        ("A photo of 12 young men.", "A photo of several young men."),
        ("A row of five uniformed officers.", "A row of uniformed officers."),
        ("Two women and a dog.", "Several women and a dog."),
        ("Soldiers in 1863 uniforms, two horses.", "Soldiers in 1863 uniforms, two horses."),  # not people
    ],
)
def test_hedge_prose_count(prose: str, hedged: str) -> None:
    """Hedging drops exact headcounts and keeps the sentence grammatical."""
    assert hedge_prose_count(prose) == hedged


def test_count_inside_a_collective_phrase_is_found_and_rewritten() -> None:
    """ "a group of six soldiers" states six: a tag-vs-prose dispute fixes it in place (#87 regression)."""
    disputes = detect_disputes("5boys", _NORWAY)
    assert [(d.kind, d.prose_says) for d in disputes] == [("count", ("6",))]
    res = Reconciler(TagPriorVerifier()).reconcile(Image.new("RGB", (8, 8)), "5boys", _NORWAY)
    assert res.prose.startswith("A group of five soldiers")


# --- text claims (#87): names and quotes the prose says it read ---------------------------------------------

_DIPLOMA = (
    "An ornate Nobel diploma with two facing pages. "
    "The right-hand page details an event for Professor Henrik Bohr. "
    "Gold borders frame the text."
)


@pytest.mark.parametrize(
    ("prose", "claims"),
    [
        (_DIPLOMA, ["Professor Henrik Bohr"]),
        ('A library stamp reads "N.Y. PUBLIC LIBRARY" in the corner.', ["N.Y. PUBLIC LIBRARY"]),
        ("A letter signed by Henri Becquerel lies on the desk.", ["Henri Becquerel"]),
        ("A sign labelled Café Rouge hangs over Dr. Smith's door. A man stands outside.", ["Café Rouge"]),
    ],
)
def test_detect_text_claims(prose: str, claims: list[str]) -> None:
    """Quoted strings, names after a reading verb, and titled names near written material are text claims."""
    assert detect_text_claims(prose) == claims


@pytest.mark.parametrize(
    "prose",
    [
        "Professor Albert Einstein sticks out his tongue.",  # a titled name with no written material
        "A soldier's cap and the men's coats.",  # apostrophes are not quotes
        "A portrait of a woman in a blue dress.",
    ],
)
def test_no_text_claim(prose: str) -> None:
    """A famous face named from memory, or prose with no text, makes no text claim."""
    assert detect_text_claims(prose) == []
    assert all(c.kind != "text" for c in detect_claims("man", prose))


@pytest.mark.parametrize(
    ("claim", "transcript", "confirmed"),
    [
        ("Professor Henrik Bohr", "PROFESSEUR HENRI BECQUEREL", False),
        ("Professor Henri Becquerel", "PROFESSEUR HENRI BECQUERE1", True),  # OCR noise
        ("Henri Becquerel", "HENRIBECQUEREL", True),  # OCR ran the words together
        ("N.Y. PUBLIC LIBRARY", "THE N.Y. PUBLIC LIBRARY", True),
        ("CALIFORNIA PUBLIC LIBRARY", "THE N.Y. PUBLIC LIBRARY", False),
    ],
)
def test_text_claim_matching(claim: str, transcript: str, confirmed: bool) -> None:
    """Every significant word of the claim must appear in the transcription, allowing OCR noise."""
    assert is_confirmed(claim, transcript) is confirmed


def test_unread_name_is_dropped_with_its_sentence() -> None:
    """A transcription without the claimed name drops the sentence that states it (#87, 556344c411724aaa)."""
    verifier = _CountVerifier("Prix Nobel ... Monsieur Henri Becquerel ... Stockholm 1903")
    res = Reconciler(verifier).reconcile(Image.new("RGB", (8, 8)), "document, paper", _DIPLOMA)
    assert res.prose == "An ornate Nobel diploma with two facing pages. Gold borders frame the text."
    assert [(c.kind, c.was, c.now) for c in res.changes] == [("text", ("Professor Henrik Bohr",), "unstated")]
    assert res.changes[0].verified.startswith("Prix Nobel")
    assert "Transcribe" in build_question(verifier.asked[0])


@pytest.mark.parametrize("answer", ["Monsieur le Professeur Henrik Bohr", None])
def test_read_or_unreadable_name_is_kept(answer: str | None) -> None:
    """A transcription containing the name, or none at all (abstain), keeps the prose."""
    res = Reconciler(_CountVerifier(answer)).reconcile(Image.new("RGB", (8, 8)), "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and res.changes == []


def test_one_sentence_prose_keeps_an_unread_name() -> None:
    """Dropping the only sentence would leave nothing, so the prose stays."""
    prose = "A diploma for Professor Henrik Bohr."
    res = Reconciler(_CountVerifier("Henri Becquerel")).reconcile(Image.new("RGB", (8, 8)), "document", prose)
    assert res.prose == prose and res.changes == []


def test_text_claim_survives_verifier_returning_none() -> None:
    """A verifier that returns None for a text claim abstains rather than crashing reconcile."""

    class _NoneVerifier:
        name = "none"

        def verify(self, image, dispute):
            return None

    res = Reconciler(_NoneVerifier()).reconcile(Image.new("RGB", (8, 8)), "document", _DIPLOMA)
    assert res.prose == _DIPLOMA
    assert res.changes == [] and res.errors == 0


def test_text_answer_parsing_and_budget() -> None:
    """A transcription passes through whitespace-collapsed; NONE abstains; text gets a longer answer budget."""
    assert parse_answer("  PRIX\n NOBEL ", "text") == "PRIX NOBEL"
    assert parse_answer("NONE.", "text") is None
    assert answer_budget("text") > answer_budget("color")


def test_florence_ocrs_text_claims() -> None:
    """Florence answers a text claim with its OCR task."""
    v = FlorenceGroundingVerifier(ocr_fn=lambda image: "HENRI\nBECQUEREL")
    claim = AttributeDispute("text", "text", (), ("Henrik Bohr",))
    assert v.verify(Image.new("RGB", (4, 4)), claim).value == "HENRI BECQUEREL"


# --- affiliation (#87): a side the prose's own uniform colour contradicts -----------------------------------


@pytest.mark.parametrize(
    ("prose", "fixed"),
    [
        (
            "A Union Army soldier in a gray uniform poses for a tintype.",
            "A soldier in a gray uniform poses for a tintype.",
        ),
        ("Union soldiers wear grey wool coats.", "Soldiers wear grey wool coats."),
        (
            "An officer of the Confederate Army in a dark blue uniform.",
            "An officer of the army in a dark blue uniform.",
        ),
        ("A Confederate officer wears a blue frock coat.", "An officer wears a blue frock coat."),
    ],
)
def test_contradicted_side_is_dropped(prose: str, fixed: str) -> None:
    """A side the prose's uniform colour contradicts is dropped, model-free, even with tag-prior (#87)."""
    res = Reconciler(TagPriorVerifier()).reconcile(Image.new("RGB", (8, 8)), "soldier, uniform", prose)
    assert res.prose == fixed
    assert [(c.kind, c.now, c.source) for c in res.changes] == [("affiliation", "unstated", "prose")]


@pytest.mark.parametrize(
    "prose",
    [
        "A Union soldier in a blue uniform.",  # consistent
        "A Confederate corporal in a gray uniform.",  # consistent
        "A Union soldier holds a rifle.",  # no uniform colour
        "A Confederate soldier in blue trousers and a gray coat.",  # both colours: ambiguous
        "A map of the Soviet Union beside a gray coat.",  # not a Civil War side
    ],
)
def test_consistent_or_unclear_side_is_kept(prose: str) -> None:
    """Without a clean clash between one side and one colour family, the prose stays."""
    assert contradicted_side(prose) is None
    res = Reconciler(TagPriorVerifier()).reconcile(Image.new("RGB", (8, 8)), "soldier", prose)
    assert res.prose == prose
