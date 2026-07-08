"""Tests for the attribute reconciliation package (issue #36)."""

from __future__ import annotations

import pytest
from PIL import Image

from argus_lens.engine import ArgusLens
from argus_lens.reconcile import Reconciler, build_verifier, detect_disputes
from argus_lens.reconcile.color_sample import dominant_color_name, nearest_color_name
from argus_lens.reconcile.questions import build_question, parse_answer
from argus_lens.reconcile.rewrite import apply_fix
from argus_lens.reconcile.types import AttributeDispute
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
    monkeypatch.setattr(v, "_ask", lambda image, q: "The dress is red.")
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
    """A raising verifier is caught; the prose is returned unchanged."""

    class _Boom:
        name = "boom"

        def verify(self, image, dispute):
            raise RuntimeError("nope")

    r = Reconciler(_Boom())
    outcome = r.reconcile(Image.new("RGB", (4, 4)), "red_dress", "a blue dress")
    assert outcome.prose == "a blue dress"
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
