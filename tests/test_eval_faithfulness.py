"""Tests for the claim-level faithfulness metric (argus_lens.eval.faithfulness)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from PIL import Image

from argus_lens.eval.dataset import EvalItem, load_dataset
from argus_lens.eval.faithfulness import OpenAICompatJudge, faithfulness, parse_score, split_claims
from argus_lens.eval.metrics import compute_metrics
from argus_lens.eval.report import compare_to_baseline, format_scorecard
from argus_lens.eval.runner import run_eval
from argus_lens.reconcile.verifiers import openai_compat
from argus_lens.types import CaptionResult
from tests.test_eval import _stub_engine, _write_image

_IMAGE = Image.new("RGB", (8, 8), (10, 20, 30))


class _KeywordJudge:
    """Scores 5 for any claim mentioning a listed word, 95 otherwise; records every call."""

    name = "keyword"
    model_id = "keyword-judge"

    def __init__(self, false_words: tuple[str, ...] = ("dragon",), unparseable: tuple[str, ...] = ()) -> None:
        self.false_words = false_words
        self.unparseable = unparseable
        self.calls: list[list[str]] = []

    def score_claims(self, image: Image.Image, claims: list[str]) -> list[float | None]:  # noqa: D102
        self.calls.append(list(claims))
        out: list[float | None] = []
        for claim in claims:
            low = claim.lower()
            if any(w in low for w in self.unparseable):
                out.append(None)
            else:
                out.append(5.0 if any(w in low for w in self.false_words) else 95.0)
        return out


class _BrokenJudge:
    """A judge whose endpoint is down."""

    name = "broken"
    model_id = "broken-judge"

    def score_claims(self, image: Image.Image, claims: list[str]) -> list[float | None]:  # noqa: D102
        raise httpx.ConnectError("connection refused")


# --------------------------------------------------------------------------- #
# Parsing and splitting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("85", 85.0),
        ("Score: 12.5", 12.5),
        ("<think>maybe 20, maybe 30</think>\n90", 90.0),
        ("<think>The image shows 2 people", None),  # cut off by max_tokens: scratch numbers are no score
        ("150", None),
        ("I cannot tell.", None),
        ("", None),
    ],
)
def test_parse_score(answer: str, expected: float | None) -> None:
    """The first 0–100 number outside a think block is the score."""
    assert parse_score(answer) == expected


def test_split_claims_drops_trigger_repeats_and_punctuation() -> None:
    """Underscores become spaces; the trigger, repeats and letterless fragments go."""
    text = "ohwx, red_dress, 1girl. A woman stands; Red dress, ...,  42 ,\nsmiling"
    assert split_claims(text, trigger_word="OHWX") == ["red dress", "1girl", "A woman stands", "smiling"]


# --------------------------------------------------------------------------- #
# Scoring one caption
# --------------------------------------------------------------------------- #


def test_faithfulness_scores_raw_and_final_and_judges_shared_claims_once() -> None:
    """Raw has a hallucinated dragon the final caption dropped; shared claims hit the judge once."""
    result = CaptionResult(
        final_caption="ohwx, a woman, red dress",
        raw_prose="A woman. A dragon overhead.",
        raw_tags="red_dress",
    )
    judge = _KeywordJudge()

    out = faithfulness(judge, _IMAGE, result, trigger_word="ohwx")

    assert len(judge.calls) == 1
    assert sorted(c.lower() for c in judge.calls[0]) == ["a dragon overhead", "a woman", "red dress"]
    assert out["raw"]["claims"] == 3
    assert out["raw"]["unsupported"] == 1
    assert out["raw"]["rate"] == pytest.approx(1 / 3)
    assert out["final"]["unsupported"] == 0
    assert out["final"]["rate"] == 0.0


def test_faithfulness_unparseable_scores_abstain_and_empty_text_is_none() -> None:
    """A claim the judge didn't score leaves the rate's denominator; no text means no side."""
    result = CaptionResult(final_caption="a cat, a hat")
    out = faithfulness(_KeywordJudge(false_words=("cat",), unparseable=("hat",)), _IMAGE, result)

    assert out["raw"] is None
    assert out["final"]["scored"] == 1
    assert out["final"]["rate"] == 1.0

    all_abstain = faithfulness(_KeywordJudge(unparseable=("cat", "hat")), _IMAGE, result)
    assert all_abstain["final"]["rate"] is None


def test_threshold_decides_unsupported() -> None:
    """A score below the threshold is unsupported; at or above it is not."""
    result = CaptionResult(final_caption="a dragon")
    assert faithfulness(_KeywordJudge(), _IMAGE, result, threshold=5.0)["final"]["unsupported"] == 0
    assert faithfulness(_KeywordJudge(), _IMAGE, result, threshold=5.1)["final"]["unsupported"] == 1


def test_judge_error_keeps_the_other_metrics() -> None:
    """An unreachable judge is recorded under faithfulness, not raised."""
    result = CaptionResult(final_caption="a cat", raw_tags="red_dress", raw_prose="a red dress")
    metrics = compute_metrics(EvalItem(image=Path("x.png")), result, image=_IMAGE, judge=_BrokenJudge())

    assert "ConnectError" in metrics["faithfulness"]["error"]
    assert metrics["contradiction"]["count"] == 0


def test_no_judge_means_no_faithfulness() -> None:
    """Without a judge the key is present and None."""
    metrics = compute_metrics(EvalItem(image=Path("x.png")), CaptionResult(final_caption="a cat"), image=_IMAGE)
    assert metrics["faithfulness"] is None


# --------------------------------------------------------------------------- #
# The OpenAI-compatible judge
# --------------------------------------------------------------------------- #


def test_openai_compat_judge_sends_one_request_per_claim_with_one_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each claim is one chat completion carrying the same image, the claim and the key."""
    requests: list[dict] = []

    def fake_post(url: str, json: dict, headers: dict, timeout: float) -> httpx.Response:
        requests.append({"url": url, "json": json, "headers": headers})
        answer = "10" if "dragon" in json["messages"][0]["content"][0]["text"] else "90"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": answer}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(openai_compat.httpx, "post", fake_post)
    judge = OpenAICompatJudge(model_id="qwen-vl", base_url="http://judge:8000/v1/", api_key="sk-test")

    assert judge.score_claims(_IMAGE, ["a dragon", "a woman"]) == [10.0, 90.0]
    assert [r["url"] for r in requests] == ["http://judge:8000/v1/chat/completions"] * 2
    assert requests[0]["json"]["model"] == "qwen-vl"
    assert requests[0]["headers"] == {"Authorization": "Bearer sk-test"}
    images = {r["json"]["messages"][0]["content"][1]["image_url"]["url"] for r in requests}
    assert len(images) == 1
    assert images.pop().startswith("data:image/png;base64,")


def test_chat_completion_tolerates_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server with no choices yields an empty answer, which parses to no score."""
    monkeypatch.setattr(
        openai_compat.httpx,
        "post",
        lambda url, **kw: httpx.Response(200, json={"choices": []}, request=httpx.Request("POST", url)),
    )
    assert openai_compat.chat_completion("http://j/v1", "m", "data:x", "q") == ""


# --------------------------------------------------------------------------- #
# Runner, report and gate
# --------------------------------------------------------------------------- #


def test_run_eval_aggregates_faithfulness(tmp_path: Path) -> None:
    """A judged run reports raw and final rates, the paired delta and the judge config."""
    _write_image(tmp_path / "one.png")
    _write_image(tmp_path / "two.png")
    engine = _stub_engine(tags="red_dress, 1girl", prose="A woman in a red dress. A dragon flies overhead.")

    scorecard = run_eval(engine, load_dataset(tmp_path), judge=_KeywordJudge(), judge_threshold=40.0)

    a = scorecard.aggregates
    first = scorecard.per_item[0].metrics["faithfulness"]
    assert a["unsupported_rate_raw_mean"] == pytest.approx(first["raw"]["rate"])
    assert a["unsupported_rate_raw_mean"] > 0
    assert a["unsupported_rate_final_mean"] == pytest.approx(first["final"]["rate"])
    assert a["assembly_unsupported_delta_mean"] == pytest.approx(first["final"]["rate"] - first["raw"]["rate"])
    assert a["faithfulness_errors"] == 0
    assert scorecard.config["judge_model"] == "keyword-judge"
    assert scorecard.config["judge_threshold"] == 40.0
    text = format_scorecard(scorecard)
    assert "unsupported claims: final" in text
    assert "judge: keyword-judge" in text


def test_run_eval_counts_judge_errors_and_omits_rows_without_a_judge(tmp_path: Path) -> None:
    """Judge errors are counted per image; an unjudged run has no faithfulness rows."""
    _write_image(tmp_path / "one.png")
    engine = _stub_engine(tags="red_dress", prose="a red dress")

    broken = run_eval(engine, load_dataset(tmp_path), judge=_BrokenJudge())
    assert broken.n_errors == 0
    assert broken.aggregates["faithfulness_errors"] == 1
    assert broken.aggregates["unsupported_rate_final_mean"] is None

    plain = run_eval(engine, load_dataset(tmp_path))
    assert plain.aggregates["faithfulness_errors"] is None
    assert plain.config["judge_model"] is None
    assert "unsupported claims" not in format_scorecard(plain)


def test_gate_flags_a_rise_in_final_unsupported_rate_only() -> None:
    """The final rate is gated; the raw rate and the delta are descriptive."""
    base = {"unsupported_rate_final_mean": 0.10, "unsupported_rate_raw_mean": 0.10}
    cur = {"unsupported_rate_final_mean": 0.20, "unsupported_rate_raw_mean": 0.50}
    comparison = compare_to_baseline(cur, base)
    assert comparison["regressions"] == ["unsupported_rate_final_mean"]


def test_split_claims_drops_a_trigger_with_underscores() -> None:
    """The trigger is normalised like the claims: ``sks_person`` is the fragment "sks person"."""
    assert split_claims("sks_person, red_dress. A woman stands", trigger_word="sks_person") == [
        "red dress",
        "A woman stands",
    ]
    assert split_claims("ohwx  style, a barn", trigger_word=" OHWX_style ") == ["a barn"]
