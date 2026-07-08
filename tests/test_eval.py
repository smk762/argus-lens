"""Tests for the evaluation harness (argus_lens.eval)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.engine import ArgusLens
from argus_lens.eval.dataset import EvalItem, load_dataset
from argus_lens.eval.metrics import (
    redundancy_rate,
    tag_coverage,
    tag_prose_contradiction,
    token_budget_adherence,
)
from argus_lens.eval.report import compare_to_baseline, format_comparison, format_scorecard
from argus_lens.eval.runner import run_eval
from argus_lens.types import CaptionResult

# --------------------------------------------------------------------------- #
# Stub backends (no models)
# --------------------------------------------------------------------------- #


class _StubTag(CaptionBackend):
    """Returns fixed WD14-style tags."""

    name = "stub-tag"
    style = "anime"
    requires_gpu = False

    def __init__(self, tags: str) -> None:
        self._tags = tags

    def load(self, device: str = "auto") -> None:  # noqa: D102
        pass

    def caption_image(self, image: Image.Image) -> str:  # noqa: D102
        return self._tags

    def unload(self) -> None:  # noqa: D102
        pass


class _StubProse(CaptionBackend):
    """Returns fixed prose."""

    name = "stub-prose"
    requires_gpu = False

    def __init__(self, prose: str) -> None:
        self._prose = prose

    def load(self, device: str = "auto") -> None:  # noqa: D102
        pass

    def caption_image(self, image: Image.Image) -> str:  # noqa: D102
        return self._prose

    def unload(self) -> None:  # noqa: D102
        pass


def _stub_engine(tags: str, prose: str) -> ArgusLens:
    """Engine backed by a hybrid of the two stubs, so both tags and prose flow."""
    pipeline = HybridPipeline(tag_backend=_StubTag(tags), prose_backend=_StubProse(prose))
    return ArgusLens(backend=pipeline)


def _write_image(path: Path, color: tuple[int, int, int] = (120, 40, 40)) -> Path:
    """Write a tiny solid-colour PNG and return its path."""
    Image.new("RGB", (8, 8), color).save(path)
    return path


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_contradiction_flags_color_and_pose() -> None:
    """Tags say red/standing, prose says blue/sitting → two contradictions."""
    r = CaptionResult(
        final_caption="x",
        raw_tags="red_dress, standing, 1girl",
        raw_prose="A woman in a blue dress sitting on a chair.",
    )
    c = tag_prose_contradiction(r)
    assert c["count"] == 2
    assert c["checked"] == 2
    assert c["rate"] == 1.0
    assert {d["kind"] for d in c["details"]} == {"color", "pose"}


def test_contradiction_none_when_consistent() -> None:
    """Matching colour/pose is not a contradiction."""
    r = CaptionResult(final_caption="x", raw_tags="red_dress, standing",
                      raw_prose="A woman in a red dress, standing.")
    assert tag_prose_contradiction(r)["count"] == 0


def test_contradiction_folds_color_synonyms() -> None:
    """grey/gray and blond/blonde must not read as contradictions."""
    assert tag_prose_contradiction(
        CaptionResult(final_caption="x", raw_tags="grey_shirt", raw_prose="a gray shirt")
    )["count"] == 0
    assert tag_prose_contradiction(
        CaptionResult(final_caption="x", raw_tags="blonde_hair", raw_prose="her blond hair")
    )["count"] == 0


def test_budget_adherence_flags_overflow() -> None:
    """A long training variant overflows the SDXL budget; a short zeroshot does not."""
    r = CaptionResult(final_caption="x", caption_variants={"training": "word, " * 60, "zeroshot": "a cat"})
    b = token_budget_adherence(r, "sdxl")
    assert b["over_budget"]["training"] is True
    assert b["over_budget"]["zeroshot"] is False
    assert b["any_over"] is True


def test_redundancy_rate() -> None:
    """rate = removed / (removed + kept fragments)."""
    r = CaptionResult(final_caption="a, b, c", removed_phrases=["x", "y"])
    assert redundancy_rate(r)["rate"] == 2 / 5


def test_tag_coverage_recall_and_none() -> None:
    """Recall over labelled tags; None when there are no labels."""
    r = CaptionResult(final_caption="woman, red dress", raw_tags="red_dress")
    cov = tag_coverage(r, ("red_dress", "blue_sky"))
    assert cov["recall"] == 0.5
    assert cov["missed"] == ["blue_sky"]
    assert tag_coverage(r, ()) is None


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def test_load_dataset_from_directory(tmp_path: Path) -> None:
    """A directory yields unlabelled items for each image."""
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    items = load_dataset(tmp_path)
    assert len(items) == 2
    assert all(not it.has_labels for it in items)


def test_load_dataset_from_manifest(tmp_path: Path) -> None:
    """A JSONL manifest yields labelled items with resolved relative image paths."""
    _write_image(tmp_path / "img.png")
    manifest = tmp_path / "golden.jsonl"
    manifest.write_text(
        json.dumps({"image": "img.png", "expected_tags": ["red_dress"], "target_caption": "a red dress"})
        + "\n"
    )
    items = load_dataset(manifest)
    assert len(items) == 1
    assert items[0].has_labels
    assert items[0].image == tmp_path / "img.png"
    assert items[0].expected_tags == ("red_dress",)


def test_load_dataset_rejects_bad_manifest(tmp_path: Path) -> None:
    """A row missing 'image' is a clear error naming the line."""
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps({"expected_tags": ["x"]}) + "\n")
    try:
        load_dataset(manifest)
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError for a manifest row missing 'image'")


# --------------------------------------------------------------------------- #
# Runner end-to-end (stub backend, no models)
# --------------------------------------------------------------------------- #


def test_run_eval_end_to_end(tmp_path: Path) -> None:
    """A full run over two images produces aggregates and detects the contradiction."""
    _write_image(tmp_path / "one.png")
    _write_image(tmp_path / "two.png")
    dataset = load_dataset(tmp_path)
    engine = _stub_engine(tags="red_dress, standing", prose="A woman in a blue dress, sitting.")

    scorecard = run_eval(engine, dataset)

    assert scorecard.n == 2
    assert scorecard.n_errors == 0
    # Every image hit the same stub → a contradiction on each.
    assert scorecard.aggregates["items_with_contradiction"] == 2
    assert scorecard.aggregates["contradiction_rate_mean"] == 1.0
    assert scorecard.aggregates["coverage_recall_mean"] is None  # unlabelled dataset
    assert scorecard.aggregates["clip_mean"] is None
    assert isinstance(format_scorecard(scorecard), str)


def test_run_eval_captures_per_image_errors(tmp_path: Path) -> None:
    """A missing image is reported as an error, not raised, and the run continues."""
    good = _write_image(tmp_path / "good.png")
    dataset = [EvalItem(image=good), EvalItem(image=tmp_path / "missing.png")]
    engine = _stub_engine(tags="red_dress", prose="a red dress")

    scorecard = run_eval(engine, dataset)

    assert scorecard.n == 2
    assert scorecard.n_errors == 1
    assert any(r.error for r in scorecard.per_item)


# --------------------------------------------------------------------------- #
# Baseline regression gate
# --------------------------------------------------------------------------- #


def test_compare_to_baseline_detects_regression_and_improvement() -> None:
    """More contradictions regresses; higher coverage improves."""
    baseline = {"contradiction_rate_mean": 0.10, "coverage_recall_mean": 0.80,
                "over_budget_pct": {"training": 0.0, "zeroshot": 0.0}}
    current = {"contradiction_rate_mean": 0.25, "coverage_recall_mean": 0.90,
               "over_budget_pct": {"training": 0.0, "zeroshot": 0.0}}
    cmp = compare_to_baseline(current, baseline)
    assert cmp["regressed"] is True
    assert "contradiction_rate_mean" in cmp["regressions"]
    assert "coverage_recall_mean" in cmp["improvements"]
    assert "REGRESSED" in format_comparison(cmp)


def test_compare_to_baseline_ok_within_tolerance() -> None:
    """A sub-tolerance wobble is neither regression nor improvement."""
    baseline = {"contradiction_rate_mean": 0.10}
    current = {"contradiction_rate_mean": 0.105}
    cmp = compare_to_baseline(current, baseline)
    assert cmp["regressed"] is False
    assert cmp["regressions"] == []
