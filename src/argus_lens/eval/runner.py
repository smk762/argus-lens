"""Run the engine over an eval dataset and aggregate a scorecard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from argus_lens.eval.faithfulness import DEFAULT_THRESHOLD
from argus_lens.eval.metrics import compute_metrics

if TYPE_CHECKING:
    from argus_lens.engine import ArgusLens
    from argus_lens.eval.dataset import EvalItem
    from argus_lens.eval.faithfulness import FaithfulnessJudge
    from argus_lens.eval.metrics import ClipScorer


@dataclass
class ItemResult:
    """Per-image outcome: the captioned metrics, or an error string."""

    image: str
    metrics: dict = field(default_factory=dict)
    final_caption: str = ""
    error: str | None = None


@dataclass
class Scorecard:
    """Aggregate metrics across a dataset run, plus the per-item detail."""

    n: int
    n_labelled: int
    n_errors: int
    aggregates: dict[str, Any]
    per_item: list[ItemResult]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (the aggregates + config are the baseline surface)."""
        return {
            "n": self.n,
            "n_labelled": self.n_labelled,
            "n_errors": self.n_errors,
            "aggregates": self.aggregates,
            "config": self.config,
            "per_item": [asdict(r) for r in self.per_item],
        }


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty list."""
    return sum(values) / len(values) if values else None


def _aggregate(items: list[ItemResult]) -> dict[str, Any]:
    """Roll per-item metrics up into the scorecard aggregates."""
    scored = [r for r in items if r.error is None]
    n = len(scored)
    if not n:
        return {}

    contra_rates = [r.metrics["contradiction"]["rate"] for r in scored]
    contra_total = sum(r.metrics["contradiction"]["count"] for r in scored)
    items_with_contra = sum(1 for r in scored if r.metrics["contradiction"]["count"] > 0)

    over_budget_pct: dict[str, float] = {}
    for variant in ("training", "zeroshot"):
        over_budget_pct[variant] = sum(1 for r in scored if r.metrics["budget"]["over_budget"].get(variant)) / n

    redundancy = [r.metrics["redundancy"]["rate"] for r in scored]
    coverage = [r.metrics["coverage"]["recall"] for r in scored if r.metrics["coverage"] is not None]
    clip = [r.metrics["clip"] for r in scored if r.metrics["clip"] is not None]

    return {
        **_aggregate_faithfulness(scored),
        "contradiction_rate_mean": _mean(contra_rates),
        "contradiction_total": contra_total,
        "items_with_contradiction": items_with_contra,
        "items_with_contradiction_pct": items_with_contra / n,
        "over_budget_pct": over_budget_pct,
        "redundancy_rate_mean": _mean(redundancy),
        "coverage_recall_mean": _mean(coverage),
        "clip_mean": _mean(clip),
    }


def _aggregate_faithfulness(scored: list[ItemResult]) -> dict[str, Any]:
    """Roll up the judge's per-item results; every value is ``None`` without a judge.

    ``assembly_unsupported_delta_mean`` is the paired mean of final − raw over the
    images where both sides were scored: negative means assembly removed
    unsupported claims, positive means it added them.
    """
    judged = [r.metrics.get("faithfulness") for r in scored]
    ok = [f for f in judged if f and "error" not in f]
    errors = sum(1 for f in judged if f and "error" in f)

    def _rate(f: dict, side: str) -> float | None:
        """The side's unsupported rate, or ``None`` if it had no scored claims."""
        return f[side]["rate"] if f[side] else None

    final = [r for f in ok if (r := _rate(f, "final")) is not None]
    raw = [r for f in ok if (r := _rate(f, "raw")) is not None]
    paired = [_rate(f, "final") - _rate(f, "raw") for f in ok if None not in (_rate(f, "final"), _rate(f, "raw"))]
    with_unsupported = [f for f in ok if f["final"] and f["final"]["unsupported"] > 0]
    return {
        "unsupported_rate_final_mean": _mean(final),
        "unsupported_rate_raw_mean": _mean(raw),
        "assembly_unsupported_delta_mean": _mean(paired),
        "items_with_unsupported_pct": len(with_unsupported) / len(ok) if ok else None,
        "faithfulness_errors": errors if any(f is not None for f in judged) else None,
    }


def run_eval(
    engine: ArgusLens,
    dataset: list[EvalItem],
    *,
    clip_scorer: ClipScorer | None = None,
    trigger_word: str = "",
    hybrid_preset: str | None = None,
    prose_bias: float | None = None,
    judge: FaithfulnessJudge | None = None,
    judge_threshold: float = DEFAULT_THRESHOLD,
    progress: Callable[[int, int, EvalItem], None] | None = None,
) -> Scorecard:
    """Caption every item and score it; return an aggregated :class:`Scorecard`.

    Per-image captioning errors are captured (not raised) so one bad file never
    aborts the run — they surface as ``n_errors`` and per-item ``error`` strings.
    With a *judge*, each caption is also scored for faithfulness to its image
    (see :mod:`argus_lens.eval.faithfulness`); a judge error is recorded on the
    item and counted in ``faithfulness_errors``, not raised.
    """
    per_item: list[ItemResult] = []
    total = len(dataset)
    for idx, item in enumerate(dataset):
        try:
            pil = Image.open(item.image).convert("RGB")
            result = engine.caption(
                pil,
                trigger_word=trigger_word,
                target_style=item.target_style,
                target_category=item.target_category,
                target_backend=item.target_backend,
                hybrid_preset=hybrid_preset,
                prose_bias=prose_bias,
            )
            metrics = compute_metrics(
                item,
                result,
                image=pil,
                clip_scorer=clip_scorer,
                judge=judge,
                judge_threshold=judge_threshold,
                trigger_word=trigger_word,
            )
            per_item.append(ItemResult(image=str(item.image), metrics=metrics, final_caption=result.final_caption))
        except Exception as exc:  # noqa: BLE001 - report per-image, keep the run going
            per_item.append(ItemResult(image=str(item.image), error=str(exc)))
        if progress is not None:
            progress(idx + 1, total, item)

    config = {
        "backend": getattr(engine, "_backend", None) and engine._backend.name,
        "hybrid_preset": hybrid_preset,
        "prose_bias": prose_bias,
        "clip_enabled": clip_scorer is not None,
        # Unsupported rates are only comparable between runs with the same judge
        # and threshold, so both go in the baseline surface.
        "judge_model": judge.model_id if judge is not None else None,
        "judge_threshold": judge_threshold if judge is not None else None,
    }
    return Scorecard(
        n=total,
        n_labelled=sum(1 for i in dataset if i.has_labels),
        n_errors=sum(1 for r in per_item if r.error is not None),
        aggregates=_aggregate(per_item),
        per_item=per_item,
        config=config,
    )
