"""Human-readable scorecard formatting and baseline regression comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus_lens.eval.runner import Scorecard

# Gate metrics and their direction. "lower" = smaller is better (contradictions,
# overflow); "higher" = larger is better (coverage recall, CLIP alignment).
# Redundancy is descriptive only and deliberately excluded from the gate.
_METRIC_DIRECTION: dict[str, str] = {
    "contradiction_rate_mean": "lower",
    "items_with_contradiction_pct": "lower",
    "over_budget_pct.training": "lower",
    "over_budget_pct.zeroshot": "lower",
    "coverage_recall_mean": "higher",
    "clip_mean": "higher",
}

DEFAULT_TOLERANCE = 0.01


def _get(aggregates: dict[str, Any], dotted: str) -> float | None:
    """Fetch a possibly nested aggregate by dotted key (e.g. ``over_budget_pct.training``)."""
    node: Any = aggregates
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


def _fmt(value: float | None) -> str:
    """Format a metric value, rendering ``None`` as ``n/a``."""
    return "n/a" if value is None else f"{value:.3f}"


def format_scorecard(scorecard: Scorecard) -> str:
    """Render a scorecard as an aligned plain-text block (no rich dependency)."""
    a = scorecard.aggregates
    lines = [
        "── Argus Lens eval scorecard ──",
        f"images: {scorecard.n}   labelled: {scorecard.n_labelled}   errors: {scorecard.n_errors}",
        f"backend: {scorecard.config.get('backend')}   preset: {scorecard.config.get('hybrid_preset')}"
        f"   prose_bias: {scorecard.config.get('prose_bias')}",
        "",
    ]
    if not a:
        lines.append("(no images scored)")
        return "\n".join(lines)

    rows: list[tuple[str, str]] = [
        ("tag↔prose contradiction rate", _fmt(a.get("contradiction_rate_mean")) + "   (lower better)"),
        (
            "images with a contradiction",
            f"{a.get('items_with_contradiction')}/{scorecard.n - scorecard.n_errors}"
            f"  ({_fmt(a.get('items_with_contradiction_pct'))})",
        ),
        ("total contradictions", str(a.get("contradiction_total"))),
        ("over-budget: training", _fmt(a.get("over_budget_pct", {}).get("training"))),
        ("over-budget: zeroshot", _fmt(a.get("over_budget_pct", {}).get("zeroshot"))),
        ("redundancy/filler rate", _fmt(a.get("redundancy_rate_mean"))),
        ("tag-coverage recall", _fmt(a.get("coverage_recall_mean")) + "   (higher better, labelled only)"),
        ("CLIPScore", _fmt(a.get("clip_mean")) + "   (higher better)"),
    ]
    width = max(len(label) for label, _ in rows)
    lines += [f"  {label.ljust(width)}  {value}" for label, value in rows]
    return "\n".join(lines)


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare current aggregates to a baseline; flag per-metric regressions.

    *current* and *baseline* are the ``aggregates`` dicts. A regression is a
    gate metric that moved in the wrong direction by more than *tolerance*.
    Returns ``{deltas, regressions, improvements, regressed}`` where
    ``regressed`` is the CI-friendly overall verdict.
    """
    deltas: dict[str, dict[str, float | None]] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    for metric, direction in _METRIC_DIRECTION.items():
        cur = _get(current, metric)
        base = _get(baseline, metric)
        if cur is None or base is None:
            continue
        delta = cur - base
        deltas[metric] = {"baseline": base, "current": cur, "delta": delta}
        worse = delta > tolerance if direction == "lower" else delta < -tolerance
        better = delta < -tolerance if direction == "lower" else delta > tolerance
        if worse:
            regressions.append(metric)
        elif better:
            improvements.append(metric)

    return {
        "deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "regressed": bool(regressions),
    }


def format_comparison(comparison: dict[str, Any]) -> str:
    """Render a baseline comparison as a plain-text diff block."""
    lines = ["── vs baseline ──"]
    for metric, d in comparison["deltas"].items():
        arrow = "→"
        mark = " "
        if metric in comparison["regressions"]:
            mark = "✗"
        elif metric in comparison["improvements"]:
            mark = "✓"
        lines.append(f"  {mark} {metric}: {d['baseline']:.3f} {arrow} {d['current']:.3f} ({d['delta']:+.3f})")
    verdict = "REGRESSED" if comparison["regressed"] else "OK"
    lines.append(f"  verdict: {verdict}")
    return "\n".join(lines)
