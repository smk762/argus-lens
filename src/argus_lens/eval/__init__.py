"""Evaluation harness for caption quality.

Measures caption quality so model swaps, hybrid presets, the reconciliation
summariser, and the VQA cross-check can be judged by numbers instead of vibes.

The harness is **reference-free first**: its flagship metric — whether the
prose contradicts the tags (the hallucination problem) — is an
*internal-consistency* check that needs no hand-labelled ground truth, so it
runs on any folder of images. Supplying a labelled manifest unlocks the
reference-based metrics (tag-coverage recall, CLIPScore against a reference).
"""

from __future__ import annotations

from argus_lens.eval.dataset import EvalItem, load_dataset
from argus_lens.eval.report import compare_to_baseline, format_scorecard
from argus_lens.eval.runner import ItemResult, Scorecard, run_eval

__all__ = [
    "EvalItem",
    "ItemResult",
    "Scorecard",
    "compare_to_baseline",
    "format_scorecard",
    "load_dataset",
    "run_eval",
]
