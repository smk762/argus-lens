"""Attribute reconciliation — fix prose colour/pose claims that contradict tags.

The pipeline (issue #36): detect the attributes where prose contradicts tags
(reusing the eval contradiction detector) → ask a pluggable
:class:`~argus_lens.reconcile.types.AttributeVerifier` to adjudicate each →
rewrite the prose to match. Verifiers range from model-free (``tag-prior``) to
VLM-backed (``florence`` grounding, ``molmo`` pointing, an ``openai-compat``
VQA endpoint). The eval harness then measures the drop in contradiction rate.
"""

from __future__ import annotations

from argus_lens.reconcile.detect import detect_disputes
from argus_lens.reconcile.reconciler import Reconciler, ReconcileResult
from argus_lens.reconcile.types import (
    AttributeDispute,
    AttributeVerifier,
    ReconcileChange,
    Verdict,
)
from argus_lens.reconcile.verifiers import VERIFIER_NAMES, build_verifier

__all__ = [
    "VERIFIER_NAMES",
    "AttributeDispute",
    "AttributeVerifier",
    "ReconcileChange",
    "ReconcileResult",
    "Reconciler",
    "Verdict",
    "build_verifier",
    "detect_disputes",
]
