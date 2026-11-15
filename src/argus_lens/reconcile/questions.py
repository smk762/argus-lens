"""Build pointed VQA questions and parse their short answers.

Shared by the VQA-style verifiers (openai-compat, Molmo). Asking a small,
specific question ("what colour is the dress?") and forcing the model to look
is far more accurate than trusting a free-form paragraph.
"""

from __future__ import annotations

import re

from argus_lens.assembly.subjects import _NUMBER_WORDS
from argus_lens.eval.metrics import _COLORS, _POSE_TERMS, _canon_color
from argus_lens.reconcile.types import AttributeDispute

_POSE_CHOICES = "standing, sitting, lying down, kneeling, or crouching"


def build_question(dispute: AttributeDispute) -> str:
    """A single-answer question that adjudicates *dispute*."""
    if dispute.kind == "color":
        return f"What is the main colour of the {dispute.subject} in this image? Answer with a single colour word."
    if dispute.kind == "count":
        return (
            "How many people are visible in this image, including partially visible people? "
            "Answer with a single integer and nothing else."
        )
    return f"What is the person's posture: {_POSE_CHOICES}? Answer with a single word."


def parse_answer(answer: str, kind: str) -> str | None:
    """Extract a palette colour, posture term or headcount from a free-text VQA *answer*.

    Returns the canonical value, or ``None`` if the answer names none (so the
    verifier abstains rather than inventing a value).
    """
    words = [w.strip(".,!?:;'\"()").lower() for w in answer.split()]
    if kind == "count":
        for w in words:
            if re.fullmatch(r"\d{1,3}", w):
                return w
            if w in _NUMBER_WORDS and w not in {"both", "pair", "couple"}:
                return str(_NUMBER_WORDS[w])
        return None
    vocab = _COLORS if kind == "color" else _POSE_TERMS
    for w in words:
        if w in vocab:
            return _canon_color(w) if kind == "color" else w
    return None
