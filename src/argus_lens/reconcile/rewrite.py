"""Rewrite prose to match a verified attribute value.

Colour fixes are confined to the clause containing the subject noun (so
"blue dress" becomes "red dress" without touching a "blue sky" elsewhere);
pose fixes apply across the prose (posture is a whole-subject property). The
colour/pose vocabularies are imported from the eval detector so the words the
reconciler rewrites are exactly the words the detector flagged.
"""

from __future__ import annotations

import re

from argus_lens.eval.metrics import _COLORS, _POSE_TERMS, _canon_color, _noun_variants
from argus_lens.reconcile.types import AttributeDispute

# Split into clauses while KEEPING the delimiters, so the prose rejoins verbatim.
_CLAUSE_SPLIT = re.compile(r"([.,;:!?]|\band\b|\bwith\b|\bbut\b|\bwhile\b)", re.IGNORECASE)
_WORD = re.compile(r"\b[a-z]+\b", re.IGNORECASE)


def _clause_has_subject(clause: str, subject: str) -> bool:
    """True when *clause* mentions the subject noun (any singular/plural form)."""
    variants = _noun_variants(subject)
    return bool({w.lower() for w in _WORD.findall(clause)} & variants)


def _replace_words(text: str, is_target, replacement: str) -> str:
    """Replace every whole word for which *is_target(word_lower)* is true."""

    def _sub(match: re.Match[str]) -> str:
        return replacement if is_target(match.group(0).lower()) else match.group(0)

    return _WORD.sub(_sub, text)


def apply_fix(prose: str, dispute: AttributeDispute, value: str) -> str:
    """Return *prose* rewritten so *dispute.subject* reads as *value*."""
    if dispute.kind == "color":
        correct = _canon_color(value.lower())
        parts = _CLAUSE_SPLIT.split(prose)
        for i, part in enumerate(parts):
            if _clause_has_subject(part, dispute.subject):
                parts[i] = _replace_words(part, lambda w: w in _COLORS and _canon_color(w) != correct, value)
        return "".join(parts)

    if dispute.kind == "pose":
        target = value.lower()
        return _replace_words(prose, lambda w: w in _POSE_TERMS and w != target, value)

    return prose
