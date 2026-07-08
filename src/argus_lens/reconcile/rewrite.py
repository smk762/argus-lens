"""Rewrite prose to match a verified attribute value.

Colour fixes recolour only the colour word **nearest the subject noun** within
its clause, so a second coloured noun sharing the clause ("blue eyes in a red
dress") keeps its own colour. Pose fixes replace only the posture terms the
detector actually flagged. The colour/pose vocabularies are imported from the
eval detector so the words the reconciler rewrites are exactly the words it
flagged.

Note: pose reconciliation is keyword-based, so a posture word used as a noun
("the sitting room") or unrelated verb ("lying about her height") can still be
rewritten — prefer a grounded verifier for pose, or keep pose disputes rare.
"""

from __future__ import annotations

import re

from argus_lens.eval.metrics import _COLORS, _canon_color, _noun_variants
from argus_lens.reconcile.types import AttributeDispute

# Split into clauses while KEEPING the delimiters, so the prose rejoins verbatim.
_CLAUSE_SPLIT = re.compile(r"([.,;:!?]|\band\b|\bwith\b|\bbut\b|\bwhile\b)", re.IGNORECASE)
_WORD = re.compile(r"\b[a-z]+\b", re.IGNORECASE)


def _cased(replacement: str, original: str) -> str:
    """Match *replacement*'s case to the *original* token it replaces."""
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _clause_has_subject(words: list[str], subject: str) -> bool:
    """True when the tokenised *words* mention the subject noun (any form)."""
    return bool(set(words) & _noun_variants(subject))


def _recolor_clause(clause: str, subject: str, correct: str, value: str) -> str:
    """Recolour only the wrong-colour word nearest an occurrence of *subject*."""
    tokens = list(_WORD.finditer(clause))
    lowered = [m.group(0).lower() for m in tokens]
    if not _clause_has_subject(lowered, subject):
        return clause
    subject_idx = [i for i, w in enumerate(lowered) if w in _noun_variants(subject)]
    wrong = [i for i, w in enumerate(lowered) if w in _COLORS and _canon_color(w) != correct]
    if not wrong:
        return clause
    # Nearest wrong colour to any subject occurrence (ties → the earlier token).
    best = min(wrong, key=lambda i: min(abs(i - s) for s in subject_idx))
    m = tokens[best]
    return clause[: m.start()] + _cased(value, m.group(0)) + clause[m.end() :]


def apply_fix(prose: str, dispute: AttributeDispute, value: str) -> str:
    """Return *prose* rewritten so *dispute.subject* reads as *value*."""
    if dispute.kind == "color":
        correct = _canon_color(value.lower())
        parts = _CLAUSE_SPLIT.split(prose)
        return "".join(_recolor_clause(part, dispute.subject, correct, value) for part in parts)

    if dispute.kind == "pose":
        # Only the posture terms the detector flagged, not every pose word in the prose.
        flagged = {p.lower() for p in dispute.prose_says}
        target = value.lower()

        def _sub(match: re.Match[str]) -> str:
            word = match.group(0).lower()
            return _cased(value, match.group(0)) if word in flagged and word != target else match.group(0)

        return _WORD.sub(_sub, prose)

    return prose
