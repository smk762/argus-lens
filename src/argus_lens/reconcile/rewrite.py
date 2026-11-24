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

from argus_lens.assembly.subjects import hedge_prose_count, rewrite_prose_count
from argus_lens.contradiction import noun_variants
from argus_lens.reconcile.types import AttributeDispute
from argus_lens.vocabulary import COLORS, canon_color

# Split into clauses while KEEPING the delimiters, so the prose rejoins verbatim.
_CLAUSE_SPLIT = re.compile(r"([.,;:!?]|\band\b|\bwith\b|\bbut\b|\bwhile\b)", re.IGNORECASE)
_WORD = re.compile(r"\b[a-z]+\b", re.IGNORECASE)


def _cased(replacement: str, original: str) -> str:
    """Match *replacement*'s case to the *original* token it replaces."""
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        # Only the first letter: str.capitalize() would lowercase the rest ("LoRA" -> "Lora").
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _clause_has_subject(words: list[str], subject: str) -> bool:
    """True when the tokenised *words* mention the subject noun (any form)."""
    return bool(set(words) & noun_variants(subject))


def _recolor_clause(clause: str, subject: str, correct: str, value: str) -> str:
    """Recolour only the wrong-colour word nearest an occurrence of *subject*."""
    tokens = list(_WORD.finditer(clause))
    lowered = [m.group(0).lower() for m in tokens]
    if not _clause_has_subject(lowered, subject):
        return clause
    subject_idx = [i for i, w in enumerate(lowered) if w in noun_variants(subject)]
    wrong = [i for i, w in enumerate(lowered) if w in COLORS and canon_color(w) != correct]
    if not wrong:
        return clause
    # Nearest wrong colour to any subject occurrence (ties → the earlier token).
    best = min(wrong, key=lambda i: min(abs(i - s) for s in subject_idx))
    m = tokens[best]
    return clause[: m.start()] + _cased(value, m.group(0)) + clause[m.end() :]


def apply_fix(prose: str, dispute: AttributeDispute, value: str) -> str:
    """Return *prose* rewritten so *dispute.subject* reads as *value*."""
    if dispute.kind == "color":
        correct = canon_color(value.lower())
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

    if dispute.kind == "count":
        # "six soldiers" -> "five soldiers"; a non-integer or < 2 verdict leaves the prose alone.
        # isascii(): str.isdigit() also accepts "²", which int() rejects.
        return rewrite_prose_count(prose, int(value)) if value.isascii() and value.isdigit() else prose

    return prose


def hedge_claim(prose: str, claim: AttributeDispute) -> str:
    """Return *prose* with an unconfirmed *claim* left out (#87).

    Used when a verifier contradicts a claim no tag backs: two model readings
    disagree, so neither value is trusted and the exact figure goes
    ("six soldiers" -> "several soldiers").
    """
    if claim.kind == "count":
        return hedge_prose_count(prose)
    if claim.kind == "posture":
        for term in claim.prose_says:
            prose = drop_posture_word(prose, term)
        return prose
    return prose


_AND_BEFORE = re.compile(r"\s+and\s+$", re.IGNORECASE)
_AND_AFTER = re.compile(r"\s+and\b", re.IGNORECASE)
_COMMA_BEFORE = re.compile(r",\s*$")
_AUX_BEFORE = re.compile(r"\b(?:is|are|was|were)\s+$", re.IGNORECASE)
_CLAUSE_END = re.compile(r"\s*(?:[.,;:!?]|$)")


def drop_posture_word(prose: str, term: str) -> str:
    """Return *prose* without the posture word *term* (and a ``down``/``up`` after it), keeping the sentence.

    "a woman sitting on a bench" -> "a woman on a bench"; "smiling and standing" -> "smiling"; "a woman, sitting,
    reads" -> "a woman reads". Where the word is the clause's whole predicate ("she is lying down.") nothing short
    of rewriting the sentence reads well, so that occurrence is kept. A sentence that started with the word gets its
    next word capitalised.
    """
    word = re.compile(rf"\b{re.escape(term)}\b(?:\s+(?:down|up)\b)?", re.IGNORECASE)
    out = prose
    for m in reversed(list(word.finditer(prose))):  # right to left, so earlier offsets stay valid
        start, end = m.start(), m.end()
        before, after = out[:start], out[end:]
        if _CLAUSE_END.match(after) and _AUX_BEFORE.search(before):
            continue  # "she is lying." -> "she is." / "she." both read worse than the word
        if (b := _AND_BEFORE.search(before)) is not None:
            start = b.start()
        elif (a := _AND_AFTER.match(after)) is not None:
            end += a.end()
        elif (b := _COMMA_BEFORE.search(before)) is not None and after.lstrip().startswith(","):
            start = b.start()
            end += after.index(",") + 1
        head, tail = out[:start], out[end:]
        if m.group(0)[:1].isupper() and (not head.strip() or head.rstrip().endswith((".", "!", "?"))):
            tail = re.sub(r"^\s*([a-z])", lambda c: c.group(1).upper(), tail, count=1)
        out = head + tail
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    return out.strip()
