"""Drop a Civil War side the prose contradicts with its own uniform colour (#87).

A small VLM wrote "a Union Army soldier in a gray uniform" for a Confederate
corporal: the colour is a direct reading of the pixels, the side is an
inference, and the two disagree. When the prose names one side and describes
the other side's uniform colour (Union blue, Confederate grey or butternut),
the side is left out: "a Union Army soldier in a gray uniform" becomes "a
soldier in a gray uniform". No model is needed, so this runs with any verifier.

Prose that names both sides, or both colours, is left alone (a blue-trousered
Confederate is plausible); so is a side that isn't attached to a military noun
("the Soviet Union", "a union meeting").
"""

from __future__ import annotations

import re

_NOUNS = (
    r"(?:soldiers?|officers?|uniforms?|infantry(?:m[ae]n)?|cavalry(?:m[ae]n)?|artillery(?:m[ae]n)?|troops?|"
    r"privates?|corporals?|sergeants?|lieutenants?|captains?|colonels?|generals?|veterans?|forces|coats?|"
    r"jackets?|kepis?|caps?|regiments?|men|man)"
)
_SIDE = r"(Union|Confederate)"
_ARMY = r"(?:\s+(?:States\s+)?Army)"
# "a Union Army soldier" -> "a soldier": the side (and "Army") goes, the noun stays; the article follows the noun.
_SIDE_BEFORE_NOUN_RE = re.compile(rf"\b(?:([Aa]n?)\s+)?{_SIDE}{_ARMY}?\s+({_NOUNS})\b")
# "a soldier of the Union Army" -> "a soldier of the army".
_SIDE_ARMY_RE = re.compile(rf"\b{_SIDE}{_ARMY}\b")
_UNIFORM_COLOR_RE = re.compile(
    r"\b(blue|navy|gr[ae]y|butternut)\b(?:[\s-]+[a-z]+){0,2}?\s+(?:uniforms?|coats?|jackets?|tunics?|frock|kepis?)\b",
    re.IGNORECASE,
)
_SIDE_COLORS = {"union": {"blue", "navy"}, "confederate": {"gray", "grey", "butternut"}}


def _sides(prose: str) -> set[str]:
    """The sides *prose* attaches to a military noun or to "Army"."""
    return {m.group(2).lower() for m in _SIDE_BEFORE_NOUN_RE.finditer(prose)} | {
        m.group(1).lower() for m in _SIDE_ARMY_RE.finditer(prose)
    }


def contradicted_side(prose: str) -> tuple[str, str] | None:
    """Return ``(side, colour)`` when the prose's one side clashes with its one uniform colour family."""
    sides = _sides(prose)
    colors = {m.group(1).lower() for m in _UNIFORM_COLOR_RE.finditer(prose)}
    if len(sides) != 1 or not colors:
        return None
    families = {side for side, family in _SIDE_COLORS.items() if colors & family}
    if len(families) != 1:
        return None  # no recognised colour, or both sides' colours: ambiguous
    (side,) = sides
    if side in families:
        return None  # consistent
    return side.capitalize(), sorted(colors)[0]


def drop_side(prose: str) -> str:
    """Remove every Union/Confederate side marker from *prose*, keeping the grammar."""

    def before_noun(match: re.Match[str]) -> str:
        article, noun = match.group(1), match.group(3)
        if not article:
            starts_sentence = not prose[: match.start()].strip() or prose[: match.start()].rstrip()[-1] in ".!?"
            return noun.capitalize() if starts_sentence else noun
        fixed = "an" if noun[:1].lower() in "aeiou" and not noun.lower().startswith("uni") else "a"  # "a uniform"
        return f"{fixed.capitalize() if article[:1].isupper() else fixed} {noun}"

    prose = _SIDE_BEFORE_NOUN_RE.sub(before_noun, prose)
    return _SIDE_ARMY_RE.sub("army", prose)
