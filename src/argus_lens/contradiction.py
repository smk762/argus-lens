"""Where prose contradicts the tags: colour, posture and headcount, with the text helpers that takes.

Used by the caption metrics (``eval``) and by the checks on captions (``reconcile``), so it imports neither.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from argus_lens.assembly.subjects import exact_prose_counts, exact_tag_count
from argus_lens.vocabulary import COLORS, POSE_GROUPS, POSE_TERMS, canon_color

if TYPE_CHECKING:
    from argus_lens.types import CaptionResult


def words_of(text: str) -> list[str]:
    """Lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+", text.lower())


def split_tags(tags: str) -> list[str]:
    """Split a comma-separated tag string into trimmed tags."""
    return [t.strip() for t in tags.split(",") if t.strip()]


# Connectives / shade-modifiers that are never the head noun of a colour tag.
# Without this, ``black_and_white`` would key on "and" and ``dark_blue`` on "dark",
# manufacturing contradictions against those ubiquitous prose words.
_NON_NOUN: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "the",
        "a",
        "an",
        "with",
        "on",
        "in",
        "of",
        "to",
        "at",
        "dark",
        "light",
        "pale",
        "bright",
        "deep",
        "very",
        "off",
    }
)


def _tag_color_map(tags: str) -> dict[str, set[str]]:
    """Map each colour-bearing noun in the tags to its asserted colour(s).

    ``red_dress`` / ``blue eyes`` / ``blonde_hair`` → ``{dress:{red}, eyes:{blue},
    hair:{blonde}}``. The head noun is the last token that is neither a colour
    nor a connective/shade-modifier; tags with no such noun (e.g. ``black_and_white``,
    ``dark_blue``) describe no object and are skipped.
    """
    mapping: dict[str, set[str]] = defaultdict(set)
    for tag in split_tags(tags):
        toks = tag.lower().replace("_", " ").split()
        colors = [canon_color(t) for t in toks if t in COLORS]
        nouns = [t for t in toks if t not in COLORS and t not in _NON_NOUN]
        if colors and nouns:
            mapping[nouns[-1]].update(colors)
    return dict(mapping)


_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")
_VOWELS = frozenset("aeiou")


# Plural-only nouns common in tags: stripping the "s" gives a different real word (shorts → short).
_PLURALIA_TANTUM: frozenset[str] = frozenset(
    {
        "bangs",
        "clothes",
        "glasses",
        "goggles",
        "jeans",
        "leggings",
        "news",
        "overalls",
        "pajamas",
        "pants",
        "scissors",
        "shorts",
        "sunglasses",
        "tights",
        "trousers",
    }
)


def noun_variants(noun: str) -> set[str]:
    """Singular/plural surface forms of *noun* for prose matching.

    Regular English plurals only, so a variant is never a different real word:
    ``-es`` follows a sibilant (dress → dresses, never cap → capes), consonant +
    ``y`` becomes ``-ies`` (sky → skies). The reverse direction recovers the
    singular of a plural tag noun (skies → sky, dresses → dress, eyes → eye).
    Plural-only nouns (``_PLURALIA_TANTUM``: shorts, glasses, bangs) have no
    singular of the same meaning, so they match only themselves: stripping
    them would make "short hair" or "a glass window" count as the tag. Nor is one
    generated from a singular tag: "glass" never matches "glasses", "short" never "shorts".
    """
    if noun in _PLURALIA_TANTUM:
        return {noun}
    variants = {noun}
    if noun.endswith(_SIBILANT_ENDINGS):
        variants.add(noun + "es")  # dress → dresses
        if noun.endswith("ch"):
            variants.add(noun + "s")  # a hard "ch" takes -s: stomach → stomachs
    elif len(noun) > 1 and noun.endswith("y") and noun[-2] not in _VOWELS:
        variants.add(noun[:-1] + "ies")  # sky → skies
    else:
        variants.add(noun + "s")  # boy → boys, eye → eyes
    if len(noun) > 3 and noun.endswith("ies"):
        variants.update({noun[:-3] + "y", noun[:-1]})  # skies → sky; ties → tie
    elif len(noun) > 2 and noun.endswith("es") and noun[:-2].endswith(_SIBILANT_ENDINGS):
        variants.update({noun[:-2], noun[:-1]})  # glasses → glass; horses → horse
    elif len(noun) > 1 and noun.endswith("s") and not noun.endswith("ss"):
        variants.add(noun[:-1])  # eyes → eye
    return variants - (_PLURALIA_TANTUM - {noun})


# Clause boundaries: commas/sentence punctuation and the connectives that start
# a new noun-phrase. Splitting here keeps a colour bound to its own clause, so
# "long hair, blue dress" never attributes "blue" to "hair".
_CLAUSE_SPLIT = re.compile(r"[.,;:!?]|\b(?:and|with|but|while|over|under|beside|near)\b")


def _prose_clauses(prose: str) -> list[list[str]]:
    """Split prose into clauses, each as a list of lowercase word tokens."""
    return [words_of(chunk) for chunk in _CLAUSE_SPLIT.split(prose.lower())]


def _prose_colors_for_noun(clauses: list[list[str]], noun: str) -> set[str]:
    """Colours that co-occur with *noun* inside the same clause.

    Because clauses are split on commas and connectives, a colour is only
    attributed to *noun* when they share a noun-phrase — no cross-noun leakage.
    """
    variants = noun_variants(noun)
    found: set[str] = set()
    for words in clauses:
        if set(words) & variants:
            found |= {canon_color(w) for w in words if w in COLORS}
    return found


def tag_prose_contradiction(result: CaptionResult) -> dict:
    """Count places where the prose contradicts the tags (colour, posture, headcount).

    Returns ``{count, checked, rate, details}``. ``checked`` is how many
    attributes could be compared (present in both signals); ``rate`` is
    ``count / checked`` (0.0 when nothing was comparable). A lower rate is
    better. Reference-free: compares the run's own ``raw_prose`` vs ``raw_tags``.
    """
    tags, prose = result.raw_tags, result.raw_prose
    prose_words = words_of(prose)
    clauses = _prose_clauses(prose)
    count = 0
    checked = 0
    details: list[dict] = []

    # Colour contradictions
    for noun, tag_colors in _tag_color_map(tags).items():
        prose_colors = _prose_colors_for_noun(clauses, noun)
        if not prose_colors:
            continue
        checked += 1
        if prose_colors.isdisjoint(tag_colors):
            count += 1
            details.append(
                {"kind": "color", "subject": noun, "tags_say": sorted(tag_colors), "prose_says": sorted(prose_colors)}
            )

    # Posture contradictions
    tag_words = set(words_of(tags.replace(",", " ")))
    prose_pose = set(prose_words) & POSE_TERMS
    for group in POSE_GROUPS:
        t = group & tag_words
        p = group & prose_pose
        if t and p:
            checked += 1
            if t.isdisjoint(p):
                count += 1
                details.append({"kind": "pose", "subject": "posture", "tags_say": sorted(t), "prose_says": sorted(p)})

    # Headcount contradictions (#87): an exact tag count (``1girl, 1boy``) vs the
    # one headcount the prose states ("three people"). Several prose headcounts
    # ("two women and three men") are not comparable to a total, so are skipped.
    tag_count = exact_tag_count(split_tags(tags))
    prose_counts = exact_prose_counts(prose)
    if tag_count is not None and len(prose_counts) == 1:
        checked += 1
        prose_count = prose_counts[0][0]
        if prose_count != tag_count:
            count += 1
            details.append(
                {"kind": "count", "subject": "people", "tags_say": [str(tag_count)], "prose_says": [str(prose_count)]}
            )

    rate = count / checked if checked else 0.0
    return {"count": count, "checked": checked, "rate": rate, "details": details}
