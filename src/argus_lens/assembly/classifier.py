"""Fragment classification — assigns text fragments to semantic categories."""

from __future__ import annotations

import functools
import re

from argus_lens.types import CategoryConfig, get_category_config_map


def _content_words(text: str) -> frozenset[str]:
    """Lower-cased, punctuation-stripped content words (stopwords excluded)."""
    return frozenset(
        w.lower().strip(".,!?;:'\"")
        for w in text.split()
        if len(w) > 1 and w.lower().strip(".,!?;:'\"") not in _STOPWORDS
    )


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "into",
        "this",
        "that",
        "it",
        "its",
        "she",
        "he",
        "they",
        "her",
        "his",
        "and",
        "or",
        "but",
        "also",
        "very",
        "quite",
        "rather",
        "some",
        "photo",
        "image",
        "picture",
        "portrait",
        "woman",
        "man",
        "person",
        "girl",
        "boy",
        "lady",
        "guy",
        "figure",
    }
)


# A clause about the light leads with it: "the lighting is even ...", "softly lit ...".
_LIGHTING_LEAD = re.compile(
    r"^(?:the\s+|a\s+|soft\s+|even\s+|natural\s+|harsh\s+|warm\s+|cool\s+|dramatic\s+|bright\s+)*"
    r"(?:lighting|light(?!-)|lit|illuminat\w*|backlit|backlighting|softly\s+lit|evenly\s+lit|brightly\s+lit)\b",
    re.I,
)
# Where a VLM sentence turns from the subject to the light: "..., with the lighting
# focused on her", "..., and the lighting highlights ...", "...; the light falls ...".
_LIGHTING_TURN = re.compile(
    r"(?:,\s*|\s+)(?:(?:with|and|while|as)\s+)?(?=the\s+(?:lighting|light(?!-))\b)"
    r"|,\s*(?=(?:softly|evenly|brightly|dimly)\s+lit\b)",
    re.I,
)
# Single-word tags that name a thing, not an activity: RAM++ "writing" is text on
# an object, "painting" an artwork. As tags they are never an action (#89).
NOUN_SENSE_TAGS: frozenset[str] = frozenset({"writing", "painting", "drawing", "carving"})


def split_lighting_clauses(clause: str) -> list[str]:
    """Split *clause* where it turns to the lighting, so each part is classified alone.

    ``"He wears a suit, and the lighting highlights the folds"`` ->
    ``["He wears a suit", "the lighting highlights the folds"]``. A clause
    with no such turn comes back unchanged (as a one-item list).
    """
    parts = [p.strip(" ,") for p in _LIGHTING_TURN.split(clause)]
    return [p for p in parts if p] or [clause]


def classify_fragment(
    fragment: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> str:
    """Classify a text fragment into the best-matching category.

    Scores each category by counting how many of its hint words appear in
    *fragment*.  Falls back to ``"identity"`` when no hints match.
    """
    return classify_fragment_detailed(fragment, categories)[0]


def category_scores(
    fragment: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> dict[str, int]:
    """Return ``{category: hint_match_count}`` for *fragment*, in definition order.

    Hints match whole words (plural ``-s``/``-es`` allowed) and underscores
    count as spaces, so ``brown_hair`` hits ``hair`` but ``chair`` does not,
    and ``hat`` no longer fires inside ``that``.
    """
    config_map = get_category_config_map(categories)
    text = fragment.lower().replace("_", " ")
    return {name: sum(1 for p in _hint_patterns(cfg.hint_words) if p.search(text)) for name, cfg in config_map.items()}


@functools.lru_cache(maxsize=128)
def _hint_patterns(hints: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile whole-word matchers for a category's hint words (cached per hint set)."""
    return tuple(re.compile(rf"(?<![a-z]){re.escape(h.replace('_', ' '))}(?:e?s)?(?![a-z])") for h in hints)


def classify_fragment_detailed(
    fragment: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> tuple[str, bool]:
    """Classify *fragment* and report whether a hint actually matched.

    Returns ``(category, matched)``. ``matched`` is False when no hint word
    hit and the fragment landed in the ``"identity"`` fallback — callers
    building a search index use this to keep unknown tags (``cup``,
    ``phone``) out of the identity bucket.
    """
    scores = category_scores(fragment, categories)
    if "lighting" in scores and _LIGHTING_LEAD.match(fragment.strip()):
        return "lighting", True  # "the lighting highlights his hair and clothing" is about the light
    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best, True

    words = _content_words(fragment)
    if {"indoors", "outdoors", "room", "street", "park"} & words:
        return "setting", True
    if {"standing", "sitting", "portrait"} & words:
        return "pose_composition", True

    return "identity", False


def content_words(text: str) -> frozenset[str]:
    """Public access to content word extraction."""
    return _content_words(text)
