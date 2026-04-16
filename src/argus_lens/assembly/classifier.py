"""Fragment classification — assigns text fragments to semantic categories."""

from __future__ import annotations

from argus_lens.types import CategoryConfig, get_category_config_map


def _content_words(text: str) -> frozenset[str]:
    """Lower-cased, punctuation-stripped content words (stopwords excluded)."""
    return frozenset(
        w.lower().strip(".,!?;:'\"")
        for w in text.split()
        if len(w) > 1 and w.lower().strip(".,!?;:'\"") not in _STOPWORDS
    )


_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "into",
    "this", "that", "it", "its", "she", "he", "they", "her", "his",
    "and", "or", "but", "also", "very", "quite", "rather", "some",
    "photo", "image", "picture", "portrait", "woman", "man", "person",
    "girl", "boy", "lady", "guy", "figure",
})


def classify_fragment(
    fragment: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> str:
    """Classify a text fragment into the best-matching category.

    Scores each category by counting how many of its hint words appear in
    *fragment*.  Falls back to ``"identity"`` when no hints match.
    """
    config_map = get_category_config_map(categories)
    lowered = fragment.lower()

    scores: dict[str, int] = {
        name: sum(1 for hint in cfg.hint_words if hint in lowered)
        for name, cfg in config_map.items()
    }

    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best

    words = _content_words(fragment)
    if {"indoors", "outdoors", "room", "street", "park"} & words:
        return "setting"
    if {"standing", "sitting", "portrait"} & words:
        return "pose_composition"

    return "identity"


def content_words(text: str) -> frozenset[str]:
    """Public access to content word extraction."""
    return _content_words(text)
