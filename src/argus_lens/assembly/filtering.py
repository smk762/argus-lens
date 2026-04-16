"""Redundancy filtering, filler stripping, and deduplication."""

from __future__ import annotations

import re

from argus_lens.assembly.classifier import _content_words

# ---------------------------------------------------------------------------
# Filler prefix removal
# ---------------------------------------------------------------------------

_FILLER_PREFIXES: tuple[str, ...] = (
    "the image shows ",
    "this image shows ",
    "the photo shows ",
    "the picture shows ",
    "there is ",
    "there are ",
    "in the image ",
    "in this image ",
    "a photo of ",
    "an image of ",
    "a picture of ",
)


def strip_filler_prefixes(text: str) -> tuple[str, list[str]]:
    """Remove common LLM/VLM filler prefixes.

    Returns ``(cleaned_text, list_of_removed_prefixes)``.
    """
    cleaned = re.sub(r"\s+", " ", text.strip())
    removed: list[str] = []
    changed = True
    while cleaned and changed:
        changed = False
        lowered = cleaned.lower()
        for prefix in _FILLER_PREFIXES:
            if lowered.startswith(prefix):
                removed.append(prefix.strip())
                cleaned = cleaned[len(prefix):].lstrip(" ,:-")
                changed = True
                break
    return cleaned.strip(" ,.;:-"), removed


def normalise_fragment(text: str) -> tuple[str, list[str]]:
    """Normalise a caption fragment: strip fillers, collapse whitespace."""
    cleaned, removed = strip_filler_prefixes(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned.lower(), removed


def split_caption_pieces(text: str) -> list[str]:
    """Split a caption string on sentence/clause boundaries."""
    if not text:
        return []
    return [piece.strip() for piece in re.split(r"[.;,]", text) if piece.strip()]


# ---------------------------------------------------------------------------
# WD14 word expansion for fuzzy matching
# ---------------------------------------------------------------------------


def wd14_word_set(tags_str: str) -> frozenset[str]:
    """Expand WD14 tag words with simple stemmed variants."""
    words: set[str] = set()
    for tag in tags_str.split(","):
        for w in _content_words(tag):
            words.add(w)
            if w.endswith("ing") and len(w) > 5:
                words.add(w[:-3])
            if w.endswith("ed") and len(w) > 4:
                words.add(w[:-2])
            if w.endswith("s") and len(w) > 4:
                words.add(w[:-1])
    return frozenset(words)


# ---------------------------------------------------------------------------
# Redundancy filtering
# ---------------------------------------------------------------------------


def filter_redundant_clauses_detailed(
    description: str,
    tags: str,
    threshold: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Remove prose clauses whose content words overlap with *tags*.

    Returns ``(kept_clauses, removed_clauses)``.
    """
    if not description or not tags:
        clauses = [c.strip() for c in re.split(r"[,.]", description) if c.strip()]
        return clauses, []

    tag_words = wd14_word_set(tags)
    raw_clauses = re.split(r"[,.]", description)

    kept: list[str] = []
    removed: list[str] = []
    for clause in raw_clauses:
        clause = clause.strip()
        if not clause:
            continue
        cw = _content_words(clause)
        if not cw:
            continue
        overlap = len(cw & tag_words) / len(cw)
        if overlap < threshold:
            kept.append(clause)
        else:
            removed.append(clause)
    return kept, removed


def filter_redundant_clauses(description: str, tags: str, threshold: float = 0.5) -> str:
    """Remove redundant clauses and return the filtered description."""
    kept, _ = filter_redundant_clauses_detailed(description, tags, threshold=threshold)
    return ", ".join(kept)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def dedupe_fragments(fragments: list[str]) -> list[str]:
    """Remove duplicate fragments (case-insensitive), preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for fragment in fragments:
        key = fragment.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(fragment)
    return ordered


# ---------------------------------------------------------------------------
# Trigger word helpers
# ---------------------------------------------------------------------------


def with_trigger(trigger_word: str, rest: str) -> str:
    """Prepend trigger word, avoiding empty leading commas."""
    tw = trigger_word.strip()
    body = rest.strip()
    if not tw:
        return body
    return f"{tw}, {body}" if body else tw
