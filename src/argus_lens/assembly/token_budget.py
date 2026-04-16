"""Token estimation and budget management for CLIP / T5 text encoders."""

from __future__ import annotations


def estimate_clip_tokens(text: str) -> int:
    """Rough CLIP-L/G token count.

    Commas and underscores each become a token; words average ~1.3 tokens
    in CLIP-L/OpenCLIP for English text.
    """
    if not text:
        return 0
    punctuation_tokens = text.count(",") + text.count("_")
    words = len(text.split())
    return int(words * 1.3) + punctuation_tokens


def estimate_t5_tokens(text: str) -> int:
    """Rough T5-XXL token count.

    T5 uses SentencePiece; English words average ~1.1 tokens.
    Punctuation is generally merged into adjacent tokens.
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.1) + text.count(",")


def estimate_tokens(text: str, backend: str | None = None) -> int:
    """Estimate token count using the appropriate method for *backend*.

    Flux, SD3, Kolors, and PixArt use T5-XXL; everything else uses CLIP.
    """
    t5_backends = {"flux", "sd3", "kolors", "pixart"}
    if (backend or "").lower().strip() in t5_backends:
        return estimate_t5_tokens(text)
    return estimate_clip_tokens(text)


def try_add_fragment(
    fragment: str,
    kept: list[str],
    used_tokens: int,
    max_segments: int,
    token_budget: int,
    backend: str | None = None,
) -> tuple[bool, int]:
    """Try to append *fragment* to *kept* without exceeding budget.

    Returns ``(added, new_used_tokens)``.
    """
    frag_tokens = estimate_tokens(fragment, backend)
    separator_cost = 1 if kept else 0
    if len(kept) >= max_segments or used_tokens + frag_tokens + separator_cost > token_budget:
        return False, used_tokens
    kept.append(fragment)
    return True, used_tokens + frag_tokens + separator_cost
