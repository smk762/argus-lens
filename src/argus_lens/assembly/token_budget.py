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


# Diffusion backends whose text encoder is T5-XXL; everything else uses CLIP.
T5_BACKENDS: frozenset[str] = frozenset({"flux", "sd3", "kolors", "pixart"})


# CLIP (SDXL, SD1.5) reads ~75 tokens: sentence-form captions for it stay at two sentences.
MAX_CLIP_SENTENCES = 2


def is_t5_backend(backend: str | None) -> bool:
    """True when *backend* encodes prompts with T5-XXL rather than CLIP."""
    return (backend or "").lower().strip() in T5_BACKENDS


def estimate_tokens(text: str, backend: str | None = None) -> int:
    """Estimate token count using the appropriate method for *backend*.

    Flux, SD3, Kolors, and PixArt use T5-XXL; everything else uses CLIP.
    """
    if is_t5_backend(backend):
        return estimate_t5_tokens(text)
    return estimate_clip_tokens(text)


def caption_tokens(
    parts: list[str],
    backend: str | None = None,
    *,
    trigger_word: str = "",
    joiner: str = ", ",
) -> int:
    """Estimate the tokens of the caption *parts* assemble into, trigger word included.

    Packing and every later check (refine, coherence, ``token_budget_adherence``,
    Studio's counter) must agree, so this measures the joined text exactly as it is
    emitted (``with_trigger(trigger_word, joiner.join(parts))``), never a sum of
    per-fragment estimates: those floor each fragment's word cost separately and
    under-count tag-heavy captions (30 one-word tags: 59 summed vs 68 whole, #80).
    """
    from argus_lens.assembly.filtering import with_trigger  # filtering is heavier; keep this module leaf-light

    return estimate_tokens(with_trigger(trigger_word, joiner.join(parts)), backend)


def fits_budget(
    fragment: str,
    kept: list[str],
    token_budget: int,
    backend: str | None = None,
    *,
    trigger_word: str = "",
    joiner: str = ", ",
) -> bool:
    """True when appending *fragment* to *kept* keeps the whole caption within *token_budget*."""
    return caption_tokens([*kept, fragment], backend, trigger_word=trigger_word, joiner=joiner) <= token_budget


def try_add_fragment(
    fragment: str,
    kept: list[str],
    max_segments: int,
    token_budget: int,
    backend: str | None = None,
    *,
    trigger_word: str = "",
) -> bool:
    """Append *fragment* to *kept* (joined with ", ") if it fits the segment cap and budget.

    The budget is checked against the whole caption so far (see :func:`caption_tokens`).
    Returns whether the fragment was added.
    """
    if len(kept) >= max_segments or not fits_budget(fragment, kept, token_budget, backend, trigger_word=trigger_word):
        return False
    kept.append(fragment)
    return True
