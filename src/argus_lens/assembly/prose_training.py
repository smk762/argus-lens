"""Natural-language training variant — for T5-encoder targets (Flux, SD3).

The tag-style ``training`` variant suits CLIP-only models such as SDXL. T5
models were trained on sentences and learn more from them, so this variant
keeps the prose as prose and applies the same intent as the tag variant:

* **Identity suppression** — clauses about hair, eyes, face, etc. are
  dropped; the LoRA learns identity visually.
* **Filler removal** — "The image shows ..." style openers are stripped.
* **Token budget** — whole sentences are kept, in order, until the target
  encoder's budget is spent.
"""

from __future__ import annotations

import re

from argus_lens.assembly.classifier import category_scores
from argus_lens.assembly.filtering import strip_filler_prefixes, with_trigger
from argus_lens.assembly.token_budget import estimate_tokens
from argus_lens.types import CategoryConfig

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
# "with long wavy brown hair" / "has blue eyes and" — excised in place so the
# rest of the clause ("a woman ... is wearing a red dress") survives.
_IDENTITY_NOUNS = r"(?:hair|eyes|freckles|beard|skin|complexion|lips|eyebrows|eyelashes|bangs|ponytail|braids?)"
# One identity noun phrase ("long wavy brown hair"), optionally coordinated
# with more ("freckles and bright green eyes", "red hair, green eyes").
_ID_NP = rf"(?:[\w-]+\s+){{0,4}}?{_IDENTITY_NOUNS}"
_ID_LIST = rf"{_ID_NP}(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+){_ID_NP})*"
_IDENTITY_PHRASE = re.compile(
    rf",?\s*\b(?:has|have|having)\s+{_ID_LIST}\s+and(?=\s)"
    rf"|,?\s*\b(?:with|has|have|having)\s+{_ID_LIST}\b",
    re.IGNORECASE,
)


def _identity_dominant(clause: str, categories: tuple[CategoryConfig, ...] | None) -> bool:
    """True when identity hints outscore every other category in *clause*."""
    scores = category_scores(clause, categories)
    identity = scores.pop("identity", 0)
    return identity > 0 and identity > max(scores.values(), default=0)


def _clean_sentence(
    sentence: str,
    categories: tuple[CategoryConfig, ...] | None,
) -> tuple[str, list[str]]:
    """Strip filler and identity clauses from one sentence.

    Returns ``(cleaned_sentence, removed_clauses)``.
    """
    text, removed = strip_filler_prefixes(sentence.strip().rstrip(".!?;"))
    removed.extend(m.group(0).strip(" ,") for m in _IDENTITY_PHRASE.finditer(text))
    text = _IDENTITY_PHRASE.sub("", text)
    kept: list[str] = []
    dropped_after: set[int] = set()  # indexes of kept clauses followed by a dropped one
    for clause in (c.strip() for c in text.split(",")):
        if not clause:
            continue
        # A bare subject left after excision ("She has green eyes" -> "She").
        if len(clause.split()) < 2 or _identity_dominant(clause, categories):
            removed.append(clause)
            if kept:
                dropped_after.add(len(kept) - 1)
            continue
        kept.append(clause)
    # "A woman, in her twenties, walking ..." minus the parenthetical reads
    # "A woman walking ...": re-join a short lead clause with what followed.
    cleaned = ""
    for i, clause in enumerate(kept):
        if not cleaned:
            cleaned = clause
        elif i - 1 in dropped_after and len(kept[i - 1].split()) <= 3:
            cleaned = f"{cleaned} {clause}"
        else:
            cleaned = f"{cleaned}, {clause}"
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned, removed


def assemble_prose_training_variant(
    trigger_word: str,
    prose: str,
    *,
    token_budget: int,
    target_backend: str | None = None,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> tuple[str, list[str]]:
    """Build a sentence-form training caption from *prose*.

    Returns ``(caption, removed)``. The caption is empty when *prose* is
    empty or nothing survives identity suppression; callers should fall back
    to the tag-style training variant in that case.
    """
    removed: list[str] = []
    sentences: list[str] = []
    used = estimate_tokens(trigger_word, target_backend)
    for raw in _SENTENCE_SPLIT.split(prose.strip()):
        if not raw.strip():
            continue
        sentence, dropped = _clean_sentence(raw, categories)
        removed.extend(dropped)
        if not sentence:
            continue
        sentence += "."
        cost = estimate_tokens(sentence, target_backend)
        if used + cost > token_budget:
            removed.append(sentence)
            continue
        sentences.append(sentence)
        used += cost
    if not sentences:
        return "", removed
    return with_trigger(trigger_word, " ".join(sentences)), removed
