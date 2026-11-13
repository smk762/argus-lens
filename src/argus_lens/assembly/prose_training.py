"""Natural-language training variant — for T5-encoder targets (Flux, SD3).

The tag-style ``training`` variant suits CLIP-only models such as SDXL. T5
models were trained on sentences and learn more from them, so this variant
keeps the prose as prose and applies the same intent as the tag variant:

* **Identity suppression** — clauses about hair, eyes, face, etc. are
  dropped; the LoRA learns identity visually.
* **Filler removal** — "The image shows ..." style openers are stripped.
* **Token budget** — whole sentences are kept, in order, until the target
  encoder's budget is spent. CLIP targets also keep at most
  :data:`~argus_lens.assembly.token_budget.MAX_CLIP_SENTENCES` sentences.
"""

from __future__ import annotations

import re

from argus_lens.assembly.classifier import category_scores
from argus_lens.assembly.coherence import end_sentence, split_sentences
from argus_lens.assembly.filtering import strip_filler_prefixes, with_trigger
from argus_lens.assembly.token_budget import MAX_CLIP_SENTENCES, estimate_tokens, is_t5_backend
from argus_lens.types import CategoryConfig

# "with long wavy brown hair" / "has blue eyes and" — excised in place so the
# rest of the clause ("a woman ... is wearing a red dress") survives.
_IDENTITY_NOUNS = (
    r"(?:hair|hairstyle|eyes|freckles|beard|mustache|moustache|goatee|sideburns|stubble|skin|complexion|lips"
    r"|eyebrows|eyelashes|bangs|ponytail|braids?)"
)
# A participle phrase describing the hair, excised with it so "a woman with dark
# hair pulled back from her face" does not leave "a woman pulled back from her face".
_ID_TAIL = (
    r"(?:\s+(?:pulled|tied|swept|slicked|combed|parted|styled|gathered|piled|braided|cropped|cut|worn)"
    r"(?:\s+(?!(?:and|or|but|while|with|as|wearing|holding)\b|[\w-]+ing\b)[\w-]+){0,5})?"
)
# One identity noun phrase ("long wavy brown hair"), optionally coordinated
# with more ("freckles and bright green eyes", "red hair, green eyes").
_ID_NP = rf"(?:[\w-]+\s+){{0,4}}?{_IDENTITY_NOUNS}{_ID_TAIL}"
_ID_LIST = rf"{_ID_NP}(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+){_ID_NP})*"
_IDENTITY_PHRASE = re.compile(
    rf",?\s*\b(?:has|have|having)\s+{_ID_LIST}\s+and(?=\s)"
    rf"|,?\s*\b(?:with|has|have|having)\s+{_ID_LIST}\b",
    re.IGNORECASE,
)
# A comma between coordinated adjectives of an identity noun: "with light-colored,
# wavy hair", "He has a thick, well-groomed beard", "His voluminous, wild white hair".
# Splitting clauses there strands the first adjective ("a woman with light-colored."),
# so the comma is removed before excision and clause splitting (#88).
_ADJECTIVE_COMMA = re.compile(
    r"\b((?:with|has|have|had|a|an|the|his|her|their|its)\s+(?:[\w-]+\s+)?[\w-]+),\s+"
    r"(?=(?!(?:and|or|but|her|his|their|its|the|a|an|she|he|they|it)\b)"
    rf"(?:[\w-]+,?\s+){{0,3}}(?:{_IDENTITY_NOUNS}|features|face|expression)\b)",
    re.IGNORECASE,
)
_LEADING_CONJUNCTION = re.compile(r"^(?:and|but|while|whereas)\s+", re.IGNORECASE)


def _join_adjective_commas(text: str) -> str:
    """Drop commas inside coordinated adjectives of an identity noun (repeat for 3+ adjectives)."""
    previous = None
    while previous != text:
        previous = text
        text = _ADJECTIVE_COMMA.sub(r"\1 ", text)
    return text


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
    text, removed = strip_filler_prefixes(sentence.strip())
    filler_removed = len(removed)
    if text.rstrip("\"'\u201d\u2019)]") == text:
        text = text.rstrip(".!?;")  # a closing quote keeps its own full stop
    text = _join_adjective_commas(text)
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
    # "His ... hair cascades ..., and he wears a coat" minus the hair clause.
    cleaned = _LEADING_CONJUNCTION.sub("", cleaned)
    if len(cleaned.split()) < 3 and len(removed) > filler_removed:
        # An orphan left after identity removal ("Calm features").
        removed.append(cleaned)
        cleaned = ""
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
    max_sentences = None if is_t5_backend(target_backend) else MAX_CLIP_SENTENCES
    for raw in split_sentences(prose.strip(), semicolons=True):
        sentence, dropped = _clean_sentence(raw, categories)
        removed.extend(dropped)
        if not sentence:
            continue
        sentence = end_sentence(sentence)
        cost = estimate_tokens(sentence, target_backend)
        if used + cost > token_budget or (max_sentences is not None and len(sentences) >= max_sentences):
            removed.append(sentence)
            continue
        sentences.append(sentence)
        used += cost
    if not sentences:
        return "", removed
    return with_trigger(trigger_word, " ".join(sentences)), removed
