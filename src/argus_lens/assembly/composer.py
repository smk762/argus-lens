"""Caption composition — orchestrates classification, filtering, and variant assembly."""

from __future__ import annotations

from argus_lens.assembly.classifier import classify_fragment
from argus_lens.assembly.filtering import (
    dedupe_fragments,
    extract_prose_tokens,
    filter_redundant_clauses_detailed,
    normalise_fragment,
    split_caption_pieces,
    wd14_word_set,
)
from argus_lens.assembly.noise import (
    RATING_TAGS,
    filter_training_noise,
)
from argus_lens.assembly.training import assemble_training_variant
from argus_lens.assembly.variants import assemble_variant
from argus_lens.assembly.zeroshot import assemble_zeroshot_variant
from argus_lens.types import (
    CaptionResult,
    CaptionTargetProfile,
    CategoryConfig,
    get_category_names,
)


def compose_caption_result(
    *,
    trigger_word: str,
    tags: str = "",
    prose: str = "",
    target_profile: CaptionTargetProfile,
    image_index: int = 0,
    categories: tuple[CategoryConfig, ...] | None = None,
    backend_name: str = "",
    prose_enrichment: bool = True,
) -> CaptionResult:
    """Build a structured ``CaptionResult`` from raw model outputs.

    This is the core assembly function.  It takes raw tag and prose strings
    from any backend (local or cloud) and produces category-bucketed
    variants plus training and zeroshot specialisations.
    """
    cat_names = get_category_names(categories)
    buckets: dict[str, list[str]] = {name: [] for name in cat_names}
    tag_buckets: dict[str, list[str]] = {name: [] for name in cat_names}
    removed_phrases: list[str] = []
    compaction_notes: list[str] = []

    # --- Process tag-based input (e.g. WD14) ---
    raw_tag_fragments: list[str] = []
    for tag in split_caption_pieces(tags):
        cleaned, removed = normalise_fragment(tag)
        removed_phrases.extend(removed)
        if not cleaned:
            continue
        raw_tag_fragments.append(cleaned)

    noise_kept, noise_removed = filter_training_noise(raw_tag_fragments, strip_identity=False)
    removed_phrases.extend(noise_removed)
    if noise_removed:
        compaction_notes.append("Stripped rating/meta tags that have no training value.")

    for fragment in noise_kept:
        cat = classify_fragment(fragment, categories)
        buckets[cat].append(fragment)
        tag_buckets[cat].append(fragment)

    # --- Process prose-based input (e.g. Florence, BLIP, OpenAI) ---
    kept_clauses, redundant = filter_redundant_clauses_detailed(prose, tags)
    removed_phrases.extend([f.lower() for f in redundant])
    if redundant:
        compaction_notes.append("Removed prose clauses that duplicated tag content.")

    for clause in kept_clauses:
        cleaned, removed = normalise_fragment(clause)
        removed_phrases.extend(removed)
        if not cleaned:
            continue
        buckets[classify_fragment(cleaned, categories)].append(cleaned)

    for name in cat_names:
        buckets[name] = dedupe_fragments(buckets[name])
        tag_buckets[name] = dedupe_fragments(tag_buckets[name])

    # --- Build category variants ---
    caption_variants: dict[str, str] = {}
    truncated_any = False
    for cat_name in cat_names:
        caption, truncated = assemble_variant(
            trigger_word, buckets, cat_name,
            target_profile.target_style, categories,
        )
        caption_variants[cat_name] = caption
        if truncated:
            truncated_any = True
            removed_phrases.extend(truncated)

    if truncated_any:
        compaction_notes.append("Trimmed lower-priority fragments to keep captions compact.")

    # --- Build training variant (tags + prose enrichment) ---
    training_buckets: dict[str, list[str]] = {}
    training_identity_stripped: list[str] = []
    for cat_name in cat_names:
        kept, stripped = filter_training_noise(tag_buckets[cat_name], strip_identity=True)
        training_buckets[cat_name] = kept
        training_identity_stripped.extend(stripped)
    if training_identity_stripped:
        compaction_notes.append("Suppressed identity traits for training variant (learned visually).")

    enrichment_tokens: list[str] = []
    if prose_enrichment and kept_clauses:
        existing_tag_words = wd14_word_set(tags)
        enrichment_tokens = extract_prose_tokens(kept_clauses, existing_tag_words)

    training_caption, training_truncated = assemble_training_variant(
        trigger_word, training_buckets,
        target_profile.target_style,
        clip_token_budget=target_profile.token_budget.budget,
        target_backend=target_profile.target_backend,
        image_index=image_index,
        categories=categories,
        enrichment=enrichment_tokens,
    )
    caption_variants["training"] = training_caption
    if training_truncated:
        removed_phrases.extend(training_truncated)

    # --- Build zeroshot variant ---
    zeroshot_buckets = {cat_name: list(buckets[cat_name]) for cat_name in cat_names}
    for fragment in noise_removed:
        normalised = fragment.lower().strip().replace(" ", "_")
        if normalised not in RATING_TAGS:
            zeroshot_buckets[classify_fragment(fragment, categories)].append(fragment)
    for cat_name in cat_names:
        zeroshot_buckets[cat_name] = dedupe_fragments(zeroshot_buckets[cat_name])

    zeroshot_caption, zeroshot_truncated = assemble_zeroshot_variant(
        trigger_word, zeroshot_buckets,
        target_profile.target_style,
        clip_token_budget=target_profile.token_budget.budget,
        target_backend=target_profile.target_backend,
        categories=categories,
    )
    caption_variants["zeroshot"] = zeroshot_caption
    if zeroshot_truncated:
        removed_phrases.extend(zeroshot_truncated)

    selected_category = target_profile.target_category
    return CaptionResult(
        final_caption=caption_variants.get(selected_category, ""),
        caption_variants=caption_variants,
        selected_category=selected_category,
        removed_phrases=dedupe_fragments(removed_phrases),
        compaction_notes=compaction_notes,
        raw_tags=tags,
        raw_prose=prose,
        backend_name=backend_name,
    )
