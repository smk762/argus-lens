"""Caption composition — orchestrates classification, filtering, and variant assembly."""

from __future__ import annotations

from argus_lens.assembly.classifier import classify_fragment, classify_fragment_detailed
from argus_lens.assembly.coherence import clean_prose, repair_caption, repair_fragment
from argus_lens.assembly.filtering import (
    dedupe_fragments,
    extract_prose_tokens,
    filter_redundant_clauses_detailed,
    normalise_fragment,
    split_caption_pieces,
    wd14_word_set,
)
from argus_lens.assembly.noise import (
    IDENTITY_REDUNDANT_TAGS,
    RATING_TAGS,
    filter_training_noise,
)
from argus_lens.assembly.prose_training import assemble_prose_training_variant
from argus_lens.assembly.subjects import count_subjects, is_subject_count_tag
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

    ``metadata`` carries the un-budgeted per-category fragments
    (``buckets``: ``{"tags": {...}, "prose": {...}}``, each with an extra
    ``"unclassified"`` key for fragments no hint matched) and the headcount
    estimate (``subjects``) — the inputs a search index needs, before the
    per-variant truncation.
    """
    raw_prose = prose
    prose, prose_notes = clean_prose(prose) if prose else ("", [])
    cat_names = get_category_names(categories)
    buckets: dict[str, list[str]] = {name: [] for name in cat_names}
    tag_buckets: dict[str, list[str]] = {name: [] for name in cat_names}
    removed_phrases: list[str] = []
    compaction_notes: list[str] = []
    if prose_notes:
        compaction_notes.append(f"Normalised prose ({', '.join(prose_notes)}).")
    # Index view: like ``buckets`` but split by source and with hint-less
    # fragments moved out of the identity fallback into "unclassified".
    index_buckets: dict[str, dict[str, list[str]]] = {
        source: {name: [] for name in (*cat_names, "unclassified")} for source in ("tags", "prose")
    }

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

    # Headcount tags (2girls, multiple_boys, ...) describe the frame, not the
    # subject: keep them out of every bucket and report them via ``subjects``.
    subject_tags = [f for f in noise_kept if is_subject_count_tag(f)]
    for fragment in noise_kept:
        if fragment in subject_tags:
            continue
        cat, matched = classify_fragment_detailed(fragment, categories)
        buckets[cat].append(fragment)
        tag_buckets[cat].append(fragment)
        index_buckets["tags"][cat if matched else "unclassified"].append(fragment)

    # --- Process prose-based input (e.g. Florence, BLIP, OpenAI) ---
    kept_clauses, redundant = filter_redundant_clauses_detailed(
        prose, tags, threshold=target_profile.redundancy_threshold
    )
    removed_phrases.extend([f.lower() for f in redundant])
    if redundant:
        compaction_notes.append("Removed prose clauses that duplicated tag content.")

    # Non-identity prose clauses feed training enrichment; identity clauses
    # would leak "brown", "hair" into a variant that suppresses identity.
    enrichment_clauses: list[str] = []
    for clause in kept_clauses:
        cleaned, removed = normalise_fragment(clause)
        removed_phrases.extend(removed)
        cleaned = repair_fragment(cleaned)
        if not cleaned:
            continue
        cat, matched = classify_fragment_detailed(cleaned, categories)
        if cat != "identity":
            enrichment_clauses.append(cleaned)
        buckets[cat].append(cleaned)
        index_buckets["prose"][cat if matched else "unclassified"].append(cleaned)

    for name in cat_names:
        buckets[name] = dedupe_fragments(buckets[name])
        tag_buckets[name] = dedupe_fragments(tag_buckets[name])

    # --- Build category variants ---
    caption_variants: dict[str, str] = {}
    truncated_any = False
    for cat_name in cat_names:
        caption, truncated = assemble_variant(
            trigger_word,
            buckets,
            cat_name,
            target_profile.target_style,
            categories,
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

    # Tags no hint matched (bouquet, computer, knife) sit in the identity
    # fallback, which training suppresses. They are objects, not identity:
    # offer them to training as the first, lowest-priority fill.
    enrichment_tokens: list[str] = [
        f for f in index_buckets["tags"]["unclassified"] if f.lower().replace(" ", "_") not in IDENTITY_REDUNDANT_TAGS
    ]
    if prose_enrichment and enrichment_clauses:
        # Filler-stripped clauses, so "the image shows ..." can't leak "shows".
        existing_tag_words = wd14_word_set(tags)
        enrichment_tokens += extract_prose_tokens(enrichment_clauses, existing_tag_words)

    training_caption, training_truncated = assemble_training_variant(
        trigger_word,
        training_buckets,
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

    # --- Build natural-language training variant (T5 targets: Flux, SD3) ---
    prose_training, prose_training_removed = assemble_prose_training_variant(
        trigger_word,
        prose,
        token_budget=target_profile.token_budget.budget,
        target_backend=target_profile.target_backend,
        categories=categories,
    )
    caption_variants["training_prose"] = prose_training or training_caption
    removed_phrases.extend(f.lower() for f in prose_training_removed)

    # --- Build zeroshot variant ---
    zeroshot_buckets = {cat_name: list(buckets[cat_name]) for cat_name in cat_names}
    for fragment in [*noise_removed, *subject_tags]:
        normalised = fragment.lower().strip().replace(" ", "_")
        if normalised not in RATING_TAGS:
            zeroshot_buckets[classify_fragment(fragment, categories)].append(fragment)
    for cat_name in cat_names:
        zeroshot_buckets[cat_name] = dedupe_fragments(zeroshot_buckets[cat_name])

    zeroshot_caption, zeroshot_truncated = assemble_zeroshot_variant(
        trigger_word,
        zeroshot_buckets,
        target_profile.target_style,
        clip_token_budget=target_profile.token_budget.budget,
        target_backend=target_profile.target_backend,
        categories=categories,
    )
    caption_variants["zeroshot"] = zeroshot_caption
    if zeroshot_truncated:
        removed_phrases.extend(zeroshot_truncated)

    # --- Post-check: repair disjointed fragments in every variant ---
    repaired_any = False
    for name, caption in caption_variants.items():
        kind = "prose" if name == "training_prose" and prose_training else "list"
        fixed, notes = repair_caption(caption, kind, trigger_word=trigger_word)
        if notes:
            caption_variants[name] = fixed
            repaired_any = True
    if repaired_any:
        compaction_notes.append("Repaired disjointed fragments in assembled captions.")

    selected_category = target_profile.target_category
    return CaptionResult(
        final_caption=caption_variants.get(selected_category, ""),
        caption_variants=caption_variants,
        selected_category=selected_category,
        removed_phrases=dedupe_fragments(removed_phrases),
        compaction_notes=compaction_notes,
        raw_tags=tags,
        raw_prose=raw_prose,
        backend_name=backend_name,
        metadata={
            "buckets": {
                source: {name: dedupe_fragments(frags) for name, frags in by_cat.items()}
                for source, by_cat in index_buckets.items()
            },
            "subjects": count_subjects(raw_tag_fragments, prose).to_dict(),
        },
    )
