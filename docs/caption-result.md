# Caption result reference

Every captioning call returns a `CaptionResult` (`argus_lens.types.CaptionResult`, a plain dataclass). The HTTP endpoints send it as JSON via `dataclasses.asdict()`, so the field names and the `caption_variants` keys below are the wire format that argus-studio, argus-lightbox and other clients read. Renaming any of them is a breaking change.

```json
{
  "final_caption": "sks_person, woman, long hair, smile",
  "caption_variants": {
    "identity": "...", "wardrobe": "...", "pose_composition": "...",
    "setting": "...", "lighting": "...", "action": "...",
    "training": "...", "training_prose": "...", "zeroshot": "..."
  },
  "selected_category": "identity",
  "removed_phrases": ["masterpiece", "the image shows"],
  "compaction_notes": ["Stripped rating/meta tags that have no training value."],
  "raw_tags": "1girl, long_hair, smile, masterpiece",
  "raw_prose": "The image shows a woman with long hair smiling.",
  "backend_name": "hybrid",
  "metadata": {"empty_categories": [], "buckets": {}, "subjects": {}, "monochrome": false,
               "reconciled_tags": "...", "tag_reconcile": {}, "refine": {}}
}
```

## Fields

| Field | Type | Meaning |
|---|---|---|
| `final_caption` | `str` | The caption to use. It is `caption_variants[selected_category]`, after refinement when that ran. This is what lens writes to the `<image_stem>.txt` sidecar. |
| `caption_variants` | `dict[str, str]` | One caption per category, plus the specialised variants. See [Variants](#variants). |
| `selected_category` | `str` | The target category that picked `final_caption` (the profile's `target_category`). An unknown name falls back to `identity`. |
| `removed_phrases` | `list[str]` | Phrases that assembly dropped, deduplicated. See the [caveat](#removed_phrases-covers-every-variant) below. |
| `compaction_notes` | `list[str]` | Short, human-readable notes on what assembly and refinement changed and why, e.g. "Removed prose clauses that duplicated tag content." or "Refined training with qwen3:8b." Meant for display, not parsing. |
| `raw_tags` | `str` | The tagger's output as received (WD14, RAM++, ...), comma-separated. It is **not** reconciled; see `metadata.reconciled_tags` for the tags assembly used. Empty for prose-only backends. |
| `raw_prose` | `str` | The prose model's output (Florence-2, BLIP-2, an OpenAI-compatible VLM, ...). When the opt-in `reconcile` pass is on, this is the prose after colour/pose fixes. Empty for tag-only backends. |
| `backend_name` | `str` | The backend that produced the raw output, e.g. `hybrid`, `ram`, `openai-compat`. |
| `metadata` | `dict` | Pipeline evidence. See [Metadata](#metadata). |

`raw_wd14` and `raw_florence` are read-only Python aliases for `raw_tags` and `raw_prose`. They are properties, so they don't appear in the JSON.

## Variants

`caption_variants` has one key per category, then three specialised variants. With the default categories (`DEFAULT_CATEGORY_CONFIGS`, also listed by `GET /profiles` as `target_categories`) the keys are:

| Key | Content |
|---|---|
| `identity` | Who or what the subject is: person nouns, hair, eyes, physical traits. It also takes tags that no category hint matched. |
| `wardrobe` | Clothing and accessories. |
| `pose_composition` | Pose, framing and camera angle. |
| `setting` | Location, background and scene. |
| `lighting` | Light quality and direction. |
| `action` | What the subject is doing. |
| `training` | A tag-form caption for LoRA training on CLIP models (SDXL). Identity traits are suppressed so the LoRA learns them visually, and it is packed to the target backend's token budget. |
| `training_prose` | A sentence-form training caption for T5 models (Flux, SD3), with identity phrases excised. It falls back to `training` when there's no prose. |
| `zeroshot` | An identity-first caption for prompting a model without a LoRA. |

Each category variant starts with the trigger word (when one is given), then all of its own category's fragments, then a few fragments from each other category, within a per-variant length cap. So the `setting` caption opens with the setting, not with hair colour. A category with nothing of its own is made only of other categories' fragments; `metadata.empty_categories` lists those.

Custom `CategoryConfig`s replace the six category keys with their own names. `training`, `training_prose` and `zeroshot` are always present.

## `removed_phrases` covers every variant

`removed_phrases` is one list for the whole result. It collects what was dropped anywhere in assembly: rating/meta tags, filler ("the image shows"), prose clauses that duplicated a tag, tags settled by reconciliation, and fragments trimmed to fit each variant's token budget. So it **over-reports for any single variant**: a phrase trimmed from `zeroshot` may still be in `identity`. To explain why a phrase is missing from one variant, compare that variant with `raw_tags`/`raw_prose` (or `metadata.buckets`) instead. Lightbox traces per variant for this reason.

## Metadata

Assembly always writes the first six keys. `refine` appears when refinement ran, and `replay` appears only on replayed results.

| Key | Shape | Meaning |
|---|---|---|
| `empty_categories` | `list[str]` | Categories with no fragments of their own, whose variants borrow from other buckets. |
| `buckets` | `{"tags": {<category>: [str], "unclassified": [str]}, "prose": {...}}` | Every fragment, split by source and category **before** any token-budget trimming. Fragments that no category hint matched go to `unclassified` here, rather than to the `identity` fallback. Use it for search indexing and faceted filtering. |
| `subjects` | `{count, multiple, tag_count, prose_count, agree, evidence}` | Headcount evidence. `count` is the best estimate of how many people are visible (`null` when neither tags nor prose mention people). `multiple` is true when either source says two or more. `tag_count` and `prose_count` are the count each source implies (`null` when absent). `agree` compares them (`null` unless both are known). `evidence` lists the tags and phrases used. Headcount tags (`2girls`, `multiple_boys`, `solo`) are kept out of every bucket and reported here. |
| `monochrome` | `bool` | Whether the image is black-and-white or toned, from a check of the pixels' colour spread. |
| `reconciled_tags` | `str` | The tags assembly actually used, after reconciliation, comma-separated. |
| `tag_reconcile` | `{"dropped": [str], "added": [str], "notes": [str]}` | How the tags were settled: `girl` dropped beside `woman`, `beard` changed to `mustache`, bare `black`/`white` folded into `monochrome` on a monochrome image, and so on. |
| `refine` | `{rewriter, model, accepted, rejected, attempts, error?}` or `{rewriter, model, skipped}` | The LLM refinement outcome. `accepted` lists the variants that were replaced by a validated rewrite. `rejected` maps each variant that kept its deterministic caption to the problems found (e.g. `"over token budget (64 > 60)"`). `attempts` counts rewrite rounds. `error` is set when the rewriter failed (e.g. a bad response). `skipped: "unavailable"` means the endpoint was unreachable shortly before, so no attempt was made. |
| `replay` | `{source, asset_sha256, asset_uri, caption_version, recorded_backend, recorded_profile}` | Only on results from the `replay` backend: which cortex lineage recording was served. `source` is `"cortex-lineage"`. Replay returns the recorded `metadata` as is, plus this key. |

`metadata` is open-ended: new keys may be added in minor releases, so clients should ignore keys they don't know.
