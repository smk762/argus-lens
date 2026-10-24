# Curate-index workflow

Caption a people-photo library once, then **group it by shared traits** (brown hair, red dress, kitchen, cooking …), **flag multi-person images** for face identification, and **export training datasets** for SDXL and Flux, all from a single inference pass.

```
source_root/
  <name>/pics/<shoot_id>/*.jpg      ~100 names, thousands of images
```

Everything is driven by [`curate_index.py`](curate_index.py). It is a plain script on top of the `argus-lens` package, not part of its API.

## Contents

- [Pipeline at a glance](#pipeline-at-a-glance)
- [Setup](#setup)
- [Choosing models](#choosing-models)
- [Stage by stage](#stage-by-stage)
- [Caption sets: identity, wardrobe, setting](#caption-sets-identity-wardrobe-setting)
- [SDXL and Flux coverage](#sdxl-and-flux-coverage)
- [LLM refinement and the post-check](#llm-refinement-and-the-post-check)
- [Querying and grouping](#querying-and-grouping)
- [Reproducibility](#reproducibility)
- [Recommended first run (pilot)](#recommended-first-run-pilot)
- [Outputs reference](#outputs-reference)

## Pipeline at a glance

| Stage | Runs models? | Reads | Writes |
|---|---|---|---|
| `inventory` | no | `source_root` | `manifest.jsonl`, `inventory.json` |
| `caption` | **yes**, once per unique image | manifest | `raw.jsonl`, `run.json`, `errors.jsonl` |
| `assemble` | only if `--refine-model` | `raw.jsonl` | `captions.<target>.jsonl`, `review_queue.csv` |
| `index` | no | captions | `index.csv` (+ `.parquet`), `groups.csv`, `identity_consensus.csv`, `identity_outliers.csv` |
| `query` | no | `index.csv` | stdout (image ids) |
| `export` | no | captions | `<dest>/<name>/*.{jpg,txt}` |

`raw.jsonl` stores the model's raw tags and prose. Every later stage recomputes from it, so you can re-tune assembly, refinement, taxonomy and export without touching the GPU.

## Setup

```bash
# from the repo root
uv venv && uv pip install -e ".[local]"       # WD14 + Florence-2 (local GPU)
# optional: fast columnar index you can query with DuckDB
uv pip install pyarrow duckdb
```

The default backend downloads its models on first use: WD14 to `~/.cache/wd14_tagger`, Florence-2 to the Hugging Face cache. A single 24 GB GPU (e.g. an RTX 3090) runs any setup below.

## Choosing models

A run combines a **tagger** (structured booru tags, the source of the index vocabulary) with a **prose model** (natural-language description, the source of Flux captions and of context the tags miss). Two optional helpers sit on top: a **refine** model for rewriting and a **count** model for headcount.

| Role | Default | Recommended upgrade | Notes |
|---|---|---|---|
| Tagger | `SmilingWolf/wd-vit-tagger-v3` | `--wd14-repo SmilingWolf/wd-eva02-large-tagger-v3` | EVA02-Large is the most accurate of the WD v3 family and slower. All WD taggers are trained on anime-style art, so on photos they are good at clothing, framing and objects but miss some scene detail. The prose model covers that. |
| Prose | Florence-2-base | **Qwen3-VL 8B** via Ollama (`qwen3-vl:8b`, about 12 GB at Q4), or **JoyCaption Beta One** via vLLM (`fancyfeast/llama-joycaption-beta-one-hf-llava`, about 17 GB in bf16) | JoyCaption is built specifically for diffusion-training captions, and the `--prompt straightforward` preset is modelled on its "Straightforward" mode. Qwen3-VL is a general-purpose VLM, useful where Florence-2's shorter descriptions miss actions or a second person. Florence-2 is small and fast, but it is the model whose colour hallucinations the `--reconcile` step exists to fix. |
| Refine | none | any instruct model behind an OpenAI-compatible API, e.g. `qwen3:8b` | Text-only is enough. See [LLM refinement](#llm-refinement-and-the-post-check). |
| Count | none (tags + prose) | `--count-model qwen3-vl:8b` | Asks "how many people?" as a third opinion for the face-ID queue. |

I found no independent head-to-head benchmark of these captioners for LoRA datasets. **Compare them on your own pilot** with `argus-lens eval` (the reference-free contradiction rate is a good first signal) before committing a full run.

Serving a prose model (pick one):

```bash
ollama pull qwen3-vl:8b                     # Ollama: http://localhost:11434/v1
vllm serve fancyfeast/llama-joycaption-beta-one-hf-llava --max-model-len 4096   # vLLM: http://localhost:8000/v1
```

## Stage by stage

Each stage takes a **run directory**, which holds all state for one configuration. Use a new run directory to try a different caption configuration.

### 1. inventory

```bash
python workflows/curate_index.py inventory /data/source_root runs/r1
```

This walks `<name>/pics/<shoot_id>/<file>` (change it with `--layout`) and hashes every image. The first 16 hex characters of the SHA-256 are the stable `image_id`, so moving or renaming files does not orphan results. Byte-identical copies are marked `duplicate_of` and captioned once. Files outside the layout are counted and skipped.

### 2. caption

```bash
# default: WD14 + Florence-2
python workflows/curate_index.py caption runs/r1

# recommended: EVA02 tags + Qwen3-VL prose + VLM headcount
python workflows/curate_index.py caption runs/r1 \
    --backend hybrid:wd14+openai-compat --wd14-repo SmilingWolf/wd-eva02-large-tagger-v3 \
    --model-id qwen3-vl:8b --prompt straightforward --count-model qwen3-vl:8b
```

- Appends one line per image to `raw.jsonl` and flushes after each one. **Interrupt and re-run at any time**: finished images are skipped.
- Failures go to `errors.jsonl` and do not stop the run. Re-running retries them.
- The first call writes `run.json` with the caption settings and the package version. A later call with different settings is **refused** (`--force` overrides), so one `raw.jsonl` never mixes two models.
- `--limit N` captions only N images, for pilots.

### 3. assemble

```bash
python workflows/curate_index.py assemble runs/r1 --trigger sks --targets sdxl,flux
```

This builds every caption variant for each target from `raw.jsonl`. It takes seconds and runs no models unless `--refine-model` is set. `--hybrid-preset` (default `keywords`) sets the balance between tags and prose: `tags`, `keywords`, `balanced`, `descriptive` or `prose`.

It also writes **`review_queue.csv`**, the images to send to face identification in [argus-curator](https://pypi.org/project/argus-curator/). An image is flagged when:

| reason | meaning |
|---|---|
| `multiple_people` | tags (`2girls`, `multiple_boys`, `solo_focus`, `1girl`+`1boy` …) or prose ("two women", "a man and a woman", "a group of …") indicate 2+ people |
| `tag_prose_count_disagree` | the tagger and the prose model give different counts |
| `vlm_sees_multiple` / `vlm_count_disagrees` | the `--count-model` answer differs (only with `--count-model`) |
| `no_people_detected` | `no_humans`: likely a scene-only shot or a miss |

The flagging deliberately leans towards false positives. A missed second person contaminates a person's LoRA, while an extra review costs a glance.

### 4. index

```bash
python workflows/curate_index.py index runs/r1 --taxonomy workflows/taxonomy.example.json
```

Produces one row per image × category × tag in `index.csv` (and `index.parquet` if pyarrow is installed). The categories are `identity`, `wardrobe`, `pose_composition`, `setting`, `lighting`, `action`, `object` (tags that match no category: cups, phones, bouquets) and `people` (`1`, `multiple`, `0`, `unknown`).

- **Taxonomy**: [`taxonomy.example.json`](taxonomy.example.json) merges synonyms (`gown` → `dress`, `brunette` → `brown hair`) and adds broader terms (`kitchen` → `indoors`, `home`), so "every image at home" is one query. Copy it and extend it as you find gaps in `groups.csv`.
- **`groups.csv`** lists each (category, tag) with its image count and the number of names it spans. Start your curation here.
- **Identity QA**: each `name` folder is one person, so their majority hair and eye colour is a strong prior. `identity_consensus.csv` lists each name's consistent traits. `identity_outliers.csv` lists images tagged with a different hair or eye colour: often a second person, a wig, a lighting shift or a misfiled photo. `--min-share` (default 0.6) sets how dominant a trait must be.

### 5. query

```bash
# red dress in a kitchen, single person only
python workflows/curate_index.py query runs/r1 --all "wardrobe:red dress" setting:kitchen --not people:multiple

# brown hair, either cooking or eating, two named people
python workflows/curate_index.py query runs/r1 --all "identity:brown hair" --any action:cooking action:eating --name alice bob
```

Terms take the form `category:tag`. A bare `tag` matches any category. Output is tab-separated `image_id  name  rel_path` on stdout, with a count on stderr, so you can save it straight to a file for `export --ids`. For anything more complex, query `index.parquet` with DuckDB:

```sql
SELECT tag, count(DISTINCT image_id) n FROM 'runs/r1/index.parquet'
WHERE category = 'setting' GROUP BY tag ORDER BY n DESC;
```

### 6. export

```bash
# SDXL LoRA dataset for one person: tag captions
python workflows/curate_index.py export runs/r1 datasets/alice-sdxl --target sdxl --name alice --single-person

# Flux dataset from a saved query: sentence captions
python workflows/curate_index.py query runs/r1 --all "wardrobe:red dress" > red_dress.tsv
python workflows/curate_index.py export runs/r1 datasets/red-dress-flux --target flux --ids red_dress.tsv
```

This writes `<dest>/<name>/<shoot>_<file>_<id>.{jpg,txt}` in the kohya_ss / ai-toolkit sidecar layout. Images are symlinked; use `--copy` for a self-contained folder. `--variant` picks any caption variant. The defaults are `training` for SDXL and `training_prose` for Flux.

## Caption sets: identity, wardrobe, setting

Every image gets one caption per category: `identity`, `wardrobe`, `setting`, `action`, `pose_composition` and `lighting`. Each **leads with its own category** and adds a little context from the others. These are the readable caption sets.

For **grouping**, the index does not use those captions. It uses the un-truncated per-category tag lists in `buckets`, which contain exactly one category each. That keeps "wardrobe: red dress" from being diluted by hair colour, and the per-caption word limit cannot drop a tag from the index.

## SDXL and Flux coverage

| | SDXL (CLIP) | Flux (T5) |
|---|---|---|
| Training caption | `training`: comma-separated tags, about 60 CLIP tokens | `training_prose`: 1–4 complete sentences, about 200 T5 tokens |
| Identity | suppressed (hair, eyes, face, body and age are learned from the pixels) | suppressed, including phrase-level excision ("a woman ~~with long brown hair~~ stands …") |
| Framing / pose | hard-protected; never dropped by the budget | kept as written |
| Objects | unclassified tags (bouquet, laptop) fill leftover budget | from the prose |
| Variety | omission cycles drop a different category on different images for disentanglement (deterministic per image) | sentence order is preserved |

`training_prose` falls back to the tag caption when an image has no prose. Both targets come from the same `raw.jsonl`; add more with `--targets sdxl,flux,sd3`.

## LLM refinement and the post-check

Deterministic assembly works on clause boundaries, and on its own it can produce stilted text. Two layers handle that.

1. **Post-check and repair (always on, deterministic).** Every caption is checked and repaired at the end of assembly. Openers ("In this image," "Photograph of …"), markdown and hedging ("appears to be", "possibly") are removed from the prose up front. Dangling words ("holding a"), orphan sentences ("She."), subjectless sentences ("Smiling at the camera." is merged into the previous sentence), duplicate tags (`red_dress` / `red dress` / `dress`) and a leading "She" whose antecedent was removed are all repaired.

2. **LLM rewrite (optional).** `--refine-model` sends the structured inputs and the draft captions to an instruct model, which rewrites them as fluent captions:

   ```bash
   python workflows/curate_index.py assemble runs/r1 --trigger sks \
       --refine-model qwen3:8b --refine-variants training,training_prose,identity,wardrobe,setting,action
   ```

   **No rewrite is trusted blindly.** Each one must pass the same post-check plus stricter rules: trigger word first, no identity terms in training captions, within the token budget, and at most 25% of content words absent from the tagger and prose output (the hallucination guard). A failing rewrite gets one retry with the problems listed. If it fails again, or the endpoint is down, the deterministic caption stays. `captions.<target>.jsonl` records what was accepted or rejected, and why, in each row's `refine` field. Refined category captions also enter the index (`source=refine`), which gives you cleaner, more reusable vocabulary for grouping.

   Requests use `temperature=0` and `seed=0`. Refinement is reproducible on servers that honour these, but treat it as model-dependent and keep the refine model fixed within a run.

## Reproducibility

- **Pin the environment**: install from the repo's `uv.lock` (`uv sync`). `run.json` records the `argus-lens` version and the caption settings.
- **Pin the models**: WD14 and Florence-2 download fixed files into a cache. For served models, pin the exact tag (`qwen3-vl:8b`, a specific vLLM revision) and record it. `run.json` stores what you passed.
- **Deterministic downstream**: `assemble`, `index`, `query` and `export` are pure functions of `raw.jsonl` plus their flags. Omission cycles use the image's position in path-sorted order, so the same inputs always give the same captions.
- **Script it**: a whole run is a handful of commands. Put them in a shell script or Makefile next to the run directory, so the run itself is the record.

## Recommended first run (pilot)

```bash
python workflows/curate_index.py inventory /data/source_root runs/pilot
python workflows/curate_index.py caption runs/pilot --limit 200         # about 2 names' worth
python workflows/curate_index.py assemble runs/pilot --trigger sks
python workflows/curate_index.py index runs/pilot --taxonomy workflows/taxonomy.example.json
```

Then review:

1. **`groups.csv`**: are the wardrobe, setting and action tags you care about present and plausibly counted? Add synonyms and parents to your taxonomy for any splits (`couch` vs `sofa`).
2. **`review_queue.csv`**: open the flagged images. If too many false positives come from `tag_prose_count_disagree`, the prose model is the weak link; try Qwen3-VL and `--count-model`.
3. **A sample of `captions.flux.jsonl`**: read `training_prose` for 20 images. If it's stilted, try `--refine-model`.
4. Fix the configuration, start a **new run directory**, and run the full set.

## Outputs reference

| File | Contents |
|---|---|
| `manifest.jsonl` | `image_id, sha256, rel_path, name, shoot_id, bytes[, duplicate_of]` |
| `run.json` | package version, Python version, caption settings |
| `raw.jsonl` | `sha256, rel_path, backend, tags, prose[, vlm_count]`, the only model output |
| `errors.jsonl` | images that failed to caption (retried on the next run) |
| `captions.<target>.jsonl` | per image: `variants` (identity, wardrobe, pose_composition, setting, lighting, action, training, training_prose, zeroshot), `buckets` (per-category tags and prose, plus `unclassified`), `subjects` (headcount with evidence), `refine`, `notes` |
| `review_queue.csv` | images for face identification, with reasons and evidence |
| `index.csv` / `index.parquet` | `image_id, name, shoot_id, rel_path, category, tag, source` |
| `groups.csv` | `category, tag, images, names` |
| `identity_consensus.csv` | per name, the identity traits shared by ≥ `--min-share` of its images |
| `identity_outliers.csv` | images whose hair or eye colour contradicts their name's consensus |
