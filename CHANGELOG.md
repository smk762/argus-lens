# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **LLM refinement is now on by default (opt-out)** in `argus-lens serve` and `argus-lens caption`.
  It rewrites **every** variant (the categories as well as `training`/`training_prose`/`zeroshot`) with
  `qwen3:8b` on local Ollama unless `ARGUS_REFINE_MODEL` / `ARGUS_REFINE_BASE_URL` say otherwise.
  Disable it with `ARGUS_REFINE=off` or `--no-refine`. `ARGUS_REFINE_VARIANTS` / `--refine-variants`
  still narrow the list (`all` is the default). The `Refiner` library default is unchanged (the two
  training variants); pass `variants=None` for all.
- An unreachable refine endpoint fails fast (5 s connect timeout) and refinement is skipped for 60 s
  (`metadata["refine"]["skipped"] = "unavailable"`), so a missing LLM doesn't stall captioning.
- `training_prose` rewrites for CLIP targets (SDXL, SD1.5) ask for one or two concise sentences
  within the CLIP budget. T5 targets (Flux, SD3) still allow up to four.

## [0.9.0] - 2026-10-26

### Added
- **RAM++ vocabulary packs** (#62): custom tags without retraining. `argus-lens ram-vocab build`
  writes 51 descriptions per tag with any OpenAI-compatible LLM (upstream's prompts; defaults to the
  `ARGUS_REFINE_*` endpoint) or takes supplied descriptions, embeds them with CLIP ViT-B/16 via
  `transformers`, and writes a pack (`manifest.json` + fp16 `embeddings.safetensors` +
  `descriptions.json`). `RamBackend(vocab_packs=[...])` / `RAM_VOCAB_PACKS` appends pack tags to the
  built-in 4585 (a pack tag replaces a built-in duplicate; a tag in two packs is an error);
  `builtin_vocab=False` uses packs alone. Pack tags default to threshold 0.5 (upstream's open-set
  value), overridable per tag, and report `source="ram:<pack>"`. `ram-vocab inspect` summarises a pack.
  `safetensors` is now listed in the `[torch]` extra.
- **`annotate_image()` on every backend** (#57). `CaptionBackend` now provides a default that wraps
  `caption_image()` by output kind: taggers (`emits_tags`) get their comma string split into
  unscored `Tag`s, everything else is returned as `prose`. WD14 overrides it to keep its real
  per-tag probabilities (as RAM++ already did), and `hybrid` returns the tag stage's scored tags
  plus the prose stage's prose, so scores survive into provenance in every mode.
- `region_to_normalized()` / `region_to_mwg_area()` helpers in `backends/output.py`: convert a
  region to 0–1 coordinates, or to the centre-based MWG area `argus_cortex.xmp.XmpRegion` uses.

### Changed
- **`Region` is now `tuple[float, float, float, float]`** (#58): `(x1, y1, x2, y2)` in absolute
  pixels of the source image (origin top-left), unrounded, which is what Florence-2 `<OD>` and most
  detectors emit. Normalised coordinates are derived where the image size is known. No backend
  emitted regions yet, so nothing changes on the wire.

### Fixed
- **`unload()` now frees registry-cached models** (#63). `wd14`, `florence2`, `blip2` and `ram`
  implemented `unload()` as a no-op, so `engine.unload()`, the idle reaper and `/admin/unload`
  reported success while the model stayed in VRAM until the registry's own idle sweep (up to
  `ARGUS_MODEL_IDLE_SECONDS`, default 300 s). `ModelRegistry.evict(key)` frees an unreferenced
  model immediately; `LocalBackend.unload()` evicts the keys that backend acquired, and leaves a
  model another caption is still using in place for the next unload.
- **First-use weight downloads no longer hold the GPU lease** (#64). A new `prefetch()` hook
  (WD14 model, RAM++ checkpoint, Florence-2 / BLIP-2 HF snapshot) runs once per engine *before*
  the coordinator lease, so a multi-GB download no longer blocks other GPU tenants.

## [0.8.0] - 2026-10-25

### Added
- **RAM++ photo tagger** (`ram`, #60). The RAM++ scaffold is now a working backend: an
  inference-only, torch-only port of recognize-anything's Swin-L RAM++ (vendored under
  `backends/_ram/` with its Apache-2.0 licence; no timm/transformers/torchvision/fairscale),
  loading `xinyu1205/recognize-anything-plus-model` unchanged (`weights_only=True`, auto-downloaded
  to `~/.cache/ram_plus/`, or `RAM_MODEL_DIR`). `annotate_image()` returns scored tags that clear
  RAM++'s calibrated per-tag thresholds (or a uniform `threshold=`), highest first. Selectable as
  `ram`, `hybrid:ram+florence2` and `hybrid:ram+openai-compat`. Reproduces upstream's published
  demo output exactly in fp32 and fp16; ~740 MB peak VRAM, ~50 ms/image on an RTX 3090.
- `CaptionBackend.emits_tags`: taggers set it so a standalone run feeds the assembly pipeline's
  tag side. Previously only `style == "anime"` or the name `wd14` did, so a photo-style tagger's
  output was treated as prose.

### Changed
- `RamBackend(threshold=)` now defaults to `None` (RAM++'s per-tag thresholds) instead of `0.35`.
- `RamBackend(model_id=)` is now `repo_id=`, so a hybrid's `--model-id` (the prose model) no longer
  becomes RAM++'s checkpoint repo. A non-default repo caches under `~/.cache/ram_plus/<org>--<name>/`.
- Model downloads (WD14, RAM++) go to a unique temp file that is removed on failure.
- The GPU lease estimate for `ram` is 1000 MB (measured), down from a 2000 MB guess.
- `docker-compose.yaml` persists `~/.cache/ram_plus`, so containers don't re-download the 3 GB
  RAM++ checkpoint.

## [0.7.0] - 2026-10-25

### Changed
- **XMP sidecars now merge instead of replace.** The XMP writer moved to
  `argus_cortex.xmp` (the suite's one implementation, shared with
  argus-curator), and `argus-cortex>=0.3.0` is now a dependency. `XmpSink` and
  the `write_xmp` endpoints replace only `dc:subject` / `dc:description` (latest
  write wins) and preserve everything else in an existing sidecar — Lightroom
  ratings, GPS, develop settings, curator face regions. Empty keywords / caption
  leave the existing value alone.
- **`xmp_overwrite` now defaults to `false` and means "discard and rewrite".**
  `false` merges (previously: per-image error on an existing sidecar); `true`
  discards the existing sidecar (previously the default). `XmpSink.write`
  no longer raises `FileExistsError`; an existing sidecar that isn't valid XMP
  raises `argus_cortex.xmp.XmpError` (a per-image error on the endpoints) and
  is left untouched.

### Fixed
- `BackendOutput.tag_string` drops empty / whitespace-only tag labels instead
  of emitting stray `", ,"` entries into the caption.
- `BackendOutput.to_caption_string` falls back to the tags when `prose` is
  whitespace-only, rather than returning a blank caption.
- The tag score threshold is now one rule (`Tag.passes`) shared by the
  flattened caption and the provenance `included` flag, so they can't disagree.

## [0.6.0] - 2026-10-24

### Added
- **Curate-index workflow** (`workflows/curate_index.py`, documented in
  `workflows/README.md`) — inventory → caption once → assemble per target →
  tag index + groups + identity QA → query → kohya-style export, for libraries
  laid out as `<name>/pics/<shoot_id>/*.jpg`. Resumable, content-hash keyed,
  and settings-locked via `run.json`; every stage after `caption` recomputes
  from the stored raw model output.
- **Index metadata on every `CaptionResult`** — `metadata["buckets"]` holds the
  un-truncated per-category fragments split by source (`tags` / `prose`), with
  hint-less fragments in a separate `unclassified` list instead of the identity
  fallback; `metadata["subjects"]` holds a headcount estimate from tagger
  headcount tags (`2girls`, `multiple_boys`, `solo_focus`, `no_humans`) cross-checked
  against the prose (`count`, `multiple`, `agree`, `evidence`).
- **`training_prose` variant** — sentence-form training caption for T5 targets
  (Flux, SD3): identity clauses and "with … hair/eyes" phrases excised,
  sentences kept whole under the target's token budget.
- **Coherence post-check** (`argus_lens.assembly.coherence`) — every variant is
  repaired at the end of assembly (openers, markdown, hedging, dangling words,
  orphan and subjectless sentences, duplicate/absorbed tags); `check_caption()`
  exposes the same rules as a QA gate.
- **LLM refinement** (`argus_lens.refine`) — `Refiner` + `OpenAICompatRewriter`
  rewrite selected variants with any OpenAI-compatible chat model; a rewrite is
  accepted only if it passes the post-check plus identity, budget and grounding
  rules, with one feedback retry and a deterministic fallback. `ArgusLens(refiner=...)`,
  CLI `--refine-model/--refine-url/--refine-variants/--refine-image`.
- **Recursive directory captioning** — `caption_directory(recursive=True)` /
  CLI `--recursive`; results keyed by relative path so equal filenames in
  different folders stay distinct, and `.txt` sidecars land beside their images.
- **openai-compat prompt presets** — `prompt=` / `ARGUS_OPENAI_COMPAT_PROMPT` /
  CLI `--prompt` accepts `detailed`, `straightforward`, `descriptive`, `booru`,
  `count` or literal text; optional `temperature`.
- **Server env configuration** — `argus-lens serve` (and the Docker image)
  enables refinement from `ARGUS_REFINE_MODEL` / `ARGUS_REFINE_BASE_URL` /
  `ARGUS_REFINE_API_KEY` / `ARGUS_REFINE_VARIANTS` / `ARGUS_REFINE_IMAGE`, and
  idle VRAM release from `ARGUS_IDLE_UNLOAD_S` (for hosts shared with e.g. ComfyUI).
- **Selectable WD14 v3 tagger** — `model_repo=` / `WD14_MODEL_REPO` / CLI
  `--wd14-repo` (e.g. `SmilingWolf/wd-eva02-large-tagger-v3`), cached per repo.

### Changed
- **Category hints match whole words.** Hints previously matched as
  substrings, so `hair` fired inside `chair`, `hat` inside `that`, `age` inside
  `image`, `tee` inside `street`. Plurals and underscored tags still match.
  Category vocabularies were expanded (jewellery, swimwear, furniture, weather,
  venues, poses, activities), so fewer tags fall back to identity.
- **Each category variant leads with its own category** (the `setting` caption
  opens with the setting, not with hair colour).
- **Prose enrichment lifts only modifier+noun compounds** (`knit_sweater`,
  `wooden_bench`) from non-identity, filler-stripped clauses. Loose words
  (`wearing`, `next`, `shows`) and identity compounds (`red_hair`) no longer
  leak into the training caption.
- **Unclassified tags (objects) reach training captions** as low-priority fill
  instead of being dropped with the identity bucket.
- Headcount tags (`2girls`, `multiple_*`) no longer land in the identity
  bucket; they still appear in the zero-shot variant.

### Security
- **eMudhra/emSign CA incident.** Outbound TLS now requires `certifi>=2026.10.19`,
  which no longer trusts the compromised emSign intermediates. `uv` is required
  at `>=0.13.5` (fixed trust store, and managed CPython builds that bundle pip
  26.2.2). Docker images distrust emSign ECC Root CA - G3 and the Comodo AAA
  cross-sign in the system store, then install hash-pinned pip 26.2.2 before
  any other package. PyPI publishing uses
  `pypa/gh-action-pypi-publish` v1.14.3, which vendors a fixed sigstore.
  See https://discuss.python.org/t/109576.

### Fixed
- `hybrid:<tag>+<prose>` passed every engine kwarg to both stages, so
  `hybrid:wd14+openai-compat --base-url/--model-id` crashed WD14 with a
  `TypeError`. Backends now receive only the arguments their constructor accepts.
- `docker-compose.yaml` did not pass `ARGUS_OPENAI_COMPAT_BASE_URL` /
  `ARGUS_OPENAI_COMPAT_MODEL` into the container, so a containerised server
  could not reach a remote OpenAI-compatible endpoint. The WD14 cache mount is
  now writable so a non-default `WD14_MODEL_REPO` can download.

## [0.5.0] - 2026-07-22

### Added
- **`replay` backend** — serves recorded captions straight out of the cortex
  Postgres lineage store (`CORTEX_PG_URL`) instead of running a model, so a
  GPU-free box (or the public argus-halo demo) can re-enact genuine captured
  output with **no GPU and no external API** (#45). Each request is keyed by the
  source image's sha256 — the content id cortex records on `source_asset` — so
  the engine hashes the bytes it ingests and returns the recorded
  `CaptionResult` **verbatim** (variants included), bypassing the assembly
  pipeline so profile knobs can't re-mangle captured output. An asset with no
  recorded caption is a `ReplayMiss` — HTTP 404 on the single `/caption`, and a
  per-item error on the multi-image paths (see below) — not a silent fall back
  to a live model. Reads the documented lineage schema directly (no `argus-cortex`
  dependency); `psycopg` lives behind the new `argus-lens[replay]` extra. Select
  it with `--backend replay` / `ARGUS_BACKEND=replay`.
- **Curator manifest 2.0 support on the manifest endpoints** —
  `POST /caption/manifest` and `POST /caption/manifest/stream` accept an
  optional `export_root` form field (the directory the curator exported into —
  the manifest itself sits at `<export_root>/manifest.jsonl`). Rows carrying
  `exported_path` (the 2.0 normative locator, already de-collided for
  flattened exports) are then read from `export_root / exported_path` (confined
  under the root — an absolute or `..`-laden `exported_path` is a per-row error,
  not an escape), which stays valid for `mode=move` exports and cross-host
  handoffs where `abs_path` points at a moved-away or unreachable source.
  Without `export_root` (or on
  pre-2.0 rows without `exported_path`) behaviour is unchanged: images are
  read from `abs_path` (shared volume with the curator). An `export_root`
  that is not an existing directory is a single 400 instead of N per-row read
  errors, and rows declaring a `manifest_version` outside the supported
  1.x/2.x majors are rejected with 400 naming the offending line, so a future
  incompatible major fails loudly instead of being misread.

### Changed
- **Engine `_load_image` now returns an `ImageAsset`** (name + original-bytes
  sha256 + source uri) alongside the decoded image, so the ingested content
  identity survives to the backend (this is what the replay lookup keys on).
  `caption_batch` / `caption_stream` gained an optional `names=` argument to set
  result keys when the source loses its filename (e.g. raw upload bytes); the
  server's `/caption/batch` and `/caption/stream` now pass original bytes plus
  filenames, which also fixes uploaded-image results collapsing under a shared
  name.
- **Replay misses are handled per image** on the multi-image paths (#48): one
  un-recorded asset no longer aborts the whole request. `/caption/batch` returns
  `200` with the captions that resolved plus an `errors` `{name: reason}` map;
  `/caption/stream` emits one error line for the miss and keeps streaming the
  rest; `argus-lens caption <dir> --backend replay` skips missing images (no
  sidecar), reports them, and captions the rest. The engine's `caption_batch` /
  `caption_directory` gained an `on_miss` callback (fail-fast when omitted), and
  `caption_stream` now yields `(name, ReplayMiss)` for a miss instead of raising.

## [0.4.0] - 2026-07-09

### Added
- **GPU lifecycle + capacity coordinator** — makes argus-lens a well-behaved
  tenant on a shared GPU; both mechanisms are opt-in and no-ops by default, so
  existing behaviour is unchanged. (#37, #38)
  - Lifecycle: `ArgusLens.unload()` / `unload_if_idle(ttl)` / `close()` with
    lazy reload on the next caption, an optional idle reaper (`idle_unload_s=`),
    and `vram_status()` (backend, loaded, coordinator, `free_vram_mb`, idle_s)
    plus a pre-load low-VRAM warning. Server: `GET /health` gains a `gpu` block
    and `POST /admin/unload` frees the model.
  - Coordinator: new `argus_lens.gpu` package — `GpuCoordinator` protocol with
    `NullCoordinator` (default), `LocalLeaseCoordinator` (POSIX `flock`, one
    heavy job at a time), and `GothmogCoordinator` (acquire/release against a
    `/v1/gpu/capacity` API). Selected via env (`ARGUS_GPU_COORDINATOR`,
    `GOTHMOG_URL`, `GOTHMOG_API_KEY`, `ARGUS_GPU_LEASE_PATH`); the engine wraps
    inference in the lease, sized by a per-backend VRAM estimate, and cloud
    backends bypass it.
  - Env knobs: `ARGUS_GPU_MIN_VRAM_MB` overrides the per-backend footprint the
    lease requests; `ARGUS_GPU_LEASE_TIMEOUT_S` caps how long the lease waits
    for a slot, raising `GpuLeaseTimeout` (a `TimeoutError` subclass) so callers
    can distinguish "no GPU slot" and back off.
- **VQA reconciliation cross-check** — detects attributes where Florence prose
  contradicts WD14 tags (colour/pose hallucinations), asks a pluggable
  `AttributeVerifier` to adjudicate, and rewrites the prose to match. Verifiers,
  selectable via `--reconcile`: `tag-prior` (model-free, deterministic default),
  `openai-compat` (VQA over any served vision model), `florence` (grounds the
  subject and samples the region's pixels), and `molmo`. Verifier errors are
  caught so a flaky verifier never breaks captioning; no verifier → inference
  unchanged. Exposed on the `caption` and `eval` CLI commands. (#36)
- **`argus-lens eval` — caption quality harness** — a reference-free-first
  evaluation package (`argus_lens.eval`) so quality-affecting changes can be
  judged by numbers. The flagship metric (tag↔prose contradiction) needs no
  ground truth and runs on any image folder; a JSONL manifest additionally
  unlocks tag-coverage recall and reference CLIPScore (behind the new `eval`
  extra). Emits a plain-text scorecard + JSON, with a baseline regression gate
  (`--baseline`, `--fail-on-regression`) for CI. (#35)
- **User-tunable tag↔prose balance presets** — a single axis controlling how
  much prose survives hybrid tag+prose fusion, exposed as named presets
  (`tags`/`keywords`/`balanced`/`descriptive`/`prose`) and a continuous
  `prose_bias` (0.0 = pure tags .. 1.0 = full prose). `balanced` reproduces the
  prior default, so behaviour is unchanged by default. Threaded through the
  engine, all `/caption*` server endpoints, and `--hybrid-preset` / `--prose-bias`
  CLI flags; advertised on `/profiles` for UI pickers.
- **`GET /health`** — service liveness/identity probe returning
  `{status, service, version, source_root}`, mirroring argus-curator's shape.
- **`GET /profiles`** — exposes the caption taxonomy (`assembly_profiles`,
  `target_styles`, `target_categories`, `target_backends`, `token_budgets`)
  derived from the registry and `types.py`, so UIs no longer hardcode it. The
  trunk assembly pipeline is registered as the `lora_training` profile, so
  `assembly_profiles` reflects what actually ships.
- **Immich HTTP endpoints** — the server now wraps the Immich connector,
  configured via `IMMICH_URL`/`IMMICH_API_KEY` (read per request; without them
  the endpoints return 503 and the rest of the server works normally; upstream
  transport failures *and* garbled/unexpected Immich responses map to 502):
  - `GET /immich/albums` — list albums with asset counts.
  - `POST /immich/pull` — download an album's originals (or a subset via
    `asset_ids`) into a folder under the source root, streaming NDJSON
    progress. Downloads run a few at a time over a pooled connection and land
    via temp file + atomic rename (an interrupted write never leaves a
    truncated image). Existing same-name files are skipped; two assets sharing
    a basename within one request are a per-asset error, not a silent skip;
    files `/caption/folder` cannot caption (e.g. HEIC/DNG originals) carry a
    `warning` on their progress line. `asset_ids: []` selects nothing, and ids
    not in the album are a 404.
  - `POST /immich/caption/stream` — caption album assets in memory (no disk
    writes), streaming NDJSON progress, optionally pushing captions back to
    Immich (`write_back`) as description + tag keywords; when only the
    write-back step fails, the progress line keeps `final_caption` alongside
    `error` (the caption was computed, and the description may already be set
    in Immich).
- **`write_xmp` option on the captioning endpoints** — `POST /caption/folder`
  (JSON body), `POST /caption/manifest`, and `POST /caption/manifest/stream`
  (form field) can now also write an `<image>.xmp` sidecar next to each source
  image via `XmpSink` (`dc:subject` = raw tag keywords, `dc:description` =
  final caption). XMP sidecars are the zero-coupling interop surface:
  Lightroom, digiKam, and Immich all ingest them natively on library scan.
  Independent of `write_sidecar` (write either or both); by default follows
  the same overwrite semantics as the `.txt` sidecars (pre-existing files on
  disk are replaced; duplicate targets within one batch — under any path
  spelling — are per-image collision errors), and `xmp_overwrite: false`
  instead turns a pre-existing `.xmp` (e.g. one Lightroom/digiKam already
  populated — XMP writes never merge) into a per-image error. Responses gain
  an additive `xmp_written` count, and successful
  results/progress lines an `xmp_path`. On `POST /immich/caption/stream`,
  `write_xmp: true` is rejected with 400 — Immich assets are captioned in
  memory with no local path; pull the album first (`/immich/pull`) and caption
  the folder, or use `write_back`.
- **`XmpSink.sidecar_path`** — public helper returning the `<image>.xmp`
  sidecar path for an `AssetRef` (used by the server endpoints).
- **Immich connector album support** — `ImmichSource.list_albums`,
  `ImmichSource.list_album_assets`, and `ImmichSource.fetch_original` (raw
  bytes download backing both the pull endpoint and `fetch_image`). Connector
  instances now reuse one pooled `httpx.Client` per instance (`close()`
  releases it) instead of a fresh TCP+TLS handshake per request — the
  per-asset pull/write-back loops make that per-call cost hot.

## [0.3.0] - 2026-07-01

Backwards compatible with `0.2.0` — no breaking API or default-behavior changes.
The public API (`ArgusLens`, `CaptionResult`, `CaptionTargetProfile`,
`CategoryConfig`, `TokenBudgetConfig`, the `CaptionBackend` protocol), export
schemas, and runtime dependencies are unchanged. All new capabilities are
additive/opt-in and are not wired into the default captioning path.

### Added
- **`openai-compat` backend** — caption via any server speaking the OpenAI
  `/chat/completions` wire format (Ollama, vLLM, LM Studio, LocalAI, llama.cpp,
  or a hosted proxy). The endpoint is fully configurable (`base_url`) and the
  API key is optional, since local servers typically need no credentials.
  Config resolves from constructor args → `ARGUS_OPENAI_COMPAT_{BASE_URL,MODEL,API_KEY}`
  env vars → defaults (Ollama localhost + `llava`). Uses only the core `httpx`
  dependency, so no new install extra is required. Adds a `--base-url` option to
  `argus-lens caption`. (#25)
- **Connectors I/O layer** — `Source`/`Sink` protocols with `FilesystemSource`
  and `XmpSink`, plus an **Immich source + sink** for pulling/pushing assets and
  writing caption sidecars. `ImmichSource.list_assets` pages through the Immich
  search API (with `since` for incremental sync) and `ImmichSink.write` pushes
  keywords (tag upsert + assign) and descriptions back to Immich, making the
  companion-service loop usable end to end. (#17, #18, #29)
- **Structured backend output** — new `BackendOutput` / `Tag` types so backends
  can emit structured tags with scores instead of bare strings. (#12)
- **Per-tag provenance** — provenance metadata built from `BackendOutput`,
  including an `included` flag marking which tags passed the threshold. (#13)
- **Taxonomy normalization** — controlled-vocabulary normalization layer to
  canonicalize tag labels. (#16)
- **Pluggable assembly profiles** — `AssemblyProfile` protocol + registry with
  registration validation, enabling intent-specific assembly behavior. (#15)
- **RAM++ backend** — scaffold for a photo-domain tagging backend (reported as
  unavailable until its dependencies/model are wired). (#14)

### Changed
- **Device placement via the `load(device)` contract** — the engine's configured
  `device` flows to backends through `load(device)`, which each backend records
  and applies to its (lazy) model loads; `caption_image` is device-free on the
  canonical path. The engine calls `load()` exactly once, lazily, and the
  check-and-set is thread-safe for engines shared across request threads. The
  torch backends (`florence2`, `blip2`) retain an **optional** `device` override
  on `caption_image` for backwards compatibility with pre-0.3 direct callers.
  Replaces the interim `caption_image` signature-sniffing introduced during
  development. (#10, #20, #21, #22)
  - Fixes an explicitly-set engine device being dropped for hybrid prose
    backends, and lets `wd14` be pinned to CPU; `wd14` selects ONNX Runtime
    providers without requiring torch (so `[wd14-gpu]` still uses CUDA) and keys
    its session cache by device intent plus the configured model directory, so
    backends pointing at different models never share a cached session.
- **CUDA OOM retry** — backend inference retries on CUDA out-of-memory with cache
  cleanup; the wait budget is configurable and observable via the new keyword-only
  `ArgusLens(oom_retry_max_wait_s=..., oom_retry_interval_s=...)` parameters
  (default 180s / 5s; set max wait to `0` to fail fast). (#9)
- `BackendOutput.raw` typed as `dict[str, Any]`. (#12)

### Fixed
- **`wd14` tagger repaired and bumped to `wd-vit-tagger-v3`** — fixes the dead
  download path and corrects tag handling: ratings are excluded by tag *category*
  (not a fragile `rating:` name prefix), preprocessing matches SmilingWolf v3
  (white square-pad → BICUBIC → BGR), and the input size is read from the model.
  Guards against a model/`selected_tags.csv` size mismatch instead of silently
  truncating. The cache key stays import-light (no `onnxruntime` import on the
  caption path). (#23, #24)
- **Connectors robustness** — Immich asset IDs are URL-encoded and headers tidied;
  XMP sidecars are protected and illegal XML characters stripped. (#17, #18)
- **Taxonomy** — immutable default and blank labels dropped. (#16)
- **Server: folder captioning confined to the source root** — `POST /caption/folder`
  now resolves its `folder` inside `--source-root` / `LENS_SOURCE_PATH` (relative
  or absolute-within-root) and rejects anything else, closing an unauthenticated
  arbitrary-directory walk + sidecar-write exposure; the standalone image sets
  `LENS_SOURCE_PATH=/data`. CORS no longer combines a wildcard origin with
  credentials.
- **Server: batch loops unified and hardened** — the three batch endpoints
  (`/caption/manifest`, `/caption/manifest/stream`, `/caption/folder`) share one
  parse + caption/sidecar helper: manifest lines that aren't JSON objects are a
  400 instead of a crash, a failed sidecar write counts the row as failed only
  (no more `captioned + failed > total`), same-stem sidecar collisions
  (`cat.jpg` + `cat.png` → `cat.txt`) are reported instead of silently
  overwritten, both manifest endpoints accept `prose_enrichment`, and the
  supported-extension list is shared with the connector layer (adds
  bmp/tiff/gif).
- **Server: no event-loop blocking** — `/caption/stream` now pulls its sync
  generator via a worker thread and `/caption/folder` walks the tree off the
  event loop, so long inference (including OOM-retry waits) no longer freezes
  the server.
- **`wd14` upgrade path** — when the model file is (re-)downloaded, a leftover
  `selected_tags.csv` from a previous model version is refreshed with it, so
  0.2.0 caches no longer trip the size-mismatch guard after upgrading.
- **OpenAI backends: guarded response parsing** — `openai` and `openai-compat`
  raise a clear `RuntimeError` on empty `choices` or null content (refusals /
  content filters) instead of an opaque `AttributeError`/`IndexError`;
  `openai-compat` also joins list-form content parts.
- **Packaging/config** — the `all` extra now includes `python-multipart` (the
  server's multipart endpoints 500'd under `pip install argus-lens[all]`);
  `serve` honours `ARGUS_BACKEND`; the documented-but-unread `ARGUS_API_KEY`
  is replaced by the real per-backend key variables in compose/docs; PyPI
  metadata gains Homepage/Issues/Changelog URLs.

### Tests / CI / Docs
- Backend class-contract smoke tests, including a variant that runs without
  optional dependencies installed. (#19)
- Wire-format tests for the `openai-compat` backend via `httpx.MockTransport`. (#25)
- Device-contract tests: `wd14` provider selection + provider-keyed cache, the
  back-compat `device` override, `resolve_device` behavior, and a single
  `load()` under concurrent first use. (#22)
- `wd14` v3 tests: rating-category exclusion + threshold filtering and the
  square-pad/BGR preprocessing, run against a fake ONNX session. (#23, #24)
- Expanded RAM++ backend tests.
- CI: pinned `ruff==0.15.16` for reproducible lint; resolved ruff findings and
  reformatted the codebase.
- Docs: documented the `openai-compat` backend in the README and `.env.example`;
  linked `awesome-immich` in the README.

## [0.2.0] - 2026-05-29

### Added
- OpenAI-compatible `/v1` server endpoint for Frigate GenAI integration.

### Docs
- Added value proposition, comparison table, and intent-aware positioning.
- Added a prominent link to the `argus-vision-demo` web UI.

## [0.1.0] - 2026-04-17

### Added
- Initial release: structured, intent-aware image captioning pipeline for LoRA
  training and dataset curation. Local backends (WD14, Florence-2, BLIP-2) and
  cloud backends (OpenAI, HuggingFace Inference, Replicate, NVIDIA NIM), hybrid
  tag+prose pipelines, the category-bucketed assembly engine, CLI, FastAPI
  server, and `.txt`/JSON/JSONL/CSV exporters.

[Unreleased]: https://github.com/smk762/argus-lens/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/smk762/argus-lens/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/smk762/argus-lens/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/smk762/argus-lens/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/smk762/argus-lens/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/smk762/argus-lens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/smk762/argus-lens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/smk762/argus-lens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-lens/releases/tag/v0.1.0
