# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/smk762/argus-lens/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/smk762/argus-lens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/smk762/argus-lens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-lens/releases/tag/v0.1.0
