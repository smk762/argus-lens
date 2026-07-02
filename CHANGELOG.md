# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
    its session cache by the effective provider to avoid duplicate sessions.
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
