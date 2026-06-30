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
  writing caption sidecars. (#17, #18)
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
- **Engine device handling** — the engine's configured `device` is now forwarded
  to backends that accept it (and to device-aware hybrid sub-backends). Backends
  that don't take a `device` keyword are detected via signature introspection and
  called exactly as before, so custom backends remain compatible. (#10, #20)
- **CUDA OOM retry** — backend inference retries on CUDA out-of-memory with cache
  cleanup; the wait budget is configurable and observable via the new keyword-only
  `ArgusLens(oom_retry_max_wait_s=..., oom_retry_interval_s=...)` parameters
  (default 180s / 5s; set max wait to `0` to fail fast). (#9)
- `BackendOutput.raw` typed as `dict[str, Any]`. (#12)

### Fixed
- **Connectors robustness** — Immich asset IDs are URL-encoded and headers tidied;
  XMP sidecars are protected and illegal XML characters stripped. (#17, #18)
- **Taxonomy** — immutable default and blank labels dropped. (#16)

### Tests / CI / Docs
- Backend class-contract smoke tests, including a variant that runs without
  optional dependencies installed. (#19)
- Wire-format tests for the `openai-compat` backend via `httpx.MockTransport`. (#25)
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
