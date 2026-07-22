# CLAUDE.md — argus-lens

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md) (it is unusually detailed and is the source of truth for behaviour); this file is the orientation an agent needs to change code safely.

## What this is

The **captioning** stage in the Argus suite. It takes images and produces *intent-aware, structured* captions — not one flat string but category-bucketed variants (identity, wardrobe, pose, setting, lighting, action) plus specialised `training` and `zeroshot` variants — and writes a `.txt` caption sidecar next to each source image.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           validate
```

The model is just an input source; the value is the **assembly pipeline** that runs after inference (classify → filter → specialise → token-budget → diversify → enrich). Read the README's "assembly pipeline" section for the *why* behind each step.

## Layout

`src/argus_lens/`:

- `engine.py` — `ArgusLens`, the entry point. Owns the `_BACKEND_REGISTRY` (string → backend, lazily imported), backend resolution (incl. `hybrid` = wd14+florence2 and `hybrid:tag+prose`), lazy device-aware model load, the GPU capacity lease, OOM-retry, and the idle-unload reaper. `caption` / `caption_batch` / `caption_stream` / `caption_directory`.
- `types.py` — the data contract, **plain `@dataclass`es, not Pydantic**: `CaptionResult` (`final_caption`, `caption_variants`, `removed_phrases`, …), `CategoryConfig` + `DEFAULT_CATEGORY_CONFIGS`, `CaptionTargetProfile`, `BACKEND_TOKEN_BUDGETS` (SDXL 60, Flux/SD3 200), `HYBRID_PRESETS`. Category names/hint-words and token budgets live here.
- `assembly/` — the pipeline: `classifier`, `filtering`, `noise`, `token_budget`, `training` (identity-suppressed variant), `zeroshot` (identity-first variant), `variants`, and `composer.compose_caption_result` (the orchestrator every backend output flows through). `profiles.py` is a pluggable `AssemblyProfile` registry (`lora_training` is the trunk).
- `backends/` — one module per source: local (`wd14`, `florence2`, `blip2`, `ram` scaffold) and cloud (`openai`, `openai_compat`, `hf_inference`, `nvidia_nim`, `replicate`) over `base.py` (`LocalBackend` / `CloudBackend`), `hybrid.py` (tag+prose fusion), `replay.py` (serves recorded captions from cortex — no model), `output.py`/provenance types. Register a new backend in `engine._register_backends`.
- `server.py` — FastAPI micro-server (`create_app`). `/caption`, `/caption/{url,batch,stream,folder,manifest,manifest/stream}`, `/folders`, `/immich/*`, `/health`, `/admin/unload`, `/profiles`, `/backends`. Optional `[server]` extra.
- `openai_compat.py` — `/v1/models` + `/v1/chat/completions` shim so the server is a drop-in Frigate GenAI provider; always mounted.
- `cli.py` — Typer app: `caption`, `backends`, `eval`, `serve`.
- `connectors/` — Source/Sink abstractions: `filesystem`, `immich` (pull/caption/write-back), `xmp` (`.xmp` sidecars for Lightroom/digiKam/Immich). `exporters/` — `.txt`/json/jsonl/csv. `eval/` — reference-free caption scoring. `reconcile/` — fix prose that contradicts tags via a pluggable verifier. `gpu/` — GPU coordinator + VRAM probes. `taxonomy.py`, `registry.py` (TTL model cache).

## Commands

```bash
make dev     # uv venv + editable install with [dev,cli]
make test    # pytest --tb=short -q  (asyncio_mode = auto)
make lint    # ruff check (pinned ruff 0.15.16)
make fmt     # ruff format + --fix
make check   # lint + test + build
```

Run a single test: `uv run --no-sync pytest tests/test_assembly.py::test_name -q`.

## Conventions & gotchas

- **The caption sidecar contract is load-bearing: lens writes `<image_stem>.txt` (the `final_caption`) next to the *source* image; forge relies on this pairing.** On a basename collision within a batch the server errors that image rather than overwriting a caption (`_caption_and_write` in `server.py`); `caption_directory` skips images that already have a `.txt` unless `overwrite=True`. Preserve both behaviours.
- **Backend selection is local (GPU/CPU) vs cloud (API) — don't blur the two.** `LocalBackend` gets its device once via `load(device)` and stays device-free after; `CloudBackend` resolves its API key from constructor arg → env var (e.g. `OPENAI_API_KEY`). Heavy GPU work goes through the coordinator lease + OOM-retry; cloud backends bypass both.
- **The `replay` backend reads the argus-cortex lineage store directly** (`CORTEX_PG_URL`, `[replay]` extra) and returns recorded `CaptionResult`s verbatim — it must **not** re-run the assembly pipeline or re-apply the profile. The dependency direction is cortex → lens, never the reverse: do not import `argus-cortex` here; rely only on the documented lineage schema. A missing recording is a `ReplayMiss` (a real error), not a cue to fall back to a live model.
- **Server safety boundaries — don't loosen casually:**
  - Every folder/manifest/immich path is confined under the configured source root (`--source-root` / `LENS_SOURCE_PATH`) via `_resolve_within` / `_confine_folder`; an absolute or `..`-laden path is a 400, not an escape. `manifest 2.0` rows resolve images as `export_root / exported_path` (confined the same way), falling back to `abs_path`.
  - **The caption endpoints have no auth** — reaching the port means writing sidecars into the source root. `/admin/unload` is gated by `ARGUS_ADMIN_TOKEN` when set (open otherwise). CORS is **off by default** (`--cors`); credentials are never allowed with a `*` origin (CORS-spec footgun) — keep that guard.
- **`CaptionResult` is the wire schema.** Endpoints `asdict()` it to JSON; the `caption_variants` keys (the category names + `training`/`zeroshot`) and field names are the API contract for argus-studio and downstream tools — renaming a field or variant key is a breaking change. Only manifest majors **1.x/2.x** are accepted (`_SUPPORTED_MANIFEST_MAJORS`); a new major must fail loudly.
- **Versioning is git-tag-derived** (`hatch-vcs`); `src/argus_lens/_version.py` is generated and gitignored. Never hand-edit a version — tag `vX.Y.Z` to release.
- `structlog` for logging; Ruff line-length 120 (`E`/`F`/`W`/`I`/`UP`/`B`/`SIM`); `interrogate` gates docstring coverage at 80% in CI. Python ≥ 3.11.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow (`python-ci.yml@v1`, ruff 0.15.16, `dev` extras, `interrogate` post-test). Release (`release.yml`) publishes to PyPI + GHCR on `v*` tags.
