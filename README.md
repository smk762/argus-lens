# Argus Lens

[![PyPI](https://img.shields.io/pypi/v/argus-lens)](https://pypi.org/project/argus-lens/)
[![Python](https://img.shields.io/pypi/pyversions/argus-lens)](https://pypi.org/project/argus-lens/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/smk762/argus-lens/blob/main/LICENSE)
[![CI](https://github.com/smk762/argus-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/smk762/argus-lens/actions/workflows/ci.yml)

One image. A hundred perspectives.

Multi-model captioning pipeline for LoRA training, dataset curation, and generative AI workflows. Unlike traditional captioners that describe an image, Argus Lens produces **intent-aware caption subsets** -- structured, filtered, and optimised for how captions are actually used downstream.

> **Looking for the web UI?** See [argus-studio](https://github.com/smk762/argus-studio) -- a thin Next.js frontend for exploring argus-lens interactively.

## Quick Start

```bash
pip install argus-lens[openai]
```

```python
from argus_lens import ArgusLens

engine = ArgusLens(backend="openai", api_key="sk-...")
result = engine.caption("photo.jpg", trigger_word="sks_person")

print(result.final_caption)
print(result.caption_variants["training"])
print(result.caption_variants["zeroshot"])
```

## Why Argus Lens?

Most captioning tools answer *"what's in this image?"* -- Argus Lens answers *"how will this caption be used?"*

BLIP gives you a sentence. WD14 gives you a flat tag list. Neither knows whether you're training a LoRA, curating a dataset, or generating zero-shot. Argus Lens produces **multiple structured caption variants from the same image**, each optimised for a different downstream task.

| | Typical captioner | Argus Lens |
|---|---|---|
| Output | Single flat string | Structured, category-bucketed variants |
| Intent awareness | None -- describes the image | Training, zero-shot, and per-category variants |
| Multi-model fusion | One model at a time | Hybrid pipelines (e.g. WD14 tags + GPT-4o prose) |
| Token budget management | Manual / none | Auto-tuned for SDXL (CLIP, 60 tokens) vs Flux/SD3 (T5, 200 tokens) |
| LoRA training support | Raw output, hope for the best | Identity suppression, omission cycles, tiered tag protection |
| Dataset workflow focus | Captioning only | Batch processing, deduplication, export to txt/JSON/JSONL/CSV |
| Transparency | Black box | Every removed phrase and compaction decision is reported |

### What the assembly pipeline does

The model is just an input source. The real value is what happens after inference:

1. **Classify** -- every tag and prose clause is bucketed into identity, wardrobe, pose/composition, setting, lighting, or action
2. **Filter** -- redundant prose (>50% overlap with tags), filler prefixes ("The image shows..."), and training noise (rating tags, meta tokens) are stripped
3. **Specialise** -- the training variant suppresses identity traits (the LoRA learns these visually; stating them conflicts with the base model's prior), while the zero-shot variant puts identity first (no LoRA to supply it)
4. **Budget** -- fragments are assembled under the correct CLIP/T5 token limit with tiered protection (framing tags are never dropped, short generic tags are shed first)
5. **Diversify** -- omission cycles systematically suppress different category buckets across images, creating stronger concept disentanglement across the dataset
6. **Enrich** -- novel compound nouns from prose output ("gray sweater", "wooden door") are extracted and appended at lowest priority to fill remaining token budget

## Features

- **Multi-model backends**: WD14, RAM++, Florence-2, BLIP-2 (local GPU/CPU) + OpenAI, HuggingFace, Replicate, NVIDIA NIM, and any OpenAI-compatible server — Ollama, vLLM, LM Studio (cloud or self-hosted API)
- **Structured captions**: Category-bucketed variants (identity, wardrobe, pose, setting, lighting, action)
- **Training-optimised**: Tiered tag protection, omission cycles, CLIP/T5 token budgets, identity suppression
- **Zero-shot variant**: Identity-first, prose-preferred captions for generation without LoRA
- **Hybrid pipelines**: Mix local + cloud backends (e.g. WD14 tags + GPT-4o prose)
- **Backend-aware budgets**: Automatic token limits for SDXL (60), Flux (200), SD3 (200)
- **Deterministic and filterable**: Category-aware outputs you can audit, filter, and override
- **CLI + Server**: Command-line tool and optional FastAPI micro-server
- **Export formats**: `.txt` sidecars, JSON, JSONL, CSV

## Installation

pip handles all Python dependencies through extras. Pick the extras that match your use case:

```bash
# Assembly engine only (no model deps)
pip install argus-lens

# Local backends (GPU inference)
pip install argus-lens[local]      # WD14 + RAM++ + Florence-2 + BLIP-2
pip install argus-lens[wd14]       # WD14 only (CPU, no torch)
pip install argus-lens[wd14-gpu]   # WD14 only, CUDA onnxruntime (no torch)
pip install argus-lens[torch]      # RAM++ / Florence-2 / BLIP-2 only

# Cloud backends (no GPU needed)
pip install argus-lens[openai]     # GPT-4o vision
pip install argus-lens[replicate]  # Replicate API
pip install argus-lens[replay]     # replay recorded captions from cortex (psycopg)
# openai-compat, hf-inference, nvidia-nim need no extra — only the core httpx dep

# CLI (the `argus-lens` command; combine with your backend extras)
pip install argus-lens[cli,openai]

# Server (FastAPI + uvicorn; add [cli] for the `argus-lens serve` command)
pip install argus-lens[cli,server,local,openai]

# Everything
pip install argus-lens[all]
```

If you're adding argus-lens to an existing project, just add e.g. `argus-lens[openai]` to your `requirements.txt` -- pip resolves all transitive deps automatically.

Cloud calls and model downloads verify TLS with `certifi`. The package depends on `certifi>=2026.10.19` so those calls do not trust the compromised eMudhra emSign intermediates ([PSRT notice](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)). That pin does not update `pip` itself: pip older than 26.2.2 carries its own CA bundle and keeps trusting those intermediates, so upgrade pip to 26.2.2 or newer before installing. Lockfile installs need `uv>=0.13.5`.

### System dependencies for local GPU backends

Cloud-only users (`[openai]`, `[replicate]`) need no system packages -- skip this section.

Local backends (`[local]`, `[wd14]`, `[torch]`) require system libraries for image processing and (optionally) CUDA for GPU acceleration. On Ubuntu/Debian:

```bash
sudo apt install -y \
    libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1
```

For GPU inference, you also need:

- NVIDIA GPU drivers (check with `nvidia-smi`)
- CUDA runtime (the `Dockerfile.gpu-base` in this repo uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` as a reference)
- NVIDIA Container Toolkit (for Docker deployment only)

If you already have torch and CUDA working in your environment, you're set -- the pip extras handle the rest.

## Usage

### Python API

Import and use directly in your code. This is the primary interface.

```python
from argus_lens import ArgusLens

# Cloud backend -- works anywhere, no GPU
engine = ArgusLens(backend="openai", api_key="sk-...")
result = engine.caption("photo.jpg", trigger_word="sks_person")

# Local backend -- needs torch + GPU/CPU
engine = ArgusLens(backend="hybrid")
result = engine.caption("photo.jpg", trigger_word="sks_person")

# Self-hosted OpenAI-compatible server (Ollama, vLLM, LM Studio, ...)
# No API key needed for local servers; base_url defaults to Ollama localhost.
engine = ArgusLens(
    backend="openai-compat",
    base_url="http://localhost:11434/v1",
    model_id="llama3.2-vision",
)
result = engine.caption("photo.jpg", trigger_word="sks_person")

# Batch processing
results = engine.caption_directory("./images/", output_format="txt")
```

### CLI

The `argus-lens` command requires the `[cli]` extra (`pip install argus-lens[cli,<backend>]`).

```bash
# Caption a single image
argus-lens caption photo.jpg --trigger sks_person --backend openai

# Caption a directory, output as txt sidecars
argus-lens caption ./images/ --format txt --backend hybrid

# Self-hosted Ollama (or any OpenAI-compatible server)
argus-lens caption photo.jpg --backend openai-compat \
    --base-url http://localhost:11434/v1 --model-id llama3.2-vision

# List available backends
argus-lens backends
```

### Evaluation

Measure caption quality so model swaps, hybrid presets, and future summariser / VQA passes can be judged by numbers instead of vibes. Requires the `[eval]` extra for CLIPScore only (`pip install argus-lens[cli,eval]`); every other metric is dependency-free.

The harness is **reference-free first** — its flagship metric, whether the prose contradicts the tags (the hallucination problem), is an internal-consistency check that needs no ground truth, so it runs on any folder of images:

```bash
# Reference-free: score any image directory
argus-lens eval ./images/ --backend hybrid

# Compare two hybrid presets on the same set
argus-lens eval ./images/ --hybrid-preset keywords -o keywords.json
argus-lens eval ./images/ --hybrid-preset descriptive --baseline keywords.json

# Full run with a labelled golden manifest + CLIPScore, gated for CI
# (copy eval/golden.example.jsonl to eval/golden.jsonl and point it at your images)
argus-lens eval eval/golden.jsonl --clip --baseline baseline.json --fail-on-regression
```

Metrics: **tag↔prose contradiction** (colour/pose — reference-free), **token-budget adherence**, **redundancy/filler rate**, **tag-coverage recall** (labelled), and optional **CLIPScore**. A labelled golden set is a JSONL manifest — see [eval/README.md](eval/README.md) for the format.

### Reconciliation (colour/pose fixes)

Florence-2 sometimes hallucinates a colour or hand position. Reconciliation finds the attributes where the prose *contradicts* the tags (the same detector the eval harness uses) and asks a pluggable **verifier** to adjudicate each, then rewrites the prose to match:

```python
from argus_lens import ArgusLens
from argus_lens.reconcile import build_verifier

# Model-free: trust the (usually more reliable) WD14 tags over the prose
engine = ArgusLens(backend="hybrid", verifier=build_verifier("tag-prior"))

# Or verify against the pixels / a VQA model:
engine = ArgusLens(backend="hybrid", verifier=build_verifier(
    "openai-compat", base_url="http://localhost:11434/v1", model_id="llama3.2-vision"))
```

| Verifier | How it adjudicates | Needs |
|---|---|---|
| `tag-prior` | trusts the tag value (no image) | nothing — default, fully deterministic |
| `openai-compat` | asks a served vision model "what colour is the dress?" | a running OpenAI-compatible endpoint (`httpx` only) |
| `florence` | grounds the subject to a box, samples the pixels | `[torch]` (reuses Florence-2's unused grounding task) |
| `molmo` | asks Molmo, which can point at pixels | `[torch]` + ~8–17 GB VRAM |

Measure the effect with the eval harness — `--reconcile` runs the verifier before scoring, so the contradiction rate shows the improvement directly:

```bash
argus-lens eval ./images/ -o before.json
argus-lens eval ./images/ --reconcile tag-prior --baseline before.json
```

### Refinement (LLM rewrite, validated)

Deterministic assembly always runs first, and every variant passes a coherence post-check that repairs dangling words, orphan sentences, openers, hedging and duplicate tags. For more fluent captions, a `Refiner` can rewrite selected variants with any OpenAI-compatible chat model. A rewrite is **accepted only if it passes the same post-check** plus rules for identity suppression, token budget and grounding in the tagger/prose output. Otherwise the deterministic caption is kept.

`argus-lens serve` and `argus-lens caption` refine **by default**: every variant is rewritten with `qwen3:8b` on local Ollama (`ARGUS_REFINE_MODEL`, `ARGUS_REFINE_BASE_URL`, `ARGUS_REFINE_API_KEY` to point elsewhere; `ARGUS_REFINE_VARIANTS` or `--refine-variants` to narrow it). Opt out with `ARGUS_REFINE=off` or `--no-refine`. If the endpoint is unreachable, captioning carries on deterministically and retries refinement after a 60 s cooldown. For CLIP targets (SDXL), `training_prose` rewrites are held to one or two short sentences.

```python
from argus_lens.refine import OpenAICompatRewriter, Refiner

refiner = Refiner(OpenAICompatRewriter(model_id="qwen3:8b"), variants=("training", "training_prose"))
engine = ArgusLens(backend="hybrid", refiner=refiner)
```

```bash
argus-lens caption ./images/ --recursive --refine-model qwen3:14b   # refine is on by default; --no-refine to skip
```

`result.metadata["refine"]` records what was accepted or rejected, and why.

### Dataset curation workflow

[`workflows/`](workflows/README.md) contains a complete, resumable pipeline for a large people-photo library (`<name>/pics/<shoot_id>/*.jpg`). It captions each image once, groups images by shared identity, wardrobe, setting and action tags, flags multi-person images for face ID, and exports SDXL and Flux training datasets. It builds on these result fields:

- `metadata["buckets"]`: un-truncated per-category tags and prose, with no-match tags kept apart in `unclassified`
- `metadata["subjects"]`: a headcount estimate (`count`, `multiple`, `agree`, `evidence`)
- `caption_variants["training_prose"]`: sentence-form training captions for T5 models (Flux, SD3), next to the tag-form `training` caption for CLIP models (SDXL)

**Photo libraries: use RAM++ for tags.** WD14 is a Danbooru tagger, right for booru/anime checkpoints but a poor index for everyday photos. `ram` is [RAM++](https://github.com/xinyu1205/recognize-anything) (Recognize Anything Plus): 4585 photo-domain tags ("beach", "christmas market", "wedding dress") with calibrated per-tag thresholds, emitted with scores. Run it alone (`ARGUS_BACKEND=ram`) or as the tag stage of a hybrid (`hybrid:ram+florence2`, `hybrid:ram+openai-compat`). It needs only `[torch]` (~1 GB VRAM in fp16, ~50 ms/image on an RTX 3090). The ~3 GB checkpoint downloads to `~/.cache/ram_plus/` on first use (override with `RAM_MODEL_DIR`), and `threshold=` swaps the calibrated cut-offs for one uniform value.

**Custom RAM++ tags (vocabulary packs).** RAM++ stores each tag as 51 CLIP-embedded text descriptions, so you can add tags it doesn't know without retraining. `argus-lens ram-vocab build tags.txt --out packs/fashion --name fashion` writes 51 descriptions per tag with any OpenAI-compatible LLM (the `ARGUS_REFINE_*` endpoint by default, e.g. a local Ollama model; same prompts as upstream), embeds them with CLIP ViT-B/16, and writes a pack (`manifest.json` + `embeddings.safetensors`, about 70 KB per tag). A tag list is one tag per line, optionally `tag<TAB>threshold` (default 0.5); a descriptions JSON (`[{"tag": ["desc", ...]}]`, upstream's format) skips the LLM step. Use it with `RamBackend(vocab_packs=[...])` or `RAM_VOCAB_PACKS=packs/fashion`. Pack tags extend the built-in 4585 (a pack tag that duplicates a built-in one replaces it), and report `source="ram:<pack>"` in provenance. `builtin_vocab=False` tags with the packs alone. `argus-lens ram-vocab inspect <pack>` summarises one. Built from upstream's 200 OpenImages-Uncommon descriptions, a pack reproduces 5 of the 6 tags in upstream's published open-set example; the sixth scores 0.498 against the 0.5 threshold.

Newer models plug in directly: `--wd14-repo SmilingWolf/wd-eva02-large-tagger-v3` for tags, and `hybrid:wd14+openai-compat --model-id qwen3-vl:8b-instruct --prompt straightforward` for prose from Qwen3-VL, JoyCaption or any served VLM.

### GPU sharing (lifecycle + coordinator)

When argus-lens shares a GPU with another process (e.g. an image generator), it can behave as a good tenant. Two independent, opt-in mechanisms:

**Lifecycle** — the model loads lazily and can be freed on demand:

```python
engine = ArgusLens(backend="hybrid", idle_unload_s=120)  # auto-unload after 2 min idle
engine.caption("photo.jpg")
engine.unload()          # free VRAM now; reloads lazily on the next caption
engine.vram_status()     # {backend, loaded, coordinator, free_vram_mb, idle_s}
```

The server exposes `GET /health` (now includes a `gpu` block) and `POST /admin/unload`, so an external scheduler can free lens's VRAM on demand.

**Coordinator** — gate heavy inference behind the suite's shared GPU lease ([`argus_cortex.gpu`](https://github.com/smk762/argus-cortex)), so lens, argus-curator and any other argus GPU service on the host take turns. Selected via env (defaults to `none`, i.e. unchanged behaviour):

| `ARGUS_GPU_COORDINATOR` | Behaviour | Needs |
|---|---|---|
| `none` (default) | no gating | — |
| `lease` | one argus GPU job at a time via a host-wide file lock, started only once the card has room | POSIX; NVML (`nvidia-ml-py`, pulled in by the `local`/`torch`/`wd14-gpu` extras) for the VRAM wait |

```bash
# Share the GPU with argus-curator: both name the same lock file
ARGUS_GPU_COORDINATOR=lease ARGUS_GPU_LEASE_PATH=/run/argus-gpu/argus-gpu.lock argus-lens serve
```

The lease takes the lock, then waits until NVML reports enough free VRAM, which accounts for tenants that never take the lock (ComfyUI, Ollama). Lens asks for its per-backend VRAM estimate only when the model still has to load; once it is resident the lease just takes the lock. Without NVML (or on a CPU host) the VRAM wait is skipped and the lease is the lock alone. Cloud backends (zero VRAM footprint) bypass the lease, and OOM retry remains the reactive backstop.

The default lock is `<tempdir>/argus-gpu.lock`, the same file argus-curator uses. Containers each have their own `/tmp`, so in Docker point `ARGUS_GPU_LEASE_PATH` at a directory every GPU service mounts (see `docker-compose.yaml`).

Extra knobs (all optional, and host-wide: every argus service should agree on them): `ARGUS_GPU_MIN_VRAM_MB` overrides the per-backend footprint the lease waits for (e.g. for Florence-2-large or a non-default hybrid); `ARGUS_GPU_LEASE_TIMEOUT_S` caps how long the lease waits (default 300 s, raising `GpuLeaseTimeout`, a `TimeoutError`); `ARGUS_GPU_LEASE_PATH` sets the lock file; `ARGUS_GPU_DEVICE` picks the card the VRAM wait probes. `ARGUS_ADMIN_TOKEN` protects `POST /admin/unload`.

### Replay backend (recorded captions, no GPU)

The `replay` backend serves captions straight out of the [cortex](https://github.com/smk762/argus-cortex) Postgres lineage store — the `source_asset → caption` DAG a real pipeline run recorded once. It loads **no model and makes no external API call**, so a GPU-free box (or the public [argus-halo](https://github.com/smk762/argus-halo) demo) can re-enact genuine captured output.

```bash
pip install argus-lens[replay,cli,server]
export CORTEX_PG_URL=postgresql://user:pass@host/db
argus-lens serve --backend replay      # or ARGUS_BACKEND=replay
```

Each request is keyed by the source image's **sha256** — the same content id cortex stores on `source_asset` — so the engine hashes the bytes it ingests and returns the recorded `CaptionResult` **verbatim** (variants included), bypassing the assembly pipeline; profile knobs like `target_style` don't re-mangle captured output. An image with no recorded caption is a `404` (`ReplayMiss`) rather than a silent fall back to a live model. The caption text lives in Postgres, so the blob store (`CORTEX_S3_*`) isn't needed for replay.

#### Recording the tape

Lens writes the tape itself. With `LENS_RECORD_LINEAGE=1` and `CORTEX_PG_URL` set, every **live** caption (from `serve`'s caption endpoints or `argus-lens caption`) is recorded as a `source_asset` (keyed on the original bytes' sha256, the key replay looks up) plus a `caption` row with its variants, raw tags/prose, backend and target profile. This is what `--backend replay`, [argus-lightbox](https://github.com/smk762/argus-lightbox) and the argus-halo tape read. With `CORTEX_S3_ENDPOINT`/`CORTEX_S3_BUCKET` also set, the original image bytes go to the cortex blob store under the same sha256, so a GPU-free host can show the image too.

```bash
pip install argus-lens[lineage,cli,server,local]
export CORTEX_PG_URL=postgresql://user:pass@host/db LENS_RECORD_LINEAGE=1
argus-lens caption ./dataset/            # each image's caption lands in the lineage store
```

Recording is opt-in through its own flag, not `CORTEX_PG_URL` alone, because a replay host reads that URL and must never write into the tape it serves; replayed captions are never recorded. The flag set without `CORTEX_PG_URL` is a startup error. A failed write is logged and the caption still returned, and a pre-decoded PIL input (no original bytes, so no sha256) is not recorded. The OpenAI-compatible `/v1` endpoints (Frigate) never record.

### HTTP Server

Run the built-in FastAPI server for frontend consumers (e.g. [argus-studio](https://github.com/smk762/argus-studio)):

```bash
pip install argus-lens[cli,server,local]
argus-lens serve --cors --port 8080
```

Endpoints:

- `POST /caption` -- multipart file upload
- `POST /caption/url` -- JSON body with image URL
- `POST /caption/batch` -- multiple file upload
- `POST /caption/stream` -- NDJSON streaming for batch
- `POST /caption/manifest` -- batch-caption an [argus-curator](https://github.com/smk762/argus-curator) JSONL manifest (shared `target_profile`, writes `.txt` sidecars; `write_xmp: true` also writes `.xmp` sidecars). Understands manifest 1.x and 2.x (unknown majors are a 400): pass the curator's export root as `export_root` and 2.0 rows are read from `export_root / exported_path` (valid even for `mode=move` exports); without it, rows fall back to `abs_path` (shared volume)
- `POST /caption/manifest/stream` -- streaming variant of `/caption/manifest`: one NDJSON progress line per image, then a completion summary (supports `write_xmp` and `export_root` too)
- `POST /caption/folder` -- batch-caption every image in a folder under the source root (optionally recursive, writes `.txt` sidecars; `write_xmp: true` also writes `.xmp` sidecars)
- `GET /folders?path=<rel>` -- browse folders under `--source-root` / `LENS_SOURCE_PATH` (for the UI folder picker)
- `GET /backends` -- list available backends
- `GET /health` -- liveness probe: `{status, service, version, source_root}`
- `GET /profiles` -- caption taxonomy for UIs: `{assembly_profiles, target_styles, target_categories, target_backends, token_budgets}`
- `GET /immich/albums` -- list Immich albums (`{albums: [{id, name, asset_count}]}`); requires `IMMICH_URL` + `IMMICH_API_KEY`
- `POST /immich/pull` -- download an Immich album (or selected `asset_ids`) into a folder under the source root; NDJSON progress stream, atomic concurrent downloads, skips existing files (in-request filename collisions are per-asset errors, and non-captionable originals like HEIC/DNG get a `warning`)
- `POST /immich/caption/stream` -- caption Immich album assets in memory (NDJSON progress); `write_back: true` pushes captions back to Immich (`write_xmp` is rejected here -- assets never touch disk; pull first, then `/caption/folder`)
- `POST /v1/chat/completions` -- OpenAI-compatible endpoint (always mounted; usable as a Frigate GenAI provider)

The `/immich/*` endpoints read `IMMICH_URL` and `IMMICH_API_KEY` from the environment at request time; if either is unset they return `503` and the rest of the server is unaffected.

**CORS and cross-site writes.** `--cors` allows only the localhost dev frontend (`http://localhost:3000`, `http://127.0.0.1:3000`). Allow-list other origins with `--cors-origin <origin>` (repeatable) or `LENS_CORS_ORIGINS` (comma-separated), or use `--cors-any` for a credential-less wildcard on public demos. A literal `*` in the allow-list is treated as `--cors-any`.

The caption endpoints have no auth and write sidecars, and CORS is not a write boundary: a `multipart/form-data` POST needs no preflight, so any page a browser visits could drive one. So every unsafe request (POST/PUT/PATCH/DELETE) is checked by Origin, with or without `--cors`: no `Origin` (curl, argus-curator, Frigate) passes, the same host as the request (a UI proxied onto this host) passes, and so does an allow-listed origin; anything else is a `403`. `--cors-any` therefore grants reads, not writes: name an origin with `--cors-origin` to let it write. The guard is the suite's shared [`argus_cortex.server.WriteGuard`](https://github.com/smk762/argus-cortex), as in argus-curator, argus-forge and argus-proof.

### Immich

Immich is strong at CLIP search and faces but weak at descriptive keywords and captions -- the gap Argus Lens fills. Immich has no in-process ML plugin hook, so Argus Lens runs as a companion service: pull assets via the Immich API, caption them, and push keywords + description back.

Create an API key in Immich under Account Settings -> API Keys, then:

```python
from argus_lens import ArgusLens
from argus_lens.connectors import ImmichSink, ImmichSource

IMMICH_URL = "http://immich.local:2283"
API_KEY = "..."

source = ImmichSource(IMMICH_URL, API_KEY)
sink = ImmichSink(IMMICH_URL, API_KEY)
engine = ArgusLens(backend="hybrid")

# Pass `since` (ISO 8601) to only process assets changed after your last run.
for ref in source.list_assets(since="2026-07-01T00:00:00Z"):
    result = engine.caption(source.fetch_image(ref))
    keywords = [t.strip() for t in result.raw_tags.split(",") if t.strip()]
    sink.write(ref, keywords=keywords, description=result.caption_variants["zeroshot"])
```

Keywords are upserted as Immich tags (existing tags are reused) and attached to the asset; the description lands in the asset's description field, both searchable in the Immich UI. Writes are idempotent, so re-running over the same assets is safe.

If you'd rather keep Argus Lens decoupled from the Immich API entirely, use `XmpSink` instead: it writes standard `.xmp` sidecars next to your originals, which Immich (as well as Lightroom and digiKam) ingests on library scan. The captioning endpoints expose this directly: pass `write_xmp: true` to `/caption/folder`, `/caption/manifest`, or `/caption/manifest/stream` and each image gets an `<image>.xmp` sidecar with the raw tags as `dc:subject` keywords and the final caption as `dc:description` -- no Immich/Lightroom/digiKam API integration required, just point their library scan at the folder. Existing `.xmp` files are **merged**: lens replaces only those two properties and keeps everything else (ratings, GPS, develop settings, curator face regions); pass `xmp_overwrite: true` to discard the existing sidecar instead. A sidecar that isn't valid XMP is reported as a per-image error and left untouched. The XMP implementation is shared across the suite via [`argus_cortex.xmp`](https://github.com/smk762/argus-cortex).

### Docker

For fresh hosts or isolated deployment with GPU passthrough. No pip install needed on the host.

```bash
# Build and run
./build-docker.sh
docker compose up
```

This builds a CUDA 12.4 base image, installs all extras into it, and runs `argus-lens serve` on port 8080. The image distrusts emSign ECC Root CA - G3 and the Comodo AAA cross-sign in the system trust store, and installs hash-pinned pip 26.2.2 before project dependencies. Rebuild without layer cache after pulling these Dockerfiles (`docker build --no-cache`, or `docker compose build --no-cache`).

#### Standalone image (GHCR)

A self-contained image built from `Dockerfile.standalone` is published to GHCR on each release — no local build needed:

```bash
docker run -p 8100:8100 -v /path/to/images:/data ghcr.io/smk762/argus-lens:latest
```

Folder browsing and captioning are confined to the mounted `/data` (override with `LENS_SOURCE_PATH`).

#### Configuration

Copy or create a `.env` file for the Docker deployment:

| Variable | Default | Description |
|---|---|---|
| `ARGUS_BACKEND` | `hybrid` | Captioning backend (`hybrid`, `wd14`, `ram`, `florence2`, `openai`, `hybrid:ram+florence2`, etc.) |
| `OPENAI_API_KEY` / `REPLICATE_API_TOKEN` / `HF_TOKEN` / `NVIDIA_API_KEY` | -- | API key for the matching cloud backend (each backend reads its own variable) |
| `IMMICH_URL` / `IMMICH_API_KEY` | -- | Immich server URL + API key for the `/immich/*` endpoints (unset: those endpoints return 503) |
| `ARGUS_PORT` | `8080` | Host port for the server |
| `WD14_MODEL_DIR` | `~/.cache/wd14_tagger/` | WD14 ONNX model directory (auto-downloads on first use) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache (auto-downloads on first use) |
| `HF_TRUST_REMOTE_CODE` | `false` | Only needed for legacy `microsoft/Florence-2-*` weights. See [Security](#security) |
| `LENS_CORS_ORIGINS` | -- | Comma-separated origins allowed CORS and cross-site writes (the image runs `--cors`, which trusts only `localhost:3000`) |
| `LENS_SOURCE_PATH` | `/data/images` | Host image folder mounted read-write at `/data/images` for the folder picker and `POST /caption/folder`; `.txt` sidecars are written next to the images |

#### GPU prerequisites

```bash
# Verify NVIDIA driver
nvidia-smi

# Install container toolkit (if not already)
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Model caching

The `docker-compose.yaml` bind-mounts `~/.cache/wd14_tagger`, `~/.cache/ram_plus` and `~/.cache/huggingface` from the host so models persist across container rebuilds. Models auto-download on first use if not already cached.

## Security

### `trust_remote_code` and Florence-2

By default, the Florence-2 backend uses [`florence-community/Florence-2-base`](https://huggingface.co/florence-community/Florence-2-base) weights which are natively supported in `transformers` -- no `trust_remote_code` needed.

The legacy [`microsoft/Florence-2-base`](https://huggingface.co/microsoft/Florence-2-base) weights require `HF_TRUST_REMOTE_CODE=true`, which executes arbitrary Python from the model repository at load time. Only enable this for models you trust. WD14 uses a static ONNX model and never runs remote code.

## Related projects

- [argus-studio](https://github.com/smk762/argus-studio) -- a thin Next.js web UI for exploring argus-lens interactively.
- [awesome-immich](https://github.com/tlwhittaker/awesome-immich) -- a curated list of Immich plugins, tools, and community projects. Argus Lens integrates with Immich as a companion tagging/captioning service -- see [Immich](#immich).

## License

MIT
