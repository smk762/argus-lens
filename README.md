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

- **Multi-model backends**: WD14, Florence-2, BLIP-2 (local GPU/CPU) + OpenAI, HuggingFace, Replicate, NVIDIA NIM, and any OpenAI-compatible server — Ollama, vLLM, LM Studio (cloud or self-hosted API)
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
pip install argus-lens[local]      # WD14 + Florence-2 + BLIP-2
pip install argus-lens[wd14]       # WD14 only (CPU, no torch)
pip install argus-lens[wd14-gpu]   # WD14 only, CUDA onnxruntime (no torch)
pip install argus-lens[torch]      # Florence-2 / BLIP-2 only

# Cloud backends (no GPU needed)
pip install argus-lens[openai]     # GPT-4o vision
pip install argus-lens[replicate]  # Replicate API
# openai-compat, hf-inference, nvidia-nim need no extra — only the core httpx dep

# CLI (the `argus-lens` command; combine with your backend extras)
pip install argus-lens[cli,openai]

# Server (FastAPI + uvicorn; add [cli] for the `argus-lens serve` command)
pip install argus-lens[cli,server,local,openai]

# Everything
pip install argus-lens[all]
```

If you're adding argus-lens to an existing project, just add e.g. `argus-lens[openai]` to your `requirements.txt` -- pip resolves all transitive deps automatically.

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
- `POST /caption/manifest` -- batch-caption an [argus-curator](https://github.com/smk762/argus-curator) JSONL manifest (shared `target_profile`, writes `.txt` sidecars; `write_xmp: true` also writes `.xmp` sidecars)
- `POST /caption/manifest/stream` -- streaming variant of `/caption/manifest`: one NDJSON progress line per image, then a completion summary (supports `write_xmp` too)
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

If you'd rather keep Argus Lens decoupled from the Immich API entirely, use `XmpSink` instead: it writes standard `.xmp` sidecars next to your originals, which Immich (as well as Lightroom and digiKam) ingests on library scan. The captioning endpoints expose this directly: pass `write_xmp: true` to `/caption/folder`, `/caption/manifest`, or `/caption/manifest/stream` and each image gets an `<image>.xmp` sidecar with the raw tags as `dc:subject` keywords and the final caption as `dc:description` -- no Immich/Lightroom/digiKam API integration required, just point their library scan at the folder. Existing `.xmp` files are replaced by default; pass `xmp_overwrite: false` to report them as per-image errors instead (XMP writes never merge, so this protects sidecars other tools populated).

### Docker

For fresh hosts or isolated deployment with GPU passthrough. No pip install needed on the host.

```bash
# Build and run
./build-docker.sh
docker compose up
```

This builds a CUDA 12.4 base image, installs all extras into it, and runs `argus-lens serve` on port 8080.

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
| `ARGUS_BACKEND` | `hybrid` | Captioning backend (`hybrid`, `wd14`, `florence2`, `openai`, etc.) |
| `OPENAI_API_KEY` / `REPLICATE_API_TOKEN` / `HF_TOKEN` / `NVIDIA_API_KEY` | -- | API key for the matching cloud backend (each backend reads its own variable) |
| `IMMICH_URL` / `IMMICH_API_KEY` | -- | Immich server URL + API key for the `/immich/*` endpoints (unset: those endpoints return 503) |
| `ARGUS_PORT` | `8080` | Host port for the server |
| `WD14_MODEL_DIR` | `~/.cache/wd14_tagger/` | WD14 ONNX model directory (auto-downloads on first use) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache (auto-downloads on first use) |
| `HF_TRUST_REMOTE_CODE` | `false` | Only needed for legacy `microsoft/Florence-2-*` weights. See [Security](#security) |

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

The `docker-compose.yaml` bind-mounts `~/.cache/wd14_tagger` and `~/.cache/huggingface` from the host so models persist across container rebuilds. Models auto-download on first use if not already cached.

## Security

### `trust_remote_code` and Florence-2

By default, the Florence-2 backend uses [`florence-community/Florence-2-base`](https://huggingface.co/florence-community/Florence-2-base) weights which are natively supported in `transformers` -- no `trust_remote_code` needed.

The legacy [`microsoft/Florence-2-base`](https://huggingface.co/microsoft/Florence-2-base) weights require `HF_TRUST_REMOTE_CODE=true`, which executes arbitrary Python from the model repository at load time. Only enable this for models you trust. WD14 uses a static ONNX model and never runs remote code.

## Related projects

- [argus-studio](https://github.com/smk762/argus-studio) -- a thin Next.js web UI for exploring argus-lens interactively.
- [awesome-immich](https://github.com/tlwhittaker/awesome-immich) -- a curated list of Immich plugins, tools, and community projects. Argus Lens integrates with Immich as a companion tagging/captioning service -- see [Immich](#immich).

## License

MIT
