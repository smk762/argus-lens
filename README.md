# Argus Lens

Structured image captioning for training and generation.

> **Looking for the web UI?** See [argus-vision-demo](https://github.com/smk762/argus-vision-demo) -- a thin Next.js frontend for exploring argus-lens interactively.

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

## Features

- **Multi-model backends**: WD14, Florence-2 (local GPU/CPU) + OpenAI, HuggingFace, Replicate, NVIDIA NIM (cloud API)
- **Structured captions**: Category-bucketed variants (identity, wardrobe, pose, setting, lighting, action)
- **Training-optimised**: Tiered tag protection, omission cycles, CLIP/T5 token budgets, identity suppression
- **Zero-shot variant**: Identity-first, prose-preferred captions for generation without LoRA
- **Hybrid pipelines**: Mix local + cloud backends (e.g. WD14 tags + GPT-4o prose)
- **Backend-aware budgets**: Automatic token limits for SDXL (60), Flux (200), SD3 (200)
- **CLI + Server**: Command-line tool and optional FastAPI micro-server
- **Export formats**: `.txt` sidecars, JSON, JSONL, CSV

## Installation

pip handles all Python dependencies through extras. Pick the extras that match your use case:

```bash
# Assembly engine only (no model deps)
pip install argus-lens

# Local backends (GPU inference)
pip install argus-lens[local]      # WD14 + Florence-2
pip install argus-lens[wd14]       # WD14 only (CPU, no torch)
pip install argus-lens[torch]      # Florence-2 only

# Cloud backends (no GPU needed)
pip install argus-lens[openai]     # GPT-4o vision
pip install argus-lens[replicate]  # Replicate API

# Server (FastAPI + uvicorn)
pip install argus-lens[server,local,openai]

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

# Batch processing
results = engine.caption_directory("./images/", output_format="txt")
```

### CLI

```bash
# Caption a single image
argus-lens caption photo.jpg --trigger sks_person --backend openai

# Caption a directory, output as txt sidecars
argus-lens caption ./images/ --format txt --backend hybrid

# List available backends
argus-lens backends
```

### HTTP Server

Run the built-in FastAPI server for frontend consumers (e.g. [argus-vision-demo](https://github.com/smk762/argus-vision-demo)):

```bash
pip install argus-lens[server,local]
argus-lens serve --cors --port 8080
```

Endpoints:

- `POST /caption` -- multipart file upload
- `POST /caption/url` -- JSON body with image URL
- `POST /caption/batch` -- multiple file upload
- `POST /caption/stream` -- NDJSON streaming for batch
- `GET /backends` -- list available backends

### Docker

For fresh hosts or isolated deployment with GPU passthrough. No pip install needed on the host.

```bash
# Build and run
./build-docker.sh
docker compose up
```

This builds a CUDA 12.4 base image, installs all extras into it, and runs `argus-lens serve` on port 8080.

#### Configuration

Copy or create a `.env` file for the Docker deployment:

| Variable | Default | Description |
|---|---|---|
| `ARGUS_BACKEND` | `hybrid` | Captioning backend (`hybrid`, `wd14`, `florence2`, `openai`, etc.) |
| `ARGUS_API_KEY` | -- | API key for cloud backends |
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

## License

MIT
