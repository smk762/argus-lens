# Argus Lens

Structured image captioning for training and generation.

## Quick Start

```bash
pip install argus-lens[all]
```

```python
from argus_lens import ArgusLens

engine = ArgusLens(backend="hybrid")
result = engine.caption("photo.jpg", trigger_word="sks_person")

print(result.final_caption)
print(result.caption_variants["training"])
print(result.caption_variants["zeroshot"])
```

## Features

- **Multi-model backends**: WD14, BLIP-2, Florence-2 (local GPU/CPU) + OpenAI, HuggingFace, Replicate, NVIDIA NIM (cloud API)
- **Structured captions**: Category-bucketed variants (identity, wardrobe, pose, setting, lighting, action)
- **Training-optimised**: Tiered tag protection, omission cycles, CLIP/T5 token budgets, identity suppression
- **Zero-shot variant**: Identity-first, prose-preferred captions for generation without LoRA
- **Hybrid pipelines**: Mix local + cloud backends (e.g. WD14 tags + GPT-4o prose)
- **Backend-aware budgets**: Automatic token limits for SDXL (60), Flux (200), SD3 (200)
- **CLI + Server**: Command-line tool and optional FastAPI micro-server
- **Export formats**: `.txt` sidecars, JSON, JSONL, CSV

## Installation

```bash
# Assembly engine only (no model deps)
pip install argus-lens

# Local backends
pip install argus-lens[local]      # WD14 + BLIP-2 + Florence-2
pip install argus-lens[wd14]       # WD14 only (CPU, no torch)
pip install argus-lens[torch]      # BLIP-2 + Florence-2

# Cloud backends
pip install argus-lens[openai]     # GPT-4o vision

# Everything
pip install argus-lens[all]
```

## License

MIT
