"""Opt-in live eval of the default prose prompt on a small VLM (#104).

Skipped unless ``LENS_LIVE_EVAL_DIR`` names a folder of photos (about 20) and an
OpenAI-compatible VLM is reachable at ``ARGUS_OPENAI_COMPAT_BASE_URL`` (default
Ollama) serving ``ARGUS_OPENAI_COMPAT_MODEL`` (e.g. ``qwen3-vl:2b-instruct``)::

    LENS_LIVE_EVAL_DIR=~/eval-photos ARGUS_OPENAI_COMPAT_MODEL=qwen3-vl:2b-instruct \\
        uv run --no-sync pytest tests/test_prompt_eval_live.py -q -s

CI never sets the variable, so this never runs there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from PIL import Image

from argus_lens.assembly.composer import compose_caption_result
from argus_lens.backends.openai_compat import OpenAICompatBackend
from argus_lens.connectors.filesystem import IMAGE_SUFFIXES
from argus_lens.types import CaptionTargetProfile

_DIR = os.environ.get("LENS_LIVE_EVAL_DIR")
pytestmark = pytest.mark.skipif(not _DIR, reason="set LENS_LIVE_EVAL_DIR to run the live prompt eval")

# The word lists from #104's measurements.
_MOOD_META = re.compile(
    r"\b(?:vibrant|serene|peaceful|tranquil|striking|stunning|picturesque|majestic|idyllic|evokes?|atmosphere|"
    r"sense of|overall|creating a|showcasing|highlighting|juxtaposition|this is a|this image|the image|"
    r"the photo|the composition|the viewer)\b",
    re.I,
)
_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s")
_MAX_TOKENS = 300


def test_default_prompt_live() -> None:
    """Mean prose <= ~600 chars, few mood/meta words, ~none in zeroshot, no empties or numbered lists."""
    paths = sorted(p for p in Path(_DIR).expanduser().iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:20]
    assert paths, f"no images in {_DIR}"
    backend = OpenAICompatBackend(max_tokens=_MAX_TOKENS)
    profile = CaptionTargetProfile(target_style="photo", target_backend="sdxl", target_category="identity")
    lengths, prose_hits, zeroshot_hits = [], 0, 0
    for path in paths:
        prose = backend.caption_image(Image.open(path))
        assert prose.strip(), f"{path.name}: empty prose"
        assert not _NUMBERED.search(prose), f"{path.name}: numbered list"
        # ~4 chars per token: a reply this long almost certainly hit max_tokens.
        assert len(prose) < _MAX_TOKENS * 4 * 0.95, f"{path.name}: reply looks cut off at max_tokens"
        result = compose_caption_result(trigger_word="", prose=prose, target_profile=profile)
        lengths.append(len(prose))
        prose_hits += len(_MOOD_META.findall(prose))
        zeroshot_hits += len(_MOOD_META.findall(result.caption_variants["zeroshot"]))
    n = len(paths)
    print(
        f"\n{n} photos: mean prose {sum(lengths) / n:.0f} chars, mood/meta {prose_hits / n:.2f}/photo raw, "
        f"{zeroshot_hits / n:.2f}/photo zeroshot"
    )
    assert sum(lengths) / n <= 600
    assert prose_hits / n <= 1.0
    assert zeroshot_hits / n <= 0.1
