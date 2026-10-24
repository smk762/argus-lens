"""Caption refinement — LLM rewrite of assembled captions, validated deterministically.

Deterministic assembly produces every caption first. A :class:`Refiner`
then asks a :class:`CaptionRewriter` (any OpenAI-compatible chat model) to
rewrite selected variants into fluent text, and accepts a rewrite only if it
passes the coherence post-check (trigger first, no suppressed identity
terms, inside the token budget, grounded in the inputs). Otherwise the
deterministic caption stays.

    from argus_lens import ArgusLens
    from argus_lens.refine import Refiner, OpenAICompatRewriter

    refiner = Refiner(OpenAICompatRewriter(model_id="qwen3:8b"))
    engine = ArgusLens(backend="hybrid", refiner=refiner)
"""

from __future__ import annotations

from argus_lens.refine.openai_compat import OpenAICompatRewriter, parse_rewrite
from argus_lens.refine.refiner import (
    IDENTITY_FORBIDDEN,
    VARIANT_SPECS,
    CaptionRewriter,
    Refiner,
    RefineRequest,
    VariantSpec,
    refiner_from_env,
    variant_spec,
)

__all__ = [
    "IDENTITY_FORBIDDEN",
    "VARIANT_SPECS",
    "CaptionRewriter",
    "OpenAICompatRewriter",
    "RefineRequest",
    "Refiner",
    "VariantSpec",
    "parse_rewrite",
    "refiner_from_env",
    "variant_spec",
]
