"""Attribute verifiers and a factory to build them by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from argus_lens.reconcile.verifiers.florence import FlorenceGroundingVerifier
from argus_lens.reconcile.verifiers.molmo import MolmoVerifier
from argus_lens.reconcile.verifiers.openai_compat import OpenAICompatVQAVerifier
from argus_lens.reconcile.verifiers.tag_prior import TagPriorVerifier

if TYPE_CHECKING:
    from argus_lens.reconcile.types import AttributeVerifier

VERIFIER_NAMES: tuple[str, ...] = ("tag-prior", "openai-compat", "florence", "molmo")


def build_verifier(
    name: str,
    *,
    base_url: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    device: str = "cuda",
) -> AttributeVerifier:
    """Construct a verifier by name.

    ``tag-prior`` is model-free; ``openai-compat`` needs *base_url*/*model_id*
    (defaults target Ollama); ``florence`` and ``molmo`` load local models on
    first use. Unknown names raise ``ValueError``.
    """
    key = name.strip().lower()
    if key == "tag-prior":
        return TagPriorVerifier()
    if key == "openai-compat":
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if model_id:
            kwargs["model_id"] = model_id
        return OpenAICompatVQAVerifier(**kwargs)
    if key == "florence":
        return FlorenceGroundingVerifier(model_id=model_id or "florence-community/Florence-2-base", device=device)
    if key == "molmo":
        return MolmoVerifier(model_id=model_id or "allenai/Molmo-7B-D-0924", device=device)
    raise ValueError(f"Unknown verifier {name!r}. Choose from: {', '.join(VERIFIER_NAMES)}")


__all__ = [
    "VERIFIER_NAMES",
    "FlorenceGroundingVerifier",
    "MolmoVerifier",
    "OpenAICompatVQAVerifier",
    "TagPriorVerifier",
    "build_verifier",
]
