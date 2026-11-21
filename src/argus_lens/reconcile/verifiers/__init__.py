"""Attribute verifiers and a factory to build them by name."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from argus_lens.reconcile.verifiers.florence import FlorenceGroundingVerifier
from argus_lens.reconcile.verifiers.molmo import MolmoVerifier
from argus_lens.reconcile.verifiers.ocr import OCR_URL_ENV, OcrVerifier, ocr_url_from_env
from argus_lens.reconcile.verifiers.openai_compat import OpenAICompatVQAVerifier
from argus_lens.reconcile.verifiers.routing import TextRoutingVerifier
from argus_lens.reconcile.verifiers.tag_prior import TagPriorVerifier

if TYPE_CHECKING:
    from argus_lens.reconcile.types import AttributeVerifier

VERIFIER_NAMES: tuple[str, ...] = ("tag-prior", "openai-compat", "florence", "molmo", "ocr")
TEXT_READER_NAMES: tuple[str, ...] = ("ocr", "florence", "openai-compat", "molmo")  # verifiers that read text
TEXT_VERIFIER_ENV = "LENS_TEXT_VERIFIER"  # which reader checks text claims; overridden by --text-verifier


def build_verifier(
    name: str,
    *,
    base_url: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    device: str = "cpu",
    text_verifier: str | None = None,
    ocr_url: str | None = None,
) -> AttributeVerifier:
    """Construct a verifier by name, with an optional separate reader for ``text`` claims (#141).

    ``tag-prior`` is model-free; ``openai-compat`` needs *base_url*/*model_id*
    (defaults target Ollama); ``florence`` and ``molmo`` load local models on
    first use. *device* defaults to ``"cpu"`` (Florence-2-base runs there; pass
    ``"cuda"`` for GPU / Molmo). ``ocr`` is argus-retina's OCR server at
    *ocr_url* (or ``LENS_OCR_URL``) and answers only text claims.

    The text reader is *text_verifier*, else ``LENS_TEXT_VERIFIER``, else
    ``ocr`` when an OCR URL is configured, else *name* itself. A reader other
    than *name* gets the text claims first; when it fails, *name* reads them
    if it can (``tag-prior`` can't), so the check is not skipped. Unknown names
    raise ``ValueError``.
    """
    ocr_url = (ocr_url or "").strip() or ocr_url_from_env()
    text = (text_verifier or os.environ.get(TEXT_VERIFIER_ENV, "")).strip().lower() or ("ocr" if ocr_url else "")
    main = _build_one(name, base_url=base_url, model_id=model_id, api_key=api_key, device=device, ocr_url=ocr_url)
    if not text or text == main.name:
        return main
    # *model_id* names one model. It is the main verifier's when that one loads a model; a reader of another
    # kind then takes its own default, since a Florence id is no Ollama tag and the reverse.
    reader_model = None if main.name in _MODEL_VERIFIERS else model_id
    reader = _build_one(text, base_url=base_url, model_id=reader_model, api_key=api_key, device=device, ocr_url=ocr_url)
    if not getattr(reader, "reads_text", False):
        raise ValueError(f"text verifier {text!r} can't read text. Choose from: {', '.join(TEXT_READER_NAMES)}")
    readers = [reader, main] if getattr(main, "reads_text", False) else [reader]
    return TextRoutingVerifier(main, readers)


_MODEL_VERIFIERS: frozenset[str] = frozenset({"openai-compat", "florence", "molmo"})  # verifiers *model_id* configures


def _build_one(
    name: str,
    *,
    base_url: str | None,
    model_id: str | None,
    api_key: str | None,
    device: str,
    ocr_url: str | None,
) -> AttributeVerifier:
    """Construct one verifier by name; unknown names raise ``ValueError``."""
    key = name.strip().lower()
    if key == "ocr":
        return OcrVerifier(ocr_url)
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
        return FlorenceGroundingVerifier(model_id=model_id, device=device)
    if key == "molmo":
        return MolmoVerifier(model_id=model_id or "allenai/Molmo-7B-D-0924", device=device)
    raise ValueError(f"Unknown verifier {name!r}. Choose from: {', '.join(VERIFIER_NAMES)}")


__all__ = [
    "OCR_URL_ENV",
    "TEXT_READER_NAMES",
    "TEXT_VERIFIER_ENV",
    "VERIFIER_NAMES",
    "FlorenceGroundingVerifier",
    "MolmoVerifier",
    "OcrVerifier",
    "OpenAICompatVQAVerifier",
    "TagPriorVerifier",
    "TextRoutingVerifier",
    "build_verifier",
]
