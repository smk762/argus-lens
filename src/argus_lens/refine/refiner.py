"""Refinement orchestration — LLM rewrite, deterministic validation, fallback.

The deterministic assembly is always run first; its captions are the draft
*and* the fallback. A :class:`CaptionRewriter` (an LLM) rewrites the
requested variants from the same structured inputs, and every rewrite must
pass :func:`~argus_lens.assembly.coherence.check_caption` — trigger first,
no suppressed identity terms, inside the token budget, no fragments, and
grounded in the inputs. A failing rewrite gets one retry with the problems
listed; if it still fails the deterministic caption is kept. Nothing an
LLM writes reaches a caption unchecked.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from argus_lens.assembly.coherence import check_caption, repair_caption

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.types import CaptionResult, CaptionTargetProfile

logger = structlog.get_logger()

# Terms a training caption must not contain (the LoRA learns identity visually).
IDENTITY_FORBIDDEN: frozenset[str] = frozenset(
    {
        "hair",
        "eyes",
        "freckles",
        "beard",
        "complexion",
        "blonde",
        "brunette",
        "redhead",
        "skin tone",
        "years old",
        "twenties",
        "thirties",
        "forties",
    }
)

# Glue words a fluent rewrite may add without them appearing in the inputs.
_GLUE_WORDS: frozenset[str] = frozenset(
    {
        "wearing",
        "wears",
        "dressed",
        "while",
        "standing",
        "sitting",
        "holding",
        "holds",
        "near",
        "beside",
        "next",
        "behind",
        "front",
        "background",
        "foreground",
        "scene",
        "person",
        "woman",
        "man",
        "women",
        "men",
        "people",
        "girl",
        "boy",
        "shot",
        "view",
        "photo",
        "photograph",
        "indoors",
        "outdoors",
        "also",
        "together",
        "there",
        "visible",
        "around",
        "against",
        "both",
        "one",
        "two",
        "three",
        "other",
        "another",
        "each",
        "she",
        "her",
        "they",
        "them",
        "him",
    }
)


@dataclass(frozen=True)
class VariantSpec:
    """How one caption variant is rewritten and validated.

    Attributes:
        kind: ``"list"`` (comma-separated) or ``"prose"`` (sentences).
        rules: Instructions for the rewriter; ``{budget}`` and ``{category}``
            are filled in.
        suppress_identity: Reject rewrites containing ``IDENTITY_FORBIDDEN``.
        forbidden: Further terms a rewrite must not contain.
        allow_empty: An empty rewrite is a valid answer ("nothing of this
            category is visible") rather than a failure.
    """

    kind: str
    rules: str
    suppress_identity: bool = False
    forbidden: frozenset[str] = frozenset()
    allow_empty: bool = False


# Headcount tags: noise in a training caption (the deterministic path strips them too).
HEADCOUNT_FORBIDDEN: frozenset[str] = frozenset(
    {"1girl", "1boy", "1other", "solo", "2girls", "2boys", "3girls", "multiple girls", "multiple boys"}
)

_LIST_STYLE = (
    "Comma-separated booru-style tags and short noun phrases (underscores or spaces are both fine), "
    "most important first. No sentences, no pronouns, no dangling words."
)
VARIANT_SPECS: dict[str, VariantSpec] = {
    "training": VariantSpec(
        "list",
        "Tag caption for SDXL/CLIP LoRA training. "
        + _LIST_STYLE
        + " Cover framing, pose, clothing, action, setting, lighting and notable objects. "
        "Never mention hair, eyes, face, skin, body type or age: the LoRA learns identity from the pixels. "
        "No headcount tags (1girl, solo). At most {budget} CLIP tokens.",
        suppress_identity=True,
        forbidden=HEADCOUNT_FORBIDDEN,
    ),
    "training_prose": VariantSpec(
        "prose",
        "Natural-language caption for Flux/T5 LoRA training: one to four complete, grammatical sentences, "
        "each with a subject and a verb. Describe framing, pose, clothing, action, setting and lighting "
        "concretely, with no hedging and no mood speculation. Never mention hair, eyes, face, skin, body "
        "type or age: the LoRA learns identity from the pixels. Refer to the subject generically "
        '("a woman", "the man"). At most {budget} T5 tokens.',
        suppress_identity=True,
    ),
    "zeroshot": VariantSpec(
        "list",
        "Prompt for generating this image without a LoRA. " + _LIST_STYLE + " Identity first (hair, eyes, "
        "age, build), then action, pose, clothing, lighting, setting. At most {budget} CLIP tokens.",
    ),
}
_CATEGORY_RULES = (
    "Tags and short noun phrases describing ONLY the {category} of the image, for search and grouping. "
    + _LIST_STYLE
    + ' Use plain, reusable vocabulary ("red dress", "kitchen", "cooking") so images group well.'
    "{identity_rule} If the image shows nothing of this category, return an empty string."
)
_NOT_IDENTITY = " Never include hair, eyes, face, skin, body or age: those belong to identity."


def variant_spec(name: str) -> VariantSpec:
    """Return the :class:`VariantSpec` for *name* (category variants included)."""
    if name in VARIANT_SPECS:
        return VARIANT_SPECS[name]
    rules = _CATEGORY_RULES.replace("{category}", name.replace("_", " and "))
    rules = rules.replace("{identity_rule}", "" if name == "identity" else _NOT_IDENTITY)
    return VariantSpec("list", rules, suppress_identity=name != "identity", allow_empty=True)


@dataclass(frozen=True)
class RefineRequest:
    """Everything a rewriter sees for one image."""

    trigger_word: str
    tags: str
    prose: str
    buckets: dict[str, Any]
    subjects: dict[str, Any]
    drafts: dict[str, str]
    rules: dict[str, str]
    token_budget: int
    target_backend: str | None
    feedback: dict[str, list[str]] = field(default_factory=dict)


@runtime_checkable
class CaptionRewriter(Protocol):
    """Rewrites draft captions; returns ``{variant_name: caption}``."""

    name: str
    model_id: str

    def rewrite(self, request: RefineRequest, image: Image.Image | None = None) -> dict[str, str]:
        """Return rewritten captions for the variants in ``request.rules``."""
        ...


_WORD = re.compile(r"[a-z0-9][a-z0-9'+-]*")


def _vocabulary(*texts: str) -> frozenset[str]:
    """Lower-cased words in *texts* (underscored tags split), plus glue words."""
    words: set[str] = set(_GLUE_WORDS)
    for text in texts:
        words.update(_WORD.findall(text.lower().replace("_", " ")))
    return frozenset(words)


class Refiner:
    """Rewrite selected variants of a :class:`CaptionResult` with an LLM, safely.

    Args:
        rewriter: The :class:`CaptionRewriter` (e.g. ``OpenAICompatRewriter``).
        variants: Variant names to rewrite. Defaults to the two training
            captions; add ``"identity"``, ``"wardrobe"``, ``"setting"`` ... to
            rewrite category captions too.
        send_image: Pass the image to the rewriter (needs a vision model). The
            grounding check is relaxed accordingly, since the model may then
            legitimately name things the tagger and captioner missed.
        max_novel_ratio: Share of content words allowed outside the inputs.
        retries: Extra attempts for variants that fail validation.
    """

    def __init__(
        self,
        rewriter: CaptionRewriter,
        *,
        variants: tuple[str, ...] = ("training", "training_prose"),
        send_image: bool = False,
        max_novel_ratio: float | None = None,
        retries: int = 1,
    ) -> None:
        self.rewriter = rewriter
        self.variants = variants
        self.send_image = send_image
        self.max_novel_ratio = max_novel_ratio if max_novel_ratio is not None else (0.4 if send_image else 0.25)
        self.retries = retries

    def _problems(self, name: str, text: str, request: RefineRequest, vocab: frozenset[str]) -> list[str]:
        """Validate one rewrite against its spec."""
        spec = variant_spec(name)
        return check_caption(
            text,
            spec.kind,
            trigger_word=request.trigger_word,
            forbidden_words=(IDENTITY_FORBIDDEN if spec.suppress_identity else frozenset()) | spec.forbidden,
            token_budget=request.token_budget,
            target_backend=request.target_backend if spec.kind == "prose" else None,
            allowed_vocabulary=vocab,
            max_novel_ratio=self.max_novel_ratio,
        )

    def refine(
        self,
        result: CaptionResult,
        profile: CaptionTargetProfile,
        *,
        trigger_word: str = "",
        image: Image.Image | None = None,
    ) -> CaptionResult:
        """Return a copy of *result* with validated rewrites applied.

        ``metadata["refine"]`` records the model, which variants were
        accepted, and the problems behind any rejection. A rewriter error
        (endpoint down, bad JSON) leaves the deterministic captions in place.
        """
        names = [n for n in self.variants if n in result.caption_variants]
        if not names:
            return result
        request = RefineRequest(
            trigger_word=trigger_word,
            tags=result.raw_tags,
            prose=result.raw_prose,
            buckets=result.metadata.get("buckets", {}),
            subjects=result.metadata.get("subjects", {}),
            drafts={n: result.caption_variants[n] for n in names},
            rules={n: variant_spec(n).rules.replace("{budget}", str(profile.token_budget.budget)) for n in names},
            token_budget=profile.token_budget.budget,
            target_backend=profile.target_backend,
        )
        vocab = _vocabulary(result.raw_tags, result.raw_prose, trigger_word)
        accepted: dict[str, str] = {}
        rejected: dict[str, list[str]] = {}
        pending = list(names)
        attempts = 0
        error: str | None = None
        while pending and attempts <= self.retries:
            attempts += 1
            try:
                out = self.rewriter.rewrite(request, image if self.send_image else None)
            except Exception as exc:  # noqa: BLE001 - any rewriter failure falls back to deterministic
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("refine.error", error=error)
                break
            feedback: dict[str, list[str]] = {}
            for name in pending:
                spec = variant_spec(name)
                text = (out.get(name) or "").strip()
                if text:
                    text, _ = repair_caption(text, spec.kind, trigger_word=trigger_word)
                if spec.allow_empty and name in out and text.strip() in ("", trigger_word.strip()):
                    accepted[name] = trigger_word.strip()  # "nothing of this category" is an answer
                    continue
                problems = self._problems(name, text, request, vocab) if text else ["missing from response"]
                if problems:
                    feedback[name] = problems
                else:
                    accepted[name] = text
            pending = list(feedback)
            rejected = feedback
            request = dataclasses.replace(
                request,
                rules={n: request.rules[n] for n in pending},
                drafts={n: request.drafts[n] for n in pending},
                feedback=feedback,
            )

        variants = {**result.caption_variants, **accepted}
        record: dict[str, Any] = {
            "rewriter": self.rewriter.name,
            "model": self.rewriter.model_id,
            "accepted": sorted(accepted),
            "rejected": rejected,
            "attempts": attempts,
        }
        if error:
            record["error"] = error
        notes = list(result.compaction_notes)
        if accepted:
            notes.append(f"Refined {', '.join(sorted(accepted))} with {self.rewriter.model_id}.")
        if rejected:
            notes.append(f"Kept deterministic {', '.join(sorted(rejected))} (refinement failed validation).")
        return dataclasses.replace(
            result,
            caption_variants=variants,
            final_caption=variants.get(result.selected_category, result.final_caption),
            compaction_notes=notes,
            metadata={**result.metadata, "refine": record},
        )


def refiner_from_env() -> Refiner | None:
    """Build a :class:`Refiner` from ``ARGUS_REFINE_*`` env vars, or ``None`` when unset.

    ``ARGUS_REFINE_MODEL`` enables it; ``ARGUS_REFINE_BASE_URL`` /
    ``ARGUS_REFINE_API_KEY`` locate the endpoint; ``ARGUS_REFINE_VARIANTS``
    (comma-separated, default ``training,training_prose``) and
    ``ARGUS_REFINE_IMAGE`` (``true`` sends the image to a vision model) tune it.
    Used by ``argus-lens serve`` so the server and its UIs get refinement
    without code changes.
    """
    model = os.environ.get("ARGUS_REFINE_MODEL")
    if not model:
        return None
    from argus_lens.refine.openai_compat import OpenAICompatRewriter

    variants = os.environ.get("ARGUS_REFINE_VARIANTS") or "training,training_prose"
    return Refiner(
        OpenAICompatRewriter(model_id=model),
        variants=tuple(v.strip() for v in variants.split(",") if v.strip()),
        send_image=os.environ.get("ARGUS_REFINE_IMAGE", "").strip().lower() in ("1", "true", "yes"),
    )
