"""Caption-quality metrics.

Reference-free (need only the image + the model's own output):

* :func:`tag_prose_contradiction` — does the prose assert a colour or posture
  that contradicts the tags? This is the flagship: it targets the hallucination
  problem directly and needs no ground truth.
* :func:`token_budget_adherence` — do the budget-bound variants fit SDXL/Flux limits?
* :func:`redundancy_rate` — how much the deterministic filters had to strip.
* :func:`clip_score` — image↔caption alignment (optional; needs the ``eval`` extra).

Reference-based (need a labelled manifest):

* :func:`tag_coverage` — recall of the expected high-confidence tags.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from argus_lens.assembly.token_budget import estimate_tokens
from argus_lens.types import BACKEND_TOKEN_BUDGETS, DEFAULT_TOKEN_BUDGET

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.eval.dataset import EvalItem
    from argus_lens.types import CaptionResult

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Canonical colours plus spelling/near-synonym folds, so "grey" vs "gray" or
# "blond" vs "blonde" never read as a contradiction.
_COLOR_CANON: dict[str, str] = {
    "grey": "gray",
    "blond": "blonde",
    "violet": "purple",
    "crimson": "red",
    "scarlet": "red",
    "golden": "gold",
}
_COLORS: frozenset[str] = frozenset(
    {
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "violet",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "blonde",
        "blond",
        "silver",
        "gold",
        "golden",
        "tan",
        "beige",
        "cyan",
        "teal",
        "navy",
        "maroon",
        "crimson",
        "scarlet",
        "turquoise",
    }
)

# Mutually-exclusive posture terms: tags asserting one and prose asserting
# another (from the same group) is a contradiction.
_POSE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"standing", "sitting", "lying", "kneeling", "crouching", "squatting"}),
)
_POSE_TERMS: frozenset[str] = frozenset().union(*_POSE_GROUPS)


def _canon_color(word: str) -> str:
    """Fold a colour word onto its canonical spelling."""
    return _COLOR_CANON.get(word, word)


def _words(text: str) -> list[str]:
    """Lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+", text.lower())


def _split_tags(tags: str) -> list[str]:
    """Split a comma-separated tag string into trimmed tags."""
    return [t.strip() for t in tags.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Tag <-> prose contradiction (reference-free)
# ---------------------------------------------------------------------------


def _tag_color_map(tags: str) -> dict[str, set[str]]:
    """Map each colour-bearing noun in the tags to its asserted colour(s).

    ``red_dress`` / ``blue eyes`` / ``blonde_hair`` → ``{dress:{red}, eyes:{blue},
    hair:{blonde}}``. The head noun is taken as the last non-colour token.
    """
    mapping: dict[str, set[str]] = defaultdict(set)
    for tag in _split_tags(tags):
        toks = tag.lower().replace("_", " ").split()
        colors = [_canon_color(t) for t in toks if t in _COLORS]
        nouns = [t for t in toks if t not in _COLORS]
        if colors and nouns:
            mapping[nouns[-1]].update(colors)
    return dict(mapping)


def _prose_colors_for_noun(prose_words: list[str], noun: str) -> set[str]:
    """Colours mentioned within a 3-word window of *noun* in the prose."""
    found: set[str] = set()
    variants = {noun, noun + "s", noun.rstrip("s")}
    for i, w in enumerate(prose_words):
        if w in variants:
            for j in range(max(0, i - 3), min(len(prose_words), i + 4)):
                if prose_words[j] in _COLORS:
                    found.add(_canon_color(prose_words[j]))
    return found


def tag_prose_contradiction(result: CaptionResult) -> dict:
    """Count places where the prose contradicts the tags (colour + posture).

    Returns ``{count, checked, rate, details}``. ``checked`` is how many
    attributes could be compared (present in both signals); ``rate`` is
    ``count / checked`` (0.0 when nothing was comparable). A lower rate is
    better. Reference-free: compares the run's own ``raw_prose`` vs ``raw_tags``.
    """
    tags, prose = result.raw_tags, result.raw_prose
    prose_words = _words(prose)
    count = 0
    checked = 0
    details: list[dict] = []

    # Colour contradictions
    for noun, tag_colors in _tag_color_map(tags).items():
        prose_colors = _prose_colors_for_noun(prose_words, noun)
        if not prose_colors:
            continue
        checked += 1
        if prose_colors.isdisjoint(tag_colors):
            count += 1
            details.append(
                {"kind": "color", "subject": noun, "tags_say": sorted(tag_colors), "prose_says": sorted(prose_colors)}
            )

    # Posture contradictions
    tag_words = set(_words(tags.replace(",", " ")))
    prose_pose = set(prose_words) & _POSE_TERMS
    for group in _POSE_GROUPS:
        t = group & tag_words
        p = group & prose_pose
        if t and p:
            checked += 1
            if t.isdisjoint(p):
                count += 1
                details.append({"kind": "pose", "subject": "posture", "tags_say": sorted(t), "prose_says": sorted(p)})

    rate = count / checked if checked else 0.0
    return {"count": count, "checked": checked, "rate": rate, "details": details}


# ---------------------------------------------------------------------------
# Token-budget adherence (reference-free)
# ---------------------------------------------------------------------------

# Variants whose length is governed by the token budget (see assembly/*).
_BUDGET_VARIANTS: tuple[str, ...] = ("training", "zeroshot")


def token_budget_adherence(result: CaptionResult, target_backend: str) -> dict:
    """Token counts for the budget-bound variants and whether they overflow.

    Returns ``{budget, tokens: {variant: n}, over_budget: {variant: bool},
    any_over: bool}``.
    """
    budget = BACKEND_TOKEN_BUDGETS.get((target_backend or "").lower().strip(), DEFAULT_TOKEN_BUDGET)
    tokens: dict[str, int] = {}
    over: dict[str, bool] = {}
    for variant in _BUDGET_VARIANTS:
        text = result.caption_variants.get(variant, "")
        n = estimate_tokens(text, target_backend)
        tokens[variant] = n
        over[variant] = n > budget
    return {"budget": budget, "tokens": tokens, "over_budget": over, "any_over": any(over.values())}


# ---------------------------------------------------------------------------
# Redundancy / filler (reference-free)
# ---------------------------------------------------------------------------


def redundancy_rate(result: CaptionResult) -> dict:
    """How much material the deterministic filters stripped, relative to what survived.

    ``rate = removed / (removed + kept)`` where *kept* is the comma-separated
    fragment count of the final caption. Higher means the raw model output was
    noisier / more redundant. Returns ``{removed, kept, rate}``.
    """
    removed = len(result.removed_phrases)
    kept = len([f for f in result.final_caption.split(",") if f.strip()])
    denom = removed + kept
    return {"removed": removed, "kept": kept, "rate": (removed / denom if denom else 0.0)}


# ---------------------------------------------------------------------------
# Tag coverage (reference-based)
# ---------------------------------------------------------------------------


def _tag_present(tag: str, haystack: str) -> bool:
    """True when every content word of *tag* appears in *haystack*."""
    parts = [w for w in tag.lower().replace("_", " ").split() if len(w) > 1]
    return bool(parts) and all(re.search(rf"\b{re.escape(w)}", haystack) for w in parts)


def tag_coverage(result: CaptionResult, expected_tags: tuple[str, ...]) -> dict | None:
    """Recall of the expected high-confidence tags in the caption output.

    Returns ``{recall, hits, total, missed}`` or ``None`` when there are no
    labels. Searches both the final caption and the raw tag output.
    """
    if not expected_tags:
        return None
    haystack = f"{result.final_caption} {result.raw_tags}".lower()
    hits = [t for t in expected_tags if _tag_present(t, haystack)]
    missed = [t for t in expected_tags if t not in hits]
    return {"recall": len(hits) / len(expected_tags), "hits": len(hits), "total": len(expected_tags), "missed": missed}


# ---------------------------------------------------------------------------
# CLIPScore (optional; needs the `eval` extra: torch + transformers)
# ---------------------------------------------------------------------------


class ClipScorer:
    """Lazy CLIP image↔text cosine similarity. Built via :func:`try_build_clip_scorer`."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        import torch  # noqa: PLC0415 - optional dep, imported only when CLIP is requested
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        self._torch = torch
        self._device = device
        self._model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)

    def score(self, image: Image.Image, caption: str) -> float:
        """Cosine similarity between the image and caption embeddings (roughly [-1, 1])."""
        torch = self._torch
        inputs = self._processor(
            text=[caption or " "], images=image, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
            return float((img * txt).sum(dim=-1).item())


def try_build_clip_scorer(device: str = "cpu") -> ClipScorer | None:
    """Return a :class:`ClipScorer`, or ``None`` if torch/transformers aren't installed."""
    try:
        return ClipScorer(device=device)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Per-item aggregation
# ---------------------------------------------------------------------------


def compute_metrics(
    item: EvalItem,
    result: CaptionResult,
    *,
    image: Image.Image | None = None,
    clip_scorer: ClipScorer | None = None,
) -> dict:
    """Compute every applicable metric for one captioned item.

    Reference-based metrics are included only when *item* carries labels;
    CLIPScore only when a *clip_scorer* and *image* are supplied.
    """
    metrics: dict = {
        "contradiction": tag_prose_contradiction(result),
        "budget": token_budget_adherence(result, item.target_backend),
        "redundancy": redundancy_rate(result),
        "coverage": tag_coverage(result, item.expected_tags),
    }
    if clip_scorer is not None and image is not None:
        caption = item.target_caption or result.final_caption
        metrics["clip"] = clip_scorer.score(image, caption)
    else:
        metrics["clip"] = None
    return metrics
