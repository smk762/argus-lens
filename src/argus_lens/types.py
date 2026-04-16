"""Core data types for Argus Lens."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Backend kind
# ---------------------------------------------------------------------------


class BackendKind(enum.Enum):
    """Discriminator for local (GPU/CPU) vs cloud (API) backends."""

    LOCAL = "local"
    CLOUD = "cloud"


# ---------------------------------------------------------------------------
# Category configuration
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "identity",
    "wardrobe",
    "pose_composition",
    "setting",
    "lighting",
    "action",
)

CAPTION_TARGET_STYLES: tuple[str, ...] = ("photo", "anime")


@dataclass(frozen=True)
class CategoryConfig:
    """Defines a classification category for fragment bucketing.

    Attributes:
        name: Unique category identifier (e.g. ``"identity"``, ``"wardrobe"``).
        hint_words: Words/phrases used to classify a text fragment into this
            category.  The classifier scores each fragment by counting hint
            matches; the category with the highest score wins.
        training_priority: Ordering weight for the training variant — lower
            numbers are filled first (higher priority).  ``0`` means
            "hard-protected" (never dropped by truncation).
        training_max_fragments: Maximum fragments kept from this bucket in
            the training variant.  ``None`` = unlimited.
        zeroshot_priority: Ordering weight for the zero-shot variant.
        zeroshot_max_fragments: Maximum fragments in the zero-shot variant.
            ``None`` = unlimited.
        exclude_from_training: When True the entire bucket is suppressed in
            the training variant (e.g. identity — learned visually).
    """

    name: str
    hint_words: tuple[str, ...] = ()
    training_priority: int = 5
    training_max_fragments: int | None = None
    zeroshot_priority: int = 5
    zeroshot_max_fragments: int | None = None
    exclude_from_training: bool = False


# Built-in category definitions with their hint word sets.

IDENTITY_HINTS: tuple[str, ...] = (
    "hair", "eyes", "face", "smile", "expression", "freckles", "glasses",
    "beard", "makeup", "blonde", "brunette", "brown hair", "black hair",
    "red hair", "looking at camera", "age", "complexion", "skin",
)

WARDROBE_HINTS: tuple[str, ...] = (
    "shirt", "t-shirt", "tee", "dress", "jacket", "coat", "hoodie",
    "sweater", "jeans", "pants", "skirt", "shorts", "shoes", "boots",
    "heels", "hat", "cap", "gloves", "uniform", "wearing", "outfit",
    "suit", "blouse", "scarf", "tie", "vest", "socks", "sandals",
)

POSE_HINTS: tuple[str, ...] = (
    "standing", "sitting", "kneeling", "leaning", "posing", "full body",
    "upper body", "close up", "close-up", "portrait", "selfie",
    "side view", "front view", "looking away", "arms crossed",
    "from above", "from below", "from side", "headshot",
)

SETTING_HINTS: tuple[str, ...] = (
    "background", "room", "living room", "bedroom", "kitchen", "window",
    "curtains", "floor", "rug", "couch", "sofa", "chair", "bed",
    "street", "park", "beach", "forest", "studio", "indoors", "outdoors",
    "wall", "garden", "office", "cafe", "restaurant", "city", "mountain",
)

LIGHTING_HINTS: tuple[str, ...] = (
    "backlight", "backlighting", "natural light", "studio lighting",
    "golden hour", "shadow", "shadows", "high key", "low key",
    "rim light", "silhouette", "soft light", "harsh light", "candlelight",
    "neon", "dramatic lighting", "sunlight", "overcast", "spotlight",
    "ambient light", "warm light", "cool light", "flash", "bokeh",
    "lens flare", "dappled light", "twilight", "dawn", "dusk",
)

ACTION_HINTS: tuple[str, ...] = (
    "reading", "writing", "cooking", "dancing", "running", "walking",
    "eating", "drinking", "playing", "swimming", "typing", "painting",
    "holding", "carrying", "reaching", "waving", "pointing", "laughing",
    "singing", "climbing", "jumping", "stretching", "working",
    "talking", "hugging", "fighting", "sleeping", "driving",
)

DEFAULT_CATEGORY_CONFIGS: tuple[CategoryConfig, ...] = (
    CategoryConfig(
        name="identity",
        hint_words=IDENTITY_HINTS,
        training_priority=99,
        zeroshot_priority=1,
        exclude_from_training=True,
    ),
    CategoryConfig(
        name="wardrobe",
        hint_words=WARDROBE_HINTS,
        training_priority=4,
        training_max_fragments=2,
        zeroshot_priority=3,
    ),
    CategoryConfig(
        name="pose_composition",
        hint_words=POSE_HINTS,
        training_priority=1,
        zeroshot_priority=2,
    ),
    CategoryConfig(
        name="setting",
        hint_words=SETTING_HINTS,
        training_priority=6,
        training_max_fragments=3,
        zeroshot_priority=5,
        zeroshot_max_fragments=3,
    ),
    CategoryConfig(
        name="lighting",
        hint_words=LIGHTING_HINTS,
        training_priority=5,
        training_max_fragments=2,
        zeroshot_priority=4,
        zeroshot_max_fragments=2,
    ),
    CategoryConfig(
        name="action",
        hint_words=ACTION_HINTS,
        training_priority=2,
        zeroshot_priority=2,
    ),
)


def get_category_config_map(
    categories: tuple[CategoryConfig, ...] | None = None,
) -> dict[str, CategoryConfig]:
    """Return a ``{name: config}`` mapping, using defaults when *categories* is None."""
    configs = categories or DEFAULT_CATEGORY_CONFIGS
    return {c.name: c for c in configs}


def get_category_names(
    categories: tuple[CategoryConfig, ...] | None = None,
) -> tuple[str, ...]:
    """Return category names in definition order."""
    configs = categories or DEFAULT_CATEGORY_CONFIGS
    return tuple(c.name for c in configs)


# ---------------------------------------------------------------------------
# Token budget configuration
# ---------------------------------------------------------------------------

# Token budgets keyed by target backend.  Values are the *usable* token
# count after reserving space for BOS/EOS and trigger word overhead.
BACKEND_TOKEN_BUDGETS: dict[str, int] = {
    "sd15": 60,
    "sdxl": 60,
    "flux": 200,
    "sd3": 200,
    "kolors": 200,
    "pixart": 200,
    "playground": 60,
}

DEFAULT_TOKEN_BUDGET = 60


@dataclass(frozen=True)
class TokenBudgetConfig:
    """Per-backend token budget with optional override."""

    backend_name: str | None = None
    budget: int = DEFAULT_TOKEN_BUDGET

    @classmethod
    def for_backend(cls, backend_name: str | None, override: int | None = None) -> TokenBudgetConfig:
        """Resolve the token budget for a target backend.

        Priority: explicit *override* > ``BACKEND_TOKEN_BUDGETS`` lookup >
        ``DEFAULT_TOKEN_BUDGET``.
        """
        if override is not None:
            return cls(backend_name=backend_name, budget=override)
        budget = BACKEND_TOKEN_BUDGETS.get((backend_name or "").lower().strip(), DEFAULT_TOKEN_BUDGET)
        return cls(backend_name=backend_name, budget=budget)


# ---------------------------------------------------------------------------
# Target profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionTargetProfile:
    """Describes the intended use of the generated caption.

    Attributes:
        target_style: ``"photo"`` or ``"anime"`` — affects tag budgets and
            classification heuristics.
        target_category: Which category variant becomes ``final_caption``.
        target_backend: Diffusion backend name (``"sdxl"``, ``"flux"``, etc.)
            — determines the CLIP/T5 token budget.
        checkpoint: Optional checkpoint name — used for style inference
            (e.g. ``"ponyDiffusion"`` implies anime).
        token_budget: Resolved token budget (computed from *target_backend*).
    """

    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str | None = "sdxl"
    checkpoint: str | None = None
    token_budget: TokenBudgetConfig = field(default_factory=lambda: TokenBudgetConfig.for_backend("sdxl"))


def normalise_target_style(
    style: str | None,
    checkpoint: str | None = None,
    target_backend: str | None = None,
) -> str:
    """Normalise a target style string, inferring from checkpoint/backend hints."""
    value = (style or "").strip().lower()
    if value in CAPTION_TARGET_STYLES:
        return value
    hint = " ".join(part for part in [checkpoint or "", target_backend or ""]).lower()
    if any(token in hint for token in ("pony", "illustrious", "anime", "booru")):
        return "anime"
    return "photo"


def normalise_target_category(
    category: str | None,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> str:
    """Normalise a category name, falling back to ``"identity"``."""
    names = get_category_names(categories)
    value = (category or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in names:
        return value
    return "identity"


def resolve_target_profile(
    *,
    target_style: str = "photo",
    target_category: str = "identity",
    target_backend: str | None = "sdxl",
    checkpoint: str | None = None,
    token_budget_override: int | None = None,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> CaptionTargetProfile:
    """Build a fully resolved target profile."""
    style = normalise_target_style(target_style, checkpoint, target_backend)
    category = normalise_target_category(target_category, categories)
    budget = TokenBudgetConfig.for_backend(target_backend, token_budget_override)
    return CaptionTargetProfile(
        target_style=style,
        target_category=category,
        target_backend=(target_backend or "").strip() or None,
        checkpoint=(checkpoint or "").strip() or None,
        token_budget=budget,
    )


# ---------------------------------------------------------------------------
# Caption result
# ---------------------------------------------------------------------------


@dataclass
class CaptionResult:
    """Structured output from the captioning pipeline.

    Attributes:
        final_caption: The caption for the selected ``target_category``.
        caption_variants: One caption per category, plus ``"training"``
            and ``"zeroshot"`` variants.
        selected_category: Which category was selected as ``final_caption``.
        removed_phrases: Phrases stripped during assembly (filler, redundant,
            noise, truncated).
        compaction_notes: Human-readable notes about what was changed and why.
        raw_tags: Raw tag output from tag-based backends (e.g. WD14).
        raw_prose: Raw prose output from prose-based backends
            (e.g. Florence-2, BLIP-2, OpenAI).
        backend_name: Name of the backend that produced the raw output.
        metadata: Arbitrary metadata from the backend or pipeline.
    """

    final_caption: str
    caption_variants: dict[str, str] = field(default_factory=dict)
    selected_category: str = "identity"
    removed_phrases: list[str] = field(default_factory=list)
    compaction_notes: list[str] = field(default_factory=list)
    raw_tags: str = ""
    raw_prose: str = ""
    backend_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_wd14(self) -> str:
        """Backward-compatible alias for raw tag output."""
        return self.raw_tags

    @property
    def raw_florence(self) -> str:
        """Backward-compatible alias for raw prose output."""
        return self.raw_prose
