"""Argus Lens — Structured image captioning for training and generation."""

from argus_lens._version import __version__
from argus_lens.engine import ArgusLens
from argus_lens.types import (
    BackendKind,
    CaptionResult,
    CaptionTargetProfile,
    CategoryConfig,
    TokenBudgetConfig,
)

__all__ = [
    "__version__",
    "ArgusLens",
    "BackendKind",
    "CaptionResult",
    "CaptionTargetProfile",
    "CategoryConfig",
    "TokenBudgetConfig",
]
