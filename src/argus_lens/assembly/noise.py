"""Training noise filtering — strips tags that harm LoRA quality."""

from __future__ import annotations

# Rating and meta tokens that carry no useful training signal.
TRAINING_NOISE_TAGS: frozenset[str] = frozenset(
    {
        "sensitive",
        "general",
        "questionable",
        "explicit",
        "rating:general",
        "rating:sensitive",
        "rating:questionable",
        "rating:explicit",
        "realistic",
        "1girl",
        "1boy",
        "1other",
        "solo",
    }
)

# Identity traits the LoRA learns visually.  Stating them in captions
# risks conflicts with the base model's prior.
IDENTITY_REDUNDANT_TAGS: frozenset[str] = frozenset(
    {
        "brown_hair",
        "brown_eyes",
        "black_hair",
        "black_eyes",
        "blonde_hair",
        "blue_eyes",
        "green_eyes",
        "red_hair",
        "long_hair",
        "short_hair",
        "medium_hair",
        "male",
        "female",
        "breasts",
        "medium_breasts",
    }
)

# Tags that are useful for zero-shot generation (no LoRA) even though
# they are stripped as noise for training.
ZEROSHOT_RESTORE_TAGS: frozenset[str] = frozenset(
    {
        "1girl",
        "1boy",
        "1other",
        "solo",
    }
)

# Rating tags that should never appear in any variant.
RATING_TAGS: frozenset[str] = frozenset(
    {
        "sensitive",
        "general",
        "questionable",
        "explicit",
        "rating:general",
        "rating:sensitive",
        "rating:questionable",
        "rating:explicit",
        "realistic",
    }
)


def filter_training_noise(
    fragments: list[str],
    *,
    strip_identity: bool = True,
) -> tuple[list[str], list[str]]:
    """Remove tags that add no value (or cause harm) during LoRA training.

    Returns ``(kept, removed)``.
    """
    noise = TRAINING_NOISE_TAGS | (IDENTITY_REDUNDANT_TAGS if strip_identity else frozenset())
    kept: list[str] = []
    removed: list[str] = []
    for fragment in fragments:
        normalised = fragment.lower().strip().replace(" ", "_")
        if normalised in noise:
            removed.append(fragment)
        else:
            kept.append(fragment)
    return kept, removed
