"""Category variant assembly — builds one caption per target category."""

from __future__ import annotations

from argus_lens.assembly.filtering import with_trigger
from argus_lens.types import (
    CategoryConfig,
    get_category_config_map,
    normalise_target_category,
    normalise_target_style,
)


def _variant_limits(target_style: str, category: str) -> tuple[int, int]:
    """Return ``(max_segments, max_words)`` for a category variant."""
    style = normalise_target_style(target_style)
    if style == "anime":
        limits = {
            "identity": (12, 28),
            "wardrobe": (12, 28),
            "camera_framing": (8, 20),
            "pose_gaze": (10, 24),
            "setting": (10, 24),
            "lighting": (8, 20),
            "action": (10, 24),
        }
    else:
        limits = {
            "identity": (8, 22),
            "wardrobe": (8, 24),
            "camera_framing": (6, 18),
            "pose_gaze": (8, 24),
            "setting": (7, 22),
            "lighting": (6, 18),
            "action": (8, 22),
        }
    return limits.get(normalise_target_category(category), (8, 22))


def _variant_bucket_plan(
    category: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> list[tuple[str, int | None]]:
    """Return the ordered bucket fill plan for a category variant.

    The selected category gets unlimited fragments; others get capped
    slots scaled by their ``zeroshot_priority``.
    """
    config_map = get_category_config_map(categories)
    cat = normalise_target_category(category, categories)

    plan: list[tuple[str, int | None]] = []
    for name, cfg in sorted(config_map.items(), key=lambda x: x[1].zeroshot_priority):
        if name == cat:
            plan.append((name, None))
        else:
            cap = max(0, 4 - cfg.zeroshot_priority) if cfg.zeroshot_priority <= 4 else 1
            plan.append((name, cap))
    return plan


def assemble_variant(
    trigger_word: str,
    buckets: dict[str, list[str]],
    category: str,
    target_style: str,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> tuple[str, list[str]]:
    """Assemble a caption variant for the given *category*.

    Returns ``(caption, removed_fragments)``.
    """
    max_segments, max_words = _variant_limits(target_style, category)
    kept: list[str] = []
    removed: list[str] = []
    used_words = len(trigger_word.split())

    for bucket_name, bucket_limit in _variant_bucket_plan(category, categories):
        fragments = buckets.get(bucket_name, [])
        if bucket_limit is not None:
            kept_fragments = fragments[:bucket_limit]
            removed.extend(fragments[bucket_limit:])
            fragments = kept_fragments
        for fragment in fragments:
            fragment_words = len(fragment.split())
            if kept and (len(kept) >= max_segments or used_words + fragment_words > max_words):
                removed.append(fragment)
                continue
            kept.append(fragment)
            used_words += fragment_words

    return with_trigger(trigger_word, ", ".join(kept)), removed
