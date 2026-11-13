"""Zero-shot variant assembly — optimised for generation without a LoRA.

Opposite priorities to the training variant:

* **Identity first** — hair colour, eye colour, age, gender, body type are
  critical because there is no LoRA to supply them visually.
* **Natural language preferred** — longer prose fragments score higher than
  short tags because the base model was trained on natural language.
* **No stochastic dropping** — every token counts; diversity noise is disabled.
* **Priority order**: identity > action > pose_composition > wardrobe >
  lighting > setting.
"""

from __future__ import annotations

from argus_lens.assembly.filtering import with_trigger
from argus_lens.assembly.token_budget import try_add_fragment
from argus_lens.types import (
    CategoryConfig,
    get_category_config_map,
    normalise_target_style,
)


def assemble_zeroshot_variant(
    trigger_word: str,
    buckets: dict[str, list[str]],
    target_style: str,
    clip_token_budget: int = 60,
    *,
    target_backend: str | None = None,
    categories: tuple[CategoryConfig, ...] | None = None,
) -> tuple[str, list[str]]:
    """Build a caption optimised for zero-shot generation (no LoRA).

    Returns ``(caption, removed_fragments)``.
    """
    style = normalise_target_style(target_style)
    max_segments = 12 if style == "anime" else 10

    config_map = get_category_config_map(categories)

    def _prose_priority(fragments: list[str]) -> list[str]:
        """Order fragments longest-first so natural-language phrases win the budget."""
        return sorted(fragments, key=lambda f: len(f.split()), reverse=True)

    plan = sorted(config_map.items(), key=lambda x: x[1].zeroshot_priority)

    kept: list[str] = []
    removed: list[str] = []

    for bucket_name, cfg in plan:
        fragments = _prose_priority(list(buckets.get(bucket_name, [])))
        cap = cfg.zeroshot_max_fragments
        if cap is not None:
            removed.extend(fragments[cap:])
            fragments = fragments[:cap]
        for fragment in fragments:
            if not try_add_fragment(
                fragment, kept, max_segments, clip_token_budget, target_backend, trigger_word=trigger_word
            ):
                removed.append(fragment)

    return with_trigger(trigger_word, ", ".join(kept)), removed
