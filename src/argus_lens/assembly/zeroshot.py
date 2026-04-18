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
from argus_lens.assembly.token_budget import estimate_tokens
from argus_lens.types import (
    CAMERA_FRAMING_HINTS,
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

    # Backward compat: distribute legacy "pose_composition" key into the two
    # new buckets so callers using the old key still get their fragments included.
    if "pose_composition" in buckets and buckets["pose_composition"]:
        _framing_hints = set(CAMERA_FRAMING_HINTS)
        _camera: list[str] = list(buckets.get("camera_framing", []))
        _gaze: list[str] = list(buckets.get("pose_gaze", []))
        for _frag in buckets["pose_composition"]:
            if any(_h in _frag.lower() for _h in _framing_hints):
                _camera.append(_frag)
            else:
                _gaze.append(_frag)
        buckets = {**buckets, "camera_framing": _camera, "pose_gaze": _gaze}

    config_map = get_category_config_map(categories)

    def _prose_priority(fragments: list[str]) -> list[str]:
        return sorted(fragments, key=lambda f: len(f.split()), reverse=True)

    plan = sorted(config_map.items(), key=lambda x: x[1].zeroshot_priority)

    trigger_tokens = estimate_tokens(trigger_word, target_backend)
    used_tokens = trigger_tokens
    kept: list[str] = []
    removed: list[str] = []

    for bucket_name, cfg in plan:
        fragments = _prose_priority(list(buckets.get(bucket_name, [])))
        cap = cfg.zeroshot_max_fragments
        if cap is not None:
            removed.extend(fragments[cap:])
            fragments = fragments[:cap]
        for fragment in fragments:
            frag_tokens = estimate_tokens(fragment, target_backend)
            separator_cost = 1 if kept else 0
            if len(kept) >= max_segments or used_tokens + frag_tokens + separator_cost > clip_token_budget:
                removed.append(fragment)
                continue
            kept.append(fragment)
            used_tokens += frag_tokens + separator_cost

    return with_trigger(trigger_word, ", ".join(kept)), removed
