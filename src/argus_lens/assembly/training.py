"""Training variant assembly — optimised for LoRA fine-tuning.

Design principles:

* **Tier 1 (hard-protected)** — framing tags (upper_body, close-up, ...) are
  never dropped by truncation or diversity noise.
* **Tier 2 (soft-protected)** — primary pose tags (standing, sitting, ...) survive
  the diversity pass and are only shed under extreme budget pressure.
* **Rescue** — pose/expression signals misclassified into the identity bucket
  are rescued before wardrobe fill.
* **Wardrobe cap** — configurable max wardrobe fragments prevent clothing
  from dominating the caption.
* **Frequency-aware diversity** — short generic tags are dropped more often
  than long descriptive phrases.
* **Omission cycles** — systematic bucket suppression per image index creates
  stronger disentanglement across the dataset.
* **Identity exclusion** — identity bucket is excluded; the LoRA learns
  identity visually from images.
"""

from __future__ import annotations

import random

from argus_lens.assembly.filtering import with_trigger
from argus_lens.assembly.token_budget import estimate_tokens, try_add_fragment
from argus_lens.types import (
    CategoryConfig,
    normalise_target_style,
)

# ---------------------------------------------------------------------------
# Tag sets for tiered protection
# ---------------------------------------------------------------------------

FRAMING_TAGS: frozenset[str] = frozenset({
    # Camera distance
    "close-up", "close_up", "closeup",
    "upper_body", "upper body",
    "full_body", "full body",
    "portrait", "headshot",
    "bust_shot", "bust shot",
    "cowboy_shot", "cowboy shot",
    "medium_shot", "medium shot",
    "wide_shot", "wide shot",
    "long_shot", "long shot",
    "waist_up", "waist up", "waist-up",
    "half_body", "half body",
    # Camera angle
    "from_above", "from above",
    "from_below", "from below",
    "from_side", "from side",
    "low_angle", "low angle",
    "high_angle", "high angle",
    "dutch_angle", "dutch angle",
    "eye_level", "eye level",
    "bird's_eye", "bird's eye", "birds_eye", "birds eye",
    "aerial_view", "aerial view",
    "overhead", "overhead_view", "overhead view",
    "top_down", "top down", "top-down",
    "profile_view", "profile view",
    "three_quarter_view", "three quarter view",
    "rear_view", "rear view",
})

PRIMARY_POSE_TAGS: frozenset[str] = frozenset({
    "standing", "sitting", "kneeling", "leaning", "walking", "running",
    "lying_down", "lying down", "crouching", "perching",
    "leaning_forward", "leaning forward",
    "leaning_back", "leaning back",
})

POSE_EXPRESSION_RESCUE: frozenset[str] = frozenset({
    # Gaze / look direction
    "looking_at_viewer", "looking at viewer",
    "looking_at_camera", "looking at camera",
    "looking_away", "looking away",
    "looking_down", "looking down",
    "looking_up", "looking up",
    "looking_left", "looking left",
    "looking_right", "looking right",
    "looking_over_shoulder", "looking over shoulder",
    "looking_back", "looking back",
    "side_glance", "side glance",
    "averted_gaze", "averted gaze",
    "eye_contact", "eye contact",
    "gaze",
    # Expression
    "closed_mouth", "open_mouth", "closed mouth", "open mouth",
    "smile", "smiling", "grin", "frown", "serious",
    "cleavage",
    # Framing (also in FRAMING_TAGS, kept here for belt-and-suspenders rescue)
    "upper_body", "upper body", "full_body", "full body",
    "close-up", "close_up", "closeup",
    "from_side", "from side", "from_above", "from above", "from_below", "from below",
    "low_angle", "low angle", "high_angle", "high angle", "dutch_angle", "dutch angle",
    # Body pose
    "arms_crossed", "arms crossed",
    "hand_on_hip", "hand on hip", "hands_on_hips", "hands on hips",
    "arms_raised", "arms raised",
    "hands_behind_back", "hands behind back",
    "sitting", "standing", "kneeling", "leaning",
    "crouching", "lying_down", "lying down",
})

OMISSION_CYCLES: tuple[dict[str, bool], ...] = (
    {},
    {"setting": True},
    {"wardrobe": True},
    {"setting": True, "wardrobe": True},
)

TRAINING_MAX_WARDROBE = 2


def is_framing_fragment(fragment: str) -> bool:
    """Tier 1 (hard-protected) framing/composition signal."""
    normalised = fragment.lower().strip().replace(" ", "_")
    if normalised in FRAMING_TAGS:
        return True
    return any(tag.replace(" ", "_") in normalised for tag in FRAMING_TAGS)


def is_primary_pose(fragment: str) -> bool:
    """Tier 2 (soft-protected) primary pose tag."""
    normalised = fragment.lower().strip().replace(" ", "_")
    return normalised in PRIMARY_POSE_TAGS


def is_rescuable(fragment: str) -> bool:
    """Pose/expression signal worth rescuing from the identity bucket."""
    normalised = fragment.lower().strip().replace(" ", "_")
    return normalised in POSE_EXPRESSION_RESCUE or is_framing_fragment(fragment)


def assemble_training_variant(
    trigger_word: str,
    buckets: dict[str, list[str]],
    target_style: str,
    clip_token_budget: int = 60,
    *,
    target_backend: str | None = None,
    rng: random.Random | None = None,
    drop_probability: float = 0.1,
    image_index: int = 0,
    categories: tuple[CategoryConfig, ...] | None = None,
    enrichment: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Build a caption optimised for LoRA training.

    *enrichment* contains tag-style tokens extracted from prose (e.g.
    Florence-2) that provide novel scene detail not covered by WD14 tags.
    These are appended at lowest priority after all tag tiers.

    Returns ``(caption, removed_fragments)``.
    """
    style = normalise_target_style(target_style)
    max_segments = 10 if style == "anime" else 8
    _rng = rng or random.Random()

    omission = OMISSION_CYCLES[image_index % len(OMISSION_CYCLES)]

    # Phase 1: rescue pose/expression signals from identity bucket
    rescued: list[str] = []
    identity_discarded: list[str] = []
    for fragment in buckets.get("identity", []):
        if is_rescuable(fragment):
            rescued.append(fragment)
        else:
            identity_discarded.append(fragment)

    # Phase 2: collect fragments by protection tier
    tier1: list[str] = []
    tier2: list[str] = []
    pose_rest: list[str] = []
    wardrobe_all: list[str] = []
    setting_all: list[str] = []
    lighting_all: list[str] = []
    action_all: list[str] = []
    removed: list[str] = []

    # camera_framing → tier1 first, pose_gaze → tier2/rest.
    # Also drain the legacy "pose_composition" key for callers using custom buckets.
    for fragment in buckets.get("camera_framing", []):
        if is_framing_fragment(fragment):
            tier1.append(fragment)
        elif is_primary_pose(fragment):
            tier2.append(fragment)
        else:
            pose_rest.append(fragment)

    for fragment in (
        list(buckets.get("pose_gaze", []))
        + list(buckets.get("pose_composition", []))
    ):
        if is_framing_fragment(fragment):
            tier1.append(fragment)
        elif is_primary_pose(fragment):
            tier2.append(fragment)
        else:
            pose_rest.append(fragment)

    for fragment in rescued:
        if is_framing_fragment(fragment):
            tier1.append(fragment)
        elif is_primary_pose(fragment):
            tier2.append(fragment)
        else:
            pose_rest.append(fragment)

    # Action fragments share priority with rescued pose
    raw_action = list(buckets.get("action", []))
    if "action" in omission:
        removed.extend(raw_action)
    else:
        action_all = raw_action

    raw_wardrobe = list(buckets.get("wardrobe", []))
    if "wardrobe" in omission:
        removed.extend(raw_wardrobe)
    else:
        wardrobe_all = raw_wardrobe[:TRAINING_MAX_WARDROBE]
        removed.extend(raw_wardrobe[TRAINING_MAX_WARDROBE:])

    raw_lighting = list(buckets.get("lighting", []))
    lighting_cap = 2
    if "lighting" in omission:
        removed.extend(raw_lighting)
    else:
        lighting_all = raw_lighting[:lighting_cap]
        removed.extend(raw_lighting[lighting_cap:])

    raw_setting = list(buckets.get("setting", []))
    setting_cap = 3
    if "setting" in omission:
        removed.extend(raw_setting)
    else:
        setting_all = raw_setting[:setting_cap]
        removed.extend(raw_setting[setting_cap:])

    # Phase 3: frequency-aware diversity pass
    def _freq_drop(fragment: str) -> bool:
        word_count = len(fragment.split())
        if word_count <= 2:
            return _rng.random() < drop_probability * 1.5
        return _rng.random() < drop_probability * 0.3

    def _diversity_pass(fragments: list[str]) -> tuple[list[str], list[str]]:
        kept_f: list[str] = []
        dropped: list[str] = []
        for f in fragments:
            if _freq_drop(f):
                dropped.append(f)
            else:
                kept_f.append(f)
        _rng.shuffle(kept_f)
        return kept_f, dropped

    pose_rest, pose_dropped = _diversity_pass(pose_rest)
    action_all, action_dropped = _diversity_pass(action_all)
    wardrobe_all, ward_dropped = _diversity_pass(wardrobe_all)
    lighting_all, light_dropped = _diversity_pass(lighting_all)
    setting_all, set_dropped = _diversity_pass(setting_all)
    removed.extend(pose_dropped + action_dropped + ward_dropped + light_dropped + set_dropped)

    # Phase 4: assemble under token budget
    trigger_tokens = estimate_tokens(trigger_word, target_backend)
    used_tokens = trigger_tokens
    kept: list[str] = []

    for fragment in tier1:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in tier2:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in pose_rest:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in action_all:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in wardrobe_all:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in lighting_all:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    for fragment in setting_all:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    # Phase 5: prose enrichment (lowest priority, budget permitting)
    for fragment in enrichment or []:
        added, used_tokens = try_add_fragment(fragment, kept, used_tokens, max_segments, clip_token_budget, target_backend)
        if not added:
            removed.append(fragment)

    removed.extend(identity_discarded)

    return with_trigger(trigger_word, ", ".join(kept)), removed
