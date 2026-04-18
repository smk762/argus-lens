"""Tests for the structured caption assembly pipeline.

Ported from imogen/tests/test_captioners.py with adaptations for the
Argus Lens API and new categories (lighting, action).
"""

from __future__ import annotations

import random

from argus_lens.assembly.classifier import classify_fragment
from argus_lens.assembly.composer import compose_caption_result
from argus_lens.assembly.filtering import (
    dedupe_fragments,
    filter_redundant_clauses,
    normalise_fragment,
    strip_filler_prefixes,
    with_trigger,
)
from argus_lens.assembly.noise import filter_training_noise
from argus_lens.assembly.token_budget import estimate_clip_tokens, estimate_t5_tokens
from argus_lens.assembly.training import assemble_training_variant
from argus_lens.assembly.zeroshot import assemble_zeroshot_variant
from argus_lens.types import resolve_target_profile

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFillerStripping:
    def test_removes_common_prefixes(self):
        text = "The image shows a woman standing in a park."
        cleaned, removed = strip_filler_prefixes(text)
        assert "the image shows" not in cleaned.lower()
        assert len(removed) > 0

    def test_preserves_content(self):
        text = "a woman standing in a park"
        cleaned, removed = strip_filler_prefixes(text)
        assert "woman" in cleaned
        assert removed == []

    def test_normalise_fragment_lowercases(self):
        cleaned, _ = normalise_fragment("The Image Shows A Tall WOMAN")
        assert cleaned == cleaned.lower()


class TestRedundancyFilter:
    def test_removes_overlapping_clauses(self):
        tags = "blonde hair, black t-shirt"
        description = "A woman with blonde hair relaxing in a park with tall trees."
        filtered = filter_redundant_clauses(description, tags)
        assert "park" in filtered

    def test_keeps_novel_clauses(self):
        tags = "blonde hair"
        description = "She is cooking dinner in a modern kitchen."
        filtered = filter_redundant_clauses(description, tags)
        assert "cooking" in filtered or "kitchen" in filtered


class TestDedup:
    def test_removes_case_insensitive_dupes(self):
        result = dedupe_fragments(["Hello", "hello", "World", "HELLO"])
        assert result == ["Hello", "World"]


class TestTrigger:
    def test_prepends_trigger(self):
        assert with_trigger("sks", "hello") == "sks, hello"

    def test_empty_trigger(self):
        assert with_trigger("", "hello") == "hello"

    def test_empty_body(self):
        assert with_trigger("sks", "") == "sks"


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


class TestNoiseFiltering:
    def test_strips_rating_and_meta_tags(self):
        fragments = ["sensitive", "1girl", "solo", "black t-shirt", "standing", "realistic"]
        kept, removed = filter_training_noise(fragments, strip_identity=False)
        assert "black t-shirt" in kept
        assert "standing" in kept
        assert "sensitive" in removed
        assert "1girl" in removed

    def test_strips_identity_redundant_tags(self):
        fragments = ["brown_hair", "brown_eyes", "black t-shirt", "looking_at_viewer", "standing"]
        kept, removed = filter_training_noise(fragments, strip_identity=True)
        assert "black t-shirt" in kept
        assert "looking_at_viewer" in kept
        assert "brown_hair" in removed

    def test_normalises_spaces_to_underscores(self):
        fragments = ["brown hair", "brown eyes", "black jacket"]
        kept, removed = filter_training_noise(fragments, strip_identity=True)
        assert "black jacket" in kept
        assert "brown hair" in removed

    def test_preserves_all_when_identity_off(self):
        fragments = ["brown_hair", "brown_eyes", "black t-shirt"]
        kept, removed = filter_training_noise(fragments, strip_identity=False)
        assert len(kept) == 3
        assert removed == []


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_clip_tokens("") == 0

    def test_basic_text(self):
        assert estimate_clip_tokens("hello world") > 0

    def test_commas_add_tokens(self):
        plain = estimate_clip_tokens("black t-shirt standing")
        with_commas = estimate_clip_tokens("black t-shirt, standing, outdoors")
        assert with_commas > plain

    def test_t5_estimation(self):
        assert estimate_t5_tokens("hello world") > 0
        assert estimate_t5_tokens("") == 0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_classifies_wardrobe(self):
        assert classify_fragment("black t-shirt") == "wardrobe"

    def test_classifies_setting(self):
        assert classify_fragment("living room with green curtains") == "setting"

    def test_classifies_pose(self):
        assert classify_fragment("standing with arms crossed") == "pose_gaze"

    def test_classifies_camera_framing(self):
        assert classify_fragment("close-up portrait") == "camera_framing"

    def test_classifies_lighting(self):
        assert classify_fragment("dramatic backlighting with rim light") == "lighting"

    def test_classifies_action(self):
        assert classify_fragment("reading a book while drinking coffee") == "action"

    def test_fallback_to_identity(self):
        assert classify_fragment("gentle dignified aura") == "identity"


# ---------------------------------------------------------------------------
# Compose: full pipeline
# ---------------------------------------------------------------------------


class TestComposeCaptionResult:
    def test_strips_filler_and_builds_variants(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="blonde hair, black t-shirt, standing",
            prose="The image shows a woman standing in a living room with green curtains and a patterned rug.",
            target_profile=profile,
        )
        assert result.final_caption.startswith("sks_eva")
        assert "the image shows" not in result.final_caption.lower()
        assert "black t-shirt" in result.caption_variants["wardrobe"]
        assert "living room" in result.caption_variants["setting"]

    def test_prefers_selected_category(self):
        profile = resolve_target_profile(target_style="photo", target_category="wardrobe")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="blonde hair, black t-shirt, blue jeans, standing",
            prose="A photo of a woman standing beside a couch in a bright room.",
            target_profile=profile,
        )
        assert result.selected_category == "wardrobe"
        assert "black t-shirt" in result.final_caption

    def test_produces_training_variant(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="sensitive, 1girl, solo, blonde hair, black t-shirt, standing",
            prose="She is wearing a black t-shirt and standing in a park.",
            target_profile=profile,
        )
        assert "training" in result.caption_variants
        training = result.caption_variants["training"]
        assert training.startswith("sks_eva")
        assert "sensitive" not in training
        assert "1girl" not in training

    def test_produces_zeroshot_variant(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="sensitive, 1girl, solo, brown_hair, brown_eyes, black dress, standing",
            prose="A young woman with brown hair standing in a park.",
            target_profile=profile,
        )
        assert "zeroshot" in result.caption_variants
        zeroshot = result.caption_variants["zeroshot"]
        assert "brown_hair" in zeroshot or "brown hair" in zeroshot
        assert "sensitive" not in zeroshot

    def test_zeroshot_restores_useful_noise(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="sensitive, 1girl, solo, brown_hair, standing",
            prose="",
            target_profile=profile,
        )
        zeroshot = result.caption_variants["zeroshot"]
        assert "1girl" in zeroshot
        assert "solo" in zeroshot
        assert "sensitive" not in zeroshot

    def test_training_vs_zeroshot_identity(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="1girl, brown_hair, brown_eyes, black dress, standing",
            prose="A young woman with brown hair.",
            target_profile=profile,
        )
        training = result.caption_variants["training"]
        zeroshot = result.caption_variants["zeroshot"]
        assert "brown_hair" not in training
        assert "brown_hair" in zeroshot or "brown hair" in zeroshot


# ---------------------------------------------------------------------------
# Training variant
# ---------------------------------------------------------------------------


class TestTrainingVariant:
    def test_excludes_identity(self):
        buckets = {
            "identity": ["blonde hair", "blue eyes"],
            "wardrobe": ["black jacket", "blue jeans"],
            "pose_composition": ["standing", "upper body"],
            "setting": ["indoors"],
            "lighting": [],
            "action": [],
        }
        caption, removed = assemble_training_variant(
            "sks_eva", buckets, "photo",
            rng=random.Random(42), drop_probability=0.0,
        )
        assert "blonde hair" not in caption
        assert "blue eyes" not in caption
        assert "blonde hair" in removed

    def test_protects_framing_tags(self):
        buckets = {
            "identity": [],
            "wardrobe": ["outfit_a", "outfit_b"],
            "pose_composition": ["upper body", "standing", "looking at viewer"],
            "setting": ["park", "sunny day", "trees"],
            "lighting": [],
            "action": [],
        }
        caption, removed = assemble_training_variant(
            "sks_eva", buckets, "photo", clip_token_budget=25,
            rng=random.Random(42), drop_probability=0.0,
        )
        assert "upper body" in caption

    def test_wardrobe_capped(self):
        buckets = {
            "identity": [],
            "wardrobe": ["black jacket", "blue jeans", "white sneakers", "red hat", "silver watch"],
            "pose_composition": ["standing"],
            "setting": ["outdoors"],
            "lighting": [],
            "action": [],
        }
        caption, _ = assemble_training_variant(
            "sks_eva", buckets, "photo",
            rng=random.Random(42), drop_probability=0.0,
        )
        wardrobe_count = sum(
            1 for f in ["black jacket", "blue jeans", "white sneakers", "red hat", "silver watch"]
            if f in caption
        )
        assert wardrobe_count <= 2

    def test_rescues_pose_from_identity(self):
        buckets = {
            "identity": ["looking_at_viewer", "smile", "upper_body", "jewelry", "brown_hair"],
            "wardrobe": ["black dress", "high heels"],
            "pose_composition": [],
            "setting": ["indoors"],
            "lighting": [],
            "action": [],
        }
        caption, removed = assemble_training_variant(
            "sks_eva", buckets, "photo", clip_token_budget=30,
            rng=random.Random(42), drop_probability=0.0,
        )
        assert "looking_at_viewer" in caption
        assert "smile" in caption
        assert "upper_body" in caption
        assert "brown_hair" in removed

    def test_omission_cycle_drops_setting(self):
        buckets = {
            "identity": [],
            "wardrobe": ["black jacket"],
            "pose_composition": ["standing"],
            "setting": ["park", "sunny day"],
            "lighting": [],
            "action": [],
        }
        caption_full, _ = assemble_training_variant(
            "sks_eva", buckets, "photo",
            rng=random.Random(42), drop_probability=0.0, image_index=0,
        )
        caption_no_setting, _ = assemble_training_variant(
            "sks_eva", buckets, "photo",
            rng=random.Random(42), drop_probability=0.0, image_index=1,
        )
        assert "park" in caption_full or "sunny day" in caption_full
        assert "park" not in caption_no_setting

    def test_respects_token_budget(self):
        buckets = {
            "identity": [],
            "wardrobe": ["jacket", "jeans"],
            "pose_composition": ["standing"],
            "setting": [f"place_{i}" for i in range(10)],
            "lighting": [],
            "action": [],
        }
        _, removed_tight = assemble_training_variant(
            "sks_eva", buckets, "photo", clip_token_budget=15,
            rng=random.Random(42), drop_probability=0.0,
        )
        _, removed_loose = assemble_training_variant(
            "sks_eva", buckets, "photo", clip_token_budget=60,
            rng=random.Random(42), drop_probability=0.0,
        )
        assert len(removed_tight) > len(removed_loose)


# ---------------------------------------------------------------------------
# Zeroshot variant
# ---------------------------------------------------------------------------


class TestZeroshotVariant:
    def test_keeps_identity(self):
        buckets = {
            "identity": ["blonde hair", "blue eyes", "freckles"],
            "wardrobe": ["black jacket"],
            "pose_composition": ["standing"],
            "setting": ["indoors"],
            "lighting": [],
            "action": [],
        }
        caption, _ = assemble_zeroshot_variant("sks_eva", buckets, "photo")
        assert "blonde hair" in caption
        assert "blue eyes" in caption

    def test_prefers_prose_over_tags(self):
        buckets = {
            "identity": [
                "a young woman with long brown hair and green eyes",
                "brown_hair",
                "green_eyes",
            ],
            "wardrobe": [],
            "pose_composition": [],
            "setting": [],
            "lighting": [],
            "action": [],
        }
        caption, _ = assemble_zeroshot_variant("sks_eva", buckets, "photo")
        parts = caption.split(", ")
        prose_idx = next(i for i, p in enumerate(parts) if "young woman" in p)
        tag_positions = [i for i, p in enumerate(parts) if p.strip() in ("brown_hair", "green_eyes")]
        for idx in tag_positions:
            assert prose_idx < idx

    def test_deterministic(self):
        buckets = {
            "identity": ["blonde hair"],
            "wardrobe": ["black jacket"],
            "pose_composition": ["standing"],
            "setting": ["indoors"],
            "lighting": [],
            "action": [],
        }
        results = {assemble_zeroshot_variant("sks_eva", buckets, "photo")[0] for _ in range(20)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Token budget per backend
# ---------------------------------------------------------------------------


class TestTokenBudgetPerBackend:
    def test_sdxl_budget(self):
        profile = resolve_target_profile(target_backend="sdxl")
        assert profile.token_budget.budget == 60

    def test_flux_budget(self):
        profile = resolve_target_profile(target_backend="flux")
        assert profile.token_budget.budget == 200

    def test_sd3_budget(self):
        profile = resolve_target_profile(target_backend="sd3")
        assert profile.token_budget.budget == 200

    def test_override(self):
        profile = resolve_target_profile(target_backend="sdxl", token_budget_override=100)
        assert profile.token_budget.budget == 100

    def test_unknown_backend_uses_default(self):
        profile = resolve_target_profile(target_backend="unknown_model")
        assert profile.token_budget.budget == 60


# ---------------------------------------------------------------------------
# New categories: lighting and action
# ---------------------------------------------------------------------------


class TestNewCategories:
    def test_lighting_in_compose_output(self):
        profile = resolve_target_profile(target_style="photo", target_category="identity")
        result = compose_caption_result(
            trigger_word="sks_eva",
            tags="blonde hair, standing",
            prose="A woman standing in golden hour sunlight with dramatic backlighting.",
            target_profile=profile,
        )
        assert "lighting" in result.caption_variants

    def test_action_classification(self):
        assert classify_fragment("reading a thick novel") == "action"
        assert classify_fragment("cooking pasta in a large pot") == "action"
        assert classify_fragment("dancing and jumping around") == "action"

    def test_lighting_classification(self):
        assert classify_fragment("dramatic backlighting with harsh shadows") == "lighting"
        assert classify_fragment("golden hour rim light and silhouette") == "lighting"
