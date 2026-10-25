"""Torch-free pieces of the RAM++ port: tag vocabulary, checkpoint keys, tag selection.

Kept apart from :mod:`argus_lens.backends._ram.model` so they can be imported and
tested without torch installed (lens CI runs without the ``[torch]`` extra).
The tag list and thresholds are recognize-anything's ``ram_tag_list.txt`` /
``ram_tag_list_threshold.txt`` (Apache-2.0, see ``LICENSE-RAM``).
"""

from __future__ import annotations

from importlib import resources

DESCRIPTIONS_PER_TAG = 51  # RAM++ embeds 51 LLM-written descriptions per tag


def load_tag_vocabulary() -> tuple[list[str], list[float]]:
    """Return RAM++'s 4585 tags and their calibrated per-tag thresholds (package data)."""
    pkg = resources.files("argus_lens.backends._ram")
    tags = pkg.joinpath("ram_tag_list.txt").read_text(encoding="utf-8").splitlines()
    thresholds = [float(line) for line in pkg.joinpath("ram_tag_list_threshold.txt").read_text().split()]
    if len(tags) != len(thresholds):
        raise RuntimeError(f"RAM++ tag list ({len(tags)}) and thresholds ({len(thresholds)}) are out of sync")
    return tags, thresholds


def prepare_state_dict(checkpoint: dict) -> dict:
    """Adapt an upstream RAM++ checkpoint to this module (mirrors ``load_checkpoint_swinlarge``).

    Drops buffers the model recomputes (``relative_position_index``, ``attn_mask``) and
    renames the legacy ``vision_multi`` prefix to ``tagging_head``.
    """
    state = checkpoint.get("model", checkpoint)
    out = {}
    for key, value in state.items():
        if "relative_position_index" in key or "attn_mask" in key:
            continue
        out[key.replace("vision_multi", "tagging_head")] = value
    return out


def unexpectedly_missing(missing: list[str]) -> list[str]:
    """Filter ``load_state_dict``'s missing keys down to ones that indicate a broken checkpoint.

    Expected gaps: the Swin buffers this module recomputes (``relative_position_index``,
    ``attn_mask``, dropped by :func:`prepare_state_dict` exactly as upstream does), and
    ``reweight_scale``, which the released checkpoint does not contain (it stores an unused
    ``logit_scale``), so upstream also runs with its initial ``log(1/0.07)``.
    """
    return [k for k in missing if k != "reweight_scale" and not k.endswith(("relative_position_index", "attn_mask"))]


def select_tags(probs: list[float], thresholds: list[float], threshold: float | None = None) -> list[tuple[int, float]]:
    """Return ``(tag_index, probability)`` for tags that clear their cut-off, highest first.

    The cut-off is each tag's calibrated threshold, or the uniform *threshold* when given.
    Upstream keeps a tag when its probability is strictly greater than the cut-off.
    """
    kept = [
        (i, p)
        for i, (p, t) in enumerate(zip(probs, thresholds, strict=True))
        if p > (t if threshold is None else threshold)
    ]
    return sorted(kept, key=lambda item: -item[1])
