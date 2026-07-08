"""Find the attribute disputes between tags and prose.

Reuses the eval harness's contradiction detector so the reconciler fixes
exactly what the eval metric measures — the same colour/pose logic drives both.
"""

from __future__ import annotations

from argus_lens.eval.metrics import tag_prose_contradiction
from argus_lens.reconcile.types import AttributeDispute
from argus_lens.types import CaptionResult


def detect_disputes(tags: str, prose: str) -> list[AttributeDispute]:
    """Return the colour/pose attributes where *prose* contradicts *tags*."""
    detail = tag_prose_contradiction(CaptionResult(final_caption="", raw_tags=tags, raw_prose=prose))
    return [
        AttributeDispute(
            kind=d["kind"],
            subject=d["subject"],
            tags_say=tuple(d["tags_say"]),
            prose_says=tuple(d["prose_says"]),
        )
        for d in detail["details"]
    ]
