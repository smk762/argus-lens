"""Find the attribute disputes between tags and prose, and the prose claims nothing checks.

Disputes reuse the eval harness's contradiction detector, so the reconciler
fixes exactly what the eval metric measures: the same colour/pose/count logic
drives both. Claims (#87) are prose facts no tag speaks to, such as a headcount
from a photo tagger that emits no headcount tags. They are not contradictions,
so the eval metric does not count them; only a verifier that looks at the image
can check them.
"""

from __future__ import annotations

from argus_lens.assembly.subjects import exact_prose_counts, exact_tag_count
from argus_lens.eval.metrics import _split_tags, tag_prose_contradiction
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


def detect_claims(tags: str, prose: str) -> list[AttributeDispute]:
    """Return the prose claims the tags can neither confirm nor contradict, for a verifier to check.

    Today that is one: a single exact headcount in the prose ("six soldiers")
    when the tags state none (RAM++, or WD14 with only ``multiple_*``). A claim
    is an :class:`AttributeDispute` with empty ``tags_say``. Several prose
    headcounts ("two women and three men") are skipped, as for disputes.
    """
    if exact_tag_count(_split_tags(tags)) is not None:
        return []  # the tags state a count: detect_disputes compares the two
    counts = exact_prose_counts(prose)
    if len(counts) != 1:
        return []
    return [AttributeDispute(kind="count", subject="people", tags_say=(), prose_says=(str(counts[0][0]),))]
