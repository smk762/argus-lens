"""Find the attribute disputes between tags and prose, and the prose claims nothing checks.

Disputes reuse the eval harness's contradiction detector, so the reconciler
fixes exactly what the eval metric measures: the same colour/pose/count logic
drives both. Claims (#87) are prose facts no tag speaks to, such as a headcount
from a photo tagger that emits no headcount tags. They are not contradictions,
so the eval metric does not count them; only a verifier that looks at the image
can check them.
"""

from __future__ import annotations

import re

from argus_lens.assembly.subjects import exact_prose_counts, exact_tag_count
from argus_lens.eval.metrics import _POSE_TERMS, _split_tags, _words, tag_prose_contradiction
from argus_lens.reconcile.text_claims import detect_text_claims
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

    A claim is an :class:`AttributeDispute` with empty ``tags_say``. Two kinds:

    * ``count``: a single exact headcount in the prose ("six soldiers") when the
      tags state none (RAM++, or WD14 with only ``multiple_*``). Several prose
      headcounts ("two women and three men") are skipped, as for disputes.
    * ``text``: the names and quoted strings the prose says it read
      ("a diploma for Professor Henrik Bohr"), all in one claim so the verifier
      transcribes the image once (see :mod:`argus_lens.reconcile.text_claims`).
    * ``posture``: the one posture word the prose uses ("a woman sitting on a
      bench") when the tags use none. Two different posture words ("one standing,
      one sitting") describe several people and are skipped, as is a posture
      word naming a thing ("a sitting room", "a standing desk"). Only a verifier
      that ``reads_pose`` (argus-puppet) is asked about it.
    """
    claims: list[AttributeDispute] = []
    if exact_tag_count(_split_tags(tags)) is None:  # with a tag count, detect_disputes compares the two
        counts = exact_prose_counts(prose)
        if len(counts) == 1:
            claims.append(
                AttributeDispute(kind="count", subject="people", tags_say=(), prose_says=(str(counts[0][0]),))
            )
    posture = _prose_posture(prose)
    if posture is not None and not set(_words(tags.replace(",", " "))) & _POSE_TERMS:
        claims.append(AttributeDispute(kind="posture", subject="posture", tags_say=(), prose_says=(posture,)))
    texts = detect_text_claims(prose)
    if texts:
        claims.append(AttributeDispute(kind="text", subject="text", tags_say=(), prose_says=tuple(texts)))
    return claims


# A posture word followed by one of these names a thing, not how someone holds themselves.
_POSTURE_NOUN_USE = re.compile(
    r"\b(?:sitting|standing|lying)\s+(?:rooms?|areas?|desks?|ovations?|water|stones?|orders?)\b",
    re.IGNORECASE,
)


def _prose_posture(prose: str) -> str | None:
    """The one posture word *prose* uses about someone, or ``None`` for none or several different ones."""
    terms = set(_words(_POSTURE_NOUN_USE.sub(" ", prose))) & _POSE_TERMS
    return next(iter(terms)) if len(terms) == 1 else None
