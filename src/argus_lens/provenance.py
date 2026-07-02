"""Build provenance metadata for ``CaptionResult.metadata`` (issue #2).

A metadata-first product needs to know *which backend* produced *which tag*
with *what confidence*. This module turns structured ``BackendOutput`` into a
serialisable provenance dict suitable for ``CaptionResult.metadata``.
"""

from __future__ import annotations

from typing import Any

from argus_lens.backends.output import BackendOutput


def build_provenance(
    output: BackendOutput,
    *,
    backend_name: str = "",
    min_score: float | None = None,
) -> dict[str, Any]:
    """Return a provenance dict describing each tag's source and confidence.

    Every tag is recorded so the provenance stays a complete audit trail. When
    *min_score* is set, each tag additionally carries an ``"included"`` flag
    marking whether it clears the threshold, using the same rule as
    ``BackendOutput.tag_string`` (unscored tags are always included). This keeps
    the meaning of *min_score* consistent across the API without discarding the
    below-threshold tags from the record.

    Args:
        output: The structured backend output.
        backend_name: Fallback source name for tags that did not set one.
        min_score: Optional confidence threshold used to compute each tag's
            ``"included"`` flag.  ``None`` marks every tag as included.
    """
    tags = [
        {
            "label": tag.label,
            "score": tag.score,
            "source": tag.source or backend_name,
            "region": list(tag.region) if tag.region is not None else None,
            "included": min_score is None or tag.score is None or tag.score >= min_score,
        }
        for tag in output.tags
    ]
    return {
        "backend": backend_name,
        "min_score": min_score,
        "tags": tags,
    }
