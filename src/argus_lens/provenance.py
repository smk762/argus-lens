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

    Args:
        output: The structured backend output.
        backend_name: Fallback source name for tags that did not set one.
        min_score: Optional confidence threshold recorded alongside the tags.
    """
    tags = [
        {
            "label": tag.label,
            "score": tag.score,
            "source": tag.source or backend_name,
            "region": list(tag.region) if tag.region is not None else None,
        }
        for tag in output.tags
    ]
    return {
        "backend": backend_name,
        "min_score": min_score,
        "tags": tags,
    }
