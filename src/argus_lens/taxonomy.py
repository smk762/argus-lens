"""Controlled-vocabulary / taxonomy normalization layer (issue #5).

Maps raw model labels onto a canonical vocabulary and optional hierarchy:

* synonym collapse  -- ``automobile`` / ``car`` -> ``car``
* hierarchy expansion -- ``mountain`` -> ``mountain`` + ``landscape`` + ``nature``

This is the defensible layer a single-model competitor can't easily copy: it
makes keywords consistent and searchable regardless of which backend produced
them. Ships with a small default scheme; callers can supply their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Taxonomy:
    """A controlled vocabulary with synonyms and a parent hierarchy.

    The dataclass is frozen so a shared instance (e.g. ``DEFAULT_TAXONOMY``)
    cannot be reassigned; use read-only mappings (see ``DEFAULT_TAXONOMY``) to
    also protect their contents from in-place mutation.

    Attributes:
        synonyms: ``{alias: canonical}`` mapping (lower-cased).  Values are
            expected to already be canonical — resolution is a single lookup,
            not transitive — so chained aliases are not collapsed further.
        parents: ``{canonical: (ancestor, ...)}`` for hierarchy expansion.
            Each entry should list the *full* ancestor chain; expansion is
            single-level (ancestors are not themselves expanded).
    """

    synonyms: Mapping[str, str] = field(default_factory=dict)
    parents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def canonical(self, label: str) -> str:
        """Return the canonical form of *label* (synonym-collapsed).

        Returns ``""`` for blank/whitespace-only input.
        """
        key = label.strip().lower()
        return self.synonyms.get(key, key)

    def expand(self, label: str) -> list[str]:
        """Return the canonical label followed by its ancestors (de-duplicated).

        Blank labels yield an empty list rather than an empty-string keyword.
        """
        canon = self.canonical(label)
        if not canon:
            return []
        out = [canon]
        for ancestor in self.parents.get(canon, ()):
            if ancestor and ancestor not in out:
                out.append(ancestor)
        return out

    def normalize(self, labels: list[str], *, expand_hierarchy: bool = True) -> list[str]:
        """Normalize a list of raw labels, preserving first-seen order.

        Blank labels are dropped.
        """
        seen: list[str] = []
        for label in labels:
            produced = self.expand(label) if expand_hierarchy else [self.canonical(label)]
            for item in produced:
                if item and item not in seen:
                    seen.append(item)
        return seen


DEFAULT_TAXONOMY = Taxonomy(
    synonyms=MappingProxyType({"automobile": "car", "auto": "car", "photograph": "photo"}),
    parents=MappingProxyType({"mountain": ("landscape", "nature"), "beach": ("coast", "nature")}),
)
