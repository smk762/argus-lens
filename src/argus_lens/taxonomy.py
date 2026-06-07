"""Controlled-vocabulary / taxonomy normalization layer (issue #5).

Maps raw model labels onto a canonical vocabulary and optional hierarchy:

* synonym collapse  -- ``automobile`` / ``car`` -> ``car``
* hierarchy expansion -- ``mountain`` -> ``mountain`` + ``landscape`` + ``nature``

This is the defensible layer a single-model competitor can't easily copy: it
makes keywords consistent and searchable regardless of which backend produced
them. Ships with a small default scheme; callers can supply their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Taxonomy:
    """A controlled vocabulary with synonyms and a parent hierarchy.

    Attributes:
        synonyms: ``{alias: canonical}`` mapping (lower-cased).
        parents: ``{canonical: (ancestor, ...)}`` for hierarchy expansion.
    """

    synonyms: dict[str, str] = field(default_factory=dict)
    parents: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def canonical(self, label: str) -> str:
        """Return the canonical form of *label* (synonym-collapsed)."""
        key = label.strip().lower()
        return self.synonyms.get(key, key)

    def expand(self, label: str) -> list[str]:
        """Return the canonical label followed by its ancestors (de-duplicated)."""
        canon = self.canonical(label)
        out = [canon]
        for ancestor in self.parents.get(canon, ()):
            if ancestor not in out:
                out.append(ancestor)
        return out

    def normalize(self, labels: list[str], *, expand_hierarchy: bool = True) -> list[str]:
        """Normalize a list of raw labels, preserving first-seen order."""
        seen: list[str] = []
        for label in labels:
            produced = self.expand(label) if expand_hierarchy else [self.canonical(label)]
            for item in produced:
                if item not in seen:
                    seen.append(item)
        return seen


DEFAULT_TAXONOMY = Taxonomy(
    synonyms={"automobile": "car", "auto": "car", "photograph": "photo"},
    parents={"mountain": ("landscape", "nature"), "beach": ("coast", "nature")},
)
