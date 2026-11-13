"""RAM++ vocabulary packs: custom tags that extend (or replace) the built-in 4585 (#62).

RAM++ represents each tag as :data:`~argus_lens.backends._ram.vocab.DESCRIPTIONS_PER_TAG`
CLIP text embeddings of LLM-written descriptions, and its tagging head is shared across
tags, so new tags need no retraining: append their embeddings to ``label_embed``.

A pack is a directory::

    manifest.json            format/version, name, tags + per-tag thresholds, CLIP model,
                             description source, licence, creation date
    embeddings.safetensors   {"label_embed": float16 (len(tags) * 51, 512)}
    descriptions.json        optional: the descriptions the embeddings came from
                             (upstream's [{tag: [desc, ...]}] shape), for reproducibility

The manifest and the merge rules here are torch-free so CI can test them; only
:func:`load_pack_embeddings` needs torch + safetensors.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from argus_lens.backends._ram.vocab import DESCRIPTIONS_PER_TAG

PACK_FORMAT = "argus-ram-vocab"
PACK_VERSION = 1
MANIFEST_FILE = "manifest.json"
EMBEDDINGS_FILE = "embeddings.safetensors"
DESCRIPTIONS_FILE = "descriptions.json"
# The text encoder RAM++'s label embeddings live in: OpenAI CLIP ViT-B/16 (HF conversion).
CLIP_MODEL = "openai/clip-vit-base-patch16"
EMBED_DIM = 512
# Upstream's open-set default: "the threshold for unseen categories is often lower".
DEFAULT_PACK_THRESHOLD = 0.5


class PackError(ValueError):
    """A vocabulary pack is malformed or conflicts with another pack."""


@dataclass(frozen=True)
class PackManifest:
    """What a pack contains and how it was made."""

    name: str
    tags: list[str]
    thresholds: list[float]
    clip_model: str = CLIP_MODEL
    descriptions_per_tag: int = DESCRIPTIONS_PER_TAG
    embed_dim: int = EMBED_DIM
    created: str = ""
    description_source: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    format: str = PACK_FORMAT
    version: int = PACK_VERSION

    def validate(self) -> PackManifest:
        """Raise :class:`PackError` unless the manifest is usable with this lens's RAM++."""
        if self.format != PACK_FORMAT:
            raise PackError(f"not a RAM++ vocabulary pack (format {self.format!r})")
        if self.version != PACK_VERSION:
            raise PackError(f"pack {self.name!r} is format version {self.version}; this lens reads {PACK_VERSION}")
        if not self.name or not self.tags:
            raise PackError("a pack needs a name and at least one tag")
        if len(self.tags) != len(self.thresholds):
            raise PackError(f"pack {self.name!r}: {len(self.tags)} tags but {len(self.thresholds)} thresholds")
        if len(set(self.tags)) != len(self.tags):
            dupes = sorted({t for t in self.tags if self.tags.count(t) > 1})
            raise PackError(f"pack {self.name!r} repeats tags: {dupes[:5]}")
        if any(not t.strip() for t in self.tags):
            raise PackError(f"pack {self.name!r} has an empty tag")
        if not all(0.0 < t <= 1.0 for t in self.thresholds):
            raise PackError(f"pack {self.name!r}: thresholds must be in (0, 1]")
        if self.descriptions_per_tag != DESCRIPTIONS_PER_TAG or self.embed_dim != EMBED_DIM:
            raise PackError(
                f"pack {self.name!r} has {self.descriptions_per_tag}×{self.embed_dim} embeddings per tag; "
                f"RAM++ needs {DESCRIPTIONS_PER_TAG}×{EMBED_DIM}"
            )
        if self.clip_model != CLIP_MODEL:
            raise PackError(f"pack {self.name!r} was embedded with {self.clip_model!r}; RAM++ needs {CLIP_MODEL!r}")
        return self

    @classmethod
    def read(cls, pack_dir: str | Path) -> PackManifest:
        """Load and validate ``manifest.json`` from *pack_dir*."""
        path = Path(pack_dir) / MANIFEST_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PackError(f"no {MANIFEST_FILE} in {pack_dir}") from exc
        except json.JSONDecodeError as exc:
            raise PackError(f"{path} is not valid JSON: {exc}") from exc
        known = {f for f in cls.__dataclass_fields__}
        try:
            manifest = cls(**{k: v for k, v in data.items() if k in known})
        except TypeError as exc:
            raise PackError(f"{path} is missing required fields: {exc}") from exc
        return manifest.validate()

    def write(self, pack_dir: str | Path) -> Path:
        """Write ``manifest.json`` into *pack_dir* (validated first)."""
        self.validate()
        path = Path(pack_dir) / MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class MergedVocabulary:
    """The vocabulary RAM++ runs with once packs are applied.

    ``builtin_keep`` lists the built-in tag indices whose embeddings stay (in order);
    pack embeddings follow them in pack order. ``sources[i]`` names where tag *i* came
    from: ``""`` for built-in, else the pack name.
    """

    tags: list[str]
    thresholds: list[float]
    sources: list[str]
    builtin_keep: list[int]


def merge_vocabularies(
    builtin_tags: Sequence[str],
    builtin_thresholds: Sequence[float],
    packs: Sequence[PackManifest],
    *,
    include_builtin: bool = True,
) -> MergedVocabulary:
    """Combine the built-in vocabulary with *packs*.

    A pack tag that is also a built-in tag **replaces** it (its descriptions and threshold
    were curated on purpose). The same tag in two packs is ambiguous and raises
    :class:`PackError`. With ``include_builtin=False`` only the packs' tags are used,
    upstream's open-set mode.
    """
    if not include_builtin and not packs:
        raise PackError("builtin_vocab=False needs at least one vocabulary pack")
    owner: dict[str, str] = {}
    for pack in packs:
        for tag in pack.tags:
            if tag in owner:
                raise PackError(f"tag {tag!r} is in both pack {owner[tag]!r} and pack {pack.name!r}")
            owner[tag] = pack.name

    keep = [i for i, tag in enumerate(builtin_tags) if tag not in owner] if include_builtin else []
    tags = [builtin_tags[i] for i in keep]
    thresholds = [builtin_thresholds[i] for i in keep]
    sources = [""] * len(keep)
    for pack in packs:
        tags.extend(pack.tags)
        thresholds.extend(pack.thresholds)
        sources.extend([pack.name] * len(pack.tags))
    return MergedVocabulary(tags=tags, thresholds=thresholds, sources=sources, builtin_keep=keep)


def load_pack_embeddings(pack_dir: str | Path, manifest: PackManifest) -> Any:
    """Return the pack's ``label_embed`` tensor, checked against its manifest (needs torch + safetensors)."""
    from safetensors.torch import load_file

    path = Path(pack_dir) / EMBEDDINGS_FILE
    if not path.exists():
        raise PackError(f"no {EMBEDDINGS_FILE} in {pack_dir}")
    embeds = load_file(str(path)).get("label_embed")
    expected = (len(manifest.tags) * DESCRIPTIONS_PER_TAG, EMBED_DIM)
    if embeds is None or tuple(embeds.shape) != expected:
        shape = None if embeds is None else tuple(embeds.shape)
        raise PackError(f"pack {manifest.name!r}: label_embed is {shape}, expected {expected}")
    return embeds
