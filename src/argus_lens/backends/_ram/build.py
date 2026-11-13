"""Build a RAM++ vocabulary pack from tags or pre-written descriptions (#62)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus_lens.backends._ram.describe import fit_descriptions
from argus_lens.backends._ram.packs import (
    DEFAULT_PACK_THRESHOLD,
    DESCRIPTIONS_FILE,
    EMBEDDINGS_FILE,
    PackError,
    PackManifest,
)


@dataclass
class TagInput:
    """Parsed builder input: tags in order, any per-tag thresholds, and any supplied descriptions."""

    tags: list[str]
    thresholds: dict[str, float]
    descriptions: dict[str, list[str]]


def read_tag_input(path: str | Path) -> TagInput:
    """Parse a tag list or a descriptions file.

    * ``.json``: upstream's ``[{"tag": ["desc", ...]}, ...]`` or a ``{"tag": ["desc", ...]}`` map.
    * anything else: one tag per line, optionally ``tag<TAB>threshold``; blank lines and
      ``#`` comments are skipped.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tags: list[str] = []
    thresholds: dict[str, float] = {}
    descriptions: dict[str, list[str]] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        items = list(data.items()) if isinstance(data, dict) else [kv for entry in data for kv in entry.items()]
        for tag, descs in items:
            if not isinstance(descs, list) or not all(isinstance(d, str) for d in descs):
                raise PackError(f"descriptions for {tag!r} must be a list of strings")
            tags.append(tag.strip())
            descriptions[tag.strip()] = descs
    else:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            tag, _, threshold = line.partition("\t")
            tags.append(tag.strip())
            if threshold.strip():
                try:
                    thresholds[tag.strip()] = float(threshold)
                except ValueError as exc:
                    raise PackError(f"bad threshold for {tag!r}: {threshold!r}") from exc
    if len(set(tags)) != len(tags):
        raise PackError(f"{path} lists a tag more than once")
    return TagInput(tags=tags, thresholds=thresholds, descriptions=descriptions)


def build_pack(
    out_dir: str | Path,
    *,
    name: str,
    tags: list[str],
    descriptions: Mapping[str, list[str]],
    embed: Callable[[list[str]], Any],
    thresholds: Mapping[str, float] | None = None,
    default_threshold: float = DEFAULT_PACK_THRESHOLD,
    description_source: dict[str, Any] | None = None,
    license: str | None = None,
) -> PackManifest:
    """Embed *descriptions* (51 per tag, fitted if needed) and write the pack to *out_dir*.

    *embed* maps a list of strings to a ``(n, 512)`` unit-normalised tensor, normally
    :meth:`argus_lens.backends._ram.embed.ClipTextEmbedder.embed`.
    """
    import torch
    from safetensors.torch import save_file

    thresholds = thresholds or {}
    missing = [t for t in tags if not descriptions.get(t)]
    if missing:
        raise PackError(f"no descriptions for {len(missing)} tag(s), e.g. {missing[:3]}")
    fitted = {tag: fit_descriptions(tag, descriptions[tag]) for tag in tags}
    manifest = PackManifest(
        name=name,
        tags=list(tags),
        thresholds=[float(thresholds.get(t, default_threshold)) for t in tags],
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        description_source=description_source or {},
        license=license,
    ).validate()

    flat = [d for tag in tags for d in fitted[tag]]
    embeds = embed(flat)
    if tuple(embeds.shape) != (len(flat), 512):
        raise PackError(f"embedder returned {tuple(embeds.shape)}, expected {(len(flat), 512)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"label_embed": embeds.to(torch.float16).contiguous()}, str(out / EMBEDDINGS_FILE))
    (out / DESCRIPTIONS_FILE).write_text(
        json.dumps([{tag: fitted[tag]} for tag in tags], indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest.write(out)
    return manifest
