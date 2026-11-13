"""Golden-set loading for the evaluation harness.

Two shapes are accepted, so the harness is useful before any labelling work:

* **A directory of images** — every image becomes a reference-free
  :class:`EvalItem` (no ``expected_tags`` / ``target_caption``). Only the
  reference-free metrics apply.
* **A JSONL manifest** — one JSON object per line, each locating an image and
  optionally carrying ground-truth ``expected_tags``, a reference
  ``target_caption``, and per-item ``target_style`` / ``target_category`` /
  ``target_backend`` overrides. Image paths resolve relative to the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from argus_lens.connectors.filesystem import IMAGE_SUFFIXES


@dataclass(frozen=True)
class EvalItem:
    """One image to caption and score.

    ``expected_tags`` and ``target_caption`` are optional: when absent the item
    is scored by the reference-free metrics only.
    """

    image: Path
    expected_tags: tuple[str, ...] = ()
    target_caption: str | None = None
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    notes: str = ""

    @property
    def has_labels(self) -> bool:
        """True when reference-based metrics can be computed for this item."""
        return bool(self.expected_tags) or bool(self.target_caption)


def _image_from_dir(directory: Path) -> list[EvalItem]:
    """Every image directly under (and below) *directory* as an unlabelled item."""
    images = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return [EvalItem(image=p) for p in images]


def _resolve_image(manifest_dir: Path, raw: str) -> Path:
    """Resolve a manifest ``image`` field relative to the manifest's own directory."""
    p = Path(raw)
    return p if p.is_absolute() else (manifest_dir / p)


def _item_from_row(manifest_dir: Path, row: dict, line_no: int) -> EvalItem:
    """Build an :class:`EvalItem` from one manifest row, validating required fields."""
    raw_image = row.get("image")
    if not isinstance(raw_image, str) or not raw_image.strip():
        raise ValueError(f"manifest line {line_no}: missing/invalid 'image'")
    tags = row.get("expected_tags") or []
    if not isinstance(tags, list):
        raise ValueError(f"manifest line {line_no}: 'expected_tags' must be a list")
    return EvalItem(
        image=_resolve_image(manifest_dir, raw_image),
        expected_tags=tuple(str(t).strip() for t in tags if str(t).strip()),
        target_caption=(str(row["target_caption"]) if row.get("target_caption") else None),
        target_style=str(row.get("target_style", "photo")),
        target_category=str(row.get("target_category", "identity")),
        target_backend=str(row.get("target_backend", "sdxl")),
        notes=str(row.get("notes", "")),
    )


def _items_from_manifest(manifest: Path) -> list[EvalItem]:
    """Parse a JSONL manifest into items, naming the offending line on error."""
    items: list[EvalItem] = []
    manifest_dir = manifest.parent
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_no}: expected a JSON object")
        items.append(_item_from_row(manifest_dir, row, line_no))
    return items


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load an eval dataset from a directory of images or a ``.jsonl`` manifest.

    A directory yields unlabelled (reference-free) items; a ``.jsonl`` file
    yields items that may carry ground-truth labels. Raises ``FileNotFoundError``
    if *path* does not exist and ``ValueError`` for a malformed manifest.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")
    if p.is_dir():
        return _image_from_dir(p)
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        return _items_from_manifest(p)
    raise ValueError(f"unsupported dataset path (want a directory or .jsonl): {path}")
