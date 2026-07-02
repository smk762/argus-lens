"""JSON / JSONL exporter."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from argus_lens.types import CaptionResult


def _result_to_dict(name: str, result: CaptionResult) -> dict:
    """Convert a ``CaptionResult`` to a dict with the image name included."""
    d = asdict(result)
    d["name"] = name
    return d


def export_json(results: dict[str, CaptionResult], output_path: Path) -> None:
    """Write all results as a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [_result_to_dict(name, r) for name, r in results.items()]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def export_jsonl(results: dict[str, CaptionResult], output_path: Path) -> None:
    """Write results as newline-delimited JSON (one line per image)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for name, result in results.items():
            f.write(json.dumps(_result_to_dict(name, result), ensure_ascii=False) + "\n")
