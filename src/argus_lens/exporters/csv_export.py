"""CSV exporter."""

from __future__ import annotations

import csv
from pathlib import Path

from argus_lens.types import CaptionResult


def export_csv(results: dict[str, CaptionResult], output_path: Path) -> None:
    """Write results as a CSV file with key columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name", "final_caption", "selected_category", "backend_name",
        "training", "zeroshot", "raw_tags", "raw_prose",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "name": name,
                "final_caption": result.final_caption,
                "selected_category": result.selected_category,
                "backend_name": result.backend_name,
                "training": result.caption_variants.get("training", ""),
                "zeroshot": result.caption_variants.get("zeroshot", ""),
                "raw_tags": result.raw_tags,
                "raw_prose": result.raw_prose,
            })
