"""Output exporters — txt sidecars, JSON/JSONL, CSV."""

from __future__ import annotations

from pathlib import Path

from argus_lens.types import CaptionResult


def export_results(
    results: dict[str, CaptionResult],
    output_dir: Path,
    fmt: str = "txt",
) -> None:
    """Export captioning results in the specified format."""
    if fmt == "txt":
        from argus_lens.exporters.text import export_txt

        export_txt(results, output_dir)
    elif fmt in ("json", "jsonl"):
        from argus_lens.exporters.json_export import export_json, export_jsonl

        if fmt == "json":
            export_json(results, output_dir / "captions.json")
        else:
            export_jsonl(results, output_dir / "captions.jsonl")
    elif fmt == "csv":
        from argus_lens.exporters.csv_export import export_csv

        export_csv(results, output_dir / "captions.csv")
    else:
        raise ValueError(f"Unknown export format: {fmt!r}. Choose from: txt, json, jsonl, csv")
