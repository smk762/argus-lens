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


def export_results_to_file(results: dict[str, CaptionResult], output_path: Path, fmt: str = "txt") -> None:
    """Export *results* to exactly *output_path* in format *fmt*.

    Unlike :func:`export_results` (which picks ``captions.<fmt>`` or per-image
    sidecars inside a directory), the caller names the file. ``txt`` writes the
    final caption(s), one per line.
    """
    if fmt == "txt":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(r.final_caption for r in results.values()), encoding="utf-8")
    elif fmt in ("json", "jsonl"):
        from argus_lens.exporters.json_export import export_json, export_jsonl

        (export_json if fmt == "json" else export_jsonl)(results, output_path)
    elif fmt == "csv":
        from argus_lens.exporters.csv_export import export_csv

        export_csv(results, output_path)
    else:
        raise ValueError(f"Unknown export format: {fmt!r}. Choose from: txt, json, jsonl, csv")
