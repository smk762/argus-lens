"""CSV exporter."""

from __future__ import annotations

import csv
from pathlib import Path

from argus_lens.types import CaptionResult

# Leading characters that make Excel, LibreOffice and Google Sheets read a cell
# as a formula (OWASP "CSV injection"). '-' is included: spreadsheets parse
# "-foo" as a formula too (it renders as #NAME? at best), so escaping it keeps
# the text intact rather than corrupting it.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _neutralise(value: object) -> object:
    """Prefix a string cell that starts with a formula trigger with ``'``.

    Names come from file/Immich asset names and captions are model output from
    image content, so neither is trusted. The leading quote makes spreadsheets
    show the cell as text; it is visible to plain CSV readers, which is the
    usual trade-off for this mitigation.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def export_csv(results: dict[str, CaptionResult], output_path: Path) -> None:
    """Write results as a CSV file with key columns.

    Every string cell starting with ``=``, ``+``, ``-``, ``@``, tab or CR is
    prefixed with a single quote so opening the file in a spreadsheet cannot
    evaluate it as a formula.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "final_caption",
        "selected_category",
        "backend_name",
        "training",
        "zeroshot",
        "raw_tags",
        "raw_prose",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in results.items():
            row = {
                "name": name,
                "final_caption": result.final_caption,
                "selected_category": result.selected_category,
                "backend_name": result.backend_name,
                "training": result.caption_variants.get("training", ""),
                "zeroshot": result.caption_variants.get("zeroshot", ""),
                "raw_tags": result.raw_tags,
                "raw_prose": result.raw_prose,
            }
            writer.writerow({k: _neutralise(v) for k, v in row.items()})
