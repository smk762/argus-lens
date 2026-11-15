"""Tests for file-targeted export and CSV formula-injection escaping."""

import csv
import json

import pytest

from argus_lens.exporters import export_results_to_file
from argus_lens.exporters.csv_export import export_csv
from argus_lens.types import CaptionResult


def _rows(path):
    """Read a CSV export back as a list of dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("fmt", ["txt", "json", "jsonl", "csv"])
def test_export_results_to_file_writes_exactly_the_named_path(tmp_path, fmt):
    """The named file is written and nothing else lands in its directory."""
    out = tmp_path / "out" / f"mine.{fmt}"
    export_results_to_file({"a.png": CaptionResult(final_caption="a cat")}, out, fmt)
    assert [p.name for p in out.parent.iterdir()] == [out.name]
    text = out.read_text(encoding="utf-8")
    assert "a cat" in text
    if fmt == "json":
        assert json.loads(text)[0]["name"] == "a.png"


def test_export_results_to_file_rejects_unknown_format(tmp_path):
    """An unknown format raises rather than silently writing nothing."""
    with pytest.raises(ValueError):
        export_results_to_file({}, tmp_path / "x", "xml")


@pytest.mark.parametrize("payload", ['=HYPERLINK("http://evil/")', "+1+1", "-2+3", "@SUM(A1)", "\tx", "\rx"])
def test_csv_escapes_formula_triggers(tmp_path, payload):
    """Cells starting with a formula trigger are prefixed with a single quote, in every column."""
    out = tmp_path / "c.csv"
    result = CaptionResult(final_caption=payload, raw_tags=payload, caption_variants={"training": payload})
    export_csv({payload: result}, out)
    row = _rows(out)[0]
    for key in ("name", "final_caption", "raw_tags", "training"):
        assert row[key] == "'" + payload


def test_csv_leaves_ordinary_text_alone(tmp_path):
    """Captions not starting with a trigger are written unchanged."""
    out = tmp_path / "c.csv"
    export_csv({"a.png": CaptionResult(final_caption="a cat, = sign inside")}, out)
    row = _rows(out)[0]
    assert row["name"] == "a.png"
    assert row["final_caption"] == "a cat, = sign inside"
