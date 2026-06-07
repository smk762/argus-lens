"""Tests for provenance metadata (issue #2)."""

from argus_lens.backends.output import BackendOutput, Tag
from argus_lens.provenance import build_provenance


def test_build_provenance_records_source_and_score():
    out = BackendOutput(
        tags=[
            Tag("mountain", score=0.91, source="ram", region=(0, 0, 10, 10)),
            Tag("nature", score=None),
        ]
    )
    meta = build_provenance(out, backend_name="ram", min_score=0.5)

    assert meta["backend"] == "ram"
    assert meta["min_score"] == 0.5
    assert meta["tags"][0] == {
        "label": "mountain",
        "score": 0.91,
        "source": "ram",
        "region": [0, 0, 10, 10],
    }
    # Falls back to backend_name when the tag has no source; region stays None.
    assert meta["tags"][1]["source"] == "ram"
    assert meta["tags"][1]["region"] is None
