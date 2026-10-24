"""End-to-end test of workflows/curate_index.py with stub backends (no models, no network)."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

import argus_lens
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline

_SCRIPT = Path(__file__).resolve().parents[1] / "workflows" / "curate_index.py"
_spec = importlib.util.spec_from_file_location("curate_index", _SCRIPT)
curate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curate)

# Per-image outputs keyed by the image's solid colour.
OUTPUTS = {
    (200, 0, 0): (
        "1girl, solo, brown_hair, red_dress, kitchen, cooking",
        "A woman cooks in a kitchen, wearing a red dress.",
    ),
    (0, 200, 0): ("1girl, solo, brown_hair, jeans, park, walking", "A woman walks through a park."),
    (0, 0, 200): ("2girls, multiple_girls, blonde_hair, red_dress, kitchen", "Two women stand in a kitchen."),
    (9, 9, 9): ("1girl, solo, brown_hair, red_dress, beach", "A woman in a red dress on a beach."),
}


class _ColourStub(CaptionBackend):
    """Backend whose output depends on the image's top-left pixel."""

    def __init__(self, index: int) -> None:
        self.name = f"stub{index}"
        self._index = index

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image) -> str:
        """Look the output up by nearest pixel colour (JPEG shifts values slightly)."""
        px = image.convert("RGB").getpixel((0, 0))
        key = min(OUTPUTS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, px, strict=True)))
        return OUTPUTS[key][self._index]

    def unload(self) -> None:
        """No-op."""


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A two-person library with colliding filenames, a duplicate, and a stubbed engine."""
    src = tmp_path / "src"
    layout = {
        "alice/pics/s1/IMG_1.jpg": (200, 0, 0),
        "alice/pics/s2/IMG_1.jpg": (0, 200, 0),
        "alice/pics/s2/IMG_2.jpg": (0, 0, 200),
        "alice/pics/s3/IMG_9.jpg": (9, 9, 9),
        "bob/pics/s1/IMG_1.jpg": (200, 0, 0),  # byte-identical to alice/s1/IMG_1 -> duplicate
        "bob/notes/readme.jpg": (0, 200, 0),  # outside the layout -> skipped
    }
    for rel, colour in layout.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), colour).save(path, quality=100)

    real = argus_lens.ArgusLens
    monkeypatch.setattr(
        argus_lens, "ArgusLens", lambda **kw: real(backend=HybridPipeline(_ColourStub(0), _ColourStub(1)))
    )
    return src, tmp_path / "run"


def _csv(path: Path) -> list[dict]:
    """Read a CSV into dicts."""
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_end_to_end(library, tmp_path, capsys) -> None:
    """inventory -> caption -> assemble -> index -> query -> export."""
    src, run = library
    curate.main(["inventory", str(src), str(run)])
    manifest = [json.loads(line) for line in (run / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 5 and sum("duplicate_of" in r for r in manifest) == 1
    assert {r["name"] for r in manifest} == {"alice", "bob"}

    curate.main(["caption", str(run)])
    assert len((run / "raw.jsonl").read_text().splitlines()) == 4  # duplicate captioned once
    curate.main(["caption", str(run)])  # resume: nothing left to do
    assert len((run / "raw.jsonl").read_text().splitlines()) == 4
    with pytest.raises(SystemExit, match="different caption settings"):
        curate.main(["caption", str(run), "--wd14-threshold", "0.5"])

    curate.main(["assemble", str(run), "--trigger", "sks"])
    for target in ("sdxl", "flux"):
        rows = [json.loads(line) for line in (run / f"captions.{target}.jsonl").read_text().splitlines()]
        assert all(r["variants"]["training"].startswith("sks") for r in rows)
    review = _csv(run / "review_queue.csv")
    assert [r["rel_path"] for r in review] == ["alice/pics/s2/IMG_2.jpg"]
    assert "multiple_people" in review[0]["reasons"]

    curate.main(["index", str(run), "--taxonomy", str(_SCRIPT.parent / "taxonomy.example.json")])
    index = _csv(run / "index.csv")
    assert {"category": "wardrobe", "tag": "dress"} in [{k: r[k] for k in ("category", "tag")} for r in index]
    assert any(r["category"] == "setting" and r["tag"] == "home" for r in index)  # kitchen -> home
    assert any(r["category"] == "people" and r["tag"] == "multiple" for r in index)
    outliers = _csv(run / "identity_outliers.csv")
    assert [o["rel_path"] for o in outliers] == ["alice/pics/s2/IMG_2.jpg"]  # blonde vs alice's brown
    assert outliers[0]["expected"] == "brown hair"

    capsys.readouterr()
    curate.main(["query", str(run), "--all", "wardrobe:red dress", "setting:kitchen", "--not", "people:multiple"])
    out = capsys.readouterr().out.strip().splitlines()
    assert [line.split("\t")[2] for line in out] == ["alice/pics/s1/IMG_1.jpg"]

    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(out))
    dest = tmp_path / "dataset"
    curate.main(["export", str(run), str(dest), "--target", "flux", "--ids", str(ids)])
    txts = list(dest.rglob("*.txt"))
    assert len(txts) == 1 and txts[0].read_text().startswith("sks, A woman cooks in a kitchen")
    assert next(dest.rglob("*.jpg")).is_symlink()

    curate.main(["export", str(run), str(tmp_path / "solo"), "--single-person"])
    assert len(list((tmp_path / "solo").rglob("*.txt"))) == 3


def test_parse_count() -> None:
    """VLM headcount answers parse from digits or number words."""
    assert curate._parse_count("2") == 2
    assert curate._parse_count("Two people.") == 2
    assert curate._parse_count("none") == 0
    assert curate._parse_count("unclear") is None


def test_run_json_records_env_configured_settings(library, monkeypatch) -> None:
    """Settings supplied via env vars land in run.json, not as nulls."""
    src, run = library
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_MODEL", "qwen3-vl:8b")
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_BASE_URL", "http://ollama.lan:11434/v1")
    curate.main(["inventory", str(src), str(run)])
    curate.main(["caption", str(run), "--limit", "1"])
    caption = json.loads((run / "run.json").read_text())["caption"]
    assert caption["model_id"] == "qwen3-vl:8b"
    assert caption["base_url"] == "http://ollama.lan:11434/v1"
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_MODEL", "llava")
    with pytest.raises(SystemExit, match="different caption settings"):
        curate.main(["caption", str(run)])
