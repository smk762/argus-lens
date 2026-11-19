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


@pytest.mark.parametrize(
    ("answer", "count"),
    [("One person, around 30 years old.", 1), ("Two people, one in their 20s.", 2), ("3 people, two seated", 3)],
)
def test_parse_count_takes_the_first_count(answer: str, count: int) -> None:
    """A later number (an age, a decade) must not override a spelled-out headcount."""
    assert curate._parse_count(answer) == count


@pytest.mark.parametrize("answer", ["No one is visible in the image.", "Nobody.", "no people", "No person is shown."])
def test_parse_count_negations_are_zero(answer: str) -> None:
    """'No one' must not parse as 1 via the word 'one'."""
    assert curate._parse_count(answer) == 0


def test_read_jsonl_skips_line_torn_mid_utf8(tmp_path, capsys) -> None:
    """A trailing append cut inside a multi-byte character is skipped, not fatal."""
    path = tmp_path / "raw.jsonl"
    good = json.dumps({"caption": "café"}, ensure_ascii=False).encode()
    torn = json.dumps({"caption": "naïve"}, ensure_ascii=False).encode()
    path.write_bytes(good + b"\n" + torn[: torn.index("ï".encode()) + 1])
    assert list(curate.read_jsonl(path)) == [{"caption": "café"}]
    assert "skipping unreadable line" in capsys.readouterr().err


def _export_run(tmp_path: Path, source_root: str) -> tuple[Path, Path]:
    """A minimal run dir (inventory + sdxl captions) for one image under *source_root*."""
    image = tmp_path / "src" / "alice" / "pics" / "s1" / "IMG_1.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image)
    run = tmp_path / "run"
    run.mkdir()
    (run / "inventory.json").write_text(json.dumps({"source_root": source_root}))
    row = {
        "image_id": "abcdef0123456789",
        "name": "alice",
        "shoot_id": "s1",
        "rel_path": "alice/pics/s1/IMG_1.jpg",
        "subjects": {},
        "variants": {"training": "sks, woman"},
    }
    (run / "captions.sdxl.jsonl").write_text(json.dumps(row) + "\n")
    return run, image


def test_export_symlinks_absolute_with_relative_source_root(tmp_path, monkeypatch) -> None:
    """A relative source_root still yields links that resolve (not relative to the link's dir)."""
    monkeypatch.chdir(tmp_path)
    run, image = _export_run(tmp_path, "src")
    dest = tmp_path / "dataset"
    curate.main(["export", str(run), str(dest)])
    link = next(dest.rglob("*.jpg"))
    assert link.is_symlink() and link.exists()
    assert link.resolve() == image.resolve()


def test_export_replaces_dangling_symlink(tmp_path) -> None:
    """Re-export over a link whose old target vanished recreates it instead of crashing."""
    run, image = _export_run(tmp_path, str(tmp_path / "src"))
    dest = tmp_path / "dataset"
    curate.main(["export", str(run), str(dest)])
    link = next(dest.rglob("*.jpg"))
    link.unlink()
    link.symlink_to(tmp_path / "gone.jpg")  # dangling, as if the library moved
    curate.main(["export", str(run), str(dest)])
    assert link.is_symlink() and link.resolve() == image.resolve()
    curate.main(["export", str(run), str(dest), "--copy"])  # copy mode replaces the link with a real file
    assert not link.is_symlink() and link.read_bytes() == image.read_bytes()


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


def test_run_json_records_count_url_and_refuses_a_different_one(library, monkeypatch) -> None:
    """The counter's endpoint is a caption setting: a resume against another one is refused."""
    from argus_lens.backends.openai_compat import OpenAICompatBackend

    monkeypatch.setattr(OpenAICompatBackend, "caption_image", lambda self, image: "1")
    src, run = library
    curate.main(["inventory", str(src), str(run)])
    curate.main(["caption", str(run), "--limit", "1", "--count-model", "m", "--count-url", "http://a:1/v1"])
    assert json.loads((run / "run.json").read_text())["caption"]["count_url"] == "http://a:1/v1"
    with pytest.raises(SystemExit, match="count_url"):
        curate.main(["caption", str(run), "--count-model", "m", "--count-url", "http://b:1/v1"])


def test_forced_resume_is_recorded_in_run_json(library) -> None:
    """--force with different settings keeps the original ones and logs the forced mix."""
    src, run = library
    curate.main(["inventory", str(src), str(run)])
    curate.main(["caption", str(run), "--limit", "1"])
    curate.main(["caption", str(run), "--limit", "1", "--wd14-threshold", "0.5", "--force"])
    config = json.loads((run / "run.json").read_text())
    assert config["caption"]["wd14_threshold"] != 0.5
    (forced,) = config["forced_resumes"]
    assert forced["caption"]["wd14_threshold"] == 0.5
    assert forced["diffs"]["wd14_threshold"]["now"] == 0.5
