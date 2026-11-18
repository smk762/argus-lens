"""The .txt sidecar exporter never writes outside its output directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus_lens.exporters.text import export_txt
from argus_lens.types import CaptionResult


def _r(caption: str = "a cat") -> CaptionResult:
    """A minimal caption result."""
    return CaptionResult(final_caption=caption)


def test_writes_nested_sidecars_inside_output_dir(tmp_path: Path) -> None:
    """Relative names, including sub-directories, still land under output_dir."""
    export_txt({"IMG_1.jpg": _r("one"), "shoot_1/IMG_2.png": _r("two")}, tmp_path)
    assert (tmp_path / "IMG_1.txt").read_text() == "one"
    assert (tmp_path / "shoot_1" / "IMG_2.txt").read_text() == "two"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.jpg",
        "a/../../escape.jpg",
        "..\\escape.jpg",
        "/etc/cron.d/x.jpg",
        "C:\\Windows\\x.jpg",
        "",
    ],
)
def test_rejects_names_that_escape(tmp_path: Path, name: str) -> None:
    """Absolute or ``..`` names raise and write nothing, not even the good keys."""
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        export_txt({"ok.jpg": _r(), name: _r("pwned")}, out)
    assert not (out / "ok.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_rejects_a_sidecar_symlinked_outside(tmp_path: Path) -> None:
    """An existing sidecar symlink pointing outside output_dir is not followed."""
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (out / "x.txt").symlink_to(victim)
    with pytest.raises(ValueError):
        export_txt({"x.jpg": _r("pwned")}, out)
    assert victim.read_text() == "keep me"


def test_names_sharing_a_sidecar_raise_and_write_nothing(tmp_path: Path) -> None:
    """IMG_1.jpg and IMG_1.png map to one IMG_1.txt: refuse rather than overwrite one caption."""
    with pytest.raises(ValueError, match="share the sidecar"):
        export_txt({"ok.jpg": _r(), "IMG_1.jpg": _r("one"), "IMG_1.png": _r("two")}, tmp_path)
    assert list(tmp_path.glob("*.txt")) == []
