"""CLI tests: single-image --output path and the serve bind address."""

import sys
import types

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

import argus_lens  # noqa: E402
from argus_lens.cli import app  # noqa: E402
from argus_lens.types import CaptionResult  # noqa: E402


class _FakeEngine:
    """Stands in for ArgusLens: no model, a fixed caption."""

    def __init__(self, **_kwargs) -> None:
        """Accept and ignore the CLI's engine kwargs."""

    def caption(self, path, **_kwargs) -> CaptionResult:
        """Return a canned caption for any image."""
        return CaptionResult(final_caption="a cat")


def test_caption_output_writes_the_named_file(tmp_path, monkeypatch):
    """--output lands at exactly the given path, and the message names it."""
    monkeypatch.setattr(argus_lens, "ArgusLens", _FakeEngine)
    monkeypatch.setenv("ARGUS_REFINE", "off")
    image = tmp_path / "img.png"
    image.write_bytes(b"")
    out = tmp_path / "out" / "mine.json"
    res = CliRunner().invoke(app, ["caption", str(image), "-f", "json", "-o", str(out), "--no-refine"])
    assert res.exit_code == 0, res.output
    assert out.exists()
    assert not (out.parent / "captions.json").exists()
    assert f"Exported to {out}" in res.output


def _capture_serve(monkeypatch, args):
    """Run `serve` with uvicorn/create_app stubbed; return the uvicorn.run kwargs."""
    seen = {}
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: seen.update(kw)))
    pytest.importorskip("fastapi")
    import argus_lens.server

    monkeypatch.setattr(argus_lens.server, "create_app", lambda **kw: object())
    res = CliRunner().invoke(app, ["serve", *args])
    assert res.exit_code == 0, res.output
    return seen


def test_serve_binds_loopback_by_default(monkeypatch):
    """A plain `serve` listens on 127.0.0.1 only."""
    assert _capture_serve(monkeypatch, [])["host"] == "127.0.0.1"


def test_serve_host_override(monkeypatch):
    """--host 0.0.0.0 still exposes the server (as the Docker images do)."""
    assert _capture_serve(monkeypatch, ["--host", "0.0.0.0"])["host"] == "0.0.0.0"


@pytest.mark.parametrize("kind", ["missing", "directory", "not_utf8"])
def test_eval_bad_baseline_fails_before_the_run(tmp_path, monkeypatch, kind):
    """An unreadable --baseline is a clean exit 1 before any captioning happens."""
    from PIL import Image

    import argus_lens.eval as lens_eval

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8)).save(images / "a.png")
    baseline = tmp_path / "baseline.json"
    if kind == "directory":
        baseline.mkdir()
    elif kind == "not_utf8":
        baseline.write_bytes(b"\xff\xfe\x00bad")
    ran: list[bool] = []

    def _fake_run_eval(*_args, **_kwargs):
        """Record that the expensive eval run started."""
        ran.append(True)
        raise RuntimeError("eval ran")

    monkeypatch.setattr(argus_lens, "ArgusLens", _FakeEngine)
    monkeypatch.setattr(lens_eval, "run_eval", _fake_run_eval)
    result = CliRunner().invoke(app, ["eval", str(images), "--baseline", str(baseline)])
    assert result.exit_code == 1
    assert not ran
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Could not read baseline" in result.output


def test_ram_vocab_build_saves_partial_descriptions_on_ctrl_c(tmp_path, monkeypatch):
    """Ctrl-C mid-way through description writing still saves the completed tags."""
    import json

    from argus_lens.backends._ram import describe

    class _InterruptedWriter:
        """Writes one tag's descriptions, then the user hits Ctrl-C."""

        model_id = "fake"
        base_url = "http://fake"

        def __init__(self, **_kwargs) -> None:
            """Accept and ignore the endpoint settings."""

        def describe_all(self, tags, *, on_progress=None, into):
            """Fill the first tag, then interrupt."""
            into[tags[0]] = ["a description"]
            raise KeyboardInterrupt

        def close(self) -> None:
            """Nothing to close."""

    monkeypatch.setattr(describe, "DescriptionWriter", _InterruptedWriter)
    source = tmp_path / "tags.txt"
    source.write_text("dress\nhat\n", "utf-8")
    out = tmp_path / "pack"
    result = CliRunner().invoke(app, ["ram-vocab", "build", str(source), "--out", str(out), "--name", "x"])
    assert result.exit_code != 0
    saved = json.loads((out / "descriptions.partial.json").read_text("utf-8"))
    assert saved == {"dress": ["a description"], "hat": []}
