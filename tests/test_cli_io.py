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
