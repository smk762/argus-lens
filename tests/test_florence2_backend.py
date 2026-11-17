"""Florence-2 backend checks that need neither torch nor transformers (both are faked)."""

from __future__ import annotations

import sys
import types

import pytest

from argus_lens.backends.florence2 import Florence2Backend


@pytest.mark.parametrize("task", ["<OD>", "<DENSE_REGION_CAPTION>", "<OCR_WITH_REGION>"])
def test_non_caption_tasks_are_rejected_up_front(task: str) -> None:
    """Tasks that return boxes/regions fail at construction, not with AttributeError on every caption."""
    with pytest.raises(ValueError, match="does not produce a caption"):
        Florence2Backend(task=task)


@pytest.mark.parametrize("task", ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>"])
def test_caption_tasks_are_accepted(task: str) -> None:
    """Every text-producing task is still allowed."""
    Florence2Backend(task=task)


def _fake_stack(monkeypatch: pytest.MonkeyPatch, version: str, *, native: bool) -> None:
    """Install fake ``torch`` / ``transformers`` modules of the given shape."""
    transformers = types.ModuleType("transformers")
    transformers.__version__ = version
    if native:
        transformers.Florence2ForConditionalGeneration = object
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_native_weights_need_the_native_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native weights on a transformers without Florence2ForConditionalGeneration are unavailable."""
    _fake_stack(monkeypatch, "4.45.0", native=False)
    backend = Florence2Backend(trust_remote_code=False)
    assert backend.is_available() is False
    assert "Florence2ForConditionalGeneration" in (backend.availability_reason() or "")

    _fake_stack(monkeypatch, "4.56.0", native=True)
    assert backend.is_available() is True
    assert backend.availability_reason() is None


def test_legacy_weights_need_old_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy remote-code weights are unavailable on transformers >= 4.52."""
    backend = Florence2Backend(trust_remote_code=True)
    _fake_stack(monkeypatch, "4.56.1", native=True)
    assert backend.is_available() is False
    assert "transformers<4.52" in (backend.availability_reason() or "")

    _fake_stack(monkeypatch, "4.51.3", native=False)
    assert backend.is_available() is True
