"""Tests for the RAM++ backend scaffold (issue #3)."""

import pytest

from argus_lens.backends.output import BackendOutput
from argus_lens.backends.ram import DEFAULT_MODEL_ID, RamBackend


def test_ram_backend_metadata():
    backend = RamBackend()
    assert backend.name == "ram"
    assert backend.style == "photo"
    assert backend.requires_gpu is True


def test_ram_backend_not_available_while_scaffold():
    backend = RamBackend()
    assert backend.is_available() is False
    assert "not yet implemented" in (backend.availability_reason() or "")


def test_constructor_defaults_and_overrides():
    default = RamBackend()
    assert default._model_id == DEFAULT_MODEL_ID
    assert default._threshold == 0.35

    custom = RamBackend(model_id="org/custom-ram", threshold=0.6)
    assert custom._model_id == "org/custom-ram"
    assert custom._threshold == 0.6


def test_build_output_sets_source_and_scores():
    backend = RamBackend()
    out = backend._build_output([("mountain", 0.9), ("lake", 0.7)])
    assert isinstance(out, BackendOutput)
    assert [t.label for t in out.tags] == ["mountain", "lake"]
    assert all(t.source == "ram" for t in out.tags)
    assert out.tags[0].score == 0.9


def test_load_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        RamBackend().load()


def test_inference_not_yet_implemented():
    backend = RamBackend()
    with pytest.raises(NotImplementedError):
        backend.annotate_image(None)
    # caption_image is a shim over annotate_image, so it must propagate too.
    with pytest.raises(NotImplementedError):
        backend.caption_image(None)
