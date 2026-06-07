"""Tests for the RAM++ backend scaffold (issue #3)."""

import pytest

from argus_lens.backends.output import BackendOutput
from argus_lens.backends.ram import RamBackend


def test_ram_backend_metadata():
    backend = RamBackend()
    assert backend.name == "ram"
    assert backend.style == "photo"
    assert backend.requires_gpu is True


def test_build_output_sets_source_and_scores():
    backend = RamBackend()
    out = backend._build_output([("mountain", 0.9), ("lake", 0.7)])
    assert isinstance(out, BackendOutput)
    assert [t.label for t in out.tags] == ["mountain", "lake"]
    assert all(t.source == "ram" for t in out.tags)
    assert out.tags[0].score == 0.9


def test_inference_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        RamBackend().load()
