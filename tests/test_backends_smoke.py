"""Backend smoke tests (issue #8).

Backends had no test coverage. These import each backend (skipping when its
optional dependency is absent) and assert the class-level contract without
loading any weights or hitting the network. Real inference tests should be
added per backend as they are implemented.
"""

import importlib

import pytest

from argus_lens.backends.base import BackendKind, CaptionBackend

# (module, class_name, optional_dependency_to_importorskip | None)
BACKENDS = [
    ("argus_lens.backends.ram", "RamBackend", None),
    ("argus_lens.backends.wd14", "WD14Backend", "onnxruntime"),
    ("argus_lens.backends.florence2", "Florence2Backend", "torch"),
    ("argus_lens.backends.blip2", "BLIP2Backend", "torch"),
    ("argus_lens.backends.openai", "OpenAIBackend", "openai"),
    ("argus_lens.backends.replicate", "ReplicateBackend", "replicate"),
    ("argus_lens.backends.hf_inference", "HFInferenceBackend", None),
    ("argus_lens.backends.nvidia_nim", "NVIDIANIMBackend", None),
]


@pytest.mark.parametrize(("module", "class_name", "dep"), BACKENDS)
def test_backend_class_contract(module, class_name, dep):
    if dep is not None:
        pytest.importorskip(dep)
    cls = getattr(importlib.import_module(module), class_name)

    assert issubclass(cls, CaptionBackend)
    assert isinstance(cls.name, str) and cls.name
    assert isinstance(cls.kind, BackendKind)
    assert cls.style in {"photo", "anime"}
    assert isinstance(cls.requires_gpu, bool)
