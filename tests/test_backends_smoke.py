"""Backend smoke tests (issue #8).

Backends had no test coverage. These import each backend and assert the
class-level contract without loading any weights or hitting the network.

Every backend imports its heavy dependency (torch, onnxruntime, the cloud
SDKs, ...) lazily inside ``load()``/inference, so the module import and these
class-level checks need no optional dependency installed — they run in full CI.
If a backend ever moves an optional dependency to module import time, we skip
on its absence rather than fail, but a missing internal module is still a hard
error. Real inference tests should be added per backend as they are implemented.
"""

import importlib

import pytest

from argus_lens.backends.base import CaptionBackend
from argus_lens.types import BackendKind

# (module, class_name)
BACKENDS = [
    ("argus_lens.backends.ram", "RamBackend"),
    ("argus_lens.backends.wd14", "WD14Backend"),
    ("argus_lens.backends.florence2", "Florence2Backend"),
    ("argus_lens.backends.blip2", "BLIP2Backend"),
    ("argus_lens.backends.openai", "OpenAIBackend"),
    ("argus_lens.backends.replicate", "ReplicateBackend"),
    ("argus_lens.backends.hf_inference", "HFInferenceBackend"),
    ("argus_lens.backends.nvidia_nim", "NVIDIANIMBackend"),
]


@pytest.mark.parametrize(("module", "class_name"), BACKENDS)
def test_backend_class_contract(module, class_name):
    """Each backend class subclasses CaptionBackend and declares valid name/kind/style/requires_gpu."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # Only skip for a genuinely-absent optional third-party dependency;
        # never mask a missing/broken internal (argus_lens) module.
        if (exc.name or "").startswith("argus_lens"):
            raise
        pytest.skip(f"optional dependency {exc.name!r} not installed for {module}")

    cls = getattr(mod, class_name)

    assert issubclass(cls, CaptionBackend)
    assert isinstance(cls.name, str) and cls.name
    assert isinstance(cls.kind, BackendKind)
    assert cls.style in {"photo", "anime"}
    assert isinstance(cls.requires_gpu, bool)
