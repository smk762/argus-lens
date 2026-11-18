"""Model-repo Python runs only when the operator allows it with ``HF_TRUST_REMOTE_CODE`` (#131).

``trust_remote_code=True`` imports whatever code a Hugging Face repo ships. Every loader
resolves the flag through :mod:`argus_lens.remote_code`; these tests pin the default
(off), check that each verifier honours it, and scan the source so a new loader can't
hardcode it on.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from argus_lens.backends import florence2 as florence2_module
from argus_lens.backends.florence2 import Florence2Backend
from argus_lens.reconcile import build_verifier
from argus_lens.reconcile.verifiers import florence as florence_verifier_module
from argus_lens.reconcile.verifiers.florence import FlorenceGroundingVerifier
from argus_lens.reconcile.verifiers.molmo import MolmoVerifier
from argus_lens.remote_code import RemoteCodeRefused, trust_remote_code_from_env

SRC = Path(__file__).resolve().parents[1] / "src" / "argus_lens"


@pytest.fixture
def remote_code_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``HF_TRUST_REMOTE_CODE`` in the environment."""
    monkeypatch.delenv("HF_TRUST_REMOTE_CODE", raising=False)


@pytest.fixture
def remote_code_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HF_TRUST_REMOTE_CODE=true``."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", "true")


@pytest.mark.parametrize(
    ("value", "expected"), [("", False), ("0", False), ("no", False), ("1", True), (" TRUE ", True)]
)
def test_env_flag_is_off_unless_set_truthy(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    """Only ``1`` / ``true`` / ``yes`` turn it on."""
    monkeypatch.setenv("HF_TRUST_REMOTE_CODE", value)
    assert trust_remote_code_from_env() is expected


class _FakeModel:
    """Stands in for a transformers model: records where it was moved."""

    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> _FakeModel:
        """Record *device* and return self, like ``nn.Module.to``."""
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        """No-op, like ``nn.Module.eval`` on a loaded model."""
        return self


def _record_loads(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType) -> list[dict[str, Any]]:
    """Replace *module*'s ``load_florence`` with one that records its arguments."""
    calls: list[dict[str, Any]] = []

    def fake_load(model_id: str, *, trust_remote_code: bool, dtype: Any = None) -> tuple[object, _FakeModel]:
        calls.append({"model_id": model_id, "trust_remote_code": trust_remote_code})
        return object(), _FakeModel()

    monkeypatch.setattr(module, "load_florence", fake_load)
    return calls


# --------------------------------------------------------------------------- #
# Florence verifier
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("remote_code_off")
def test_florence_verifier_loads_native_weights_without_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default the verifier loads the native weights through the native class."""
    calls = _record_loads(monkeypatch, florence_verifier_module)
    verifier = FlorenceGroundingVerifier(device="cpu")
    verifier._ensure_model()
    assert calls == [{"model_id": "florence-community/Florence-2-base", "trust_remote_code": False}]
    assert verifier._model.device == "cpu"


@pytest.mark.usefixtures("remote_code_off")
def test_florence_verifier_refuses_legacy_weights_without_the_flag() -> None:
    """Naming the remote-code weights without permission fails at build time, not as a silent abstain."""
    with pytest.raises(RemoteCodeRefused, match="HF_TRUST_REMOTE_CODE"):
        FlorenceGroundingVerifier(model_id="microsoft/Florence-2-large")
    with pytest.raises(RemoteCodeRefused):
        build_verifier("florence", model_id="microsoft/Florence-2-base")


@pytest.mark.usefixtures("remote_code_on")
def test_florence_verifier_uses_legacy_weights_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag set the verifier matches the caption backend: legacy weights, remote code on."""
    calls = _record_loads(monkeypatch, florence_verifier_module)
    FlorenceGroundingVerifier()._ensure_model()
    assert calls == [{"model_id": "microsoft/Florence-2-base", "trust_remote_code": True}]


# --------------------------------------------------------------------------- #
# Molmo verifier
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("remote_code_off")
def test_molmo_verifier_is_refused_without_the_flag() -> None:
    """Molmo always needs remote code, so without permission it can't be built at all."""
    with pytest.raises(RemoteCodeRefused, match="HF_TRUST_REMOTE_CODE"):
        MolmoVerifier()
    with pytest.raises(RemoteCodeRefused):
        build_verifier("molmo")


@pytest.mark.usefixtures("remote_code_off")
def test_molmo_verifier_with_an_injected_answer_needs_no_flag() -> None:
    """An injected *answer_fn* loads no model, so it needs no permission."""
    MolmoVerifier(answer_fn=lambda image, question: "red")


@pytest.mark.usefixtures("remote_code_on")
def test_molmo_verifier_builds_when_allowed() -> None:
    """With the flag set Molmo builds; the model itself still loads lazily."""
    verifier = build_verifier("molmo")
    assert isinstance(verifier, MolmoVerifier)
    assert verifier._model is None


@pytest.mark.usefixtures("remote_code_off")
def test_explicit_argument_overrides_the_environment() -> None:
    """``trust_remote_code=True`` passed in code is an explicit opt-in."""
    MolmoVerifier(trust_remote_code=True)
    assert FlorenceGroundingVerifier(trust_remote_code=True).model_id == "microsoft/Florence-2-base"


# --------------------------------------------------------------------------- #
# Caption backend shares the loader
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("remote_code_off")
def test_caption_backend_loads_through_the_shared_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend and the verifier go through one ``load_florence``, so they can't drift apart."""
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "fp16", "fp32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = _record_loads(monkeypatch, florence2_module)
    _, model, device = Florence2Backend()._loader("cpu")
    assert calls == [{"model_id": "florence-community/Florence-2-base", "trust_remote_code": False}]
    assert (model.device, device) == ("cpu", "cpu")


# --------------------------------------------------------------------------- #
# Source scan
# --------------------------------------------------------------------------- #

# The only places allowed to pass trust_remote_code=True: the shared Florence loader's legacy
# branch and Molmo, whose constructor refuses unless the operator allowed it.
ALLOWED = {"backends/florence2.py": 2, "reconcile/verifiers/molmo.py": 2}


def test_no_loader_hardcodes_remote_code_on() -> None:
    """A literal ``trust_remote_code=True`` argument appears only where the flag has already been checked."""
    found = {}
    for path in SRC.rglob("*.py"):
        hits = sum(
            1
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.keyword)
            and node.arg == "trust_remote_code"
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        )
        if hits:
            found[str(path.relative_to(SRC))] = hits
    assert found == ALLOWED
