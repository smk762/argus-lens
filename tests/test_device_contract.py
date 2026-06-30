"""Device-placement contract tests (#21).

Covers the `load(device)` contract refactor and its review follow-ups:

* wd14 ONNX provider selection (CPU pinning, CUDA preference, fallback) and the
  provider-normalised session cache key — all torch-free.
* Backwards-compatible optional `device` kwarg on the torch backends.
* `LocalBackend.resolve_device` override-vs-remembered behaviour.
* Thread-safe single `load()` under concurrent first use.
"""

import inspect
import sys
import threading
import time
import types

import pytest
from PIL import Image

from argus_lens.backends.base import CaptionBackend, LocalBackend
from argus_lens.backends.blip2 import BLIP2Backend
from argus_lens.backends.florence2 import Florence2Backend
from argus_lens.backends.wd14 import WD14Backend
from argus_lens.engine import ArgusLens


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), (1, 2, 3))


def _fake_onnxruntime(available: list[str]) -> types.ModuleType:
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: list(available)  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def ort_with_cuda(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _fake_onnxruntime(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )


@pytest.fixture
def ort_cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_onnxruntime(["CPUExecutionProvider"]))


# ── wd14 provider selection (torch-free) ────────────────────────────────────


def test_wd14_explicit_cpu_pins_cpu_even_with_cuda(ort_with_cuda):
    assert WD14Backend._select_providers("cpu") == ["CPUExecutionProvider"]


def test_wd14_auto_prefers_cuda_when_available(ort_with_cuda):
    assert WD14Backend._select_providers("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_wd14_cuda_intent_uses_cuda(ort_with_cuda):
    assert WD14Backend._select_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_wd14_auto_falls_back_to_cpu_without_cuda(ort_cpu_only):
    # No torch involved: provider choice is driven purely by ONNX Runtime, so
    # the [wd14-gpu] (onnxruntime-gpu, no torch) install still works.
    assert WD14Backend._select_providers("auto") == ["CPUExecutionProvider"]


def test_wd14_cache_key_collapses_equivalent_devices():
    # GPU-targeting intents ("auto"/"cuda") share one cache key; explicit CPU is
    # distinct. Derived from the device string alone (no onnxruntime import), so
    # the inference entrypoint stays import-light.
    assert WD14Backend._device_key("auto") == WD14Backend._device_key("cuda") == "gpu"
    assert WD14Backend._device_key("cuda:1") == "gpu"
    assert WD14Backend._device_key("cpu") == "cpu"


# ── Backwards compatibility: optional device kwarg on torch backends ────────


@pytest.mark.parametrize("cls", [Florence2Backend, BLIP2Backend])
def test_caption_image_keeps_optional_device_kwarg(cls):
    # Pre-0.3 callers used caption_image(img, device=...); keep it working.
    params = inspect.signature(cls.caption_image).parameters
    assert "device" in params, f"{cls.__name__}.caption_image dropped the device kwarg"
    assert params["device"].default is None


def test_resolve_device_override_vs_remembered():
    class _Dummy(LocalBackend):
        def caption_image(self, image: Image.Image, device: str | None = None) -> str:
            return ""

        def unload(self) -> None:
            pass

    b = _Dummy()
    b.load("cpu")
    assert b.resolve_device() == "cpu"  # remembered via load()
    assert b.resolve_device("cuda:1") == "cuda:1"  # explicit override wins (back-compat)


# ── Thread-safe single load() ───────────────────────────────────────────────


class _SlowLoadBackend(CaptionBackend):
    """Records load() calls; load() is slow to widen the race window."""

    name = "slowload"
    style = "photo"

    def __init__(self) -> None:
        self.load_calls = 0
        self.seen_device: str | None = None

    def load(self, device: str = "auto") -> None:
        time.sleep(0.05)
        self.load_calls += 1
        self.seen_device = device

    def caption_image(self, image: Image.Image) -> str:
        return "a test caption"

    def unload(self) -> None:
        pass


def test_load_called_once_under_concurrent_first_use():
    backend = _SlowLoadBackend()
    engine = ArgusLens(backend=backend, device="cpu")

    n = 8
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            engine.caption(_img())
        except BaseException as exc:  # noqa: BLE001 - surface in assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"worker errors: {errors}"
    assert backend.load_calls == 1
    assert backend.seen_device == "cpu"
