"""Engine core: backend kwarg mapping, the residency/lease race, and directory discovery."""

from __future__ import annotations

import contextlib
from pathlib import Path

from PIL import Image
from structlog.testing import capture_logs

from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.florence2 import Florence2Backend
from argus_lens.engine import ArgusLens, _resolve_backend


class _Local(CaptionBackend):
    """Local backend stub counting load/unload calls (florence2 footprint, so a lease is taken)."""

    name = "florence2"
    style = "photo"
    requires_gpu = True

    def __init__(self) -> None:
        """Zero the counters."""
        self.loads = 0
        self.unloads = 0

    def load(self, device: str = "auto") -> None:
        """Count the load."""
        self.loads += 1

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed caption."""
        return "a person standing in a kitchen"

    def unload(self) -> None:
        """Count the unload."""
        self.unloads += 1


# --------------------------------------------------------------------------- #
# florence_model_id mapping
# --------------------------------------------------------------------------- #


def test_florence_model_id_never_reaches_other_backends() -> None:
    """A non-Florence backend keeps its own model_id even when florence_model_id is also given."""
    backend = _resolve_backend("openai-compat", model_id="gpt-4o", florence_model_id="microsoft/Florence-2-large")
    assert backend.model_id == "gpt-4o"


def test_florence_model_id_targets_florence_in_a_shared_hybrid() -> None:
    """hybrid:florence2+openai-compat: Florence gets florence_model_id, the prose stage gets model_id."""
    hybrid = _resolve_backend(
        "hybrid:florence2+openai-compat", model_id="gpt-4o", florence_model_id="microsoft/Florence-2-large"
    )
    assert isinstance(hybrid._tag, Florence2Backend)
    assert hybrid._tag._model_id == "microsoft/Florence-2-large"
    assert hybrid._prose.model_id == "gpt-4o"


def test_florence2_accepts_plain_model_id() -> None:
    """Without florence_model_id, Florence-2 still takes a plain model_id."""
    assert _resolve_backend("florence2", model_id="custom/florence")._model_id == "custom/florence"


# --------------------------------------------------------------------------- #
# Residency vs lease
# --------------------------------------------------------------------------- #


def test_model_cannot_be_unloaded_between_residency_check_and_lease() -> None:
    """A lease sized for a resident model (0 MB) never ends up covering a reload."""
    backend = _Local()
    leases: list[int] = []

    class _UnloadingCoordinator:
        """Simulates the idle reaper firing as the lease is requested."""

        name = "rec"
        engine: ArgusLens | None = None

        def lease(self, *, caller: str, min_vram_mb: int, **_):
            leases.append(min_vram_mb)
            if self.engine is not None:
                self.engine.unload()
            return contextlib.nullcontext()

    coordinator = _UnloadingCoordinator()
    engine = ArgusLens(backend=backend, coordinator=coordinator)
    engine.caption(Image.new("RGB", (4, 4)))  # first caption loads under a full-footprint lease
    assert leases == [2500] and backend.loads == 1
    coordinator.engine = engine
    engine.caption(Image.new("RGB", (4, 4)))
    # Sized as resident (0 MB), so the reaper must not have freed it and forced a reload.
    assert leases[1] == 0
    assert backend.unloads == 0
    assert backend.loads == 1


# --------------------------------------------------------------------------- #
# caption_directory discovery
# --------------------------------------------------------------------------- #


def _write_image(path: Path, fmt: str = "PNG") -> None:
    """Write a tiny image at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path, format=fmt)


def test_directory_matches_extensions_case_insensitively(tmp_path: Path) -> None:
    """Camera exports (IMG_1.JPG) are found by the default glob, once each."""
    _write_image(tmp_path / "IMG_1.JPG", "JPEG")
    _write_image(tmp_path / "b.Png")
    _write_image(tmp_path / "sub" / "IMG_2.JPEG", "JPEG")
    (tmp_path / "notes.TXT").write_text("not an image")
    results = ArgusLens(backend=_Local()).caption_directory(tmp_path, recursive=True)
    assert sorted(results) == ["IMG_1.JPG", "b.Png", "sub/IMG_2.JPEG"]
    assert (tmp_path / "IMG_1.txt").exists()
    assert (tmp_path / "sub" / "IMG_2.txt").exists()


def test_directory_overlapping_patterns_yield_each_image_once(tmp_path: Path) -> None:
    """jpg and JPG patterns (or a case-insensitive filesystem) never caption one file twice."""
    _write_image(tmp_path / "a.jpg", "JPEG")
    results = ArgusLens(backend=_Local()).caption_directory(tmp_path, glob="*.{jpg,JPG}")
    assert list(results) == ["a.jpg"]


def test_directory_stem_collision_skips_later_image_loudly(tmp_path: Path) -> None:
    """IMG_1.jpg and IMG_1.png share IMG_1.txt: the first owns it, the second is skipped with a warning."""
    _write_image(tmp_path / "IMG_1.jpg", "JPEG")
    _write_image(tmp_path / "IMG_1.png")
    with capture_logs() as logs:
        results = ArgusLens(backend=_Local()).caption_directory(tmp_path)
    assert list(results) == ["IMG_1.jpg"]
    assert (tmp_path / "IMG_1.txt").exists()
    warnings = [e for e in logs if e["event"] == "caption_sidecar_collision_skipped"]
    assert [(w["image"], w["kept"]) for w in warnings] == [("IMG_1.png", "IMG_1.jpg")]


def test_directory_still_skips_existing_sidecars(tmp_path: Path) -> None:
    """Without overwrite an existing <stem>.txt is left alone; with it, the image is re-captioned."""
    _write_image(tmp_path / "a.jpg", "JPEG")
    (tmp_path / "a.txt").write_text("hand-written")
    engine = ArgusLens(backend=_Local())
    assert engine.caption_directory(tmp_path) == {}
    assert (tmp_path / "a.txt").read_text() == "hand-written"
    assert list(engine.caption_directory(tmp_path, overwrite=True)) == ["a.jpg"]
