"""Tests for the RAM++ backend (#3, #60).

The first block runs in full CI with no optional dependencies: registration,
routing, the tag vocabulary, thresholding, checkpoint handling and download.
Tests that need torch skip without it, and the end-to-end test additionally
needs the real checkpoint on disk (``RAM_MODEL_DIR`` or ``~/.cache/ram_plus``).
"""

import contextlib
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from argus_lens.backends import ram as ram_module
from argus_lens.backends._ram.vocab import (
    load_tag_vocabulary,
    prepare_state_dict,
    select_tags,
    unexpectedly_missing,
)
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.backends.output import BackendOutput
from argus_lens.backends.ram import CHECKPOINT_FILENAME, DEFAULT_MODEL_ID, RamBackend
from argus_lens.engine import ArgusLens, _resolve_backend
from argus_lens.gpu import resolve_min_vram_mb
from argus_lens.registry import ModelRegistry

# -- contract & configuration (no optional deps) ------------------------------


def test_ram_backend_metadata():
    """RAM++ is a photo-domain tagger: its output feeds the tag side of assembly."""
    backend = RamBackend()
    assert backend.name == "ram"
    assert backend.style == "photo"
    assert backend.emits_tags is True
    assert backend.requires_gpu is True
    assert resolve_min_vram_mb("ram") == 1000


def test_constructor_defaults_and_overrides(tmp_path):
    """Defaults to RAM++'s calibrated per-tag thresholds; a uniform threshold and model dir can be set."""
    default = RamBackend()
    assert default._repo_id == DEFAULT_MODEL_ID
    assert default._threshold is None

    custom = RamBackend(repo_id="org/custom-ram", threshold=0.6, model_dir=tmp_path)
    assert custom._repo_id == "org/custom-ram"
    assert custom._threshold == 0.6
    assert custom._model_dir == tmp_path


def test_availability_tracks_torch(monkeypatch):
    """Available whenever torch imports (weights download lazily); otherwise names the extra."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    assert RamBackend().is_available() is False
    assert "argus-lens[torch]" in (RamBackend().availability_reason() or "")

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert RamBackend().is_available() is True
    assert RamBackend().availability_reason() is None


def test_build_output_sets_source_and_scores():
    """_build_output produces BackendOutput tags with rounded scores and source "ram"."""
    out = RamBackend()._build_output([("mountain", 0.912345), ("lake", 0.7)])
    assert isinstance(out, BackendOutput)
    assert [t.label for t in out.tags] == ["mountain", "lake"]
    assert all(t.source == "ram" for t in out.tags)
    assert out.tags[0].score == 0.9123
    assert out.tag_string() == "mountain, lake"


# -- vocabulary, thresholds, checkpoint keys ------------------------------------


def test_tag_vocabulary_is_ram_plus():
    """The packaged vocabulary is RAM++'s 4585 photo tags with a calibrated threshold each."""
    tags, thresholds = load_tag_vocabulary()
    assert len(tags) == len(thresholds) == 4585
    assert len(set(tags)) == len(tags)
    assert {"beach", "sunset", "wedding dress", "living room", "dog"} <= set(tags)
    assert all(0.0 < t <= 1.0 for t in thresholds)
    assert sorted(thresholds)[len(thresholds) // 2] == 0.65  # most tags use the 0.65 default


def test_select_tags_per_class_and_uniform():
    """Per-tag thresholds by default, a uniform cut-off when given; strict `>`; highest score first."""
    probs = [0.9, 0.7, 0.66, 0.65, 0.2]
    thresholds = [0.8, 0.8, 0.65, 0.65, 0.1]
    assert select_tags(probs, thresholds) == [(0, 0.9), (2, 0.66), (4, 0.2)]  # 0.65 is not > 0.65
    assert select_tags(probs, thresholds, threshold=0.6) == [(0, 0.9), (1, 0.7), (2, 0.66), (3, 0.65)]
    assert select_tags(probs, thresholds, threshold=0.95) == []


def test_prepare_state_dict_mirrors_upstream_loader():
    """Unwraps "model", drops recomputed Swin buffers, renames the legacy vision_multi prefix."""
    checkpoint = {
        "model": {
            "visual_encoder.layers.0.blocks.0.attn.relative_position_index": 1,
            "visual_encoder.layers.0.blocks.1.attn_mask": 2,
            "visual_encoder.patch_embed.proj.weight": 3,
            "vision_multi.encoder.layer.0.output.dense.weight": 4,
        }
    }
    assert prepare_state_dict(checkpoint) == {
        "visual_encoder.patch_embed.proj.weight": 3,
        "tagging_head.encoder.layer.0.output.dense.weight": 4,
    }
    assert prepare_state_dict({"fc.weight": 5}) == {"fc.weight": 5}  # already unwrapped


def test_unexpectedly_missing_ignores_only_known_gaps():
    """Recomputed buffers and reweight_scale are expected gaps; any real weight is not."""
    missing = [
        "reweight_scale",
        "visual_encoder.layers.0.blocks.0.attn.relative_position_index",
        "visual_encoder.layers.0.blocks.1.attn_mask",
        "fc.weight",
    ]
    assert unexpectedly_missing(missing) == ["fc.weight"]


# -- checkpoint discovery & download ---------------------------------------------


def test_checkpoint_search_order(tmp_path, monkeypatch):
    """model_dir beats RAM_MODEL_DIR beats the default cache; nothing downloads when found."""
    monkeypatch.setattr(ram_module, "_download_checkpoint", lambda *a: pytest.fail("should not download"))
    env_dir, arg_dir = tmp_path / "env", tmp_path / "arg"
    for d in (env_dir, arg_dir):
        d.mkdir()
        (d / CHECKPOINT_FILENAME).write_bytes(b"x")
    monkeypatch.setenv("RAM_MODEL_DIR", str(env_dir))
    assert RamBackend()._ensure_checkpoint() == env_dir / CHECKPOINT_FILENAME
    assert RamBackend(model_dir=arg_dir)._ensure_checkpoint() == arg_dir / CHECKPOINT_FILENAME


def test_missing_checkpoint_downloads_to_first_search_dir(tmp_path, monkeypatch):
    """With no checkpoint anywhere, it downloads into the highest-priority directory."""
    monkeypatch.delenv("RAM_MODEL_DIR", raising=False)
    monkeypatch.setattr(ram_module, "_DEFAULT_CACHE_DIR", tmp_path / "cache")
    calls = []
    monkeypatch.setattr(ram_module, "_download_checkpoint", lambda dest, repo: calls.append((dest, repo)))
    target = tmp_path / "models"
    assert RamBackend(model_dir=target, repo_id="org/ram")._ensure_checkpoint() == target / CHECKPOINT_FILENAME
    assert calls == [(target / CHECKPOINT_FILENAME, "org/ram")]


def test_custom_repo_gets_its_own_cache_dir(tmp_path, monkeypatch):
    """A non-default repo never picks up the default repo's cached checkpoint; it downloads beside it."""
    monkeypatch.delenv("RAM_MODEL_DIR", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / CHECKPOINT_FILENAME).write_bytes(b"default")
    monkeypatch.setattr(ram_module, "_DEFAULT_CACHE_DIR", cache)
    calls = []
    monkeypatch.setattr(ram_module, "_download_checkpoint", lambda dest, repo: calls.append((dest, repo)))

    assert RamBackend()._ensure_checkpoint() == cache / CHECKPOINT_FILENAME
    assert calls == []
    expected = cache / "org--custom-ram" / CHECKPOINT_FILENAME
    assert RamBackend(repo_id="org/custom-ram")._ensure_checkpoint() == expected
    assert calls == [(expected, "org/custom-ram")]


def test_download_checkpoint_is_atomic(tmp_path, monkeypatch):
    """The download streams to a .part file and only appears under its real name once complete."""
    seen = {}

    class _Resp:
        headers = {"content-length": "6"}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size):
            yield b"abc"
            assert not (tmp_path / CHECKPOINT_FILENAME).exists()  # not visible mid-download
            yield b"def"

    @contextlib.contextmanager
    def _stream(method, url, **kwargs):
        seen["url"] = url
        yield _Resp()

    monkeypatch.setattr("httpx.stream", _stream)
    dest = tmp_path / CHECKPOINT_FILENAME
    ram_module._download_checkpoint(dest, "org/ram")
    assert dest.read_bytes() == b"abcdef"
    assert seen["url"] == f"https://huggingface.co/org/ram/resolve/main/{CHECKPOINT_FILENAME}"
    assert [p.name for p in tmp_path.iterdir()] == [CHECKPOINT_FILENAME]  # no temp file left behind


def test_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    """An interrupted download removes its temp file and never creates the checkpoint."""

    class _Resp:
        headers = {}

        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size):
            yield b"abc"
            raise ConnectionError("connection dropped")

    @contextlib.contextmanager
    def _stream(method, url, **kwargs):
        yield _Resp()

    monkeypatch.setattr("httpx.stream", _stream)
    with pytest.raises(ConnectionError):
        ram_module._download_checkpoint(tmp_path / CHECKPOINT_FILENAME, "org/ram")
    assert list(tmp_path.iterdir()) == []


# -- engine integration ------------------------------------------------------------


def test_engine_registers_ram_standalone_and_in_hybrids():
    """`ram`, `hybrid:ram+florence2` and `hybrid:ram+openai-compat` resolve; kwargs reach the right stage."""
    backend = _resolve_backend("ram", threshold=0.5)
    assert isinstance(backend, RamBackend) and backend._threshold == 0.5

    for prose in ("florence2", "openai-compat"):
        hybrid = _resolve_backend(
            f"hybrid:ram+{prose}", base_url="http://localhost:11434/v1", model_id="qwen3-vl:8b-instruct"
        )
        assert isinstance(hybrid, HybridPipeline)
        assert isinstance(hybrid._tag, RamBackend)
        assert hybrid._prose.name == prose
        # --model-id names the prose model; it must not become RAM++'s checkpoint repo.
        assert hybrid._tag._repo_id == DEFAULT_MODEL_ID


class _Tagger(CaptionBackend):
    """Photo-style stub tagger (like RAM++), optionally flagged as emitting tags."""

    name = "stub-tagger"
    style = "photo"

    def __init__(self, emits_tags: bool) -> None:
        """Set the emits_tags flag under test."""
        self.emits_tags = emits_tags

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image: Image.Image) -> str:
        """Return a fixed photo-tag string."""
        return "beach, sunset, ocean, person"

    def unload(self) -> None:
        """No-op."""


def test_engine_routes_tagger_output_to_the_tag_side():
    """A standalone emits_tags backend feeds raw_tags; a photo backend without the flag is prose (#60)."""
    image = Image.new("RGB", (32, 32), "orange")
    tagged = ArgusLens(backend=_Tagger(emits_tags=True)).caption(image)
    assert tagged.raw_tags == "beach, sunset, ocean, person"
    assert tagged.raw_prose == ""
    assert "beach" in tagged.caption_variants["setting"]

    prose = ArgusLens(backend=_Tagger(emits_tags=False)).caption(image)
    assert prose.raw_tags == ""
    assert prose.raw_prose


# -- torch: model structure and inference wiring -----------------------------------


def test_ram_plus_forward_shape_and_checkpoint_layout():
    """RamPlus (random weights) maps a 384px image to one probability per tag, under the checkpoint's key names."""
    torch = pytest.importorskip("torch")
    from argus_lens.backends._ram.model import IMAGE_SIZE, RamPlus, preprocess

    tags, _ = load_tag_vocabulary()
    torch.manual_seed(0)
    model = RamPlus(num_tags=len(tags)).eval()
    keys = set(model.state_dict())
    for key in (
        "visual_encoder.patch_embed.proj.weight",
        "visual_encoder.layers.2.blocks.17.attn.qkv.weight",
        "image_proj.weight",
        "label_embed",
        "reweight_scale",
        "wordvec_proj.weight",
        "tagging_head.encoder.layer.1.crossattention.self.query.weight",
        "tagging_head.encoder.layer.1.crossattention.output.LayerNorm.weight",
        "tagging_head.encoder.layer.1.intermediate.dense.weight",
        "tagging_head.encoder.layer.1.output.dense.weight",
        "fc.weight",
    ):
        assert key in keys, key
    assert model.label_embed.shape == (len(tags) * 51, 512)

    pixels = preprocess(Image.new("RGB", (640, 480), "teal"), "cpu", torch.float32)
    assert pixels.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    probs = model(pixels)
    assert probs.shape == (1, len(tags))
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_annotate_image_thresholds_and_sorts(monkeypatch):
    """annotate_image keeps tags above their cut-off, highest score first, using the cached model."""
    torch = pytest.importorskip("torch")

    tags = ["beach", "dog", "sunset"]

    def _fake_loader(self, device):
        model = lambda pixels: torch.tensor([[0.9, 0.5, 0.95]])  # noqa: E731
        return model, tags, [0.65, 0.65, 0.8], torch.float32

    monkeypatch.setattr(RamBackend, "_loader", _fake_loader)
    image = Image.new("RGB", (64, 64))
    # A private registry, so the fake model is never cached where a real-weights test would get it.
    registry = ModelRegistry()
    backend = RamBackend(registry=registry)
    backend.load("cpu")
    out = backend.annotate_image(image)
    assert [(t.label, t.score) for t in out.tags] == [("sunset", 0.95), ("beach", 0.9)]
    assert backend.caption_image(image) == "sunset, beach"
    lenient = RamBackend(threshold=0.4, registry=registry)
    lenient.load("cpu")
    assert lenient.caption_image(image) == "sunset, beach, dog"


# -- real weights (skipped unless the checkpoint is on disk) ------------------------


def _checkpoint_present() -> bool:
    dirs = [Path(os.environ["RAM_MODEL_DIR"])] if os.environ.get("RAM_MODEL_DIR") else []
    dirs.append(ram_module._DEFAULT_CACHE_DIR)
    return any((d / CHECKPOINT_FILENAME).exists() for d in dirs)


@pytest.mark.skipif(not _checkpoint_present(), reason="RAM++ checkpoint not downloaded")
def test_real_weights_tag_a_scene():
    """With the real checkpoint, RAM++ returns scored in-vocabulary tags that clear their thresholds."""
    torch = pytest.importorskip("torch")
    image = Image.new("RGB", (640, 480), "#87ceeb")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 300, 640, 480], fill="#e3c08d")  # sand
    draw.ellipse([480, 60, 580, 160], fill="#ffd27a")  # sun

    backend = RamBackend()
    backend.load("cuda" if torch.cuda.is_available() else "cpu")
    out = backend.annotate_image(image)

    tags, thresholds = load_tag_vocabulary()
    cutoff = dict(zip(tags, thresholds, strict=True))
    assert out.tags, "expected at least one tag for a sky-and-sun scene"
    scores = [t.score for t in out.tags]
    assert scores == sorted(scores, reverse=True)
    assert all(t.label in cutoff and t.score >= round(cutoff[t.label], 4) for t in out.tags)
    assert {"sky", "sun", "blue", "yellow"} & {t.label for t in out.tags}
