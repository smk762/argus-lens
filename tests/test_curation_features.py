"""Tests for curation-oriented engine features: recursion, backend kwargs, model selection."""

import json

import httpx
from PIL import Image

from argus_lens import ArgusLens
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.backends.openai_compat import PROMPT_PRESETS, OpenAICompatBackend
from argus_lens.backends.wd14 import WD14Backend
from argus_lens.engine import _resolve_backend


class _Stub(CaptionBackend):
    """Backend returning a fixed string."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def load(self, device: str = "auto") -> None:
        """No-op."""

    def caption_image(self, image) -> str:
        """Return the fixed text."""
        return self._text

    def unload(self) -> None:
        """No-op."""


def _engine() -> ArgusLens:
    """Engine over a stub hybrid pipeline."""
    return ArgusLens(backend=HybridPipeline(_Stub("t", "1girl, red_dress, kitchen"), _Stub("p", "A woman cooks.")))


def _tree(root):
    """Create name/pics/shoot/IMG_1.png twice with colliding filenames."""
    for shoot, colour in (("s1", (255, 0, 0)), ("s2", (0, 255, 0))):
        d = root / "alice" / "pics" / shoot
        d.mkdir(parents=True)
        Image.new("RGB", (8, 8), colour).save(d / "IMG_1.png")


def test_caption_directory_recursive_keys_by_relative_path(tmp_path) -> None:
    """Colliding filenames in different shoots stay distinct, and sidecars land beside images."""
    _tree(tmp_path)
    results = _engine().caption_directory(tmp_path, recursive=True)
    assert sorted(results) == ["alice/pics/s1/IMG_1.png", "alice/pics/s2/IMG_1.png"]
    assert (tmp_path / "alice/pics/s1/IMG_1.txt").exists()
    assert (tmp_path / "alice/pics/s2/IMG_1.txt").exists()


def test_caption_directory_flat_by_default(tmp_path) -> None:
    """Without ``recursive`` sub-directories are ignored, as before."""
    _tree(tmp_path)
    assert _engine().caption_directory(tmp_path) == {}


def test_caption_directory_recursive_jsonl_carries_metadata(tmp_path) -> None:
    """JSONL export includes the index buckets and headcount."""
    _tree(tmp_path)
    _engine().caption_directory(tmp_path, recursive=True, output_format="jsonl")
    rows = [json.loads(line) for line in (tmp_path / "captions.jsonl").read_text().splitlines()]
    assert {r["name"] for r in rows} == {"alice/pics/s1/IMG_1.png", "alice/pics/s2/IMG_1.png"}
    assert rows[0]["metadata"]["buckets"]["tags"]["wardrobe"] == ["red_dress"]
    assert rows[0]["metadata"]["subjects"]["count"] == 1


def test_hybrid_shorthand_routes_kwargs_per_stage() -> None:
    """``hybrid:wd14+openai-compat`` must not hand base_url/model_id to WD14."""
    backend = _resolve_backend(
        "hybrid:wd14+openai-compat", base_url="http://llm/v1", model_id="qwen3-vl:8b", prompt="straightforward"
    )
    assert isinstance(backend._tag, WD14Backend)
    assert backend._prose._base_url == "http://llm/v1"
    assert backend._prose.model_id == "qwen3-vl:8b"
    assert backend._prose.user_prompt == PROMPT_PRESETS["straightforward"]


def test_openai_compat_prompt_preset_and_literal(monkeypatch) -> None:
    """``prompt`` accepts a preset name, literal text, or the env var; temperature is optional."""
    assert OpenAICompatBackend().user_prompt == PROMPT_PRESETS["detailed"]
    assert OpenAICompatBackend(prompt="Describe the outfit.").user_prompt == "Describe the outfit."
    monkeypatch.setenv("ARGUS_OPENAI_COMPAT_PROMPT", "booru")
    assert OpenAICompatBackend().user_prompt == PROMPT_PRESETS["booru"]

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the body and reply with a caption."""
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a woman"}}]})

    backend = OpenAICompatBackend(prompt="straightforward", temperature=0.0)
    backend._client = httpx.Client(base_url="http://x/v1", transport=httpx.MockTransport(handler))
    backend.caption_image(Image.new("RGB", (8, 8)))
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["messages"][1]["content"][0]["text"] == PROMPT_PRESETS["straightforward"]


def test_wd14_model_repo_selects_file_and_cache(monkeypatch, tmp_path) -> None:
    """Non-default repos get their own filename, cache subdir, and registry key."""
    from argus_lens.backends import wd14 as wd14_mod

    monkeypatch.setattr(wd14_mod, "_DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.delenv("WD14_MODEL_DIR", raising=False)
    eva = WD14Backend(model_repo="SmilingWolf/wd-eva02-large-tagger-v3")
    assert eva._model_filename == "wd-eva02-large-tagger-v3.onnx"
    assert eva._search_dirs()[-1] == tmp_path / "wd-eva02-large-tagger-v3"
    assert eva._cache_key("cpu") != WD14Backend()._cache_key("cpu")
    assert WD14Backend()._search_dirs()[-1] == tmp_path  # default unchanged
    monkeypatch.setenv("WD14_MODEL_REPO", "SmilingWolf/wd-swinv2-tagger-v3")
    assert WD14Backend()._model_filename == "wd-swinv2-tagger-v3.onnx"
