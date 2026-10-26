"""Tests for LLM caption refinement: validation, retry, fallback, and wiring."""

import json

import httpx
import pytest
from PIL import Image

from argus_lens import ArgusLens
from argus_lens.assembly.composer import compose_caption_result
from argus_lens.backends.base import CaptionBackend
from argus_lens.backends.hybrid import HybridPipeline
from argus_lens.refine import OpenAICompatRewriter, Refiner, parse_rewrite
from argus_lens.types import resolve_target_profile

TAGS = "1girl, solo, brown_hair, red_dress, kitchen, standing, cup"
PROSE = "A woman with brown hair stands in a kitchen wearing a red dress, holding a cup."
GOOD = {
    "training": "sks, standing, red_dress, kitchen, holding cup",
    "training_prose": "sks, A woman stands in a kitchen, wearing a red dress and holding a cup.",
}


class _ScriptedRewriter:
    """Rewriter returning queued replies (or raising) and recording requests."""

    name = "scripted"
    model_id = "test-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def rewrite(self, request, image=None):
        """Pop the next scripted reply."""
        self.requests.append((request, image))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _result(target_backend: str = "sdxl"):
    """Deterministic result for the shared fixture."""
    profile = resolve_target_profile(target_backend=target_backend)
    return compose_caption_result(trigger_word="sks", tags=TAGS, prose=PROSE, target_profile=profile), profile


def test_valid_rewrites_are_accepted() -> None:
    """Rewrites that pass the post-check replace the deterministic captions."""
    result, profile = _result("flux")
    refined = Refiner(_ScriptedRewriter(GOOD)).refine(result, profile, trigger_word="sks")
    assert refined.caption_variants["training"] == GOOD["training"]
    assert refined.caption_variants["training_prose"] == GOOD["training_prose"]
    assert refined.metadata["refine"]["accepted"] == ["training", "training_prose"]
    assert refined.metadata["refine"]["model"] == "test-model"
    assert refined.metadata["buckets"] == result.metadata["buckets"]  # index data untouched


def test_identity_leak_is_retried_with_feedback_then_accepted() -> None:
    """A rewrite mentioning hair is rejected, the problem is fed back, and the retry is used."""
    leaky = {**GOOD, "training_prose": "sks, A woman with brown hair stands in a kitchen."}
    rewriter = _ScriptedRewriter(leaky, {"training_prose": GOOD["training_prose"]})
    result, profile = _result("flux")
    refined = Refiner(rewriter).refine(result, profile, trigger_word="sks")
    assert refined.caption_variants["training_prose"] == GOOD["training_prose"]
    retry_request = rewriter.requests[1][0]
    assert list(retry_request.rules) == ["training_prose"]
    assert any("suppressed terms: hair" in p for p in retry_request.feedback["training_prose"])
    assert refined.metadata["refine"]["attempts"] == 2


def test_persistent_failure_keeps_deterministic_caption() -> None:
    """When retries are exhausted the deterministic caption stays and the rejection is recorded."""
    hallucinated = {"training": "sks, dragon, volcano, castle, wizard, spaceship"}
    result, profile = _result()
    refined = Refiner(_ScriptedRewriter(hallucinated, hallucinated), variants=("training",)).refine(
        result, profile, trigger_word="sks"
    )
    assert refined.caption_variants["training"] == result.caption_variants["training"]
    assert "not grounded" in " ".join(refined.metadata["refine"]["rejected"]["training"])


def test_rewriter_error_falls_back() -> None:
    """An unreachable endpoint never breaks captioning."""
    result, profile = _result()
    refined = Refiner(_ScriptedRewriter(httpx.ConnectError("down"))).refine(result, profile, trigger_word="sks")
    assert refined.caption_variants == result.caption_variants
    assert "ConnectError" in refined.metadata["refine"]["error"]


def test_rewrites_are_repaired_before_validation() -> None:
    """Small slips (dangling ``and``) are repaired rather than rejected."""
    result, profile = _result()
    reply = {"training": "sks, standing, red_dress, kitchen, holding cup, and"}
    refined = Refiner(_ScriptedRewriter(reply), variants=("training",)).refine(result, profile, trigger_word="sks")
    assert refined.caption_variants["training"] == GOOD["training"]


def test_category_variants_can_be_refined() -> None:
    """Category captions (for grouping) are refinable with their own rules."""
    result, profile = _result()
    rewriter = _ScriptedRewriter({"wardrobe": "red dress"})
    refined = Refiner(rewriter, variants=("wardrobe",)).refine(result, profile)
    assert refined.caption_variants["wardrobe"] == "red dress"
    assert "ONLY the wardrobe" in rewriter.requests[0][0].rules["wardrobe"]


def test_image_only_sent_when_enabled() -> None:
    """The image goes to the rewriter only with ``send_image=True``."""
    result, profile = _result()
    image = Image.new("RGB", (8, 8))
    for send, expected in ((False, None), (True, image)):
        rewriter = _ScriptedRewriter(GOOD)
        Refiner(rewriter, send_image=send).refine(result, profile, trigger_word="sks", image=image)
        assert rewriter.requests[0][1] is expected


@pytest.mark.parametrize(
    "content",
    ['{"training": "a, b"}', '```json\n{"training": "a, b"}\n```', 'Sure! Here you go: {"training": "a, b"} Thanks'],
)
def test_parse_rewrite_tolerates_wrapping(content: str) -> None:
    """Code fences and chatter around the JSON are ignored."""
    assert parse_rewrite(content) == {"training": "a, b"}


def test_parse_rewrite_rejects_non_json() -> None:
    """A reply with no JSON object raises (and the Refiner falls back)."""
    with pytest.raises(ValueError):
        parse_rewrite("I cannot help with that.")


def test_openai_compat_rewriter_request_shape() -> None:
    """The rewriter posts a deterministic chat completion with the structured inputs."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the body and answer with a JSON caption map."""
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)}}]})

    client = httpx.Client(base_url="http://llm/v1", transport=httpx.MockTransport(handler))
    rewriter = OpenAICompatRewriter(base_url="http://llm/v1", model_id="qwen3:8b", client=client)
    result, profile = _result("flux")
    refined = Refiner(rewriter).refine(result, profile, trigger_word="sks")
    body = captured["body"]
    assert body["model"] == "qwen3:8b" and body["temperature"] == 0 and body["seed"] == 0
    user = json.loads(body["messages"][1]["content"])
    assert user["tags"] == TAGS and set(user["captions_to_write"]) == {"training", "training_prose"}
    assert refined.metadata["refine"]["accepted"] == ["training", "training_prose"]


# --- engine wiring ---


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


def _engine(**kwargs) -> ArgusLens:
    """Engine over a stub hybrid pipeline with the shared tags/prose."""
    return ArgusLens(backend=HybridPipeline(_Stub("tags", TAGS), _Stub("prose", PROSE)), **kwargs)


def test_engine_applies_refiner_on_every_path(tmp_path) -> None:
    """caption, caption_batch, and caption_stream all refine."""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(img)
    engine = _engine(refiner=Refiner(_ScriptedRewriter(GOOD, GOOD, GOOD)))
    assert engine.caption(img, trigger_word="sks").caption_variants["training"] == GOOD["training"]
    batch = engine.caption_batch([img], trigger_word="sks")
    assert batch["a.png"].caption_variants["training"] == GOOD["training"]
    ((_, streamed),) = list(engine.caption_stream([img], trigger_word="sks"))
    assert streamed.caption_variants["training"] == GOOD["training"]


def test_engine_without_refiner_is_deterministic(tmp_path) -> None:
    """No refiner, no refine metadata."""
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    assert "refine" not in _engine().caption(img).metadata


def test_refiner_from_env(monkeypatch) -> None:
    """Refinement is on by default (all variants, local Ollama); ARGUS_REFINE=off opts out."""
    from argus_lens.refine import refiner_from_env

    for var in ("ARGUS_REFINE", "ARGUS_REFINE_MODEL", "ARGUS_REFINE_BASE_URL", "ARGUS_REFINE_VARIANTS"):
        monkeypatch.delenv(var, raising=False)
    default = refiner_from_env()
    assert default.variants is None
    assert default.rewriter.model_id == "qwen3:8b"
    assert default.rewriter.base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("ARGUS_REFINE_MODEL", "qwen3:14b")
    monkeypatch.setenv("ARGUS_REFINE_BASE_URL", "http://ollama.lan:11434/v1")
    monkeypatch.setenv("ARGUS_REFINE_VARIANTS", "training, wardrobe")
    refiner = refiner_from_env()
    assert refiner.variants == ("training", "wardrobe")
    assert refiner.rewriter.model_id == "qwen3:14b"
    assert refiner.rewriter.base_url == "http://ollama.lan:11434/v1"
    for off in ("off", "0", "false", "no", "none"):
        monkeypatch.setenv("ARGUS_REFINE", off)
        assert refiner_from_env() is None


def test_all_variants_are_refined_when_unrestricted() -> None:
    """``variants=None`` asks for every variant, categories included."""
    result, profile = _result()
    rewriter = _ScriptedRewriter({}, {})
    Refiner(rewriter, variants=None).refine(result, profile, trigger_word="sks")
    assert set(rewriter.requests[0][0].rules) == set(result.caption_variants)


def test_unreachable_endpoint_is_skipped_during_cooldown() -> None:
    """A refused connection pauses refinement instead of stalling every image."""
    result, profile = _result()
    rewriter = _ScriptedRewriter(httpx.ConnectError("refused"), GOOD)
    refiner = Refiner(rewriter, unavailable_cooldown_s=60)
    assert "ConnectError" in refiner.refine(result, profile, trigger_word="sks").metadata["refine"]["error"]
    skipped = refiner.refine(result, profile, trigger_word="sks")
    assert skipped.metadata["refine"]["skipped"] == "unavailable"
    assert skipped.caption_variants == result.caption_variants
    assert len(rewriter.requests) == 1
    refiner._unavailable_until = 0.0  # cooldown over
    assert refiner.refine(result, profile, trigger_word="sks").metadata["refine"]["accepted"]


def test_prose_rules_follow_the_text_encoder() -> None:
    """CLIP targets (SDXL) ask for short prose; T5 targets (Flux) allow more."""
    from argus_lens.refine import render_rules

    clip = render_rules("training_prose", 60, "sdxl")
    t5 = render_rules("training_prose", 200, "flux")
    assert "one or two concise" in clip and "60 CLIP tokens" in clip
    assert "one to four" in t5 and "200 T5 tokens" in t5
    assert "{" not in clip + t5


def test_rewriter_disables_reasoning_and_falls_back_on_400() -> None:
    """``reasoning_effort: none`` is sent by default; a 400 drops it for good."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Reject requests carrying reasoning_effort, accept the rest."""
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unknown parameter"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)}}]})

    client = httpx.Client(base_url="http://llm/v1", transport=httpx.MockTransport(handler))
    rewriter = OpenAICompatRewriter(base_url="http://llm/v1", client=client)
    result, profile = _result("flux")
    Refiner(rewriter).refine(result, profile, trigger_word="sks")
    Refiner(rewriter).refine(result, profile, trigger_word="sks")
    assert bodies[0]["reasoning_effort"] == "none"
    assert ["reasoning_effort" in b for b in bodies] == [True, False, False]


def test_rewriter_reasoning_effort_can_be_disabled(monkeypatch) -> None:
    """An empty value never sends the parameter."""
    monkeypatch.setenv("ARGUS_REFINE_REASONING_EFFORT", "")
    assert OpenAICompatRewriter().reasoning_effort is None
    assert OpenAICompatRewriter(reasoning_effort="low").reasoning_effort == "low"


def test_category_rewrites_reject_identity_and_accept_empty() -> None:
    """A wardrobe rewrite naming hair is rejected; an empty answer means 'none visible'."""
    result, profile = _result()
    rewriter = _ScriptedRewriter({"wardrobe": "sks, red_hair", "setting": "sks, kitchen"}, {"wardrobe": ""})
    refined = Refiner(rewriter, variants=("wardrobe", "setting")).refine(result, profile, trigger_word="sks")
    assert any("suppressed terms: hair" in p for p in rewriter.requests[1][0].feedback["wardrobe"])
    assert refined.caption_variants["wardrobe"] == "sks"
    assert refined.caption_variants["setting"] == "sks, kitchen"


def test_training_rewrite_rejects_headcount_tags() -> None:
    """``1girl``/``solo`` are noise in a training caption."""
    result, profile = _result()
    reply = {"training": "sks, 1girl, standing, red_dress, kitchen"}
    refined = Refiner(_ScriptedRewriter(reply, reply), variants=("training",)).refine(
        result, profile, trigger_word="sks"
    )
    assert "1girl" in " ".join(refined.metadata["refine"]["rejected"]["training"])
