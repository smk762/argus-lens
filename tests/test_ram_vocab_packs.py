"""RAM++ vocabulary packs (#62): manifest, merge rules, builder input, description writer, backend wiring.

Everything above the torch section runs in CI without optional dependencies.
"""

import json
import os

import httpx
import pytest

from argus_lens.backends._ram.build import read_tag_input
from argus_lens.backends._ram.describe import UPSTREAM_PROMPTS, DescriptionWriter, fit_descriptions, template
from argus_lens.backends._ram.packs import (
    CLIP_MODEL,
    DEFAULT_PACK_THRESHOLD,
    PackError,
    PackManifest,
    merge_vocabularies,
)
from argus_lens.backends._ram.vocab import DESCRIPTIONS_PER_TAG
from argus_lens.backends.ram import RamBackend
from argus_lens.engine import _resolve_backend


def _manifest(name="fashion", tags=("puff sleeve", "a-line skirt"), thresholds=None):
    return PackManifest(name=name, tags=list(tags), thresholds=list(thresholds or [0.5] * len(tags)))


# -- manifest -------------------------------------------------------------------------------


def test_manifest_round_trips_and_validates(tmp_path):
    """A written manifest reads back identically; RAM++'s geometry and CLIP model are recorded."""
    manifest = _manifest()
    manifest.write(tmp_path)
    read = PackManifest.read(tmp_path)
    assert read == manifest
    assert (read.clip_model, read.descriptions_per_tag, read.embed_dim) == (CLIP_MODEL, 51, 512)
    assert DEFAULT_PACK_THRESHOLD == 0.5  # upstream's open-set default


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tags": ["a", "a"], "thresholds": [0.5, 0.5]}, "repeats tags"),
        ({"thresholds": [0.5]}, "2 tags but 1 thresholds"),
        ({"thresholds": [0.5, 1.5]}, r"\(0, 1\]"),
        ({"clip_model": "openai/clip-vit-large-patch14"}, "needs 'openai/clip-vit-base-patch16'"),
        ({"descriptions_per_tag": 10}, "needs 51"),
        ({"format": "something-else"}, "not a RAM\\+\\+ vocabulary pack"),
        ({"version": 2}, "format version 2"),
        ({"tags": [], "thresholds": []}, "at least one tag"),
    ],
)
def test_manifest_rejects_unusable_packs(change, message):
    """Mismatched geometry, models, versions, or bad tag lists fail loudly."""
    data = {**_manifest().__dict__, **change}
    with pytest.raises(PackError, match=message):
        PackManifest(**data).validate()


def test_manifest_read_errors(tmp_path):
    """A missing or corrupt manifest is a PackError, not a crash."""
    with pytest.raises(PackError, match="no manifest.json"):
        PackManifest.read(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(PackError, match="not valid JSON"):
        PackManifest.read(tmp_path)


# -- merge rules ------------------------------------------------------------------------------


def test_merge_appends_packs_and_replaces_builtin_duplicates():
    """Pack tags follow the kept built-ins; a pack tag that is also built-in replaces it."""
    merged = merge_vocabularies(
        ["beach", "dress", "dog"], [0.9, 0.65, 0.9], [_manifest(tags=("dress", "puff sleeve"), thresholds=[0.4, 0.5])]
    )
    assert merged.tags == ["beach", "dog", "dress", "puff sleeve"]
    assert merged.thresholds == [0.9, 0.9, 0.4, 0.5]
    assert merged.sources == ["", "", "fashion", "fashion"]
    assert merged.builtin_keep == [0, 2]


def test_merge_rejects_a_tag_in_two_packs():
    """The same tag in two packs is ambiguous."""
    with pytest.raises(PackError, match="in both pack 'a' and pack 'b'"):
        merge_vocabularies(["x"], [0.5], [_manifest("a", ["shared"]), _manifest("b", ["shared"])])


def test_merge_packs_only_mode():
    """include_builtin=False uses only pack tags (upstream's open-set mode), and needs a pack."""
    merged = merge_vocabularies(["beach"], [0.9], [_manifest()], include_builtin=False)
    assert merged.tags == ["puff sleeve", "a-line skirt"] and merged.builtin_keep == []
    with pytest.raises(PackError, match="needs at least one"):
        merge_vocabularies(["beach"], [0.9], [], include_builtin=False)


# -- builder input ----------------------------------------------------------------------------


def test_read_tag_list_with_thresholds_and_comments(tmp_path):
    """Plain lists take one tag per line, an optional TAB threshold, and skip comments/blank lines."""
    src = tmp_path / "tags.txt"
    src.write_text("# fashion pack\npuff sleeve\t0.45\n\na-line skirt  # common\n")
    parsed = read_tag_input(src)
    assert parsed.tags == ["puff sleeve", "a-line skirt"]
    assert parsed.thresholds == {"puff sleeve": 0.45}
    assert parsed.descriptions == {}


@pytest.mark.parametrize(
    "payload",
    [
        [{"Go-kart": ["a photo of a go-kart.", "a small racing car"]}],  # upstream's shape
        {"Go-kart": ["a photo of a go-kart.", "a small racing car"]},  # plain mapping
    ],
)
def test_read_descriptions_json(tmp_path, payload):
    """Both upstream's list-of-maps and a plain mapping are accepted."""
    src = tmp_path / "d.json"
    src.write_text(json.dumps(payload))
    parsed = read_tag_input(src)
    assert parsed.tags == ["Go-kart"]
    assert parsed.descriptions["Go-kart"][1] == "a small racing car"


def test_read_tag_input_rejects_duplicates_and_bad_thresholds(tmp_path):
    """Repeated tags and non-numeric thresholds are errors."""
    (tmp_path / "dup.txt").write_text("a\na\n")
    with pytest.raises(PackError, match="more than once"):
        read_tag_input(tmp_path / "dup.txt")
    (tmp_path / "bad.txt").write_text("a\thigh\n")
    with pytest.raises(PackError, match="bad threshold"):
        read_tag_input(tmp_path / "bad.txt")


def test_fit_descriptions_keeps_full_sets_and_pads_short_ones():
    """A full set (as upstream ships) is used verbatim; a short one gets the template and is cycled."""
    full = [f"desc {i}" for i in range(60)]
    assert fit_descriptions("cat", full) == full[:51]
    short = fit_descriptions("cat", ["a tabby"])
    assert len(short) == DESCRIPTIONS_PER_TAG
    assert short[:3] == [template("cat"), "a tabby", template("cat")]
    # An existing template (any case) isn't duplicated.
    assert fit_descriptions("Cat", ["a photo of a cat.", "x"])[:2] == ["a photo of a cat.", "x"]


# -- description writer ---------------------------------------------------------------------------


def test_description_writer_follows_upstream_recipe(monkeypatch):
    """Template + 5 upstream prompts × 10 samples = 51, one user-turn request per sample."""
    monkeypatch.setenv("ARGUS_REFINE_MODEL", "qwen3:8b")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": f"  sample {len(seen)}  "}}]})

    client = httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))
    descs = DescriptionWriter(client=client).describe("go-kart")
    assert len(descs) == DESCRIPTIONS_PER_TAG
    assert descs[0] == "a photo of a go-kart." and descs[1] == "sample 1"
    assert len(seen) == len(UPSTREAM_PROMPTS) * 10
    assert seen[0]["model"] == "qwen3:8b"
    assert seen[0]["messages"] == [{"role": "user", "content": "Describe concisely what a(n) go-kart looks like:"}]
    assert (seen[0]["temperature"], seen[0]["max_tokens"]) == (0.99, 77)


# -- backend wiring ---------------------------------------------------------------------------------


def test_backend_pack_options_and_cache_key(tmp_path, monkeypatch):
    """Packs come from the argument or RAM_VOCAB_PACKS, and change the registry key."""
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("RAM_VOCAB_PACKS", f"{a}{os.pathsep}{b}")
    from_env = RamBackend()
    assert from_env._vocab_packs == [a.resolve(), b.resolve()]

    plain = RamBackend(vocab_packs=[])
    only = RamBackend(vocab_packs=[a], builtin_vocab=False)
    keys = {plain._cache_key("cpu"), from_env._cache_key("cpu"), only._cache_key("cpu")}
    assert len(keys) == 3
    assert plain._cache_key("cpu") == "ram:cpu:xinyu1205/recognize-anything-plus-model"


def test_engine_forwards_vocab_packs(tmp_path):
    """ArgusLens(backend='ram', vocab_packs=[...]) reaches the backend."""
    backend = _resolve_backend("ram", vocab_packs=[tmp_path])
    assert backend._vocab_packs == [tmp_path.resolve()]


def test_pack_tags_report_their_source():
    """Built-in tags say 'ram'; pack tags say 'ram:<pack>'."""
    backend = RamBackend()
    assert backend._source("") == "ram"
    assert backend._source("fashion") == "ram:fashion"


def test_cli_inspect(tmp_path):
    """`argus-lens ram-vocab inspect` summarises a pack and flags built-in overrides."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from argus_lens.cli import app

    _manifest(tags=("dress", "puff sleeve")).write(tmp_path)
    result = CliRunner().invoke(app, ["ram-vocab", "inspect", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fashion: 2 tag(s)" in result.output
    assert "replaces 1 built-in tag(s): dress" in result.output


# -- torch: building and applying packs --------------------------------------------------------------


def test_build_pack_and_apply_it_to_a_model(tmp_path):
    """build_pack writes fp16 embeddings + manifest; the backend splices them into label_embed."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from argus_lens.backends._ram.build import build_pack
    from argus_lens.backends._ram.packs import load_pack_embeddings

    def fake_embed(texts):
        return torch.nn.functional.normalize(torch.randn(len(texts), 512), dim=-1)

    pack = tmp_path / "pack"
    manifest = build_pack(
        pack,
        name="fashion",
        tags=["dress", "puff sleeve"],
        descriptions={"dress": ["a dress"], "puff sleeve": ["a sleeve gathered at the shoulder"]},
        embed=fake_embed,
        thresholds={"dress": 0.4},
        license="CC0-1.0",
    )
    assert manifest.thresholds == [0.4, 0.5]
    embeds = load_pack_embeddings(pack, PackManifest.read(pack))
    assert embeds.dtype == torch.float16 and tuple(embeds.shape) == (2 * 51, 512)
    saved = json.loads((pack / "descriptions.json").read_text())
    assert len(saved[1]["puff sleeve"]) == 51

    class _Model:
        label_embed = torch.nn.Parameter(torch.arange(3 * 51 * 512, dtype=torch.float32).view(3 * 51, 512))
        num_tags = 3

    model = _Model()
    builtin = ["beach", "dress", "dog"]
    tags, thresholds, sources = RamBackend(vocab_packs=[pack])._apply_vocab_packs(model, builtin, [0.9, 0.65, 0.9])
    assert tags == ["beach", "dog", "dress", "puff sleeve"]
    assert thresholds == [0.9, 0.9, 0.4, 0.5]
    assert sources == ["", "", "fashion", "fashion"]
    assert model.num_tags == 4 and tuple(model.label_embed.shape) == (4 * 51, 512)
    # "dog" (built-in index 2) kept its own rows; the pack's "dress" replaced the built-in one.
    assert torch.equal(model.label_embed[51:102], torch.arange(2 * 51 * 512, 3 * 51 * 512).view(51, 512).float())
    assert torch.allclose(model.label_embed[102:153], embeds[:51].float())
