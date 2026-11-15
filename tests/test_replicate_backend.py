"""Replicate backend: per-instance client auth and output flattening (no network, fake SDK)."""

from __future__ import annotations

import os
import sys
import types

import pytest
from PIL import Image

from argus_lens.backends.replicate import ReplicateBackend


class _FakeClient:
    """Stand-in for ``replicate.Client`` that records its token and returns a canned output."""

    output: object = "a caption"

    def __init__(self, api_token: str) -> None:
        """Record the token the backend authorised with."""
        self.api_token = api_token

    def run(self, ref: str, input: dict) -> object:  # noqa: A002 - mirrors the SDK signature
        """Return the class-level canned output."""
        return type(self).output


@pytest.fixture
def fake_replicate(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    """Install a fake ``replicate`` module exposing :class:`_FakeClient`."""
    client_cls = type("Client", (_FakeClient,), {})
    monkeypatch.setitem(sys.modules, "replicate", types.SimpleNamespace(Client=client_cls))
    return client_cls


def _img() -> Image.Image:
    """Return a tiny grey test image."""
    return Image.new("RGB", (8, 8), "gray")


def test_explicit_api_key_wins_over_env_and_env_is_untouched(fake_replicate, monkeypatch):
    """An explicit key authorises the client even with REPLICATE_API_TOKEN set; the env is not mutated."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token")
    a = ReplicateBackend(api_key="key-a")
    b = ReplicateBackend(api_key="key-b")
    a.load()
    b.load()
    assert a._client.api_token == "key-a"
    assert b._client.api_token == "key-b"
    assert os.environ["REPLICATE_API_TOKEN"] == "env-token"


def test_env_token_used_without_explicit_key(fake_replicate, monkeypatch):
    """With no api_key the env token is used."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token")
    backend = ReplicateBackend()
    backend.load()
    assert backend._client.api_token == "env-token"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("  a dog on a beach ", "a dog on a beach"),
        (["A", " dog", " on", " a", " beach"], "A dog on a beach"),
        (iter(["Hel", "lo"]), "Hello"),
        ({"text": "a cat", "img": None}, "a cat"),
    ],
)
def test_output_flattening(fake_replicate, monkeypatch, output, expected):
    """Token streams are joined without spurious spaces, generators are consumed, dicts yield their text."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
    fake_replicate.output = output
    assert ReplicateBackend().caption_image(_img()) == expected


def test_dict_output_without_text_raises(fake_replicate, monkeypatch):
    """A dict with no text field is an error, not a caption of its repr."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
    fake_replicate.output = {"img": "https://example/x.png"}
    with pytest.raises(ValueError, match="no text field"):
        ReplicateBackend().caption_image(_img())
