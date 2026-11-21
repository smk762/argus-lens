"""Claim-level faithfulness: what share of a caption's claims does a VLM judge reject?

The other reference-free metrics check a caption against itself (tags vs prose)
or against CLIP. This one checks it against the image. The caption is split into
short claims, a vision-language judge rates each one 0–100 for how well the image
supports it, and a claim below the threshold counts as *unsupported*. This follows
the HalDec-Bench protocol (arXiv:2603.15253): one image, one claim, one score, no
reference caption. That benchmark's leaderboard is the place to pick a judge.

It scores two texts per image:

* **raw**: what the backend returned (``raw_prose`` plus ``raw_tags``), before assembly;
* **final**: the ``final_caption`` the sidecar gets.

Comparing the two shows whether the assembly pipeline removes hallucinations or
adds them. A claim that appears in both is judged once, so the comparison isn't
blurred by judge noise on identical text.

Limitations: claims are comma and sentence fragments judged without their
surrounding caption, so a fragment like "standing" is judged on its own. Judges
are imperfect (the best HalDec-Bench AUROC is about 0.8), so read the rate as a
trend between runs with the same judge, not as ground truth.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.types import CaptionResult

DEFAULT_THRESHOLD = 50.0
"""Scores below this (0–100) mark a claim unsupported."""

_DEFAULT_BASE_URL = "http://localhost:11434/v1"

PROMPT = (
    "You are checking an image caption for hallucinations.\n"
    'Claim: "{claim}"\n'
    "Looking only at the image, how well does it support this claim? "
    "Answer with one integer from 0 (clearly false, or not visible) to 100 (clearly true). "
    "Output only the number."
)

_CLAIM_SPLIT = re.compile(r"[.!?;,\n]+")
# An unclosed block (cut off by max_tokens) runs to the end: its scratch numbers are no score either.
_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class FaithfulnessJudge(Protocol):
    """Rates how well an image supports each claim."""

    name: str
    model_id: str

    def score_claims(self, image: Image.Image, claims: list[str]) -> list[float | None]:
        """Return a 0–100 score per claim, ``None`` where the answer couldn't be parsed.

        Transport errors propagate; the caller records them per image.
        """
        ...


def parse_score(answer: str) -> float | None:
    """Pull the 0–100 score out of a judge answer, or ``None`` if there isn't one.

    Drops a ``<think>`` block first, so a reasoning model's scratch numbers aren't
    read as the score. The first number left is the score; one above 100 is ``None``.
    """
    match = _NUMBER.search(_THINK.sub("", answer))
    if match is None:
        return None
    value = float(match.group())
    return value if value <= 100 else None


def split_claims(text: str, *, trigger_word: str = "") -> list[str]:
    """Split a caption into claims: its sentence and comma fragments.

    Tag underscores become spaces. The trigger word, fragments without letters,
    and repeats (case-insensitive) are dropped, so order is kept and each claim is
    judged once.
    """
    # Normalised like the claims below, or "sks_person" never equals the fragment "sks person".
    trigger = " ".join(trigger_word.replace("_", " ").split()).lower()
    seen: set[str] = set()
    claims: list[str] = []
    for chunk in _CLAIM_SPLIT.split(text.replace("_", " ")):
        claim = " ".join(chunk.split())
        key = claim.lower()
        if not re.search(r"[a-z]", key) or key == trigger or key in seen:
            continue
        seen.add(key)
        claims.append(claim)
    return claims


class OpenAICompatJudge:
    """A judge served behind any OpenAI-compatible vision endpoint (Ollama, vLLM, a cloud API)."""

    name = "openai-compat"

    def __init__(
        self,
        model_id: str,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 16,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def score_claims(self, image: Image.Image, claims: list[str]) -> list[float | None]:
        """Ask the judge about each claim; the image is encoded once."""
        # Imported here: the reconcile package imports eval.metrics, which imports this module.
        from argus_lens.reconcile.verifiers.openai_compat import chat_completion, encode_data_url  # noqa: PLC0415

        image_url = encode_data_url(image)
        return [
            parse_score(
                chat_completion(
                    self.base_url,
                    self.model_id,
                    image_url,
                    PROMPT.format(claim=claim),
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
            )
            for claim in claims
        ]


def _side(claims: list[str], scores: dict[str, float | None], threshold: float) -> dict | None:
    """Summarise one text's claims: counts, the unsupported rate and per-claim detail."""
    if not claims:
        return None
    details = [{"claim": c, "score": scores[c.lower()]} for c in claims]
    scored = [d for d in details if d["score"] is not None]
    unsupported = [d for d in scored if d["score"] < threshold]
    return {
        "claims": len(claims),
        "scored": len(scored),
        "unsupported": len(unsupported),
        "rate": len(unsupported) / len(scored) if scored else None,
        "details": details,
    }


def faithfulness(
    judge: FaithfulnessJudge,
    image: Image.Image,
    result: CaptionResult,
    *,
    trigger_word: str = "",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Judge the raw backend output and the final caption against *image*.

    Returns ``{"raw": side, "final": side}``. Each side is ``None`` when the text
    has no claims, else ``{claims, scored, unsupported, rate, details}``. ``rate``
    is unsupported / scored (lower is better), ``None`` if the judge answered none.
    """
    raw = split_claims(f"{result.raw_prose}\n{result.raw_tags}", trigger_word=trigger_word)
    final = split_claims(result.final_caption, trigger_word=trigger_word)
    unique: dict[str, str] = {}
    for claim in raw + final:
        unique.setdefault(claim.lower(), claim)
    scores = dict(zip(unique, judge.score_claims(image, list(unique.values())), strict=True))
    return {"raw": _side(raw, scores, threshold), "final": _side(final, scores, threshold)}
