"""Core types for attribute reconciliation (issue #36).

The reconciler resolves the specific attributes where the prose (Florence, etc.)
*contradicts* the tags (WD14) — the colour/pose hallucinations — by asking a
pluggable :class:`AttributeVerifier` to adjudicate each dispute, then rewriting
the prose to match the verified answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class AttributeDispute:
    """A single attribute where prose and tags disagree.

    ``kind`` is ``"color"`` or ``"pose"``; ``subject`` is the thing in question
    (``"dress"``, ``"posture"``); ``tags_say`` / ``prose_says`` are the
    conflicting values from each signal.
    """

    kind: str
    subject: str
    tags_say: tuple[str, ...]
    prose_says: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    """A verifier's ruling on a dispute.

    ``value`` is the resolved attribute (e.g. ``"red"``, ``"standing"``), or
    ``None`` to abstain (leave the prose unchanged). ``source`` names the
    verifier; ``confidence`` is an optional 0–1 score.
    """

    subject: str
    value: str | None
    source: str
    confidence: float = 1.0


@runtime_checkable
class AttributeVerifier(Protocol):
    """Adjudicates an :class:`AttributeDispute` for one image.

    Implementations range from model-free (trust the tags) to VLM-backed
    (Florence grounding, Molmo pointing, an OpenAI-compatible VQA endpoint).
    A verifier may ``return Verdict(..., value=None, ...)`` to abstain.
    """

    name: str

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Return a :class:`Verdict` resolving *dispute* for *image*."""
        ...


@dataclass(frozen=True)
class ReconcileChange:
    """A record of one prose edit the reconciler made."""

    kind: str
    subject: str
    was: tuple[str, ...]
    now: str
    source: str
