"""Route claims by kind: ``text`` to a chain of readers (#141), posture to argus-puppet, the rest to one verifier.

``--reconcile openai-compat`` with retina OCR for text: counts and colours go to the VQA model, names and quotes go
to OCR first. A reader that fails (unreachable, a bad answer) hands the claim to the next one, and the verdict's
``source`` names the reader that decided, which ``ReconcileChange.source`` records. When every reader fails the
error propagates, so the reconciler counts it: a text check is never skipped silently.

With ``LENS_POSE_URL`` set, :class:`PoseRoutingVerifier` sends posture to argus-puppet first. Unlike a text reader,
a failed pose read has no fallback: the error propagates and the posture word stays as written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute, AttributeVerifier, Verdict

logger = structlog.get_logger()


class TextRoutingVerifier:
    """Route ``text`` disputes to *text_readers* in order, with fallback; everything else to *default*."""

    reads_text = True

    def __init__(self, default: AttributeVerifier, text_readers: list[AttributeVerifier]) -> None:
        if not text_readers:
            raise ValueError("TextRoutingVerifier needs at least one text reader")
        self.default = default
        self.text_readers = list(text_readers)
        self.name = f"{default.name}+text:{'>'.join(r.name for r in self.text_readers)}"
        self.reads_pose = bool(getattr(default, "reads_pose", False))  # posture claims reach a pose default

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ask the first text reader that answers, or the default verifier for any other kind."""
        if dispute.kind != "text":
            return self.default.verify(image, dispute)
        last: Exception | None = None
        for i, reader in enumerate(self.text_readers):
            try:
                return reader.verify(image, dispute)
            except Exception as exc:  # noqa: BLE001 - any reader failure falls through to the next one
                last = exc
                following = self.text_readers[i + 1].name if i + 1 < len(self.text_readers) else None
                logger.warning("text_reader_failed", reader=reader.name, fallback=following, error=str(exc))
        assert last is not None
        raise last


class PoseRoutingVerifier:
    """Route posture checks to argus-puppet's *pose* verifier first; everything else to *default*.

    ``posture`` claims (a prose posture word no tag speaks to) go to *pose* alone. A ``pose`` dispute (tags and
    prose disagree) goes to *pose*; when it abstains (it cannot judge that word, several people, an ``unknown``
    stance, a 1.0 server) *default* decides as it would without puppet. A *pose* failure propagates, with no
    fallback, so the reconciler counts it and leaves the posture word as written: a failed pose read never rewrites
    or drops one.
    """

    reads_pose = True

    def __init__(self, default: AttributeVerifier, pose: AttributeVerifier) -> None:
        self.default = default
        self.pose = pose
        self.name = f"{default.name}+pose:{pose.name}"
        self.reads_text = bool(getattr(default, "reads_text", False))

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ask *pose* about posture (falling back to *default* on an abstained dispute), *default* about the rest."""
        if dispute.kind == "posture":
            return self.pose.verify(image, dispute)
        if dispute.kind != "pose":
            return self.default.verify(image, dispute)
        verdict = self.pose.verify(image, dispute)  # raises on failure: no fallback, the word stays
        if verdict is not None and verdict.value is not None:
            return verdict
        return self.default.verify(image, dispute)
