"""Send ``text`` claims to a chain of readers and every other dispute to one verifier (#141).

``--reconcile openai-compat`` with retina OCR for text: counts and colours go to the VQA model, names and quotes go
to OCR first. A reader that fails (unreachable, a bad answer) hands the claim to the next one, and the verdict's
``source`` names the reader that decided, which ``ReconcileChange.source`` records. When every reader fails the
error propagates, so the reconciler counts it: a text check is never skipped silently.
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
