"""Read an image's text with argus-retina's OCR server, for ``text`` claims only (#141).

A general VLM misreads text and fills the gap with a plausible name, which is the failure the text check exists to
catch. ``ghcr.io/smk762/argus-retina-ocr`` (RapidOCR PP-OCRv6 small, on CPU) only transcribes what is printed. Lens
talks to it over HTTP through cortex's :class:`~argus_cortex.moderation.OcrClient` and never imports argus-retina.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute

from argus_cortex.peers import peer_env, peer_url

from argus_lens.reconcile.types import Verdict

OCR_URL_ENV = "LENS_OCR_URL"  # the retina OCR server's origin, e.g. http://ocr:8133; no default host
OCR_PEER = "ocr"  # cortex's peer name: the suite-wide ARGUS_OCR_URL is read after LENS_OCR_URL


def ocr_url_from_env() -> str | None:
    """The OCR server URL: ``LENS_OCR_URL``, then the suite-wide ``ARGUS_OCR_URL``, or ``None`` when neither is set.

    Resolved by :func:`argus_cortex.peers.peer_url`, so each name is read from the environment and then the suite
    file. A blank value falls through to the next name and ``off`` stops there (no OCR server for lens). A value
    that is not an absolute http(s) URL raises :class:`~argus_cortex.peers.PeerRefused`, naming the variable.
    """
    return peer_url(OCR_PEER, OCR_URL_ENV)


class OcrVerifier:
    """Answer ``text`` claims with a transcription from argus-retina's OCR server; abstain on everything else.

    No legible text is a ``None`` verdict (abstain). A server that can't be reached, or answers badly, raises
    :class:`~argus_cortex.moderation.OcrCheckFailed`: an empty read would drop every claimed name, so a failed one
    must never look like one.
    """

    name = "ocr"
    reads_text = True

    def __init__(self, base_url: str | None = None, *, timeout: float = 30.0, client: Any = None) -> None:
        if client is None:
            url = base_url or ocr_url_from_env()
            if not url:
                raise ValueError(
                    f"the ocr verifier needs the OCR server's URL: pass --ocr-url or set {OCR_URL_ENV} "
                    f"(or the suite-wide {peer_env(OCR_PEER)})"
                )
            from argus_cortex.moderation import OcrClient  # noqa: PLC0415 - only this verifier needs it

            client = OcrClient(url, timeout=timeout)
        self._client = client

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Transcribe *image* for a ``text`` claim, lines joined in reading order; abstain on any other kind."""
        if dispute.kind != "text":
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")  # lossless: JPEG artefacts blur small print
        transcript = " ".join(self._client.read(buf.getvalue(), "image.png").transcript().split())
        return Verdict(subject=dispute.subject, value=transcript or None, source=self.name)
