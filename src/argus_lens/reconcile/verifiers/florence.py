"""Florence-2 grounding verifier: localise the subject, then sample its colour.

Uses Florence-2's ``<CAPTION_TO_PHRASE_GROUNDING>`` task (which the captioning
backend never calls) to find the subject's bounding box, then reads the actual
pixels for the colour — grounded truth rather than free-form prose. Text claims
(#87) are answered with Florence's ``<OCR>`` task. Posture and headcount aren't
grounding tasks, so those abstain. The model calls are guarded; inject
*ground_fn* / *ocr_fn* to unit-test without a GPU.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from argus_lens.backends.florence2 import default_model_id, load_florence
from argus_lens.reconcile.color_sample import Box, dominant_color_name
from argus_lens.reconcile.types import Verdict
from argus_lens.remote_code import ENV_VAR, RemoteCodeRefused, resolve_trust_remote_code

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute

_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
_OCR_TASK = "<OCR>"


class FlorenceGroundingVerifier:
    """Resolve colour disputes by grounding the subject and sampling pixels."""

    name = "florence"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
        ground_fn: Callable[[Image.Image, str], list[Box]] | None = None,
        ocr_fn: Callable[[Image.Image], str] | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        """Pick the weights the same way the caption backend does: native unless remote code is allowed.

        *trust_remote_code* defaults to ``HF_TRUST_REMOTE_CODE`` (off). Naming the legacy
        ``microsoft/Florence-2-*`` weights without it is refused here, not at first use,
        because the reconciler turns a load error into a silent abstain.
        """
        self.trust_remote_code = resolve_trust_remote_code(trust_remote_code)
        self.model_id = model_id or default_model_id(self.trust_remote_code)
        if not self.trust_remote_code and self.model_id.startswith("microsoft/Florence-2"):
            raise RemoteCodeRefused(
                f"{self.model_id} runs Python from its model repo; set {ENV_VAR}=true to allow it, "
                "or use the native florence-community/Florence-2-* weights"
            )
        self.device = device
        self._ground_fn = ground_fn
        self._ocr_fn = ocr_fn
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def _ground(self, image: Image.Image, phrase: str) -> list[Box]:
        """Return bounding boxes for *phrase* via Florence phrase-grounding."""
        if self._ground_fn is not None:
            return self._ground_fn(image, phrase)
        self._ensure_model()
        prompt = f"{_TASK}{phrase}"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
        )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(text, task=_TASK, image_size=(image.width, image.height))
        return [tuple(b) for b in parsed.get(_TASK, {}).get("bboxes", [])]

    def _ocr(self, image: Image.Image) -> str:
        """Return the text Florence's ``<OCR>`` task reads in *image*."""
        if self._ocr_fn is not None:
            return self._ocr_fn(image)
        self._ensure_model()
        inputs = self._processor(text=_OCR_TASK, images=image, return_tensors="pt").to(self.device)
        generated = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False,
            num_beams=3,
        )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(text, task=_OCR_TASK, image_size=(image.width, image.height))
        return str(parsed.get(_OCR_TASK, ""))

    def _ensure_model(self) -> None:
        """Lazily load the Florence model/processor (needs the ``torch`` extra).

        Thread-safe: a shared engine may reconcile from many request threads, so
        loading is locked and ``self._model`` is published **last** — no thread
        sees a set model with a still-``None`` processor, and the model loads once.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            processor, model = load_florence(self.model_id, trust_remote_code=self.trust_remote_code)
            model = model.to(self.device)
            self._processor = processor
            self._model = model

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ground the subject and name its dominant colour, or OCR a text claim; abstain otherwise."""
        if dispute.kind == "text":
            text = " ".join(self._ocr(image).split())
            return Verdict(subject=dispute.subject, value=text or None, source=self.name)
        if dispute.kind != "color":
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        boxes = self._ground(image, f"the {dispute.subject}")
        if not boxes:
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        color = dominant_color_name(image, boxes[0])  # None for a degenerate box → abstain
        return Verdict(subject=dispute.subject, value=color, source=self.name)
