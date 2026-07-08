"""Florence-2 grounding verifier: localise the subject, then sample its colour.

Uses Florence-2's ``<CAPTION_TO_PHRASE_GROUNDING>`` task (which the captioning
backend never calls) to find the subject's bounding box, then reads the actual
pixels for the colour — grounded truth rather than free-form prose. Colour only;
posture isn't a grounding task, so pose disputes abstain. The model call is
guarded; inject *ground_fn* to unit-test the box→colour path without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from argus_lens.reconcile.color_sample import Box, dominant_color_name
from argus_lens.reconcile.types import Verdict

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute

_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


class FlorenceGroundingVerifier:
    """Resolve colour disputes by grounding the subject and sampling pixels."""

    name = "florence"

    def __init__(
        self,
        model_id: str = "florence-community/Florence-2-base",
        device: str = "cpu",
        ground_fn: Callable[[Image.Image, str], list[Box]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._ground_fn = ground_fn
        self._model = None
        self._processor = None

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

    def _ensure_model(self) -> None:
        """Lazily load the Florence model/processor (needs the ``torch`` extra)."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ground the subject and name its dominant colour; abstain on pose."""
        if dispute.kind != "color":
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        boxes = self._ground(image, f"the {dispute.subject}")
        if not boxes:
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        color = dominant_color_name(image, boxes[0])
        return Verdict(subject=dispute.subject, value=color, source=self.name)
