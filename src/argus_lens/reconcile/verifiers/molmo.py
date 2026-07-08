"""Molmo verifier: ask the model a pointed question and parse the answer.

Molmo (AllenAI) is strong at grounded visual QA and can point at pixels, which
makes it well suited to both colour and posture disputes. Here it answers a
single short question per dispute. The model call is guarded; inject *answer_fn*
to unit-test the parsing path without a GPU.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from argus_lens.reconcile.questions import build_question, parse_answer
from argus_lens.reconcile.types import Verdict

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute


class MolmoVerifier:
    """Resolve colour/pose disputes by asking Molmo a single question each."""

    name = "molmo"

    def __init__(
        self,
        model_id: str = "allenai/Molmo-7B-D-0924",
        device: str = "cuda",
        answer_fn: Callable[[Image.Image, str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._answer_fn = answer_fn
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def _answer(self, image: Image.Image, question: str) -> str:
        """Return Molmo's free-text answer to *question* about *image*."""
        if self._answer_fn is not None:
            return self._answer_fn(image, question)
        self._ensure_model()
        import torch  # noqa: PLC0415

        inputs = self._processor.process(images=[image], text=question)
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate_from_batch(
                inputs, max_new_tokens=20, stop_strings="<|endoftext|>", tokenizer=self._processor.tokenizer
            )
        tokens = out[0, inputs["input_ids"].size(1) :]
        return self._processor.tokenizer.decode(tokens, skip_special_tokens=True)

    def _ensure_model(self) -> None:
        """Lazily load Molmo (needs the ``torch`` extra + ~8-17GB VRAM).

        Thread-safe: locked so a shared engine's request threads load the ~16GB
        model once (not once per thread → OOM), publishing ``self._model`` last.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, device_map=self.device)
            self._processor = processor
            self._model = model

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Ask Molmo the dispute's question and map the answer to the vocabulary."""
        answer = self._answer(image, build_question(dispute))
        return Verdict(subject=dispute.subject, value=parse_answer(answer, dispute.kind), source=self.name)
