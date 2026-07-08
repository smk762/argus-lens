"""Orchestrate detect → verify → rewrite for one image's tags + prose."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from argus_lens.reconcile.detect import detect_disputes
from argus_lens.reconcile.rewrite import apply_fix
from argus_lens.reconcile.types import ReconcileChange

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeVerifier

logger = structlog.get_logger()


@dataclass
class ReconcileResult:
    """Outcome of a reconciliation pass."""

    prose: str
    changes: list[ReconcileChange] = field(default_factory=list)
    errors: int = 0  # disputes where the verifier raised (so the caller can clean up)


class Reconciler:
    """Fix prose colour/pose claims that contradict the tags, using a verifier.

    For each dispute the verifier is asked to adjudicate; when it returns a
    concrete value that differs from what the prose said, the prose is rewritten
    to match. A verifier that abstains (``value=None``) or confirms the prose
    leaves it untouched.
    """

    def __init__(self, verifier: AttributeVerifier) -> None:
        self.verifier = verifier

    def reconcile(self, image: Image.Image, tags: str, prose: str) -> ReconcileResult:
        """Return the (possibly rewritten) prose plus the list of changes made."""
        disputes = detect_disputes(tags, prose)
        changes: list[ReconcileChange] = []
        errors = 0
        current = prose
        for dispute in disputes:
            try:
                verdict = self.verifier.verify(image, dispute)
                value = verdict.value if verdict is not None else None
            except Exception as exc:  # noqa: BLE001 - a flaky verifier must not break captioning
                logger.warning("reconcile_verify_failed", subject=dispute.subject, error=str(exc))
                errors += 1
                continue
            if not value or value.lower() in {p.lower() for p in dispute.prose_says}:
                continue  # abstained, or confirmed what the prose already said
            rewritten = apply_fix(current, dispute, value)
            if rewritten == current:
                continue  # verdict was a spelling variant / no textual change — no phantom edit
            current = rewritten
            changes.append(
                ReconcileChange(
                    kind=dispute.kind,
                    subject=dispute.subject,
                    was=dispute.prose_says,
                    now=value,
                    source=verdict.source,
                )
            )
        return ReconcileResult(prose=current, changes=changes, errors=errors)
