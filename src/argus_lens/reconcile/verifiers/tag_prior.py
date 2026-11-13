"""Model-free verifier: trust the tags over the prose.

WD14 tags are generally more reliable than Florence prose on concrete colour
and pose attributes, so when they disagree this verifier resolves in favour of
the tag. No image is inspected — it's deterministic, needs no GPU, and is the
sensible default (the reconciliation summariser from the original design).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from argus_lens.reconcile.types import Verdict

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute


class TagPriorVerifier:
    """Resolve every dispute in favour of the tag value."""

    name = "tag-prior"

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Return the first tag value as the verdict (or abstain if none)."""
        value = dispute.tags_say[0] if dispute.tags_say else None
        return Verdict(subject=dispute.subject, value=value, source=self.name)
