"""One switch for running Python that ships inside a model repository.

``from_pretrained(..., trust_remote_code=True)`` imports and runs whatever code the
Hugging Face repo carries. Lens allows that only when the operator says so, with
``HF_TRUST_REMOTE_CODE=true`` (or an explicit ``trust_remote_code=True`` argument).
Every loader that can take remote code resolves the flag here, so no loader can
quietly turn it on.
"""

from __future__ import annotations

import os

ENV_VAR = "HF_TRUST_REMOTE_CODE"


class RemoteCodeRefused(ValueError):
    """A model needs ``trust_remote_code`` and the operator hasn't allowed it."""


def trust_remote_code_from_env() -> bool:
    """Return True only when ``HF_TRUST_REMOTE_CODE`` is set to ``1``, ``true`` or ``yes``."""
    return os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def resolve_trust_remote_code(explicit: bool | None) -> bool:
    """Use an explicit argument when given, else the environment (off by default)."""
    return trust_remote_code_from_env() if explicit is None else explicit
