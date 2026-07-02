"""Backend protocol and base classes for local and cloud captioners."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from argus_lens.types import BackendKind


class CaptionBackend(ABC):
    """Protocol that all captioning backends must implement.

    Subclasses set class-level attributes (*name*, *kind*, *style*,
    *requires_gpu*) and implement the lifecycle methods.

    Two base classes are provided for convenience:

    * ``LocalBackend`` — loads models into GPU/CPU memory.
    * ``CloudBackend`` — calls hosted APIs via HTTP.

    Custom backends can subclass either, or implement ``CaptionBackend``
    directly.
    """

    name: str = "base"
    kind: BackendKind = BackendKind.LOCAL
    style: str = "photo"
    requires_gpu: bool = False

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load model weights / initialise client.

        For local backends this loads into GPU/CPU.  For cloud backends
        this may validate the API key or initialise the HTTP client.
        Called lazily on first use.
        """

    @abstractmethod
    def caption_image(self, image: Image.Image) -> str:
        """Generate a raw caption for a single image.

        Returns unstructured text (tags or prose) that will be fed into
        the assembly pipeline.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release resources (GPU memory, HTTP connections, etc.)."""

    def is_available(self) -> bool:
        """Return True if this backend can be used right now.

        Override to check for model files, API keys, etc.
        """
        return True

    def availability_reason(self) -> str | None:
        """Human-readable reason if not available."""
        return None


class LocalBackend(CaptionBackend):
    """Base class for backends that run inference locally (GPU/CPU).

    Device placement follows the :class:`CaptionBackend` contract: the
    target device is supplied once via :meth:`load` and remembered on the
    instance, so ``caption_image`` stays device-free. Subclasses read the
    remembered intent via ``self._device`` (lazy model loaders should pass
    it through :meth:`resolve_device`).
    """

    kind = BackendKind.LOCAL
    requires_gpu = True

    # Raw device intent ("auto" | "cpu" | "cuda" | "cuda:N"), set by load().
    _device: str = "auto"

    def load(self, device: str = "auto") -> None:
        """Remember the target device for subsequent (lazy) model loads."""
        self._device = device

    def resolve_device(self, device: str | None = None) -> str:
        """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``.

        With no argument, resolves the device remembered by :meth:`load`.
        """
        if device is None:
            device = self._device
        if device != "auto":
            return device
        from argus_lens.retry import resolve_device

        return resolve_device()


class CloudBackend(CaptionBackend):
    """Base class for backends that call hosted APIs.

    Handles API key resolution: constructor > env var > config file.
    """

    kind = BackendKind.CLOUD
    requires_gpu = False

    env_var: str = ""
    estimated_cost_per_image: float | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store credentials and prompt overrides; nothing is validated until :meth:`load`."""
        self._api_key = api_key
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._extra = kwargs

    def resolve_api_key(self) -> str:
        """Resolve API key from constructor, env var, or raise."""
        if self._api_key:
            return self._api_key
        if self.env_var:
            key = os.environ.get(self.env_var, "")
            if key:
                return key
        raise ValueError(
            f"{self.name} backend requires an API key. Pass api_key= or set {self.env_var} environment variable."
        )

    def load(self, device: str = "auto") -> None:
        """Validate the API key eagerly; *device* is ignored for cloud backends."""
        self.resolve_api_key()

    def unload(self) -> None:
        """Do nothing by default; subclasses close HTTP clients here."""
        pass

    def is_available(self) -> bool:
        """Return True if an API key can be resolved."""
        try:
            self.resolve_api_key()
            return True
        except ValueError:
            return False

    def availability_reason(self) -> str | None:
        """Return a hint naming the missing env var, or None if a key is configured."""
        if self.is_available():
            return None
        return f"API key not configured (set {self.env_var})"

    @property
    def default_system_prompt(self) -> str:
        """Generic captioning instructions used when no custom prompt is supplied."""
        return (
            "You are an image captioning assistant. Describe the image in detail, "
            "focusing on the subject's appearance, clothing, pose, expression, "
            "the setting, lighting, and any notable actions. "
            "Be specific and concise. Do not start with 'The image shows' or similar filler."
        )

    @property
    def system_prompt(self) -> str:
        """Effective system prompt: constructor override or the default."""
        return self._system_prompt or self.default_system_prompt
