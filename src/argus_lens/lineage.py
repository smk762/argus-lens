"""Record captions into the argus-cortex lineage store (the replay tape's writer).

Lens is the stage that produces captions, so it is the one that writes the
``source_asset → caption`` rows the rest of the suite reads back: lens's own
``replay`` backend, argus-lightbox, and the argus-halo tape. Each live caption
becomes one ``source_asset`` (de-duplicated on the original bytes' sha256, the
key replay looks up) plus one ``caption`` row, in one transaction. When the
cortex blob store is configured the original bytes go there too, under the
same sha256, so a GPU-free host can show the image as well as its caption.

Opt-in and separate from ``CORTEX_PG_URL``: the replay backend reads that same
URL, and a replay host must never write into the tape it serves. Recording
needs ``LENS_RECORD_LINEAGE=1`` *and* ``CORTEX_PG_URL`` (``[lineage]`` extra).

Recording is best-effort per caption: a failed write is logged and the caption
is still returned, so a database blip never fails a captioning run. An image
with no recoverable original bytes (a pre-decoded PIL input) is not recorded,
because without a sha256 replay could never find it.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from argus_lens.engine import ImageAsset
    from argus_lens.types import CaptionResult

logger = structlog.get_logger()

ENV_RECORD = "LENS_RECORD_LINEAGE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


class LineageRecorder:
    """Write each live caption into a cortex lineage store (and blob store).

    *store* is an ``argus_cortex.store.LineageStore`` and *blobs* an optional
    ``BlobStore``; both are duck-typed so tests can pass fakes. The schema is
    created (idempotently) at construction, so a misconfigured store fails at
    startup rather than on every caption.
    """

    def __init__(self, store: Any, blobs: Any | None = None) -> None:
        self._store = store
        self._blobs = blobs if blobs is not None and blobs.is_enabled else None  # a property
        store.ensure_schema()

    def record(
        self,
        asset: ImageAsset,
        result: CaptionResult,
        *,
        name: str,
        profile: Any,
        params: dict[str, Any],
    ) -> str | None:
        """Record *result* for *asset*; return the caption row id, or ``None`` if skipped/failed."""
        if asset.sha256 is None:
            logger.debug("lineage_skip_no_hash", name=name)
            return None
        from argus_cortex.store import Caption, SourceAsset  # noqa: PLC0415 - [lineage] extra

        try:
            if self._blobs is not None and asset.data is not None:
                self._blobs.put_content_addressed(asset.data)
            caption = Caption.from_caption_result(
                result,
                profile=asdict(profile) if is_dataclass(profile) else (profile or {}),
                params=params,
            )
            with self._store.transaction():
                # An upload has no path or URL; name it so the row says where it came from.
                uri = asset.uri or f"upload:{name}"
                asset_id = self._store.record_asset(SourceAsset(uri=uri, sha256=asset.sha256))
                if asset_id is None:
                    return None
                return self._store.record_caption(asset_id, caption)
        except Exception as exc:  # noqa: BLE001 - recording must never fail the caption
            logger.warning("lineage_record_failed", name=name, sha256=asset.sha256, error=str(exc))
            return None

    def close(self) -> None:
        """Release the store's connection pool (and the blob client)."""
        self._store.close()
        if self._blobs is not None:
            self._blobs.close()


def recorder_from_env() -> LineageRecorder | None:
    """Build a recorder when ``LENS_RECORD_LINEAGE`` is on; ``None`` otherwise.

    Raises ``ValueError`` when the flag is on but ``CORTEX_PG_URL`` is unset, or
    the flag is set to something that is neither true nor false: an explicit
    opt-in that silently records nothing would leave a gap in the tape that is
    only noticed GPU-hours later.
    """
    raw = os.environ.get(ENV_RECORD, "").strip().lower()
    if raw in _FALSY:
        return None
    if raw not in _TRUTHY:
        raise ValueError(f"{ENV_RECORD}={raw!r} is neither true nor false")
    if not os.environ.get("CORTEX_PG_URL", "").strip():
        raise ValueError(f"{ENV_RECORD} is on but CORTEX_PG_URL is unset")
    try:
        from argus_cortex.store import open_blob_store, open_lineage_store  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError("Recording lineage requires: pip install argus-lens[lineage]") from exc
    return LineageRecorder(open_lineage_store(), open_blob_store())
