"""Replay backend — serve recorded captions from the cortex lineage store.

The public demo ([argus-halo](https://github.com/smk762/argus-halo)) re-enacts a
pipeline that already ran once: cortex's Postgres lineage store captured the
``source_asset → caption`` DAG, and *that recording is the tape*. This backend
replays it — given an image it returns the recorded ``final_caption`` (and the
full variant set) **without loading a model**, so the demo runs with no GPU and
no external API (issue #45).

Assets are keyed by content: cortex de-duplicates ``source_asset`` on the
sha256 of the original **file bytes** (``content_key() == source_asset.sha256``),
so the engine hashes the bytes it ingested and this backend looks the caption up
by that hash. A hash-less input (a pre-decoded PIL image, whose original bytes
are gone) falls back to the asset ``uri``; if neither resolves, the lookup is a
:class:`ReplayMiss` — every image the demo serves is in the tape, so a miss is a
real error rather than something to paper over with a live model.

Reads are direct: this backend runs one small ``SELECT`` joining ``source_asset``
and ``caption`` against ``CORTEX_PG_URL``. It does **not** depend on the
``argus-cortex`` package (the suite's dependency direction is cortex → lens,
never the reverse); it only relies on the documented, stable lineage schema.
``psycopg`` lives behind the ``argus-lens[replay]`` extra and is imported lazily.

The caption text lives in Postgres, so ``CORTEX_S3_*`` (the blob store) is not
needed here — that store owns image *bytes* for other consumers, not caption
text.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

from PIL import Image

from argus_lens.backends.base import CaptionBackend
from argus_lens.types import BackendKind, CaptionResult

# Latest caption per asset: prefer the highest version, then the most recent row.
_LOOKUP_SQL = """
    SELECT c.final_caption, c.variants, c.raw_tags, c.raw_prose,
           c.backend, c.profile, c.metadata, c.version
    FROM caption c
    JOIN source_asset a ON a.id = c.asset_id
    WHERE a.{key} = %s
    ORDER BY c.version DESC, c.created_at DESC
    LIMIT 1
"""


class ReplayMiss(LookupError):
    """Raised when no recorded caption exists for the requested asset.

    Carries the identity that was looked up so callers (e.g. the server) can
    surface a precise 404 instead of a bare "not found".
    """

    def __init__(self, *, sha256: str | None = None, uri: str | None = None, name: str | None = None) -> None:
        """Store the asset identity and build a human-readable message."""
        self.sha256 = sha256
        self.uri = uri
        self.name = name
        ident = sha256 and f"sha256={sha256}" or uri and f"uri={uri}" or name or "<unknown asset>"
        super().__init__(f"no recorded caption for {ident} in the cortex lineage store")


class ReplayBackend(CaptionBackend):
    """Serve recorded captions from the cortex Postgres lineage store.

    This is a *lookup* backend, not an inference one: the engine short-circuits
    to :meth:`lookup` (keyed by the ingested asset's sha256) and returns the
    recorded :class:`~argus_lens.types.CaptionResult` verbatim, bypassing the
    assembly pipeline — replaying captured output, not re-deriving it.

    A single connection is opened lazily on :meth:`load` and reused under a lock
    (the read load of a replay demo is light); a dropped connection is replaced
    on the next call. Tests inject ``connection=`` to bypass the driver.
    """

    name = "replay"
    kind = BackendKind.CLOUD  # no model weights, no GPU — like the cloud backends
    style = "photo"
    requires_gpu = False

    def __init__(self, *, dsn: str | None = None, connection: Any = None, **_: Any) -> None:
        """Store the DSN (or ``CORTEX_PG_URL``); nothing connects until :meth:`load`.

        An injected *connection* (tests) bypasses ``psycopg`` and the network.
        """
        self._dsn = dsn or os.environ.get("CORTEX_PG_URL") or ""
        self._conn = connection
        self._owns_conn = connection is None
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------

    def load(self, device: str = "auto") -> None:
        """Open the Postgres connection eagerly so misconfiguration fails fast.

        *device* is ignored — there is no model to place.
        """
        if self._conn is None:
            self._conn = self._connect()

    def _connect(self) -> Any:
        """Open an autocommit psycopg connection to ``CORTEX_PG_URL``."""
        if not self._dsn:
            raise ValueError(
                "replay backend requires a cortex lineage DSN. Set CORTEX_PG_URL (e.g. postgresql://user:pass@host/db)."
            )
        try:
            import psycopg  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - exercised via the extra
            raise ImportError(
                "replay backend needs psycopg. Install it with: pip install 'argus-lens[replay]'"
            ) from exc
        return psycopg.connect(self._dsn, autocommit=True)

    def unload(self) -> None:
        """Close the connection if this backend owns it (injected ones are left alone)."""
        if self._conn is not None and self._owns_conn:
            with contextlib.suppress(Exception):  # closing a dead connection must not raise
                self._conn.close()
        if self._owns_conn:
            self._conn = None

    # -- availability ----------------------------------------------------------

    def is_available(self) -> bool:
        """True when a DSN is configured (or a connection was injected)."""
        return bool(self._dsn) or self._conn is not None

    def availability_reason(self) -> str | None:
        """Name the missing config when unavailable, else ``None``."""
        return None if self.is_available() else "cortex lineage store not configured (set CORTEX_PG_URL)"

    # -- lookup ----------------------------------------------------------------

    def lookup(self, *, sha256: str | None = None, uri: str | None = None) -> CaptionResult | None:
        """Return the recorded caption for an asset, or ``None`` if absent.

        Resolves by ``source_asset.sha256`` first (the content identity), then by
        ``uri`` when no hash is available. Returns the latest caption (highest
        version, newest row) as a fully-populated :class:`CaptionResult`.
        """
        row = None
        if sha256:
            row = self._query("sha256", sha256)
        if row is None and uri:
            row = self._query("uri", uri)
        if row is None:
            return None
        return self._row_to_result(row, sha256=sha256, uri=uri)

    def _query(self, key: str, value: str) -> tuple[Any, ...] | None:
        """Run the lookup for a single key column, reconnecting once on a dropped conn."""
        sql = _LOOKUP_SQL.format(key=key)
        with self._lock:
            for attempt in (1, 2):
                if self._conn is None:
                    self._conn = self._connect()
                try:
                    with self._conn.cursor() as cur:
                        cur.execute(sql, (value,))
                        return cur.fetchone()
                except Exception:  # noqa: BLE001 - retry once on a stale pooled connection
                    # Only an owned connection can be reopened; an injected one
                    # can't, so retrying it just re-runs on the dead handle and
                    # masks the original error — re-raise immediately instead.
                    if not self._owns_conn or attempt == 2:
                        raise
                    self.unload()  # drop the stale connection; attempt 2 reconnects
        return None

    @staticmethod
    def _row_to_result(row: tuple[Any, ...], *, sha256: str | None, uri: str | None) -> CaptionResult:
        """Map a ``(final_caption, variants, ...)`` row onto a :class:`CaptionResult`."""
        final_caption, variants, raw_tags, raw_prose, backend, profile, metadata, version = row
        variants = dict(variants or {})
        profile = dict(profile or {})
        metadata = dict(metadata or {})
        # Stamp replay provenance so consumers can tell a replayed result from a
        # freshly-inferred one, and can see which recording it came from.
        metadata["replay"] = {
            "source": "cortex-lineage",
            "asset_sha256": sha256,
            "asset_uri": uri,
            "caption_version": version,
            "recorded_backend": backend,
            "recorded_profile": profile,
        }
        return CaptionResult(
            final_caption=final_caption or "",
            caption_variants=variants,
            selected_category=str(profile.get("target_category") or "identity"),
            raw_tags=raw_tags or "",
            raw_prose=raw_prose or "",
            backend_name=backend or ReplayBackend.name,
            metadata=metadata,
        )

    def caption_image(self, image: Image.Image) -> str:
        """Not supported: the engine short-circuits replay to :meth:`lookup`.

        A pixel-only path cannot recover the original file's sha256, so replay is
        keyed at ingestion. This is only reached if a replay backend is wired into
        a pixel pipeline (e.g. as a hybrid leg), which it does not support.
        """
        raise NotImplementedError(
            "ReplayBackend serves recorded captions by asset identity; it has no "
            "pixel-level caption path. Use it as the sole backend so the engine "
            "can key the lookup by the ingested image's sha256."
        )
