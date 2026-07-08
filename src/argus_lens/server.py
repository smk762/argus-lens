"""Optional FastAPI micro-server for Argus Lens."""

from __future__ import annotations

import asyncio
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError("Server requires: pip install argus-lens[server]") from exc

import httpx
import structlog
from PIL import Image

from argus_lens._version import __version__
from argus_lens.assembly.profiles import available_profiles
from argus_lens.connectors.base import AssetRef
from argus_lens.connectors.filesystem import IMAGE_SUFFIXES
from argus_lens.connectors.immich import ImmichSink, ImmichSource
from argus_lens.connectors.xmp import XmpSink
from argus_lens.engine import ArgusLens
from argus_lens.openai_compat import create_openai_router
from argus_lens.types import (
    BACKEND_TOKEN_BUDGETS,
    CAPTION_TARGET_STYLES,
    DEFAULT_HYBRID_PRESET,
    HYBRID_PRESETS,
    CaptionResult,
    get_category_names,
)

logger = structlog.get_logger()

# One source of truth for what counts as an image, shared with the connector layer.
SUPPORTED_EXTS = IMAGE_SUFFIXES
_XMP_SINK = XmpSink()  # stateless; shared by every endpoint that writes XMP sidecars
_COUNT_CAP = 5000  # per-folder recursive image-count ceiling (keeps browsing snappy)
_PULL_CONCURRENCY = 4  # parallel Immich downloads per /immich/pull request


class CaptionURLRequest(BaseModel):
    """Caption a single image fetched from a URL."""

    image_url: str
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    hybrid_preset: str | None = None
    prose_bias: float | None = None
    prose_enrichment: bool = True


class CaptionFolderRequest(BaseModel):
    """Batch-caption every image under a server-side folder path."""

    folder: str
    recursive: bool = False
    write_sidecar: bool = True
    write_xmp: bool = False
    xmp_overwrite: bool = True
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    checkpoint: str | None = None
    hybrid_preset: str | None = None
    prose_bias: float | None = None
    prose_enrichment: bool = True


class ImmichPullRequest(BaseModel):
    """Pull an Immich album's assets into a folder under the source root."""

    album_id: str
    asset_ids: list[str] | None = None
    dest_folder: str


class ImmichCaptionRequest(BaseModel):
    """Caption an Immich album's assets in memory, optionally writing back."""

    album_id: str
    asset_ids: list[str] | None = None
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    checkpoint: str | None = None
    hybrid_preset: str | None = None
    prose_bias: float | None = None
    prose_enrichment: bool = True
    write_back: bool = False
    write_xmp: bool = False


def _immich_config() -> tuple[str, str]:
    """Read Immich connection settings from the environment at request time.

    Returns ``(url, api_key)`` or raises HTTP 503 when either is unset, so the
    server can start (and serve every other endpoint) without Immich.
    """
    url = os.environ.get("IMMICH_URL")
    api_key = os.environ.get("IMMICH_API_KEY")
    if not url or not api_key:
        raise HTTPException(
            status_code=503,
            detail="Immich is not configured: set IMMICH_URL and IMMICH_API_KEY",
        )
    return url, api_key


# Failures talking to Immich that must surface as 502, not a 500 traceback:
# transport errors, a malformed IMMICH_URL, and 200-but-garbled bodies —
# json.JSONDecodeError is a ValueError (e.g. a reverse proxy answering with an
# HTML login page) and KeyError covers responses missing expected fields.
_IMMICH_UPSTREAM_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError)


def _immich_502(exc: Exception) -> HTTPException:
    """502 for a failed or garbled Immich exchange."""
    return HTTPException(status_code=502, detail=f"Immich request failed: {exc!r}")


def _immich_album_assets(
    source: ImmichSource,
    album_id: str,
    asset_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Resolve an album's ``{"id", "name"}`` assets, filtered to *asset_ids* when given.

    ``asset_ids=None`` means the whole album; an explicit list selects exactly
    those assets (``[]`` selects none — a UI posting an emptied selection must
    not operate on the full album). Ids missing from the album raise 404
    instead of being silently dropped as a zero-work success.
    """
    assets = source.list_album_assets(album_id)
    if asset_ids is not None:
        wanted = set(asset_ids)
        assets = [a for a in assets if a["id"] in wanted]
        missing = wanted - {a["id"] for a in assets}
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"asset ids not found in album {album_id!r}: {sorted(missing)}",
            )
    return assets


def _result_to_dict(result: CaptionResult) -> dict[str, Any]:
    """Convert a ``CaptionResult`` dataclass into a JSON-serialisable dict."""
    return asdict(result)


def _parse_keywords(raw_tags: str) -> list[str]:
    """Split a comma-separated raw-tags string into trimmed, non-empty keywords."""
    return [t.strip() for t in raw_tags.split(",") if t.strip()]


def _resolve_within(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse path traversal escapes."""
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status_code=400, detail="path escapes the source root")
    return candidate


def _confine_folder(source_root: str | None, folder: str) -> Path:
    """Resolve a requested caption folder inside the configured source root.

    ``folder`` may be relative to the root or an absolute path within it;
    anything else is rejected so the endpoint cannot be used to walk (or write
    sidecars into) arbitrary server-side directories.
    """
    if not source_root:
        raise HTTPException(
            status_code=400,
            detail="no source root configured (set --source-root or LENS_SOURCE_PATH)",
        )
    root = Path(source_root)
    requested = Path(folder)
    if requested.is_absolute():
        candidate = requested.resolve()
        root_resolved = root.resolve()
        if root_resolved not in candidate.parents and candidate != root_resolved:
            raise HTTPException(status_code=400, detail="folder is outside the configured source root")
        return candidate
    return _resolve_within(root, folder)


def _parse_manifest(raw: bytes) -> list[dict[str, Any]]:
    """Decode and validate an uploaded JSONL manifest into row dicts.

    Raises HTTP 400 for non-UTF-8 bytes, invalid JSON, or lines that are valid
    JSON but not objects (e.g. a bare ``null``), naming the offending line.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"manifest not UTF-8: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON on line {i + 1}: {exc}") from exc
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail=f"line {i + 1} is not a JSON object")
        rows.append(row)
    return rows


def _caption_and_write(
    engine: ArgusLens,
    image_path: str,
    rel_path: str,
    *,
    trigger_word: str,
    target_style: str,
    target_category: str,
    target_backend: str,
    checkpoint: str | None,
    prose_enrichment: bool,
    write_sidecar: bool,
    write_xmp: bool,
    xmp_overwrite: bool,
    written: set[Path],
    hybrid_preset: str | None = None,
    prose_bias: float | None = None,
) -> dict[str, Any]:
    """Caption one image and (optionally) write its ``.txt``/``.xmp`` sidecars.

    Returns ``{"rel_path", "final_caption"}`` on success (plus ``xmp_path``
    when an XMP sidecar was written) or ``{"rel_path", "error"}`` on failure —
    never both, so batch counts stay consistent (an XMP failure after the
    ``.txt`` sidecar landed says so in the error text, since that file is
    already on disk). ``written`` tracks sidecar paths already written in this
    batch, resolved so two spellings of one target still collide: same-stem
    images (``cat.jpg`` + ``cat.png``) map to the same ``cat.txt``, and the
    collision is reported as an error instead of silently overwriting the
    first caption. With ``xmp_overwrite`` (the default) an existing ``.xmp``
    on disk is replaced like the ``.txt`` sidecars; without it a pre-existing
    sidecar (e.g. written by Lightroom/digiKam, whose metadata is not merged)
    is a per-image error instead.
    """
    try:
        result = engine.caption(
            image_path,
            trigger_word=trigger_word,
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            hybrid_preset=hybrid_preset,
            prose_bias=prose_bias,
            prose_enrichment=prose_enrichment,
        )
    except Exception as exc:  # noqa: BLE001 - report per-image, keep going
        return {"rel_path": rel_path, "error": str(exc)}
    if write_sidecar:
        sidecar = Path(image_path).with_suffix(".txt")
        # Collision keys are resolved: '..' segments or symlinked spellings of
        # the same target must not sneak past the duplicate check.
        sidecar_key = sidecar.resolve()
        if sidecar_key in written:
            return {
                "rel_path": rel_path,
                "error": f"sidecar collision: {sidecar.name} was already written for another image in this batch",
            }
        try:
            sidecar.write_text(result.final_caption, encoding="utf-8")
        except OSError as exc:
            return {"rel_path": rel_path, "error": f"sidecar write failed: {exc}"}
        written.add(sidecar_key)
    outcome: dict[str, Any] = {"rel_path": rel_path, "final_caption": result.final_caption}
    if write_xmp:
        ref = AssetRef(id=rel_path, path=image_path)
        xmp_path = XmpSink.sidecar_path(ref)
        xmp_key = xmp_path.resolve()
        # If the .txt sidecar landed above, that is on-disk state the XMP error
        # rows below must not hide from the caller.
        txt_note = " (the .txt sidecar for this image was already written)" if write_sidecar else ""
        if xmp_key in written:
            return {
                "rel_path": rel_path,
                "error": (
                    f"xmp sidecar collision: {xmp_path.name} was already written "
                    f"for another image in this batch{txt_note}"
                ),
            }
        keywords = _parse_keywords(result.raw_tags)
        try:
            # xmp_overwrite=True mirrors the .txt sidecar semantics above (a
            # pre-existing file on disk is replaced); False protects sidecars
            # other tools already populated, since XmpSink never merges.
            _XMP_SINK.write(ref, keywords=keywords, description=result.final_caption, overwrite=xmp_overwrite)
        except FileExistsError:
            return {
                "rel_path": rel_path,
                "error": f"xmp sidecar already exists: {xmp_path.name} (xmp_overwrite is false){txt_note}",
            }
        except OSError as exc:
            return {"rel_path": rel_path, "error": f"xmp write failed: {exc}{txt_note}"}
        written.add(xmp_key)
        outcome["xmp_path"] = str(xmp_path)
    return outcome


def _caption_manifest_row(
    engine: ArgusLens,
    row: dict[str, Any],
    *,
    trigger_word: str,
    write_sidecar: bool,
    write_xmp: bool,
    xmp_overwrite: bool,
    prose_enrichment: bool,
    written: set[Path],
) -> dict[str, Any]:
    """Caption one manifest row via its ``abs_path`` and ``target_profile``."""
    abs_path = row.get("abs_path")
    rel_path = row.get("rel_path") or abs_path or "<unknown>"
    if not abs_path:
        return {"rel_path": rel_path, "error": "row missing abs_path"}
    profile = row.get("target_profile") or {}
    if not isinstance(profile, dict):
        return {"rel_path": rel_path, "error": "target_profile must be a JSON object"}
    return _caption_and_write(
        engine,
        abs_path,
        rel_path,
        trigger_word=trigger_word,
        target_style=profile.get("target_style", "photo"),
        target_category=profile.get("target_category", "identity"),
        target_backend=profile.get("target_backend", "sdxl"),
        checkpoint=profile.get("checkpoint"),
        prose_enrichment=prose_enrichment,
        write_sidecar=write_sidecar,
        write_xmp=write_xmp,
        xmp_overwrite=xmp_overwrite,
        written=written,
    )


def _count_images(directory: Path, cap: int = _COUNT_CAP) -> int:
    """Recursive count of supported images under *directory* (capped)."""
    n = 0
    try:
        for p in directory.rglob("*"):
            if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
                n += 1
                if n >= cap:
                    break
    except OSError:
        pass
    return n


def _browse_folders(root: Path, rel: str) -> dict[str, Any]:
    """List sub-directories (with recursive image counts) under root/rel."""
    base = _resolve_within(root, rel)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail=f"not a directory: {rel or '.'}")

    folders: list[dict[str, Any]] = []
    direct_images = 0
    try:
        for entry in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                sub_rel = str(Path(rel) / entry.name) if rel else entry.name
                subfolders = sum(1 for c in entry.iterdir() if c.is_dir() and not c.name.startswith("."))
                folders.append(
                    {
                        "name": entry.name,
                        "rel_path": sub_rel,
                        "abs_path": str(entry.resolve()),
                        "image_count": _count_images(entry),
                        "subfolder_count": subfolders,
                    }
                )
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTS:
                direct_images += 1
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot read directory: {exc}") from exc

    parent = None
    if rel:
        parent_path = str(Path(rel).parent)
        parent = "" if parent_path == "." else parent_path

    return {
        "root": str(root.resolve()),
        "path": rel,
        "abs_path": str(base.resolve()),
        "parent": parent,
        "direct_image_count": direct_images,
        "folders": folders,
    }


def _require_admin_token(authorization: str | None) -> None:
    """Enforce ``Authorization: Bearer <ARGUS_ADMIN_TOKEN>`` when the env var is set.

    When ``ARGUS_ADMIN_TOKEN`` is unset the admin endpoints stay open (dev
    default); setting it locks them down for a networked deployment.
    """
    token = os.environ.get("ARGUS_ADMIN_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="admin token required")


def create_app(
    default_backend: str = "hybrid",
    cors: bool = False,
    cors_origins: list[str] | None = None,
    source_root: str | None = None,
    **kwargs: Any,
) -> FastAPI:
    """Create a FastAPI application for image captioning."""

    app = FastAPI(
        title="Argus Lens",
        description="Structured image captioning API",
        version="0.1.0",
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        origins = cors_origins or ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            # Credentials with a wildcard origin is invalid per the CORS spec
            # (Starlette would echo the caller's origin, silently granting any
            # site credentialed access) — only allow it for explicit origins.
            allow_credentials="*" not in origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    engine = ArgusLens(backend=default_backend, **kwargs)
    # Root for GET /folders browsing + relative folder captioning (UI folder picker).
    default_source = source_root or os.environ.get("LENS_SOURCE_PATH")

    # OpenAI-compatible /v1 endpoints (Frigate GenAI provider).
    # Always mounted — Frigate's genai block uses POST /v1/chat/completions.
    # Engine kwargs are forwarded so model_dir / florence_model_id are honoured.
    app.include_router(create_openai_router(**kwargs), prefix="/v1")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Service liveness/identity probe (mirrors argus-curator's /health shape).

        Includes GPU residency + free VRAM so a coordinator (e.g. gothmog) can
        attribute and reclaim this backend's memory (#37). The VRAM probe runs
        off the event loop (its first CUDA call can trigger context init).
        """
        return {
            "status": "ok",
            "service": "argus-lens",
            "version": __version__,
            "source_root": str(Path(default_source).resolve()) if default_source else None,
            "gpu": await asyncio.to_thread(engine.vram_status),
        }

    @app.post("/admin/unload")
    async def admin_unload(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Unload the model to free VRAM for a co-resident GPU tenant (#37).

        Matches the ``/unload`` contract gothmog's idle reaper / ``/v1/gpu/evict``
        expect; the model reloads lazily on the next caption. Set ``ARGUS_ADMIN_TOKEN``
        to require ``Authorization: Bearer <token>`` (otherwise the endpoint is open —
        keep it off untrusted networks to avoid unload/reload-thrash abuse).
        """
        _require_admin_token(authorization)
        unloaded = await asyncio.to_thread(engine.unload)
        return {"unloaded": unloaded, "gpu": await asyncio.to_thread(engine.vram_status)}

    @app.get("/profiles")
    async def profiles() -> dict[str, Any]:
        """Expose the caption taxonomy so UIs don't have to hardcode it.

        Values are derived from the real sources of truth: the assembly-profile
        registry, ``CAPTION_TARGET_STYLES``, ``DEFAULT_CATEGORY_CONFIGS``, and
        ``BACKEND_TOKEN_BUDGETS``.
        """
        return {
            "assembly_profiles": list(available_profiles()),
            "target_styles": list(CAPTION_TARGET_STYLES),
            "target_categories": list(get_category_names()),
            "target_backends": list(BACKEND_TOKEN_BUDGETS),
            "token_budgets": dict(BACKEND_TOKEN_BUDGETS),
            "hybrid_presets": dict(HYBRID_PRESETS),
            "default_hybrid_preset": DEFAULT_HYBRID_PRESET,
        }

    @app.get("/backends")
    async def list_backends() -> dict[str, Any]:
        """List all registered captioning backends with availability status."""
        return {"backends": engine.available_backends()}

    @app.post("/caption")
    async def caption_image(
        file: UploadFile = File(...),
        trigger_word: str = Form(""),
        target_style: str = Form("photo"),
        target_category: str = Form("identity"),
        target_backend: str = Form("sdxl"),
        hybrid_preset: str | None = Form(None),
        prose_bias: float | None = Form(None),
    ) -> dict[str, Any]:
        """Caption a single uploaded image and return the structured result."""
        data = await file.read()
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        result = await asyncio.to_thread(
            engine.caption,
            pil,
            trigger_word=trigger_word,
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            hybrid_preset=hybrid_preset,
            prose_bias=prose_bias,
        )
        return _result_to_dict(result)

    @app.post("/caption/url")
    async def caption_url(req: CaptionURLRequest) -> dict[str, Any]:
        """Caption an image from a URL (JSON body, no file upload needed)."""
        try:
            result = await asyncio.to_thread(
                engine.caption,
                req.image_url,
                trigger_word=req.trigger_word,
                target_style=req.target_style,
                target_category=req.target_category,
                target_backend=req.target_backend,
                hybrid_preset=req.hybrid_preset,
                prose_bias=req.prose_bias,
                prose_enrichment=req.prose_enrichment,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _result_to_dict(result)

    @app.post("/caption/batch")
    async def caption_batch(
        files: list[UploadFile] = File(...),
        trigger_word: str = Form(""),
        target_style: str = Form("photo"),
        target_category: str = Form("identity"),
        target_backend: str = Form("sdxl"),
        hybrid_preset: str | None = Form(None),
        prose_bias: float | None = Form(None),
    ) -> dict[str, Any]:
        """Caption multiple uploaded images in one request; unreadable files are skipped."""
        images: list[tuple[str, Image.Image]] = []
        for f in files:
            data = await f.read()
            try:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                images.append((f.filename or "image", pil))
            except Exception:
                continue

        results = await asyncio.to_thread(
            engine.caption_batch,
            [img for _, img in images],
            trigger_word=trigger_word,
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            hybrid_preset=hybrid_preset,
            prose_bias=prose_bias,
        )
        return {"results": {k: _result_to_dict(v) for k, v in results.items()}}

    @app.post("/caption/stream")
    async def caption_stream(
        files: list[UploadFile] = File(...),
        trigger_word: str = Form(""),
        target_style: str = Form("photo"),
        target_category: str = Form("identity"),
        target_backend: str = Form("sdxl"),
        hybrid_preset: str | None = Form(None),
        prose_bias: float | None = Form(None),
    ) -> StreamingResponse:
        """Caption uploaded images, streaming one NDJSON result line per image."""
        images: list[Image.Image] = []
        for f in files:
            data = await f.read()
            try:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                images.append(pil)
            except Exception:
                continue

        async def _ndjson():
            """Yield one JSON line per captioned image, running inference off the event loop."""
            stream = engine.caption_stream(
                images,
                trigger_word=trigger_word,
                target_style=target_style,
                target_category=target_category,
                target_backend=target_backend,
                hybrid_preset=hybrid_preset,
                prose_bias=prose_bias,
            )
            sentinel = object()
            while True:
                # caption_stream is a sync generator doing blocking CPU/GPU work
                # (including OOM-retry sleeps) — pull each item in a worker
                # thread so the event loop stays responsive.
                item = await asyncio.to_thread(next, stream, sentinel)
                if item is sentinel:
                    break
                name, result = item
                yield json.dumps({"name": name, **_result_to_dict(result)}) + "\n"

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    @app.post("/caption/manifest")
    async def caption_manifest(
        manifest: UploadFile = File(...),
        trigger_word: str = Form(""),
        write_sidecar: bool = Form(True),
        write_xmp: bool = Form(False),
        xmp_overwrite: bool = Form(True),
        prose_enrichment: bool = Form(True),
    ) -> dict[str, Any]:
        """Batch-caption an argus-curator JSONL manifest.

        Each line carries ``abs_path`` and the shared ``target_profile``; images
        are captioned with that profile (no category remapping) and, by default,
        a ``.txt`` sidecar is written next to each image. ``write_xmp`` also
        writes an ``<image>.xmp`` sidecar (dc:subject keywords +
        dc:description caption) that Lightroom/digiKam/Immich ingest natively;
        it is independent of ``write_sidecar``, and ``xmp_overwrite: false``
        turns a pre-existing ``.xmp`` (e.g. one another tool populated) into a
        per-image error instead of replacing it. Assumes the images are
        reachable at ``abs_path`` (e.g. a shared volume with the curator).
        """
        rows = _parse_manifest(await manifest.read())

        def _run() -> dict[str, Any]:
            """Caption every manifest row sequentially, collecting results and per-row errors."""
            written: set[Path] = set()
            results: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for row in rows:
                outcome = _caption_manifest_row(
                    engine,
                    row,
                    trigger_word=trigger_word,
                    write_sidecar=write_sidecar,
                    write_xmp=write_xmp,
                    xmp_overwrite=xmp_overwrite,
                    prose_enrichment=prose_enrichment,
                    written=written,
                )
                (errors if "error" in outcome else results).append(outcome)
            return {
                "total": len(rows),
                "captioned": len(results),
                "failed": len(errors),
                "xmp_written": sum(1 for r in results if "xmp_path" in r),
                "results": results,
                "errors": errors,
            }

        return await asyncio.to_thread(_run)

    @app.post("/caption/manifest/stream")
    async def caption_manifest_stream(
        manifest: UploadFile = File(...),
        trigger_word: str = Form(""),
        write_sidecar: bool = Form(True),
        write_xmp: bool = Form(False),
        xmp_overwrite: bool = Form(True),
        prose_enrichment: bool = Form(True),
    ) -> StreamingResponse:
        """Streaming variant of /caption/manifest for live progress.

        Yields one NDJSON object per image as it is captioned
        (``{type:"progress", done, total, rel_path, final_caption|error}``,
        plus ``xmp_path`` when ``write_xmp`` wrote a sidecar), then a final
        ``{type:"complete", total, captioned, failed, xmp_written}`` line. As
        with /caption/manifest, images are read from ``abs_path`` and a ``.txt``
        sidecar is written next to each (shared volume with the curator).
        """
        rows = _parse_manifest(await manifest.read())
        total = len(rows)
        written: set[Path] = set()

        async def _ndjson() -> Any:
            """Yield a progress line per row, then a final completion summary line."""
            captioned = 0
            failed = 0
            xmp_written = 0
            for i, row in enumerate(rows):
                # Caption is blocking CPU/GPU work — run off the event loop so the
                # stream flushes each line promptly.
                outcome = await asyncio.to_thread(
                    _caption_manifest_row,
                    engine,
                    row,
                    trigger_word=trigger_word,
                    write_sidecar=write_sidecar,
                    write_xmp=write_xmp,
                    xmp_overwrite=xmp_overwrite,
                    prose_enrichment=prose_enrichment,
                    written=written,
                )
                if "error" in outcome:
                    failed += 1
                else:
                    captioned += 1
                    if "xmp_path" in outcome:
                        xmp_written += 1
                yield json.dumps({"type": "progress", "done": i + 1, "total": total, **outcome}) + "\n"
            yield (
                json.dumps(
                    {
                        "type": "complete",
                        "total": total,
                        "captioned": captioned,
                        "failed": failed,
                        "xmp_written": xmp_written,
                    }
                )
                + "\n"
            )

        return StreamingResponse(
            _ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/folders")
    async def folders(path: str = "") -> dict[str, Any]:
        """Browse mounted folders under the configured source root (UI picker)."""
        if not default_source:
            raise HTTPException(status_code=400, detail="no source root configured (set LENS_SOURCE_PATH)")
        root = Path(default_source)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"source root is not a directory: {default_source}")
        return await asyncio.to_thread(_browse_folders, root, path)

    @app.post("/caption/folder")
    async def caption_folder(req: CaptionFolderRequest) -> dict[str, Any]:
        """Batch-caption every image in a folder under the source root.

        ``folder`` may be relative to the configured source root or an absolute
        path inside it; anything outside the root is rejected. Walks the folder
        (optionally recursively), captions each image with the given target
        profile, and — by default — writes a ``.txt`` sidecar next to each
        image. ``write_xmp`` additionally (and independently) writes an
        ``<image>.xmp`` sidecar for Lightroom/digiKam/Immich ingestion;
        ``xmp_overwrite: false`` makes a pre-existing ``.xmp`` a per-image
        error instead of replacing it. Returns the same shape as
        ``/caption/manifest``.
        """
        root = _confine_folder(default_source, req.folder)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {req.folder}")

        def _run() -> dict[str, Any]:
            """Walk and caption off the event loop, collecting results and per-image errors."""
            walker = root.rglob("*") if req.recursive else root.iterdir()
            images = sorted(p for p in walker if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
            written: set[Path] = set()
            results: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for p in images:
                outcome = _caption_and_write(
                    engine,
                    str(p),
                    str(p.relative_to(root)),
                    trigger_word=req.trigger_word,
                    target_style=req.target_style,
                    target_category=req.target_category,
                    target_backend=req.target_backend,
                    checkpoint=req.checkpoint,
                    hybrid_preset=req.hybrid_preset,
                    prose_bias=req.prose_bias,
                    prose_enrichment=req.prose_enrichment,
                    write_sidecar=req.write_sidecar,
                    write_xmp=req.write_xmp,
                    xmp_overwrite=req.xmp_overwrite,
                    written=written,
                )
                (errors if "error" in outcome else results).append(outcome)
            return {
                "total": len(images),
                "captioned": len(results),
                "failed": len(errors),
                "xmp_written": sum(1 for r in results if "xmp_path" in r),
                "results": results,
                "errors": errors,
            }

        return await asyncio.to_thread(_run)

    @app.get("/immich/albums")
    async def immich_albums() -> dict[str, Any]:
        """List Immich albums with asset counts (requires IMMICH_URL/IMMICH_API_KEY)."""
        url, api_key = _immich_config()
        source = ImmichSource(url, api_key)
        try:
            albums = await asyncio.to_thread(source.list_albums)
        except _IMMICH_UPSTREAM_ERRORS as exc:
            raise _immich_502(exc) from exc
        finally:
            source.close()
        return {"albums": albums}

    @app.post("/immich/pull")
    async def immich_pull(req: ImmichPullRequest) -> StreamingResponse:
        """Download Immich album assets into a folder under the source root.

        Streams NDJSON: one ``{type:"progress", done, total, name}`` line per
        asset (with an ``error`` field on per-asset failure, and a ``warning``
        when the pulled file's extension is one ``/caption/folder`` will not
        pick up, e.g. HEIC/DNG originals), then a final ``{type:"complete",
        folder, downloaded, skipped, failed}`` line. Downloads run a few at a
        time and land via a temp file + atomic rename, so an interrupted write
        never leaves a truncated image behind. Files that already exist with
        the same name are skipped; when two assets in the same request share a
        basename, the second is reported as a failure instead of being
        silently dropped. ``dest_folder`` must resolve inside the configured
        source root.
        """
        url, api_key = _immich_config()
        dest = _confine_folder(default_source, req.dest_folder)
        source = ImmichSource(url, api_key)
        try:
            try:
                assets = await asyncio.to_thread(_immich_album_assets, source, req.album_id, req.asset_ids)
            except _IMMICH_UPSTREAM_ERRORS as exc:
                raise _immich_502(exc) from exc
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, NotADirectoryError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"dest_folder is not a directory: {req.dest_folder}"
                ) from exc
        except HTTPException:
            source.close()
            raise
        total = len(assets)
        root_resolved = Path(default_source).resolve()  # default_source is set: _confine_folder passed
        folder_rel = str(dest.relative_to(root_resolved))
        logger.info("immich_pull_start", album_id=req.album_id, total=total, folder=folder_rel)

        def _download(asset: dict[str, Any], target: Path) -> None:
            """Fetch one asset's original bytes and write them via temp file + atomic rename.

            A direct write would leave a truncated file on ENOSPC/kill, which
            the exists()-skip below would then treat as complete on every
            later pull.
            """
            data = source.fetch_original(AssetRef(id=asset["id"]))
            tmp = target.with_name(target.name + ".part")
            try:
                tmp.write_bytes(data)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)

        async def _ndjson() -> Any:
            """Yield a progress line per asset, then a final completion summary line."""
            downloaded = skipped = failed = 0
            sem = asyncio.Semaphore(_PULL_CONCURRENCY)

            async def _fetch(asset: dict[str, Any], target: Path) -> None:
                """Download one asset under the concurrency cap, off the event loop."""
                async with sem:
                    await asyncio.to_thread(_download, asset, target)

            # Plan every asset first (duplicate/skip decisions are order-
            # dependent), then let downloads overlap under the semaphore while
            # progress lines still stream in album order.
            plans: list[tuple[str, str, Any]] = []
            seen_names: set[str] = set()
            for asset in assets:
                # basename only: an Immich filename can never climb out of dest
                name = Path(asset["name"]).name
                if name in seen_names:
                    # Two assets in this request share a basename: skipping would
                    # silently drop the second one, so report it as a failure
                    # (mirrors the sidecar collision handling in _caption_and_write).
                    plans.append((name, "collision", None))
                    continue
                seen_names.add(name)
                target = dest / name
                if target.exists():
                    plans.append((name, "skip", None))
                else:
                    plans.append((name, "fetch", asyncio.create_task(_fetch(asset, target))))
            try:
                for i, (name, action, task) in enumerate(plans):
                    line: dict[str, Any] = {"type": "progress", "done": i + 1, "total": total, "name": name}
                    if action == "collision":
                        failed += 1
                        line["error"] = (
                            f"filename collision: {name} was already written for another asset in this request"
                        )
                    elif action == "skip":
                        skipped += 1
                    else:
                        try:
                            await task
                            downloaded += 1
                        except Exception as exc:  # noqa: BLE001 - report per-asset, keep going
                            failed += 1
                            line["error"] = str(exc)
                    suffix = Path(name).suffix.lower()
                    if "error" not in line and suffix not in SUPPORTED_EXTS:
                        # The pull succeeds, but /caption/folder only walks
                        # SUPPORTED_EXTS — say so instead of letting the
                        # pull-then-caption workflow dead-end silently.
                        line["warning"] = (
                            f"not captionable: {suffix!r} is not a supported image type"
                            if suffix
                            else "not captionable: file has no extension"
                        )
                    yield json.dumps(line) + "\n"
                logger.info("immich_pull_done", downloaded=downloaded, skipped=skipped, failed=failed)
                yield (
                    json.dumps(
                        {
                            "type": "complete",
                            "folder": folder_rel,
                            "downloaded": downloaded,
                            "skipped": skipped,
                            "failed": failed,
                        }
                    )
                    + "\n"
                )
            finally:
                pending = [t for _, _, t in plans if t is not None]
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                source.close()

        return StreamingResponse(
            _ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/immich/caption/stream")
    async def immich_caption_stream(req: ImmichCaptionRequest) -> StreamingResponse:
        """Caption Immich album assets in memory, streaming NDJSON progress.

        Each asset is fetched via the Immich API (no disk writes) and captioned
        with the requested target profile. Streams one ``{type:"progress",
        done, total, asset_id, name, final_caption|error}`` line per asset,
        then a final ``{type:"complete", total, captioned, failed}`` line.
        When ``write_back`` is true, the caption is pushed back to Immich as
        the asset description, with the raw tags (when available) as keywords;
        if only the write-back step fails, the progress line keeps
        ``final_caption`` alongside ``error`` (and counts as failed).

        ``write_xmp`` is rejected here (400): assets are fetched via the Immich
        API and never touch local disk, so there is no image path to place a
        sidecar next to. To get XMP sidecars for Immich assets, pull them to a
        folder first (``POST /immich/pull``) and caption that folder with
        ``POST /caption/folder`` and ``write_xmp: true``.
        """
        if req.write_xmp:
            raise HTTPException(
                status_code=400,
                detail=(
                    "write_xmp is not supported on /immich/caption/stream: Immich assets are "
                    "captioned in memory and have no local path for a sidecar. Pull the album "
                    "into a folder with POST /immich/pull, then caption it via POST /caption/folder "
                    "with write_xmp=true (or use write_back to store captions in Immich itself)."
                ),
            )
        url, api_key = _immich_config()
        source = ImmichSource(url, api_key)
        sink = ImmichSink(url, api_key) if req.write_back else None

        def _close() -> None:
            """Release the pooled Immich connections."""
            source.close()
            if sink is not None:
                sink.close()

        try:
            try:
                assets = await asyncio.to_thread(_immich_album_assets, source, req.album_id, req.asset_ids)
            except _IMMICH_UPSTREAM_ERRORS as exc:
                raise _immich_502(exc) from exc
        except HTTPException:
            _close()
            raise
        total = len(assets)
        logger.info("immich_caption_start", album_id=req.album_id, total=total, write_back=req.write_back)

        def _caption_one(asset: dict[str, Any]) -> dict[str, Any]:
            """Fetch, caption, and optionally write back one asset.

            Returns ``{"asset_id", "name", "final_caption"}`` on success or
            ``{"asset_id", "name", "error"}`` when fetching or captioning
            fails. A write-back failure keeps ``final_caption`` and adds
            ``error``: the caption was computed (don't discard GPU work), and
            Immich may already hold part of the write (the description is set
            before the tag upsert), so the line must not read as a no-op.
            """
            ref = AssetRef(id=asset["id"])
            try:
                pil = source.fetch_image(ref)
                result = engine.caption(
                    pil,
                    trigger_word=req.trigger_word,
                    target_style=req.target_style,
                    target_category=req.target_category,
                    target_backend=req.target_backend,
                    checkpoint=req.checkpoint,
                    hybrid_preset=req.hybrid_preset,
                    prose_bias=req.prose_bias,
                    prose_enrichment=req.prose_enrichment,
                )
            except Exception as exc:  # noqa: BLE001 - report per-asset, keep going
                return {"asset_id": asset["id"], "name": asset["name"], "error": str(exc)}
            outcome = {"asset_id": asset["id"], "name": asset["name"], "final_caption": result.final_caption}
            if sink is not None:
                try:
                    sink.write(ref, keywords=_parse_keywords(result.raw_tags), description=result.final_caption)
                except Exception as exc:  # noqa: BLE001 - report per-asset, keep going
                    outcome["error"] = f"write_back failed (caption computed; description may already be set): {exc}"
            return outcome

        async def _ndjson() -> Any:
            """Yield a progress line per asset, then a final completion summary line."""
            captioned = 0
            failed = 0
            try:
                for i, asset in enumerate(assets):
                    # Fetch + caption + write-back is blocking network/GPU work —
                    # run off the event loop so the stream flushes each line promptly.
                    outcome = await asyncio.to_thread(_caption_one, asset)
                    if "error" in outcome:
                        failed += 1
                    else:
                        captioned += 1
                    yield json.dumps({"type": "progress", "done": i + 1, "total": total, **outcome}) + "\n"
                logger.info("immich_caption_done", captioned=captioned, failed=failed)
                yield json.dumps({"type": "complete", "total": total, "captioned": captioned, "failed": failed}) + "\n"
            finally:
                _close()

        return StreamingResponse(
            _ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
