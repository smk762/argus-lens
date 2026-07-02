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
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError("Server requires: pip install argus-lens[server]") from exc

from PIL import Image

from argus_lens.connectors.filesystem import IMAGE_SUFFIXES
from argus_lens.engine import ArgusLens
from argus_lens.openai_compat import create_openai_router
from argus_lens.types import CaptionResult

# One source of truth for what counts as an image, shared with the connector layer.
SUPPORTED_EXTS = IMAGE_SUFFIXES
_COUNT_CAP = 5000  # per-folder recursive image-count ceiling (keeps browsing snappy)


class CaptionURLRequest(BaseModel):
    """Caption a single image fetched from a URL."""

    image_url: str
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    prose_enrichment: bool = True


class CaptionFolderRequest(BaseModel):
    """Batch-caption every image under a server-side folder path."""

    folder: str
    recursive: bool = False
    write_sidecar: bool = True
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    checkpoint: str | None = None
    prose_enrichment: bool = True


def _result_to_dict(result: CaptionResult) -> dict[str, Any]:
    """Convert a ``CaptionResult`` dataclass into a JSON-serialisable dict."""
    return asdict(result)


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
    written: set[Path],
) -> dict[str, Any]:
    """Caption one image and (optionally) write its ``.txt`` sidecar.

    Returns ``{"rel_path", "final_caption"}`` on success or ``{"rel_path",
    "error"}`` on failure — never both, so batch counts stay consistent.
    ``written`` tracks sidecar paths already written in this batch: same-stem
    images (``cat.jpg`` + ``cat.png``) map to the same ``cat.txt``, and the
    collision is reported as an error instead of silently overwriting the
    first caption.
    """
    try:
        result = engine.caption(
            image_path,
            trigger_word=trigger_word,
            target_style=target_style,
            target_category=target_category,
            target_backend=target_backend,
            checkpoint=checkpoint,
            prose_enrichment=prose_enrichment,
        )
    except Exception as exc:  # noqa: BLE001 - report per-image, keep going
        return {"rel_path": rel_path, "error": str(exc)}
    if write_sidecar:
        sidecar = Path(image_path).with_suffix(".txt")
        if sidecar in written:
            return {
                "rel_path": rel_path,
                "error": f"sidecar collision: {sidecar.name} was already written for another image in this batch",
            }
        try:
            sidecar.write_text(result.final_caption, encoding="utf-8")
        except OSError as exc:
            return {"rel_path": rel_path, "error": f"sidecar write failed: {exc}"}
        written.add(sidecar)
    return {"rel_path": rel_path, "final_caption": result.final_caption}


def _caption_manifest_row(
    engine: ArgusLens,
    row: dict[str, Any],
    *,
    trigger_word: str,
    write_sidecar: bool,
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
        )
        return {"results": {k: _result_to_dict(v) for k, v in results.items()}}

    @app.post("/caption/stream")
    async def caption_stream(
        files: list[UploadFile] = File(...),
        trigger_word: str = Form(""),
        target_style: str = Form("photo"),
        target_category: str = Form("identity"),
        target_backend: str = Form("sdxl"),
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
        prose_enrichment: bool = Form(True),
    ) -> dict[str, Any]:
        """Batch-caption an argus-curator JSONL manifest.

        Each line carries ``abs_path`` and the shared ``target_profile``; images
        are captioned with that profile (no category remapping) and, by default,
        a ``.txt`` sidecar is written next to each image. Assumes the images are
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
                    prose_enrichment=prose_enrichment,
                    written=written,
                )
                (errors if "error" in outcome else results).append(outcome)
            return {
                "total": len(rows),
                "captioned": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors,
            }

        return await asyncio.to_thread(_run)

    @app.post("/caption/manifest/stream")
    async def caption_manifest_stream(
        manifest: UploadFile = File(...),
        trigger_word: str = Form(""),
        write_sidecar: bool = Form(True),
        prose_enrichment: bool = Form(True),
    ) -> StreamingResponse:
        """Streaming variant of /caption/manifest for live progress.

        Yields one NDJSON object per image as it is captioned
        (``{type:"progress", done, total, rel_path, final_caption|error}``),
        then a final ``{type:"complete", total, captioned, failed}`` line. As
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
            for i, row in enumerate(rows):
                # Caption is blocking CPU/GPU work — run off the event loop so the
                # stream flushes each line promptly.
                outcome = await asyncio.to_thread(
                    _caption_manifest_row,
                    engine,
                    row,
                    trigger_word=trigger_word,
                    write_sidecar=write_sidecar,
                    prose_enrichment=prose_enrichment,
                    written=written,
                )
                if "error" in outcome:
                    failed += 1
                else:
                    captioned += 1
                yield json.dumps({"type": "progress", "done": i + 1, "total": total, **outcome}) + "\n"
            yield json.dumps({"type": "complete", "total": total, "captioned": captioned, "failed": failed}) + "\n"

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
        image. Returns the same shape as ``/caption/manifest``.
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
                    prose_enrichment=req.prose_enrichment,
                    write_sidecar=req.write_sidecar,
                    written=written,
                )
                (errors if "error" in outcome else results).append(outcome)
            return {
                "total": len(images),
                "captioned": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors,
            }

        return await asyncio.to_thread(_run)

    return app
