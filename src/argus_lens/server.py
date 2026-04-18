"""Optional FastAPI micro-server for Argus Lens."""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import asdict
from typing import Any

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError("Server requires: pip install argus-lens[server]") from exc

from PIL import Image

from argus_lens.engine import ArgusLens
from argus_lens.types import CaptionResult, get_category_names


class CaptionURLRequest(BaseModel):
    image_url: str
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    prose_enrichment: bool = True


def _stabilize_caption_variants(cv: dict[str, Any]) -> dict[str, Any]:
    """Ensure every category key exists so clients (e.g. web UIs) can render split pose rows."""
    out = dict(cv)
    for name in get_category_names():
        out.setdefault(name, "")
    for key in ("training", "zeroshot", "pose_composition"):
        out.setdefault(key, "")
    return out


def _result_to_dict(result: CaptionResult) -> dict[str, Any]:
    d = asdict(result)
    d["caption_variants"] = _stabilize_caption_variants(d["caption_variants"])
    return d


def _package_version() -> dict[str, str | None]:
    """Version from hatch-vcs generated _version.py, or importlib.metadata fallback."""
    try:
        from argus_lens import _version as v

        cid = getattr(v, "commit_id", None) or getattr(v, "__commit_id__", None)
        return {"version": v.__version__, "commit": cid}
    except Exception:
        try:
            from importlib.metadata import version as pkg_version

            return {"version": pkg_version("argus-lens"), "commit": None}
        except Exception:
            return {"version": "unknown", "commit": None}


def create_app(
    default_backend: str = "hybrid",
    cors: bool = False,
    cors_origins: list[str] | None = None,
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

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/version")
    async def version_info() -> dict[str, str | None]:
        """Installed argus-lens package version (for UI / health checks)."""
        return _package_version()

    engine = ArgusLens(backend=default_backend, **kwargs)

    @app.get("/backends")
    async def list_backends() -> dict[str, Any]:
        return {"backends": engine.available_backends()}

    @app.post("/caption")
    async def caption_image(
        file: UploadFile = File(...),
        trigger_word: str = Form(""),
        target_style: str = Form("photo"),
        target_category: str = Form("identity"),
        target_backend: str = Form("sdxl"),
    ) -> dict[str, Any]:
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
        images: list[Image.Image] = []
        for f in files:
            data = await f.read()
            try:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                images.append(pil)
            except Exception:
                continue

        async def _ndjson():
            for name, result in engine.caption_stream(
                images,
                trigger_word=trigger_word,
                target_style=target_style,
                target_category=target_category,
                target_backend=target_backend,
            ):
                line = json.dumps({"name": name, **_result_to_dict(result)}) + "\n"
                yield line

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    return app
