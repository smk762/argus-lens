"""Optional FastAPI micro-server for Argus Lens."""

from __future__ import annotations

import asyncio
import io
import json
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

from argus_lens.engine import ArgusLens
from argus_lens.openai_compat import create_openai_router
from argus_lens.types import CaptionResult


class CaptionURLRequest(BaseModel):
    image_url: str
    trigger_word: str = ""
    target_style: str = "photo"
    target_category: str = "identity"
    target_backend: str = "sdxl"
    prose_enrichment: bool = True


def _result_to_dict(result: CaptionResult) -> dict[str, Any]:
    return asdict(result)


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

    engine = ArgusLens(backend=default_backend, **kwargs)

    # OpenAI-compatible /v1 endpoints (Frigate GenAI provider).
    # Always mounted — Frigate's genai block uses POST /v1/chat/completions.
    # Engine kwargs are forwarded so model_dir / florence_model_id are honoured.
    app.include_router(create_openai_router(**kwargs), prefix="/v1")

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

    @app.post("/caption/manifest")
    async def caption_manifest(
        manifest: UploadFile = File(...),
        trigger_word: str = Form(""),
        write_sidecar: bool = Form(True),
    ) -> dict[str, Any]:
        """Batch-caption an argus-curator JSONL manifest.

        Each line carries ``abs_path`` and the shared ``target_profile``; images
        are captioned with that profile (no category remapping) and, by default,
        a ``.txt`` sidecar is written next to each image. Assumes the images are
        reachable at ``abs_path`` (e.g. a shared volume with the curator).
        """
        raw = await manifest.read()
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
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"invalid JSON on line {i + 1}: {exc}") from exc

        def _run() -> dict[str, Any]:
            results: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for row in rows:
                abs_path = row.get("abs_path")
                rel_path = row.get("rel_path") or abs_path or "<unknown>"
                if not abs_path:
                    errors.append({"rel_path": rel_path, "error": "row missing abs_path"})
                    continue
                profile = row.get("target_profile") or {}
                try:
                    result = engine.caption(
                        abs_path,
                        trigger_word=trigger_word,
                        target_style=profile.get("target_style", "photo"),
                        target_category=profile.get("target_category", "identity"),
                        target_backend=profile.get("target_backend", "sdxl"),
                        checkpoint=profile.get("checkpoint"),
                    )
                except Exception as exc:  # noqa: BLE001 - report per-row, keep going
                    errors.append({"rel_path": rel_path, "error": str(exc)})
                    continue
                if write_sidecar:
                    try:
                        sidecar = Path(abs_path).with_suffix(".txt")
                        sidecar.write_text(result.final_caption, encoding="utf-8")
                    except OSError as exc:
                        errors.append({"rel_path": rel_path, "error": f"sidecar write failed: {exc}"})
                results.append({"rel_path": rel_path, "final_caption": result.final_caption})
            return {
                "total": len(rows),
                "captioned": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors,
            }

        return await asyncio.to_thread(_run)

    return app
