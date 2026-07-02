"""Argus Lens CLI — command-line interface for image captioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import typer
    from typer import Argument, Option
except ImportError as _exc:
    print("CLI requires: pip install argus-lens[cli]", file=sys.stderr)
    raise SystemExit(1) from _exc

app = typer.Typer(
    name="argus-lens",
    help="Structured image captioning for training and generation.",
    no_args_is_help=True,
)


@app.command()
def caption(
    path: Path = Argument(..., help="Image file or directory to caption"),
    backend: str = Option("hybrid", "--backend", "-b", help="Captioning backend"),
    trigger: str = Option("", "--trigger", "-t", help="Trigger word to prepend"),
    style: str = Option("photo", "--style", "-s", help="Target style: photo or anime"),
    category: str = Option("identity", "--category", "-c", help="Target category for final_caption"),
    target_backend: str = Option("sdxl", "--target-backend", help="Diffusion backend (sdxl, flux, sd3)"),
    output: Path | None = Option(None, "--output", "-o", help="Output file (for json/jsonl/csv)"),
    fmt: str = Option("txt", "--format", "-f", help="Output format: txt, json, jsonl, csv"),
    overwrite: bool = Option(False, "--overwrite", help="Overwrite existing caption files"),
    api_key: str | None = Option(None, "--api-key", help="API key for cloud backends"),
    model_id: str | None = Option(None, "--model-id", help="Model ID override"),
    base_url: str | None = Option(
        None, "--base-url", help="Endpoint URL for openai-compat backend (e.g. http://localhost:11434/v1)"
    ),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Caption images in a file or directory."""
    from argus_lens import ArgusLens

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if model_id:
        if backend in ("florence2", "blip2"):
            kwargs["florence_model_id" if backend == "florence2" else "model_id"] = model_id
        else:
            kwargs["model_id"] = model_id

    engine = ArgusLens(backend=backend, **kwargs)

    if path.is_file():
        result = engine.caption(
            path,
            trigger_word=trigger,
            target_style=style,
            target_category=category,
            target_backend=target_backend,
        )
        if output:
            from argus_lens.exporters import export_results

            export_results({path.name: result}, output.parent, fmt)
            typer.echo(f"Exported to {output}")
        else:
            typer.echo(result.final_caption)
            if verbose:
                typer.echo(f"\nVariants: {json.dumps(result.caption_variants, indent=2)}")
    elif path.is_dir():
        count = 0

        def _progress(current: int, total: int, name: str, _result: object) -> None:
            """Track completion count and echo per-image progress when verbose."""
            nonlocal count
            count = current
            if verbose:
                typer.echo(f"  [{current}/{total}] {name}")

        results = engine.caption_directory(
            path,
            trigger_word=trigger,
            target_style=style,
            target_category=category,
            target_backend=target_backend,
            output_format=fmt,
            overwrite=overwrite,
            progress=_progress if verbose else None,
        )
        typer.echo(f"Captioned {len(results)} images -> {fmt}")
    else:
        typer.echo(f"Error: {path} is not a file or directory", err=True)
        raise typer.Exit(1)


@app.command()
def backends() -> None:
    """List available captioning backends and their status."""
    from argus_lens import ArgusLens

    engine = ArgusLens.__new__(ArgusLens)
    info = engine.available_backends()

    for name, meta in sorted(info.items()):
        status = "available" if meta.get("available") else "unavailable"
        kind = meta.get("kind", "?")
        reason = meta.get("reason") or ""
        marker = "+" if meta.get("available") else "-"
        line = f"  [{marker}] {name:<16} ({kind:<10}) {status}"
        if reason:
            line += f"  -- {reason}"
        typer.echo(line)


@app.command()
def serve(
    port: int = Option(8080, "--port", "-p", help="Port to listen on"),
    host: str = Option("0.0.0.0", "--host", help="Host to bind to"),
    backend: str = Option(
        "hybrid", "--backend", "-b", envvar="ARGUS_BACKEND", help="Default backend for /caption endpoints"
    ),
    cors: bool = Option(False, "--cors", help="Enable CORS (allow all origins)"),
    source_root: str | None = Option(None, "--source-root", help="Root folder for /folders browsing (UI picker)"),
) -> None:
    """Start the Argus Lens micro-server (FastAPI).

    Exposes both the native /caption endpoints and an OpenAI-compatible
    /v1/chat/completions endpoint suitable for use as a Frigate GenAI provider:

        genai:
          enabled: true
          provider: openai
          base_url: http://<this-host>:<port>/v1
          model: argus-hybrid
    """
    try:
        import uvicorn
    except ImportError as _exc:
        typer.echo("Server requires: pip install argus-lens[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_lens.server import create_app

    app = create_app(default_backend=backend, cors=cors, source_root=source_root)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
