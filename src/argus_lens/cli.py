"""Argus Lens CLI — command-line interface for image captioning."""

from __future__ import annotations

import json
import os
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


def _make_verifier(
    reconcile: str | None,
    verifier_url: str | None,
    verifier_model: str | None,
    verifier_key: str | None,
    verifier_device: str,
) -> object | None:
    """Build an attribute verifier from CLI flags, or ``None`` when disabled."""
    if not reconcile:
        return None
    from argus_lens.reconcile import build_verifier

    return build_verifier(
        reconcile,
        base_url=verifier_url,
        model_id=verifier_model,
        api_key=verifier_key,
        device=verifier_device,
    )


@app.command()
def caption(
    path: Path = Argument(..., help="Image file or directory to caption"),
    backend: str = Option("hybrid", "--backend", "-b", help="Captioning backend"),
    trigger: str = Option("", "--trigger", "-t", help="Trigger word to prepend"),
    style: str = Option("photo", "--style", "-s", help="Target style: photo or anime"),
    category: str = Option("identity", "--category", "-c", help="Target category for final_caption"),
    target_backend: str = Option("sdxl", "--target-backend", help="Diffusion backend (sdxl, flux, sd3)"),
    hybrid_preset: str | None = Option(
        None, "--hybrid-preset", help="Tag/prose balance: tags, keywords, balanced, descriptive, prose"
    ),
    prose_bias: float | None = Option(
        None, "--prose-bias", help="Continuous tag/prose balance: 0.0 (pure tags) .. 1.0 (full prose)"
    ),
    output: Path | None = Option(None, "--output", "-o", help="Output file (for json/jsonl/csv)"),
    fmt: str = Option("txt", "--format", "-f", help="Output format: txt, json, jsonl, csv"),
    overwrite: bool = Option(False, "--overwrite", help="Overwrite existing caption files"),
    recursive: bool = Option(False, "--recursive", "-r", help="Walk sub-directories (results keyed by relative path)"),
    api_key: str | None = Option(None, "--api-key", help="API key for cloud backends"),
    model_id: str | None = Option(None, "--model-id", help="Model ID override"),
    base_url: str | None = Option(
        None, "--base-url", help="Endpoint URL for openai-compat backend (e.g. http://localhost:11434/v1)"
    ),
    prompt: str | None = Option(
        None,
        "--prompt",
        help="openai-compat user prompt: preset (detailed, straightforward, descriptive, booru) or literal text",
    ),
    wd14_repo: str | None = Option(
        None, "--wd14-repo", help="WD14 v3 tagger repo, e.g. SmilingWolf/wd-eva02-large-tagger-v3"
    ),
    reconcile: str | None = Option(
        None,
        "--reconcile",
        help="Fix prose colour/pose vs tags via a verifier: tag-prior, openai-compat, florence, molmo",
    ),
    verifier_url: str | None = Option(None, "--verifier-url", help="Base URL for the openai-compat verifier"),
    verifier_model: str | None = Option(None, "--verifier-model", help="Model id for the verifier"),
    verifier_key: str | None = Option(None, "--verifier-key", help="API key for the verifier"),
    verifier_device: str = Option("cpu", "--verifier-device", help="Device for florence/molmo verifiers (cpu|cuda)"),
    refine: bool = Option(
        True, "--refine/--no-refine", help="Rewrite captions with an LLM, validated (on unless ARGUS_REFINE=off)"
    ),
    refine_model: str | None = Option(
        None, "--refine-model", help="OpenAI-compatible chat model for refinement [ARGUS_REFINE_MODEL, qwen3:8b]"
    ),
    refine_url: str | None = Option(
        None, "--refine-url", help="Base URL for the refine model [ARGUS_REFINE_BASE_URL, local Ollama]"
    ),
    refine_key: str | None = Option(None, "--refine-key", help="API key for the refine model"),
    refine_variants: str | None = Option(
        None, "--refine-variants", help="Comma-separated variants to rewrite, or 'all' [ARGUS_REFINE_VARIANTS, all]"
    ),
    refine_image: bool = Option(False, "--refine-image", help="Send the image to the refine model (vision models)"),
    verbose: bool = Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Caption images in a file or directory."""
    from argus_lens import ArgusLens
    from argus_lens.backends.replay import ReplayMiss

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if prompt:
        kwargs["prompt"] = prompt
    if wd14_repo:
        kwargs["model_repo"] = wd14_repo
    if model_id:
        if backend in ("florence2", "blip2"):
            kwargs["florence_model_id" if backend == "florence2" else "model_id"] = model_id
        else:
            kwargs["model_id"] = model_id

    verifier = _make_verifier(reconcile, verifier_url, verifier_model, verifier_key, verifier_device)
    refiner = None
    from argus_lens.refine import refine_opted_out

    if refine and not refine_opted_out():
        from argus_lens.refine import OpenAICompatRewriter, Refiner, parse_variants

        refiner = Refiner(
            OpenAICompatRewriter(base_url=refine_url, model_id=refine_model, api_key=refine_key),
            variants=parse_variants(refine_variants or os.environ.get("ARGUS_REFINE_VARIANTS") or "all"),
            send_image=refine_image or os.environ.get("ARGUS_REFINE_IMAGE", "").strip().lower() in ("1", "true", "yes"),
        )
    engine = ArgusLens(backend=backend, verifier=verifier, refiner=refiner, **kwargs)

    if path.is_file():
        try:
            result = engine.caption(
                path,
                trigger_word=trigger,
                target_style=style,
                target_category=category,
                target_backend=target_backend,
                hybrid_preset=hybrid_preset,
                prose_bias=prose_bias,
            )
        except ReplayMiss as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
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
        misses: list[str] = []

        def _progress(current: int, total: int, name: str, _result: object) -> None:
            """Track completion count and echo per-image progress when verbose."""
            nonlocal count
            count = current
            if verbose:
                typer.echo(f"  [{current}/{total}] {name}")

        def _on_miss(name: str, exc: ReplayMiss) -> None:
            """Report a replay miss and keep captioning the rest of the directory (#48)."""
            misses.append(name)
            typer.echo(f"  [miss] {name}: {exc}", err=True)

        results = engine.caption_directory(
            path,
            trigger_word=trigger,
            target_style=style,
            target_category=category,
            target_backend=target_backend,
            hybrid_preset=hybrid_preset,
            prose_bias=prose_bias,
            output_format=fmt,
            overwrite=overwrite,
            recursive=recursive,
            progress=_progress if verbose else None,
            on_miss=_on_miss,
        )
        typer.echo(f"Captioned {len(results)} images -> {fmt}")
        if misses:
            typer.echo(f"{len(misses)} image(s) had no recorded caption (replay miss)", err=True)
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


ram_vocab_app = typer.Typer(help="Build and inspect RAM++ vocabulary packs (custom tags, #62).", no_args_is_help=True)
app.add_typer(ram_vocab_app, name="ram-vocab")


@ram_vocab_app.command("build")
def ram_vocab_build(
    source: Path = Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Tag list (one per line, optional TAB threshold) or descriptions JSON ([{tag: [desc, ...]}]).",
    ),
    out: Path = Option(..., "--out", "-o", help="Directory to write the pack into."),
    name: str = Option(..., "--name", "-n", help="Pack name; pack tags report source 'ram:<name>'."),
    threshold: float = Option(0.5, help="Threshold for tags without their own (upstream's open-set default)."),
    license: str | None = Option(None, help="Licence of the tag list / descriptions, recorded in the manifest."),
    base_url: str | None = Option(
        None, help="OpenAI-compatible endpoint for writing descriptions [ARGUS_REFINE_BASE_URL]."
    ),
    model: str | None = Option(None, help="LLM that writes the descriptions [ARGUS_REFINE_MODEL]."),
    api_key: str | None = Option(None, help="API key for the endpoint [ARGUS_REFINE_API_KEY]."),
    device: str = Option("auto", help="Device for the CLIP text encoder."),
) -> None:
    """Build a pack: write 51 descriptions per tag with an LLM (or use supplied ones), embed them with CLIP."""
    from argus_lens.backends._ram.build import build_pack, read_tag_input
    from argus_lens.backends._ram.embed import ClipTextEmbedder
    from argus_lens.backends._ram.packs import CLIP_MODEL

    parsed = read_tag_input(source)
    descriptions = dict(parsed.descriptions)
    todo = [t for t in parsed.tags if not descriptions.get(t)]
    source_info: dict[str, object] = {"input": source.name}
    if todo:
        from argus_lens.backends._ram.describe import DescriptionWriter

        writer = DescriptionWriter(base_url=base_url, model_id=model, api_key=api_key)
        typer.echo(f"Writing descriptions for {len(todo)} tag(s) with {writer.model_id} at {writer.base_url} ...")
        try:
            descriptions.update(
                writer.describe_all(todo, on_progress=lambda i, n, tag: typer.echo(f"  [{i}/{n}] {tag}"))
            )
        finally:
            writer.close()
        source_info.update(llm=writer.model_id, endpoint=writer.base_url, prompts="upstream")
    else:
        source_info["descriptions"] = "supplied"

    typer.echo(f"Embedding {len(parsed.tags) * 51} descriptions with {CLIP_MODEL} ...")
    manifest = build_pack(
        out,
        name=name,
        tags=parsed.tags,
        descriptions=descriptions,
        embed=ClipTextEmbedder(device=device).embed,
        thresholds=parsed.thresholds,
        default_threshold=threshold,
        description_source=source_info,
        license=license,
    )
    typer.echo(f"Wrote pack {manifest.name!r} with {len(manifest.tags)} tag(s) to {out}")


@ram_vocab_app.command("inspect")
def ram_vocab_inspect(pack: Path = Argument(..., exists=True, file_okay=False, help="Pack directory.")) -> None:
    """Show a pack's manifest and check it against the built-in vocabulary."""
    from argus_lens.backends._ram.packs import PackManifest
    from argus_lens.backends._ram.vocab import load_tag_vocabulary

    manifest = PackManifest.read(pack)
    builtin = set(load_tag_vocabulary()[0])
    overrides = [t for t in manifest.tags if t in builtin]
    typer.echo(f"{manifest.name}: {len(manifest.tags)} tag(s), created {manifest.created or 'unknown'}")
    typer.echo(f"  licence: {manifest.license or 'unspecified'}")
    typer.echo(f"  description source: {json.dumps(manifest.description_source)}")
    if overrides:
        typer.echo(f"  replaces {len(overrides)} built-in tag(s): {', '.join(overrides[:10])}")
    for tag, thr in list(zip(manifest.tags, manifest.thresholds, strict=True))[:20]:
        typer.echo(f"  {thr:.2f}  {tag}")
    if len(manifest.tags) > 20:
        typer.echo(f"  ... and {len(manifest.tags) - 20} more")


@app.command("eval")
def eval_command(
    path: Path = Argument(..., help="Image directory (reference-free) or .jsonl golden manifest"),
    backend: str = Option("hybrid", "--backend", "-b", help="Captioning backend"),
    style: str = Option("photo", "--style", "-s", help="Target style for directory datasets"),
    category: str = Option("identity", "--category", "-c", help="Target category for directory datasets"),
    target_backend: str = Option("sdxl", "--target-backend", help="Diffusion backend (token budget)"),
    trigger: str = Option("", "--trigger", "-t", help="Trigger word to prepend"),
    hybrid_preset: str | None = Option(None, "--hybrid-preset", help="Tag/prose balance preset"),
    prose_bias: float | None = Option(None, "--prose-bias", help="Tag/prose balance 0.0..1.0"),
    clip: bool = Option(False, "--clip", help="Also compute CLIPScore (needs the 'eval' extra)"),
    clip_device: str = Option("cpu", "--clip-device", help="Device for the CLIP model"),
    output: Path | None = Option(None, "--output", "-o", help="Write the full scorecard JSON here"),
    baseline: Path | None = Option(None, "--baseline", help="Baseline scorecard JSON to compare against"),
    fail_on_regression: bool = Option(
        False, "--fail-on-regression", help="Exit non-zero if any gate metric regressed vs --baseline"
    ),
    api_key: str | None = Option(None, "--api-key", help="API key for cloud backends"),
    model_id: str | None = Option(None, "--model-id", help="Model ID override"),
    base_url: str | None = Option(None, "--base-url", help="Endpoint URL for openai-compat backend"),
    reconcile: str | None = Option(
        None,
        "--reconcile",
        help="Apply an attribute verifier before scoring: tag-prior, openai-compat, florence, molmo",
    ),
    verifier_url: str | None = Option(None, "--verifier-url", help="Base URL for the openai-compat verifier"),
    verifier_model: str | None = Option(None, "--verifier-model", help="Model id for the verifier"),
    verifier_key: str | None = Option(None, "--verifier-key", help="API key for the verifier"),
    verifier_device: str = Option("cpu", "--verifier-device", help="Device for florence/molmo verifiers (cpu|cuda)"),
    verbose: bool = Option(False, "--verbose", "-v", help="Per-image progress"),
) -> None:
    """Score caption quality over a dataset and print a scorecard.

    Works reference-free on any image directory (tag/prose contradiction,
    token-budget adherence, redundancy). A ``.jsonl`` manifest with
    ``expected_tags`` / ``target_caption`` additionally unlocks tag-coverage
    recall and reference CLIPScore.
    """
    from dataclasses import replace

    from argus_lens import ArgusLens
    from argus_lens.eval import compare_to_baseline, format_scorecard, load_dataset, run_eval
    from argus_lens.eval.metrics import try_build_clip_scorer
    from argus_lens.eval.report import format_comparison

    try:
        dataset = load_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Could not load dataset: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not dataset:
        typer.echo(f"No images found in {path}", err=True)
        raise typer.Exit(1)
    if path.is_dir():
        dataset = [
            replace(it, target_style=style, target_category=category, target_backend=target_backend) for it in dataset
        ]

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
    verifier = _make_verifier(reconcile, verifier_url, verifier_model, verifier_key, verifier_device)
    engine = ArgusLens(backend=backend, verifier=verifier, **kwargs)

    clip_scorer = None
    if clip:
        clip_scorer = try_build_clip_scorer(device=clip_device)
        if clip_scorer is None:
            typer.echo("CLIPScore skipped: install argus-lens[eval] (torch + transformers)", err=True)

    def _progress(current: int, total: int, item: object) -> None:
        """Echo per-image progress when verbose."""
        typer.echo(f"  [{current}/{total}] {getattr(item, 'image', '')}")

    scorecard = run_eval(
        engine,
        dataset,
        clip_scorer=clip_scorer,
        trigger_word=trigger,
        hybrid_preset=hybrid_preset,
        prose_bias=prose_bias,
        progress=_progress if verbose else None,
    )

    typer.echo(format_scorecard(scorecard))

    if output:
        output.write_text(json.dumps(scorecard.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"\nScorecard written to {output}")

    # A run where every image errored scored nothing: always a hard failure, so a
    # broken run never exits 0 (regardless of --baseline / --fail-on-regression).
    all_failed = scorecard.n_errors == scorecard.n and scorecard.n > 0
    if all_failed:
        typer.echo(f"\nAll {scorecard.n} images errored — no metrics computed.", err=True)

    if baseline:
        try:
            base = json.loads(baseline.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            typer.echo(f"Baseline is not valid JSON: {exc}", err=True)
            raise typer.Exit(1) from exc
        if not isinstance(base, dict):
            typer.echo(f"Baseline must be a JSON object, got {type(base).__name__}", err=True)
            raise typer.Exit(1)
        comparison = compare_to_baseline(scorecard.aggregates, base.get("aggregates", base))
        typer.echo("\n" + format_comparison(comparison))
        if fail_on_regression and comparison["regressed"]:
            raise typer.Exit(1)
    elif fail_on_regression:
        typer.echo("--fail-on-regression needs --baseline to compare against; skipping regression gate.", err=True)

    if all_failed:
        raise typer.Exit(1)


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
