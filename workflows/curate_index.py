#!/usr/bin/env python3
"""Curate-index workflow: caption a people-photo library once, then group, query and export it.

Stages (each resumable; see workflows/README.md):

    inventory  walk SOURCE_ROOT/<name>/pics/<shoot_id>/*.jpg -> manifest.jsonl
    caption    one model pass per unique image            -> raw.jsonl (+ run.json)
    assemble   raw -> captions per target (sdxl, flux)    -> captions.<target>.jsonl, review_queue.csv
    index      captions -> long tag table + identity QA   -> index.csv, groups.csv, identity_*.csv
    query      AND/OR tag search over the index
    export     training dataset (image symlinks + .txt)   -> <dest>/<name>/...

Inference happens only in ``caption``. Everything downstream recomputes from
``raw.jsonl``, so changing the assembly, refinement, taxonomy or export
settings never re-runs the GPU stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_LAYOUT = "{name}/pics/{shoot_id}/{file}"
# Settings that change what ``caption`` produces. A resumed run must match them.
CAPTION_KEYS = ("backend", "wd14_repo", "wd14_threshold", "model_id", "base_url", "prompt", "count_model")
INDEX_CATEGORIES = ("identity", "wardrobe", "pose_composition", "setting", "lighting", "action")


# --------------------------------------------------------------------------- io helpers


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows of a JSONL file (nothing if it does not exist), skipping a torn last line."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping unreadable line in {path.name}", file=sys.stderr)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically write *rows* to *path*; return the row count."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> int:
    """Atomically write a CSV; return the row count."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    tmp.replace(path)
    return count


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes (the stable image id)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_tag(tag: str) -> str:
    """Index form of a tag: lower-case, spaces not underscores (``red_dress`` -> ``red dress``)."""
    return re.sub(r"\s+", " ", tag.replace("_", " ")).strip().lower()


def layout_regex(layout: str) -> re.Pattern[str]:
    """Compile a ``{name}/pics/{shoot_id}/{file}`` layout into a relative-path matcher."""
    pattern = re.escape(layout)
    for field in ("name", "shoot_id", "file"):
        pattern = pattern.replace(re.escape("{" + field + "}"), f"(?P<{field}>[^/]+)")
    return re.compile(f"^{pattern}$")


# --------------------------------------------------------------------------- inventory


def cmd_inventory(args: argparse.Namespace) -> None:
    """Walk the source tree and write manifest.jsonl (one row per file, duplicates marked)."""
    root = Path(args.source_root).resolve()
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    matcher = layout_regex(args.layout)
    rows: list[dict[str, Any]] = []
    first_seen: dict[str, str] = {}
    skipped = 0
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for i, path in enumerate(paths, 1):
        rel = path.relative_to(root).as_posix()
        match = matcher.match(rel)
        if not match:
            skipped += 1
            continue
        digest = sha256_file(path)
        row = {
            "image_id": digest[:16],
            "sha256": digest,
            "rel_path": rel,
            "name": match["name"],
            "shoot_id": match["shoot_id"],
            "bytes": path.stat().st_size,
        }
        if digest in first_seen:
            row["duplicate_of"] = first_seen[digest]
        else:
            first_seen[digest] = rel
        rows.append(row)
        if i % 500 == 0:
            print(f"  hashed {i}/{len(paths)}")
    write_jsonl(run / "manifest.jsonl", rows)
    meta = {"source_root": str(root), "layout": args.layout, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run / "inventory.json").write_text(json.dumps(meta, indent=2))
    names = {r["name"] for r in rows}
    dupes = sum(1 for r in rows if "duplicate_of" in r)
    print(f"{len(rows)} images ({len(first_seen)} unique, {dupes} duplicates) across {len(names)} names")
    if skipped:
        print(f"{skipped} image files did not match layout {args.layout!r} and were skipped")


# --------------------------------------------------------------------------- caption


def _load_run_config(run: Path, wanted: dict[str, Any], force: bool) -> dict[str, Any]:
    """Create or check run.json so a resumed run cannot silently mix settings."""
    path = run / "run.json"
    import argus_lens

    if path.exists():
        existing = json.loads(path.read_text())
        diffs = {
            k: (existing["caption"].get(k), wanted.get(k))
            for k in CAPTION_KEYS
            if existing["caption"].get(k) != wanted.get(k)
        }
        if diffs and not force:
            raise SystemExit(
                f"run.json was created with different caption settings: {diffs}. "
                "Use a new run directory, or --force to continue anyway."
            )
        return existing
    config = {
        "argus_lens_version": argus_lens.__version__,
        "python": sys.version.split()[0],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caption": wanted,
    }
    path.write_text(json.dumps(config, indent=2))
    return config


def _parse_count(text: str) -> int | None:
    """First integer in a headcount answer ("2", "Two people." -> 2)."""
    words = {"zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in words:
            return words[word]
    return None


def cmd_caption(args: argparse.Namespace) -> None:
    """Run inference once per unique image, appending raw tags/prose to raw.jsonl."""
    from argus_lens import ArgusLens

    run = Path(args.run_dir)
    inventory = json.loads((run / "inventory.json").read_text())
    root = Path(inventory["source_root"])
    wanted = {
        "backend": args.backend,
        "wd14_repo": args.wd14_repo,
        "wd14_threshold": args.wd14_threshold,
        "model_id": args.model_id,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "count_model": args.count_model,
    }
    _load_run_config(run, wanted, args.force)

    kwargs: dict[str, Any] = {"threshold": args.wd14_threshold}
    for key, value in (
        ("model_repo", args.wd14_repo),
        ("model_id", args.model_id),
        ("base_url", args.base_url),
        ("prompt", args.prompt),
    ):
        if value:
            kwargs[key] = value
    engine = ArgusLens(backend=args.backend, **kwargs)

    counter = None
    if args.count_model:
        from argus_lens.backends.openai_compat import OpenAICompatBackend

        counter = OpenAICompatBackend(
            model_id=args.count_model,
            base_url=args.count_url or args.base_url,
            prompt="count",
            temperature=0.0,
            max_tokens=8,
        )

    raw_path = run / "raw.jsonl"
    done = {r["sha256"] for r in read_jsonl(raw_path)}
    todo = [r for r in read_jsonl(run / "manifest.jsonl") if "duplicate_of" not in r and r["sha256"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already captioned, {len(todo)} to go")

    started = time.monotonic()
    with raw_path.open("a", encoding="utf-8") as out, (run / "errors.jsonl").open("a", encoding="utf-8") as err:
        for i, row in enumerate(todo, 1):
            path = root / row["rel_path"]
            try:
                result = engine.caption(path)
                record = {
                    "sha256": row["sha256"],
                    "rel_path": row["rel_path"],
                    "backend": result.backend_name,
                    "tags": result.raw_tags,
                    "prose": result.raw_prose,
                }
                if counter is not None:
                    from PIL import Image

                    with Image.open(path) as img:
                        answer = counter.caption_image(img)
                    record["vlm_count"] = _parse_count(answer)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            except Exception as exc:  # noqa: BLE001 - record and keep going; rerun retries failures
                err.write(json.dumps({"rel_path": row["rel_path"], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                err.flush()
                print(f"  error {row['rel_path']}: {exc}", file=sys.stderr)
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.monotonic() - started, 1e-6)
                print(f"  [{i}/{len(todo)}] {rate:.2f} img/s, eta {(len(todo) - i) / max(rate, 1e-6) / 60:.0f} min")


# --------------------------------------------------------------------------- assemble


def _build_refiner(args: argparse.Namespace):
    """Return a Refiner when --refine-model is set, else None."""
    if not args.refine_model:
        return None
    from argus_lens.refine import OpenAICompatRewriter, Refiner

    return Refiner(
        OpenAICompatRewriter(base_url=args.refine_url, model_id=args.refine_model, api_key=args.refine_key),
        variants=tuple(v.strip() for v in args.refine_variants.split(",") if v.strip()),
    )


def _needs_review(subjects: dict[str, Any], vlm_count: int | None) -> list[str]:
    """Reasons an image should go to face identification / manual review."""
    reasons = []
    if subjects.get("multiple"):
        reasons.append("multiple_people")
    if subjects.get("agree") is False:
        reasons.append("tag_prose_count_disagree")
    if vlm_count is not None:
        if vlm_count >= 2 and not subjects.get("multiple"):
            reasons.append("vlm_sees_multiple")
        if subjects.get("count") is not None and vlm_count != subjects["count"]:
            reasons.append("vlm_count_disagrees")
    if subjects.get("count") == 0:
        reasons.append("no_people_detected")
    return reasons


def cmd_assemble(args: argparse.Namespace) -> None:
    """Compose captions for each target from raw.jsonl (no inference unless refining)."""
    from argus_lens.assembly.composer import compose_caption_result
    from argus_lens.types import resolve_target_profile

    run = Path(args.run_dir)
    manifest = {r["sha256"]: r for r in read_jsonl(run / "manifest.jsonl") if "duplicate_of" not in r}
    raw = sorted(read_jsonl(run / "raw.jsonl"), key=lambda r: r["rel_path"])
    refiner = _build_refiner(args)
    review: dict[str, tuple[dict[str, Any], dict[str, Any], list[str]]] = {}
    for target in (t.strip() for t in args.targets.split(",") if t.strip()):
        profile = resolve_target_profile(target_backend=target, hybrid_preset=args.hybrid_preset)
        rows = []
        for idx, rec in enumerate(raw):
            result = compose_caption_result(
                trigger_word=args.trigger,
                tags=rec["tags"],
                prose=rec["prose"],
                target_profile=profile,
                image_index=idx,  # deterministic omission cycles across the dataset
                backend_name=rec.get("backend", ""),
            )
            if refiner is not None:
                result = refiner.refine(result, profile, trigger_word=args.trigger)
            item = manifest.get(rec["sha256"], {})
            rows.append(
                {
                    "image_id": rec["sha256"][:16],
                    "rel_path": rec["rel_path"],
                    "name": item.get("name"),
                    "shoot_id": item.get("shoot_id"),
                    "target": target,
                    "variants": result.caption_variants,
                    "buckets": result.metadata["buckets"],
                    "subjects": result.metadata["subjects"],
                    "vlm_count": rec.get("vlm_count"),
                    "refine": result.metadata.get("refine"),
                    "notes": result.compaction_notes,
                }
            )
            reasons = _needs_review(result.metadata["subjects"], rec.get("vlm_count"))
            if reasons:
                review[rec["sha256"]] = (rows[-1], result.metadata["subjects"], reasons)
        written = write_jsonl(run / f"captions.{target}.jsonl", rows)
        refined = sum(1 for r in rows if r["refine"] and r["refine"]["accepted"])
        print(f"{target}: {written} captions" + (f", {refined} refined" if refiner else ""))
    write_csv(
        run / "review_queue.csv",
        [
            "image_id",
            "rel_path",
            "name",
            "shoot_id",
            "count",
            "tag_count",
            "prose_count",
            "vlm_count",
            "reasons",
            "evidence",
        ],
        (
            [
                r["image_id"],
                r["rel_path"],
                r["name"],
                r["shoot_id"],
                s.get("count"),
                s.get("tag_count"),
                s.get("prose_count"),
                r["vlm_count"],
                ";".join(reasons),
                ";".join(s.get("evidence", [])),
            ]
            for r, s, reasons in sorted(review.values(), key=lambda v: v[0]["rel_path"])
        ),
    )
    print(f"review_queue.csv: {len(review)} images flagged (multi-person / count disagreements)")


# --------------------------------------------------------------------------- index


def _load_taxonomy(path: str | None):
    """Load a taxonomy JSON ({"synonyms": {...}, "parents": {...}}) or the package default."""
    from argus_lens.taxonomy import DEFAULT_TAXONOMY, Taxonomy

    if not path:
        return DEFAULT_TAXONOMY
    data = json.loads(Path(path).read_text())
    return Taxonomy(
        synonyms={norm_tag(k): norm_tag(v) for k, v in data.get("synonyms", {}).items()},
        parents={norm_tag(k): tuple(norm_tag(p) for p in v) for k, v in data.get("parents", {}).items()},
    )


def _refined_terms(caption: str, trigger: str) -> list[str]:
    """Split a refined category caption into terms, minus the trigger word."""
    terms = [t.strip() for t in caption.split(",") if t.strip()]
    return [t for t in terms if t != trigger]


def index_rows(
    captions: Iterable[dict[str, Any]], taxonomy, *, trigger: str = "", objects: bool = True
) -> Iterator[list[Any]]:
    """Yield ``[image_id, name, shoot_id, rel_path, category, tag, source]`` rows."""
    for row in captions:
        base = [row["image_id"], row["name"], row["shoot_id"], row["rel_path"]]
        seen: set[tuple[str, str]] = set()
        buckets = row["buckets"]["tags"]
        for category in (*INDEX_CATEGORIES, "unclassified"):
            if category == "unclassified" and not objects:
                continue
            label = "object" if category == "unclassified" else category
            for tag in buckets.get(category, []):
                for term in taxonomy.normalize([norm_tag(tag)]):
                    if (label, term) not in seen:
                        seen.add((label, term))
                        yield [*base, label, term, "tags"]
        accepted = (row.get("refine") or {}).get("accepted", [])
        for category in INDEX_CATEGORIES:
            if category in accepted:
                for term in taxonomy.normalize(
                    [norm_tag(t) for t in _refined_terms(row["variants"][category], trigger)]
                ):
                    if (category, term) not in seen:
                        seen.add((category, term))
                        yield [*base, category, term, "refine"]
        count = row["subjects"].get("count")
        people = "unknown" if count is None else ("multiple" if row["subjects"].get("multiple") else str(count))
        yield [*base, "people", people, "subjects"]


_COLOURS = r"(?:black|brown|blonde|red|orange|grey|gray|white|silver|pink|purple|blue|green|aqua|light brown|dark brown|auburn)"
_IDENTITY_SLOTS = {
    "hair colour": re.compile(rf"^{_COLOURS} hair$"),
    "eye colour": re.compile(rf"^{_COLOURS} eyes$"),
}


def identity_qa(index: list[list[Any]], min_share: float) -> tuple[list[list[Any]], list[list[Any]]]:
    """Per-name identity consensus, and images whose hair/eye colour contradicts it.

    Each ``name`` folder is one person, so its majority hair and eye colour is
    a strong prior. An image tagged with a *different* colour is often a
    second person, a wig/lighting change, or a misfiled photo — worth a look.
    """
    images_per_name: dict[str, set[str]] = defaultdict(set)
    tags_per_image: dict[str, set[str]] = defaultdict(set)
    where: dict[str, tuple[str, str]] = {}
    for image_id, name, _shoot, rel, category, tag, _src in index:
        images_per_name[name].add(image_id)
        where[image_id] = (name, rel)
        if category == "identity":
            tags_per_image[image_id].add(tag)
    consensus_rows, outlier_rows = [], []
    for name in sorted(images_per_name):
        ids = images_per_name[name]
        counts = Counter(tag for i in ids for tag in tags_per_image[i])
        for tag, n in counts.most_common():
            share = n / len(ids)
            if share >= min_share:
                consensus_rows.append([name, tag, n, len(ids), round(share, 3)])
        for slot, pattern in _IDENTITY_SLOTS.items():
            slot_counts = Counter({t: n for t, n in counts.items() if pattern.match(t)})
            if not slot_counts:
                continue
            majority, n = slot_counts.most_common(1)[0]
            if n / len(ids) < min_share:
                continue
            for image_id in sorted(ids):
                found = sorted(t for t in tags_per_image[image_id] if pattern.match(t))
                if found and majority not in found:
                    outlier_rows.append([image_id, name, where[image_id][1], slot, majority, ";".join(found)])
    return consensus_rows, outlier_rows


def cmd_index(args: argparse.Namespace) -> None:
    """Build index.csv (one row per image x category x tag), groups.csv and identity QA files."""
    run = Path(args.run_dir)
    captions = list(read_jsonl(run / f"captions.{args.target}.jsonl"))
    if not captions:
        raise SystemExit(f"no captions.{args.target}.jsonl - run 'assemble' first")
    taxonomy = _load_taxonomy(args.taxonomy)
    header = ["image_id", "name", "shoot_id", "rel_path", "category", "tag", "source"]
    rows = list(index_rows(captions, taxonomy, trigger=args.trigger, objects=not args.no_objects))
    write_csv(run / "index.csv", header, rows)

    groups = Counter((r[4], r[5]) for r in rows)
    names = defaultdict(set)
    for r in rows:
        names[(r[4], r[5])].add(r[1])
    write_csv(
        run / "groups.csv",
        ["category", "tag", "images", "names"],
        ([c, t, n, len(names[(c, t)])] for (c, t), n in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1]))),
    )
    consensus, outliers = identity_qa(rows, args.min_share)
    write_csv(run / "identity_consensus.csv", ["name", "tag", "images", "total", "share"], consensus)
    write_csv(run / "identity_outliers.csv", ["image_id", "name", "rel_path", "slot", "expected", "found"], outliers)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({h: [r[i] for r in rows] for i, h in enumerate(header)}), run / "index.parquet")
        print("index.parquet written (query with DuckDB: SELECT ... FROM 'index.parquet')")
    except ImportError:
        pass
    print(f"index.csv: {len(rows)} rows over {len(captions)} images; groups.csv: {len(groups)} groups")
    print(f"identity_outliers.csv: {len(outliers)} images contradict their name's hair/eye colour consensus")


# --------------------------------------------------------------------------- query


def _parse_terms(terms: list[str]) -> list[tuple[str | None, str]]:
    """``wardrobe:red dress`` -> ("wardrobe", "red dress"); bare terms match any category."""
    parsed = []
    for term in terms:
        category, _, tag = term.partition(":") if ":" in term else ("", "", term)
        parsed.append((category or None, norm_tag(tag)))
    return parsed


def cmd_query(args: argparse.Namespace) -> None:
    """Print images matching all --all terms, at least one --any term, and no --not term."""
    run = Path(args.run_dir)
    tags: dict[str, set[tuple[str, str]]] = defaultdict(set)
    info: dict[str, tuple[str, str]] = {}
    with (run / "index.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tags[row["image_id"]].add((row["category"], row["tag"]))
            info[row["image_id"]] = (row["name"], row["rel_path"])

    def hit(have: set[tuple[str, str]], term: tuple[str | None, str]) -> bool:
        category, tag = term
        return any(t == tag and (category is None or c == category) for c, t in have)

    all_terms, any_terms, not_terms = _parse_terms(args.all), _parse_terms(args.any), _parse_terms(args.exclude)
    matches = sorted(
        image_id
        for image_id, have in tags.items()
        if all(hit(have, t) for t in all_terms)
        and (not any_terms or any(hit(have, t) for t in any_terms))
        and not any(hit(have, t) for t in not_terms)
        and (not args.name or info[image_id][0] in args.name)
    )
    for image_id in matches[: args.limit or None]:
        name, rel = info[image_id]
        print(f"{image_id}\t{name}\t{rel}")
    by_name = Counter(info[i][0] for i in matches)
    print(f"# {len(matches)} images across {len(by_name)} names", file=sys.stderr)


# --------------------------------------------------------------------------- export


def cmd_export(args: argparse.Namespace) -> None:
    """Write a kohya-style dataset: image symlinks (or copies) with .txt captions."""
    import shutil

    run = Path(args.run_dir)
    root = Path(json.loads((run / "inventory.json").read_text())["source_root"])
    dest = Path(args.dest)
    rows = list(read_jsonl(run / f"captions.{args.target}.jsonl"))
    if args.name:
        rows = [r for r in rows if r["name"] in args.name]
    if args.ids:
        wanted = {line.split("\t", 1)[0].strip() for line in Path(args.ids).read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r["image_id"] in wanted]
    if args.single_person:
        rows = [r for r in rows if not r["subjects"].get("multiple")]
    written = 0
    for row in rows:
        caption = row["variants"].get(args.variant)
        if caption is None:
            raise SystemExit(f"unknown variant {args.variant!r}; have {sorted(row['variants'])}")
        src = root / row["rel_path"]
        out_dir = dest / row["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{row['shoot_id']}_{Path(row['rel_path']).stem}_{row['image_id'][:8]}"
        image_out = out_dir / f"{stem}{src.suffix.lower()}"
        if not image_out.exists():
            if args.copy:
                shutil.copy2(src, image_out)
            else:
                os.symlink(src, image_out)
        (out_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        written += 1
    print(f"exported {written} images to {dest} ({args.target}/{args.variant})")


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all stages."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="walk the source tree into manifest.jsonl")
    p.add_argument("source_root")
    p.add_argument("run_dir")
    p.add_argument("--layout", default=DEFAULT_LAYOUT, help=f"relative path layout (default {DEFAULT_LAYOUT})")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("caption", help="run inference once per unique image")
    p.add_argument("run_dir")
    p.add_argument("--backend", default="hybrid", help="hybrid (WD14+Florence-2), hybrid:wd14+openai-compat, ...")
    p.add_argument("--wd14-repo", default=None, help="e.g. SmilingWolf/wd-eva02-large-tagger-v3")
    p.add_argument("--wd14-threshold", type=float, default=0.35)
    p.add_argument("--model-id", default=None, help="prose model for openai-compat (e.g. qwen3-vl:8b)")
    p.add_argument("--base-url", default=None, help="openai-compat endpoint (default Ollama localhost)")
    p.add_argument("--prompt", default=None, help="openai-compat prompt preset or text (e.g. straightforward)")
    p.add_argument("--count-model", default=None, help="vision model asked 'how many people?' as a third opinion")
    p.add_argument("--count-url", default=None, help="endpoint for --count-model (default: --base-url)")
    p.add_argument("--limit", type=int, default=0, help="caption at most N images (pilot runs)")
    p.add_argument("--force", action="store_true", help="resume even if settings differ from run.json")
    p.set_defaults(func=cmd_caption)

    p = sub.add_parser("assemble", help="compose captions per target from raw.jsonl")
    p.add_argument("run_dir")
    p.add_argument("--targets", default="sdxl,flux")
    p.add_argument("--trigger", default="", help="LoRA trigger word prepended to every caption")
    p.add_argument("--hybrid-preset", default="keywords", help="tags, keywords, balanced, descriptive, prose")
    p.add_argument("--refine-model", default=None, help="OpenAI-compatible chat model for validated rewrites")
    p.add_argument("--refine-url", default=None)
    p.add_argument("--refine-key", default=None)
    p.add_argument(
        "--refine-variants",
        default="training,training_prose",
        help="variants to rewrite; add identity,wardrobe,setting,action to refine the index vocabulary",
    )
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("index", help="build the tag index, groups and identity QA")
    p.add_argument("run_dir")
    p.add_argument("--target", default="sdxl", help="which captions.<target>.jsonl to index")
    p.add_argument("--taxonomy", default=None, help="JSON with synonyms/parents (see taxonomy.example.json)")
    p.add_argument("--trigger", default="")
    p.add_argument("--min-share", type=float, default=0.6, help="share of a name's images for identity consensus")
    p.add_argument("--no-objects", action="store_true", help="leave unclassified tags out of the index")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("query", help="search the index")
    p.add_argument("run_dir")
    p.add_argument("--all", nargs="*", default=[], metavar="CAT:TAG", help="every term must match")
    p.add_argument("--any", nargs="*", default=[], metavar="CAT:TAG", help="at least one must match")
    p.add_argument("--not", dest="exclude", nargs="*", default=[], metavar="CAT:TAG", help="none may match")
    p.add_argument("--name", nargs="*", default=[], help="restrict to these name folders")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("export", help="write a training dataset (symlinks + .txt)")
    p.add_argument("run_dir")
    p.add_argument("dest")
    p.add_argument("--target", default="sdxl")
    p.add_argument("--variant", default=None, help="default: training (sdxl) / training_prose (flux)")
    p.add_argument("--name", nargs="*", default=[], help="only these name folders")
    p.add_argument("--ids", default=None, help="file of image_ids (e.g. saved 'query' output)")
    p.add_argument("--single-person", action="store_true", help="skip images flagged as multi-person")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "export" and args.variant is None:
        args.variant = "training_prose" if args.target in ("flux", "sd3") else "training"
    args.func(args)


if __name__ == "__main__":
    main()
