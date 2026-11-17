"""Text sidecar exporter — writes one .txt file per image."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from argus_lens.types import CaptionResult


def _sidecar_path(output_dir: Path, name: str) -> Path:
    """Return the ``.txt`` path for *name* under *output_dir*, refusing escapes.

    *name* is a result key, which may come from a caller (an upload's
    filename). An absolute name would make pathlib discard *output_dir*, and
    a ``..`` part climbs out of it; both raise ``ValueError``, as does a
    sidecar that resolves outside *output_dir* (e.g. via an existing symlink).
    """
    parts = PurePosixPath(name).parts + PureWindowsPath(name).parts
    if not name or PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe result name for a sidecar: {name!r}")
    txt_path = output_dir / Path(name).with_suffix(".txt")
    if not txt_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(f"sidecar for {name!r} would land outside {output_dir}")
    return txt_path


def export_txt(results: dict[str, CaptionResult], output_dir: Path) -> None:
    """Write .txt sidecar files alongside images.

    Each file is named ``{image_stem}.txt`` and contains the
    ``final_caption``.  Compatible with kohya_ss, EveryDream,
    and other LoRA training tools.

    A *name* with directory parts (``shoot_1/IMG_1.jpg``, as produced by a
    recursive ``caption_directory``) writes its sidecar into the matching
    sub-directory of *output_dir*. An absolute name, a ``..`` part, or a
    sidecar resolving outside *output_dir* raises ``ValueError``. Every name
    is checked before anything is written, so a bad key writes nothing.
    Two names that map to one sidecar (``IMG_1.jpg`` and ``IMG_1.png``) raise
    ``ValueError`` too, rather than one caption silently replacing the other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [(_sidecar_path(output_dir, name), result) for name, result in results.items()]
    owners: dict[Path, str] = {}
    for (txt_path, _), name in zip(planned, results, strict=True):
        other = owners.setdefault(txt_path, name)
        if other != name:
            raise ValueError(f"{other!r} and {name!r} would share the sidecar {txt_path}")
    for txt_path, result in planned:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(result.final_caption, encoding="utf-8")
