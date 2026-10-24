"""Text sidecar exporter — writes one .txt file per image."""

from __future__ import annotations

from pathlib import Path

from argus_lens.types import CaptionResult


def export_txt(results: dict[str, CaptionResult], output_dir: Path) -> None:
    """Write .txt sidecar files alongside images.

    Each file is named ``{image_stem}.txt`` and contains the
    ``final_caption``.  Compatible with kohya_ss, EveryDream,
    and other LoRA training tools.

    A *name* with directory parts (``shoot_1/IMG_1.jpg``, as produced by a
    recursive ``caption_directory``) writes its sidecar into the matching
    sub-directory of *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        txt_path = output_dir / Path(name).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(result.final_caption, encoding="utf-8")
