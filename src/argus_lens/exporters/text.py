"""Text sidecar exporter — writes one .txt file per image."""

from __future__ import annotations

from pathlib import Path

from argus_lens.types import CaptionResult


def export_txt(results: dict[str, CaptionResult], output_dir: Path) -> None:
    """Write .txt sidecar files alongside images.

    Each file is named ``{image_stem}.txt`` and contains the
    ``final_caption``.  Compatible with kohya_ss, EveryDream,
    and other LoRA training tools.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        stem = Path(name).stem
        txt_path = output_dir / f"{stem}.txt"
        txt_path.write_text(result.final_caption, encoding="utf-8")
