"""XMP sidecars and the GPU coordinator belong to argus-cortex; lens must not grow its own copies.

``argus_cortex.xmp`` (shared with curator) does the merge-not-clobber writing, and
``argus_cortex.gpu`` holds the host-wide lease every Argus service takes. A local copy
of either loses the shared behaviour without any test failing, so these checks scan the
source for the signs of one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "argus_lens"
SOURCES = sorted(p for p in SRC.rglob("*.py") if "_ram" not in p.parts)

# Raw XMP packets are built from these; argus_cortex.xmp owns them.
XMP_MARKERS = ("adobe:ns:meta", "ns.adobe.com/xap", "x:xmpmeta")
# Modules a hand-rolled XMP writer or GPU lock would need.
BANNED_IMPORTS = {"fcntl", "xml.etree", "xml.etree.ElementTree", "lxml", "defusedxml"}
BANNED_CLASS_SUFFIXES = ("Coordinator", "Lease", "Broker")


def _imports(tree: ast.AST) -> set[str]:
    """Every module name imported anywhere in *tree*."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_sources_were_found() -> None:
    """Guard against the scan silently matching nothing."""
    assert len(SOURCES) > 20


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_local_xmp_writer_or_gpu_coordinator(path: Path) -> None:
    """No module builds XMP itself, takes its own GPU lock, or defines a coordinator class."""
    text = path.read_text(encoding="utf-8")
    assert not [m for m in XMP_MARKERS if m in text], "write XMP through argus_cortex.xmp"
    tree = ast.parse(text)
    assert not _imports(tree) & BANNED_IMPORTS, "use argus_cortex.xmp / argus_cortex.gpu"
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert not [c for c in classes if c.endswith(BANNED_CLASS_SUFFIXES)], (
        "the GPU coordinator lives in argus_cortex.gpu"
    )


# Lens stays MIT by talking to argus-retina over HTTP (cortex's OcrClient), never importing it or an OCR engine.
OCR_ENGINE_IMPORTS = ("argus_retina", "rapidocr", "rapidocr_onnxruntime", "paddleocr", "easyocr", "pytesseract")


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_in_process_ocr_engine_or_retina_import(path: Path) -> None:
    """Lens never imports argus-retina or an OCR engine: text is read over HTTP (#141)."""
    imported = _imports(ast.parse(path.read_text(encoding="utf-8")))
    assert not [m for m in imported if m.split(".")[0] in OCR_ENGINE_IMPORTS], (
        "read text through argus_cortex.moderation.OcrClient"
    )
