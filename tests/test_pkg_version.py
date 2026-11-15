"""``import argus_lens`` works without the generated ``_version.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_without_generated_version_file(tmp_path: Path) -> None:
    """A copied source tree lacking ``_version.py`` still imports, with a fallback version."""
    src = Path(__file__).resolve().parents[1] / "src" / "argus_lens"
    pkg = tmp_path / "argus_lens"
    pkg.mkdir()
    for name in ("__init__.py", "_pkg_version.py"):
        (pkg / name).write_text((src / name).read_text())
    # Stub the heavy imports __init__ pulls in; only the version wiring is under test.
    (pkg / "engine.py").write_text("ArgusLens = object\n")
    (pkg / "types.py").write_text(
        "BackendKind = CaptionResult = CaptionTargetProfile = CategoryConfig = TokenBudgetConfig = object\n"
    )
    code = "import argus_lens, sys; sys.stdout.write(argus_lens.__version__)"
    out = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(tmp_path)!r}); {code}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout  # installed metadata version, or the 0.0.0+unknown placeholder
