from __future__ import annotations

import py_compile
from pathlib import Path


def test_industry_scripts_compile():
    root = Path(__file__).resolve().parents[1]
    for name in [
        "vgib_doctor.py",
        "vgib_smoke_runner.py",
        "vgib_result_gate.py",
        "vgib_release_manifest.py",
        "vgib_repository_audit.py",
    ]:
        py_compile.compile(str(root / "scripts" / name), doraise=True)
