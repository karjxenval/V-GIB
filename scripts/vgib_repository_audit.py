#!/usr/bin/env python3
"""Audit whether the V-GIB repository has the minimum public-software files."""

from __future__ import annotations

from pathlib import Path

REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    ".github/workflows/ci.yml",
    "src/vgib/__init__.py",
    "src/vgib/geometry.py",
    "src/vgib/reproducibility.py",
    "src/vgib/reporting.py",
    "tests/test_static_syntax.py",
    "tests/test_geometry_utilities.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [p for p in REQUIRED if not (root / p).exists()]
    if missing:
        print("Missing files:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)
    print("Repository audit passed: required public-software files are present.")


if __name__ == "__main__":
    main()
