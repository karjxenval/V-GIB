from __future__ import annotations

from pathlib import Path


def test_metadata_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "CITATION.cff",
        "CONTRIBUTING.md",
    ]
    missing = [p for p in required if not (root / p).exists()]
    assert not missing, f"Missing metadata files: {missing}"


def test_readme_mentions_smoke_test():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8").lower()
    assert "smoke" in text
    assert "run_real_benchmarks.py" in text
