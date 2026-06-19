"""Release-manifest utilities for V-GIB."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for one file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_release_files(root: str | Path, include_suffixes: Iterable[str] | None = None):
    """Yield source/documentation files that should be tracked in a release manifest."""
    root = Path(root)
    suffixes = set(include_suffixes or [".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cff"])
    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "data", "runs", "logs", "figures", "tables", "build", "dist"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() in suffixes or path.name in {"Makefile", "LICENSE"}:
            yield path


def write_manifest(root: str | Path, out: str | Path) -> None:
    """Write SHA256 manifest relative to repository root."""
    root = Path(root).resolve()
    out = Path(out)
    rows = []
    for path in iter_release_files(root):
        rel = path.resolve().relative_to(root).as_posix()
        rows.append(f"{sha256_file(path)}  {rel}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
