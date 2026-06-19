"""Result loading and validation helpers for V-GIB benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_OUTPUTS = (
    "all_runs.csv",
    "summary_mean_std.csv",
    "run_config.json",
    "summary.md",
)


@dataclass
class RunTables:
    all_runs: pd.DataFrame
    summary: pd.DataFrame
    config: dict[str, Any]
    summary_markdown: str


def validate_run_outputs(run_dir: str | Path, min_rows: int = 1) -> None:
    """Raise a clear error if a benchmark run directory is incomplete."""
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {root}")

    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required output files in {root}: {missing}")

    all_runs = pd.read_csv(root / "all_runs.csv")
    summary = pd.read_csv(root / "summary_mean_std.csv")
    if len(all_runs) < min_rows:
        raise ValueError(f"all_runs.csv has {len(all_runs)} rows, expected at least {min_rows}")
    if len(summary) < 1:
        raise ValueError("summary_mean_std.csv is empty")

    with (root / "run_config.json").open("r", encoding="utf-8") as f:
        json.load(f)


def load_run_tables(run_dir: str | Path) -> RunTables:
    """Load standard V-GIB output tables after validation."""
    root = Path(run_dir)
    validate_run_outputs(root)
    with (root / "run_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return RunTables(
        all_runs=pd.read_csv(root / "all_runs.csv"),
        summary=pd.read_csv(root / "summary_mean_std.csv"),
        config=config,
        summary_markdown=(root / "summary.md").read_text(encoding="utf-8"),
    )


def compact_quality_report(run_dir: str | Path) -> str:
    """Return a short text report suitable for terminal output or README logs."""
    tables = load_run_tables(run_dir)
    lines = [
        f"Run directory: {Path(run_dir)}",
        f"Rows in all_runs.csv: {len(tables.all_runs)}",
        f"Rows in summary_mean_std.csv: {len(tables.summary)}",
    ]
    for col in ["dataset", "method", "fraction", "seed"]:
        if col in tables.all_runs.columns:
            vals = sorted(map(str, tables.all_runs[col].dropna().unique().tolist()))
            lines.append(f"{col}: {', '.join(vals[:12])}")
    return "\n".join(lines)
