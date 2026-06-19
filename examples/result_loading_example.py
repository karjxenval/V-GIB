"""Load and validate a V-GIB benchmark run."""

from __future__ import annotations

from pathlib import Path

from vgib.reporting import compact_quality_report, load_run_tables, validate_run_outputs


def main() -> None:
    run_dir = Path("runs/smoke_industry")
    validate_run_outputs(run_dir)
    tables = load_run_tables(run_dir)
    print(compact_quality_report(run_dir))
    print(tables.summary.head())


if __name__ == "__main__":
    main()
