#!/usr/bin/env python3
"""Quality gate for V-GIB benchmark outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-columns", nargs="*", default=[])
    parser.add_argument("--min-accuracy", type=float, default=None, help="Optional minimum accuracy/acc threshold")
    args = parser.parse_args()

    _ensure_src_on_path()
    from vgib.reporting import compact_quality_report, load_run_tables, validate_run_outputs

    validate_run_outputs(args.run_dir, min_rows=args.min_rows)
    tables = load_run_tables(args.run_dir)
    df = tables.all_runs

    missing_cols = [c for c in args.require_columns if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")

    if args.min_accuracy is not None:
        acc_col = "accuracy" if "accuracy" in df.columns else "acc" if "acc" in df.columns else None
        if acc_col is None:
            raise SystemExit("No accuracy/acc column found for --min-accuracy check")
        finite = [float(v) for v in df[acc_col].dropna().tolist() if math.isfinite(float(v))]
        if not finite:
            raise SystemExit(f"No finite values in {acc_col}")
        if max(finite) < args.min_accuracy:
            raise SystemExit(f"Best {acc_col}={max(finite):.4f} below threshold {args.min_accuracy}")

    print("V-GIB result gate passed.")
    print(compact_quality_report(args.run_dir))


if __name__ == "__main__":
    main()
