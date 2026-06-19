#!/usr/bin/env python3
"""Run the safest V-GIB smoke test and verify expected outputs.

This wrapper avoids users needing to remember the long command. It uses the
small scikit-learn breast-cancer dataset by default, so it does not require
image downloads.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data", help="Dataset cache root")
    parser.add_argument("--outdir", default="./runs/smoke_industry", help="Output directory")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=300)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts" / "run_real_benchmarks.py"
    if not runner.exists():
        raise FileNotFoundError(f"Cannot find benchmark runner: {runner}")

    cmd = [
        args.python,
        str(runner),
        "--root", args.root,
        "--outdir", args.outdir,
        "--datasets", "breast_cancer",
        "--methods", "erm", "vib", "vgib",
        "--fractions", "0.20",
        "--seeds", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-train-samples", str(args.max_train_samples),
        "--max-eval-samples", str(args.max_eval_samples),
        "--device", args.device,
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    _ensure_src_on_path()
    from vgib.reporting import compact_quality_report, validate_run_outputs

    validate_run_outputs(args.outdir, min_rows=1)
    print("\nSmoke test outputs verified.")
    print(compact_quality_report(args.outdir))


if __name__ == "__main__":
    main()
