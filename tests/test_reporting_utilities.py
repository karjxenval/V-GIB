from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vgib.reporting import compact_quality_report, load_run_tables, validate_run_outputs


def test_validate_and_load_run_outputs(tmp_path: Path):
    pd.DataFrame([{"dataset": "toy", "method": "erm", "accuracy": 0.5}]).to_csv(
        tmp_path / "all_runs.csv", index=False
    )
    pd.DataFrame([{"dataset": "toy", "method": "erm", "accuracy_mean": 0.5}]).to_csv(
        tmp_path / "summary_mean_std.csv", index=False
    )
    (tmp_path / "run_config.json").write_text(json.dumps({"seed": 13}), encoding="utf-8")
    (tmp_path / "summary.md").write_text("# Summary\n", encoding="utf-8")

    validate_run_outputs(tmp_path)
    tables = load_run_tables(tmp_path)
    assert len(tables.all_runs) == 1
    assert tables.config["seed"] == 13
    assert "Rows in all_runs.csv" in compact_quality_report(tmp_path)
