# Results schema

The benchmark runner should write results under a chosen output directory, usually `runs/<name>/`.

## Required files

| File | Purpose |
|---|---|
| `all_runs.csv` | Per-dataset, per-method, per-seed, per-label-fraction metrics. |
| `summary_mean_std.csv` | Aggregate summaries across seeds/runs. |
| `run_config.json` | Reproducibility metadata and command settings. |
| `summary.md` | Human-readable summary. |

## Recommended columns for `all_runs.csv`

At minimum, downstream tools expect some of these columns:

- `dataset`
- `method`
- `fraction`
- `seed`
- `accuracy` or `acc`
- `balanced_accuracy` or `bal_acc`
- `macro_f1`
- `auroc`
- `kl_proxy` or `compression_proxy`
- `curvature_proxy`
- `dimension_ratio`
- `utility_proxy`

The result gate is intentionally tolerant: it checks that required files exist, CSVs are readable, and tables are not empty. You may add stricter thresholds for a manuscript-specific release.
