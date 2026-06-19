# Reproducibility guide

## Recommended environment

```bash
conda env create -f environment.yml
conda activate vgib
pip install -e .
pip install -r requirements-dev.txt
```

## Environment check

```bash
python scripts/vgib_doctor.py
```

## Smoke test

```bash
python scripts/vgib_smoke_runner.py --outdir runs/smoke_industry --device cpu
python scripts/vgib_result_gate.py --run-dir runs/smoke_industry --min-rows 1
```

## Reproducibility metadata

Each benchmark run should produce:

- `run_config.json` with run settings.
- `all_runs.csv` with per-run metrics.
- `summary_mean_std.csv` with aggregated metrics.
- `summary.md` with a readable report.

## Release manifest

Before publishing a release:

```bash
python scripts/vgib_release_manifest.py --out RELEASE_MANIFEST_SHA256.txt
```

This records file hashes for source, scripts, tests, and docs.
