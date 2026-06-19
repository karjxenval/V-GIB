# README patch for V-GIB

Paste the following section near the top of `README.md`, after the project description.

```markdown
## Reusable software status

V-GIB is organized as a reusable research-software package, not only a collection of experiment scripts. The public interface is:

- `src/vgib/` for reusable utilities and package code.
- `scripts/run_real_benchmarks.py` for benchmark execution.
- `scripts/vgib_smoke_runner.py` for a reproducible laptop smoke test.
- `scripts/vgib_result_gate.py` for checking whether a run produced complete outputs.
- `scripts/vgib_doctor.py` for environment and installation diagnostics.
- `docs/` for API, reproducibility, and result-schema documentation.

### Quick industrial smoke test

```bash
python scripts/vgib_doctor.py
python scripts/vgib_smoke_runner.py --outdir runs/smoke_industry --device cpu
python scripts/vgib_result_gate.py --run-dir runs/smoke_industry --min-rows 1
```

Expected core outputs:

- `all_runs.csv`
- `summary_mean_std.csv`
- `run_config.json`
- `summary.md`

### Development checks

```bash
pip install -e .
pip install -r requirements-dev.txt
python -m pytest
python scripts/vgib_release_manifest.py --out RELEASE_MANIFEST_SHA256.txt
```

### Interpretation warning

The information, curvature, intrinsic-dimension, and utility quantities reported by V-GIB are empirical diagnostics. They are designed for controlled comparison of representation-learning methods under data constraints, not as exact population-level information-theoretic estimates.
```
