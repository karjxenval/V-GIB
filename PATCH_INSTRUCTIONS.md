# V-GIB industry-grade add-on pack

This pack is an overlay for the existing `karjxenval/V-GIB` repository. Copy the folders and files in this pack into the root of your V-GIB repository, then commit them.

## What this pack adds

1. Continuous integration with GitHub Actions.
2. Environment diagnostics for Windows/Anaconda and Linux.
3. A reproducible smoke-test runner.
4. A result-quality gate for checking benchmark outputs.
5. A release manifest generator with SHA256 hashes.
6. Public API utility modules under `src/vgib/`.
7. Stronger tests for geometry utilities, packaging metadata, output schema, and script syntax.
8. Documentation for reusable software, reproducibility, result schemas, and industry-readiness.
9. Security, issue templates, pull-request template, code of conduct, changelog, and release checklist.
10. A README patch you can paste into the current README.

## Copy into the repo

From the folder that contains this add-on pack and the V-GIB repo:

```bat
xcopy /E /I /Y vgib_industry_grade_addons V-GIB
```

Or manually copy the contents of this folder into the root of `V-GIB/`.

## Recommended commands after copying

```bat
cd V-GIB
conda create -n vgib python=3.10 -y
conda activate vgib
pip install -e .
pip install -r requirements-dev.txt
python scripts\vgib_doctor.py
python -m pytest
python scripts\vgib_smoke_runner.py --outdir runs\smoke_industry --device cpu
python scripts\vgib_result_gate.py --run-dir runs\smoke_industry --min-rows 1
python scripts\vgib_release_manifest.py --out RELEASE_MANIFEST_SHA256.txt
```

## Git upload

```bat
git status
git add .
git commit -m "Harden V-GIB as reusable research software"
git push
```

## Optional release tag

```bat
git tag -a v0.1.0 -m "V-GIB v0.1.0 research software release"
git push origin v0.1.0
```
