# Release checklist

Before tagging a release:

1. Run the environment doctor.
2. Run tests.
3. Run the smoke benchmark.
4. Run the result gate.
5. Update `CITATION.cff` with manuscript or release details.
6. Update `CHANGELOG.md`.
7. Generate `RELEASE_MANIFEST_SHA256.txt`.
8. Commit all changes.
9. Tag the release.

Commands:

```bash
python scripts/vgib_doctor.py
python -m pytest
python scripts/vgib_smoke_runner.py --outdir runs/smoke_industry --device cpu
python scripts/vgib_result_gate.py --run-dir runs/smoke_industry --min-rows 1
python scripts/vgib_release_manifest.py --out RELEASE_MANIFEST_SHA256.txt
git add .
git commit -m "Prepare V-GIB release"
git tag -a v0.1.0 -m "V-GIB v0.1.0"
git push origin main --tags
```
