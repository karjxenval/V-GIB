# Add-on manifest

This folder contains the following overlay categories:

- Repository metadata: `pyproject.toml`, `requirements.txt`, `environment.yml`, `requirements-dev.txt`, `.gitignore`, `Makefile`.
- CI and contribution infrastructure: `.github/`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- Reusable modules: `src/vgib/reproducibility.py`, `src/vgib/reporting.py`, `src/vgib/release.py`.
- Operational scripts: `scripts/vgib_doctor.py`, `scripts/vgib_smoke_runner.py`, `scripts/vgib_result_gate.py`, `scripts/vgib_release_manifest.py`, `scripts/vgib_repository_audit.py`.
- Examples: `examples/minimal_geometry_api.py`, `examples/result_loading_example.py`.
- Tests: `tests/test_geometry_utilities.py`, `tests/test_reporting_utilities.py`, `tests/test_repository_metadata.py`, `tests/test_scripts_compile_industry.py`.
- Documentation: `docs/`.

Copy these into the V-GIB repository root and commit.
