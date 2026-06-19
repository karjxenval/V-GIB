# Industry-readiness checklist for V-GIB

A serious public research-software repository should satisfy four levels.

## Level 1: Reproducible research code

- Clear README.
- Install instructions.
- Smoke test.
- Results written to stable file names.
- Seeded experiments.

## Level 2: Reusable software

- Installable package with `pyproject.toml`.
- Public modules under `src/vgib/`.
- Examples separate from core code.
- Tests for public utilities.
- Documentation of outputs and assumptions.

## Level 3: Reliable engineering workflow

- CI on push and pull requests.
- Linting and formatting.
- Environment diagnostic script.
- Result-quality gate.
- Release manifest with file hashes.
- Clear contribution and security policy.

## Level 4: Industrial/scientific-computing credibility

- Transparent limitations.
- Output schemas stable enough for downstream use.
- Run metadata saved with each benchmark.
- Release tags.
- Citation metadata.
- Clear statement that diagnostic proxies are not exact population-level estimates.

This add-on pack moves V-GIB from Level 1/2 toward Level 3. Level 4 requires accumulating verified benchmark outputs, tagged releases, and stable APIs over time.
