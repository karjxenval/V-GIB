# V-GIB model card

## Project

Variational Geometric Information Bottleneck (V-GIB) for data-efficient representation learning.

## Intended use

Research comparison of ERM, VIB, V-GIB, and ablated geometric bottleneck variants under low-label regimes.

## Not intended for

Production decision-making without independent validation, external audits, and domain-specific testing.

## Main outputs

- Predictive metrics: accuracy, balanced accuracy, F1, AUROC.
- Diagnostic proxies: KL/compression, curvature, intrinsic dimension, and utility proxy.

## Important limitation

The diagnostic quantities are empirical proxies. They are not exact population-level mutual information, curvature, or intrinsic dimension estimates.

## Reproducibility

Use `run_config.json`, `all_runs.csv`, `summary_mean_std.csv`, and `RELEASE_MANIFEST_SHA256.txt` to document runs.
