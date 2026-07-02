# Manifest

## Main scripts

- `src/vgib_real_benchmarks.py`: main reviewer-facing real benchmark suite.
- `src/complete_stronger_validation.py`: earlier complete validation pipeline.
- `src/vgib_experiments.py`: unified script for synthetic Swiss-roll, FashionMNIST, and CIFAR-10 experiments.

## Plotting scripts

- `scripts/vgib_plots.py`: generates summary tables and diagnostic plots from `vgib_experiments.py` logs.
- `scripts/plot_validation.py`: generates CIFAR validation summaries from compatible CSV logs.

## Legacy/provenance scripts

- `legacy/train_cifar_geom.py`: earlier CIFAR-specific V-GIB training script.
- `legacy/vib_vgib_sanity_checks.py`: synthetic sanity-check script.
- `legacy/untitled7.py`: empty notebook-export stub retained for provenance.
