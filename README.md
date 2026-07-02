# V-GIB: Variational Geometric Information Bottleneck

This repository contains the reproducibility code for the manuscript:

**Geometry as a Missing Axis of Representation Quality: The Variational Geometric Information Bottleneck under Data Scarcity**

The code supports experiments for geometry-aware information bottleneck learning under label scarcity. It includes real benchmark runs, diagnostic plotting scripts, and legacy/sanity-check scripts used during development.

## Repository layout

```text
V-GIB-reproducibility/
├── src/
│   ├── vgib_real_benchmarks.py          # Main reviewer-facing benchmark suite
│   ├── complete_stronger_validation.py  # Earlier complete validation pipeline
│   └── vgib_experiments.py              # Unified synthetic/FashionMNIST/CIFAR experiments
├── scripts/
│   ├── vgib_plots.py                    # Plots/tables from vgib_experiments.py logs
│   └── plot_validation.py               # CIFAR validation plotting utility
├── legacy/
│   ├── train_cifar_geom.py              # Earlier CIFAR-specific training script
│   ├── vib_vgib_sanity_checks.py        # Synthetic sanity checks
│   └── untitled7.py                     # Empty notebook-export stub retained for provenance
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── NOTES_FOR_MANUSCRIPT.md
├── tools/
│   └── verify_syntax.py
├── requirements.txt
├── .gitignore
├── DESCRIPTION.md
└── MANIFEST.md
```

## Recommended entry point

Use `src/vgib_real_benchmarks.py` as the main script for the paper. It trains matched baselines and V-GIB variants on real datasets under low-label regimes and writes CSV summaries/plots.

### Smoke test

```bash
python src/vgib_real_benchmarks.py   --root ./data   --outdir ./runs/smoke   --datasets fashionmnist breast_cancer   --methods erm vib vgib   --fractions 0.05   --seeds 13   --epochs 3   --batch-size 64   --download
```

### Main benchmark-style run

```bash
python src/vgib_real_benchmarks.py   --root ./data   --outdir ./runs/vgib_real_full   --datasets breast_cancer fashionmnist cifar10 covtype   --methods erm vib vgib vgib_no_curv vgib_no_dim   --fractions 0.01 0.05 0.10 0.20   --seeds 13 29 47   --epochs 25   --batch-size 128   --latent-dim 64   --curvature-mode jacobian   --download
```

Use PCam only if your local `torchvision`/`h5py`/dataset setup works reliably. The manuscript should not rely on incomplete PCam runs unless complete matched outputs are available.

## Basic syntax check

```bash
python tools/verify_syntax.py
```

This only verifies Python syntax. It does not run the full experiments.

## Notes

- The repository does not include downloaded datasets or trained model checkpoints.
- Large outputs should stay outside Git or be attached through a release/Zenodo archive.
- Before public release, choose a license, for example MIT, BSD-3-Clause, Apache-2.0, or a more restrictive academic license.
