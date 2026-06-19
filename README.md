# Variational Geometric Information Bottleneck (V-GIB)

Research code for validating **Variational Geometric Information Bottleneck (V-GIB)** models under data-constrained learning. The repository contains synthetic manifold experiments, image benchmarks, tabular benchmarks, geometric diagnostics, and plotting utilities.

The core idea is to compare ordinary empirical risk minimization, variational bottleneck baselines, and geometry-aware bottleneck variants using predictive performance plus diagnostic proxies for compression, curvature, intrinsic dimension, and utility.

## Repository status

This repository is organized from the original research scripts into a more public-facing structure. The main recommended entry point is:

```bash
python scripts/run_real_benchmarks.py --help
```

Use the `scripts/` directory for the cleaned public workflow. The `legacy/` directory keeps older exploratory scripts for reproducibility and traceability, but these are not the preferred public interface.

## Main features

- Real benchmark suite for Fashion-MNIST, CIFAR-10, PCAM, Covertype, and Breast Cancer.
- Matched method comparison: ERM, VIB, V-GIB, and V-GIB ablations.
- Low-label regimes with multiple seeds.
- Geometric diagnostics: Jacobian/Hessian proxies, participation-ratio intrinsic dimension, KL compression proxy, and utility proxy.
- CSV logs, aggregate summaries, Markdown summaries, and publication-style figures.

## Installation

Create a fresh environment:

```bash
conda create -n vgib python=3.10 -y
conda activate vgib
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build matching your CUDA version from the official PyTorch installation instructions, then install the remaining requirements.

## Smoke test

This is the safest first test on a normal laptop:

```bash
python scripts/run_real_benchmarks.py \
  --root ./data \
  --outdir ./runs/smoke \
  --datasets breast_cancer \
  --methods erm vib vgib \
  --fractions 0.20 \
  --seeds 13 \
  --epochs 2 \
  --batch-size 32 \
  --max-train-samples 300 \
  --max-eval-samples 200 \
  --device cpu
```

A slightly heavier smoke test using Fashion-MNIST:

```bash
python scripts/run_real_benchmarks.py \
  --root ./data \
  --outdir ./runs/fashion_smoke \
  --datasets fashionmnist \
  --methods erm vib vgib \
  --fractions 0.05 \
  --seeds 13 \
  --epochs 3 \
  --batch-size 64 \
  --download
```

## Full benchmark example

```bash
python scripts/run_real_benchmarks.py \
  --root ./data \
  --outdir ./runs/vgib_real_full \
  --datasets cifar10 pcam covtype \
  --methods erm vib vgib vgib_no_curv vgib_no_dim \
  --fractions 0.01 0.05 0.10 0.20 \
  --seeds 13 29 47 \
  --epochs 25 \
  --batch-size 128 \
  --latent-dim 64 \
  --curvature-mode jacobian \
  --download \
  --make-embeddings
```

## Plotting

For logs produced by the older unified experiment script:

```bash
python scripts/plot_experiment_logs.py --base-dir ./runs --out-dir ./figures
```

For CIFAR validation CSV logs with columns such as `epoch`, `test_acc`, `align_mi`, and `eff_ratio`:

```bash
python scripts/plot_validation.py --logdir ./logs --outdir ./figs --tablesdir ./tables
```

## Outputs

Typical outputs are written under `runs/`, `figures/`, or `tables/`:

- `all_runs.csv`: raw per-run metrics.
- `summary_mean_std.csv`: aggregate results across seeds.
- `run_config.json`: reproducibility metadata.
- `summary.md`: readable run summary.
- `.png` or `.pdf` figures for accuracy, F1, AUROC, curvature, dimension ratio, and utility diagnostics.

## Recommended citation

A `CITATION.cff` file is included as a placeholder. Update the manuscript title, author details, DOI/arXiv link, and release version before making the repository public.

## Important notes

The geometric and information quantities in this repository are diagnostic proxies, not exact population-level estimates. They should be interpreted as empirical validation tools for comparing methods under controlled experimental settings.
