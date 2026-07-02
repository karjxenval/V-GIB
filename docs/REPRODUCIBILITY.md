# Reproducibility notes

## Environment

Create a fresh Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU use, install the `torch` and `torchvision` builds matching your CUDA version.

## Determinism

The main benchmark script sets Python, NumPy, and PyTorch seeds and turns on deterministic CuDNN settings where possible. Exact equality across machines is still not guaranteed because GPU kernels, package versions, and hardware may differ.

## Outputs

Typical outputs are stored under the selected `--outdir`, including:

- per-run CSV logs,
- aggregate summaries,
- diagnostic plots,
- optional embedding visualizations.

Do not commit large datasets, checkpoints, or raw run directories to Git. Use `.gitignore` or attach them through a release/archive.

## Recommended reporting

For manuscript tables, report mean ± standard deviation over seeds. Keep CovType as supporting optimization evidence unless full validation/test summaries are available.
