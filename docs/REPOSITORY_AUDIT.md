# Repository audit

## Files promoted to public workflow

- `scripts/run_real_benchmarks.py`: strongest public entry point. Supports real datasets, multiple methods, fractions, seeds, summaries, figures, and config saving.
- `scripts/run_complete_validation.py`: broader validation script covering Swiss-roll, Fashion-MNIST, and CIFAR-10.
- `scripts/run_cifar_vgg_vgib.py`: CIFAR-10 VGG-style experiment. Defaults were changed from machine-specific Windows paths to portable relative paths.
- `scripts/plot_validation.py`: plotting and table generation for CIFAR validation CSV logs.
- `scripts/plot_experiment_logs.py`: plotting utilities for legacy experiment logs.

## Files kept as legacy

- `legacy/vgib_experiments_legacy.py`: useful historical script, but contains large commented sections and partially disabled wrappers.
- `legacy/vib_vgib_sanity_checks_legacy.py`: useful exploratory notebook-style script, but contains notebook cells, seaborn styling, and top-level execution blocks.
- `legacy/untitled7_do_not_publish.py`: not suitable for public presentation.

## Main cleanup still recommended before public release

1. Consolidate duplicate geometry utilities into `src/vgib/geometry.py`.
2. Move dataset-loading code into `src/vgib/datasets.py`.
3. Move model classes into `src/vgib/models.py`.
4. Keep experiment scripts thin: argument parsing plus calls into reusable modules.
5. Add a small CPU-only test that does not download large datasets.
6. Add result cards or sample figures after a verified run.
