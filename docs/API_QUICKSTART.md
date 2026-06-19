# V-GIB API quickstart

The preferred public workflow is still the benchmark runner:

```bash
python scripts/run_real_benchmarks.py --help
```

The reusable utility layer lives under `src/vgib/`.

```python
import torch
from vgib.geometry import kl_standard_normal, participation_ratio_idim
from vgib.reproducibility import set_global_seed, environment_report

set_global_seed(13)
mu = torch.zeros(8, 4)
logvar = torch.zeros(8, 4)
kl = kl_standard_normal(mu, logvar)
print(kl.shape)

z = torch.randn(64, 10)
print(participation_ratio_idim(z))
print(environment_report())
```

For result checking:

```python
from pathlib import Path
from vgib.reporting import load_run_tables, validate_run_outputs

run_dir = Path("runs/smoke_industry")
validate_run_outputs(run_dir)
tables = load_run_tables(run_dir)
print(tables.summary.head())
```
