"""Reproducibility and environment utilities for V-GIB."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    executable: str
    platform: str
    working_directory: str
    packages: dict[str, str]
    cuda_available: bool | None
    torch_device_count: int | None


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set Python, NumPy, and Torch seeds when Torch is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch may be intentionally unavailable in documentation-only environments.
        pass


def _version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def environment_report() -> dict[str, Any]:
    """Return a JSON-serializable environment report."""
    packages = {
        name: _version(name)
        for name in [
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "torch",
            "torchvision",
            "tqdm",
            "h5py",
        ]
    }

    cuda_available = None
    torch_device_count = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_device_count = int(torch.cuda.device_count())
    except Exception:
        pass

    report = EnvironmentReport(
        python=sys.version.replace("\n", " "),
        executable=sys.executable,
        platform=platform.platform(),
        working_directory=os.getcwd(),
        packages=packages,
        cuda_available=cuda_available,
        torch_device_count=torch_device_count,
    )
    return asdict(report)


def write_environment_report(path: str | Path) -> None:
    """Write environment metadata as pretty JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(environment_report(), indent=2, sort_keys=True), encoding="utf-8")
