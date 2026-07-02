#!/usr/bin/env python3
"""
vgib_real_benchmarks.py

Reviewer-facing experimental suite for validating the Variational Geometric
Information Bottleneck (V-GIB) on real datasets.

What this script does
---------------------
1. Downloads or loads established real datasets.
2. Trains matched baselines and V-GIB variants under low-label regimes.
3. Computes predictive metrics, geometric diagnostics, and utility proxies.
4. Produces publication-ready CSV summaries and plots.
5. Optionally saves embedding visualizations.

Supported datasets
------------------
Image:
    - fashionmnist
    - cifar10
    - pcam
Tabular:
    - covtype
    - breast_cancer

Supported methods
-----------------
    - erm           : deterministic encoder, supervised only
    - vib           : variational bottleneck, no geometric penalty
    - vgib          : variational bottleneck + curvature + dimension penalty
    - vgib_no_curv  : V-GIB ablation without curvature term
    - vgib_no_dim   : V-GIB ablation without dimension term

Notes
-----
- The training objective is practical rather than a literal optimizer for the
  population utility U(phi) = I(phi(X);Y) - beta C(phi).
- The script uses tractable surrogates:
      * predictive information lower bound via H(Y) - cross_entropy
      * bottleneck compression via KL(q(z|x) || N(0, I))
      * geometry via Jacobian or Hutchinson-style Hessian proxy
      * intrinsic dimension via participation ratio
- The output "utility_proxy" is intended for diagnostics and comparisons, not
  as an exact estimate of the theoretical quantity.

Example full run
----------------
python vgib_real_benchmarks.py \
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

Recommended first smoke test
----------------------------
python vgib_real_benchmarks.py --root ./data --outdir ./runs/smoke \
    --datasets fashionmnist breast_cancer --methods erm vib vgib \
    --fractions 0.05 --seeds 13 --epochs 3 --batch-size 64 --download
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# Matplotlib only, as requested by system policy.
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_covtype, load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import CIFAR10, FashionMNIST
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchvision is required. Install it with: pip install torchvision"
    ) from exc


# =========================
# Configuration containers
# =========================

@dataclass
class RunConfig:
    root: str
    outdir: str
    datasets: List[str]
    methods: List[str]
    fractions: List[float]
    seeds: List[int]
    epochs: int
    batch_size: int
    latent_dim: int
    lr: float
    weight_decay: float
    num_workers: int
    device: str
    download: bool
    make_embeddings: bool
    image_size: int
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    curvature_mode: str
    curvature_probes: int
    curvature_batches_eval: int
    log_interval: int
    early_stop_patience: int
    amp: bool
    beta_kl: float
    beta_curv: float
    gamma_dim: float
    input_noise_std: float


@dataclass
class MethodConfig:
    name: str
    variational: bool
    beta_kl: float
    beta_curv: float
    gamma_dim: float


@dataclass
class DatasetBundle:
    name: str
    task_type: str  # image or tabular
    num_classes: int
    input_shape: Tuple[int, ...]
    train_data: Dataset
    val_data: Dataset
    test_data: Dataset
    class_names: Optional[List[str]] = None


# =========================
# Reproducibility utilities
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Filesystem helpers
# =========================


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =========================
# Dataset helpers
# =========================


class TransformedSubset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class WrappedDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = torch.from_numpy(xs).float()
        self.ys = torch.from_numpy(ys).long()

    def __len__(self) -> int:
        return self.xs.shape[0]

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]


class EnsureRGBTensor:
    def __init__(self, image_size: int):
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.resize(img)
        x = self.to_tensor(img)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x


class EnsureGrayTensor:
    def __init__(self, image_size: int):
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.resize(img)
        x = self.to_tensor(img)
        if x.shape[0] == 3:
            x = x.mean(dim=0, keepdim=True)
        return x


class IdentityTransform:
    def __call__(self, x):
        return x



def infer_targets(dataset: Dataset, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    # Torchvision datasets usually expose .targets, PCAM does not reliably expose one,
    # so we fall back to iteration when needed.
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        targets = np.asarray(targets)
    elif hasattr(dataset, "labels"):
        targets = np.asarray(getattr(dataset, "labels"))
    else:
        if indices is None:
            indices = list(range(len(dataset)))
        ys = []
        for idx in indices:
            _, y = dataset[idx]
            ys.append(int(y))
        return np.asarray(ys)
    if indices is not None:
        targets = targets[np.asarray(indices)]
    return targets.astype(int)



def stratified_split_indices(y: np.ndarray, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros_like(y), y))
    return train_idx, val_idx



def limit_training_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    if fraction >= 1.0:
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    train_idx, _ = next(splitter.split(np.zeros_like(y), y))
    return np.asarray(train_idx)



def maybe_cap_indices(indices: np.ndarray, y: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    y_sub = y[indices]
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    keep, _ = next(splitter.split(np.zeros_like(y_sub), y_sub))
    return indices[np.asarray(keep)]



def make_image_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "fashionmnist":
        train_tf = transforms.Compose([
            EnsureGrayTensor(image_size),
        ])
        eval_tf = transforms.Compose([
            EnsureGrayTensor(image_size),
        ])
    else:
        train_tf = transforms.Compose([
            EnsureRGBTensor(image_size),
        ])
        eval_tf = transforms.Compose([
            EnsureRGBTensor(image_size),
        ])
    return train_tf, eval_tf



def load_fashionmnist(root: str, image_size: int, download: bool, seed: int) -> DatasetBundle:
    train_tf, eval_tf = make_image_transforms("fashionmnist", image_size)
    base_train = FashionMNIST(root=root, train=True, transform=None, download=download)
    base_test = FashionMNIST(root=root, train=False, transform=None, download=download)
    y_train = np.asarray(base_train.targets)
    tr_idx, va_idx = stratified_split_indices(y_train, val_size=0.1, seed=seed)
    train_ds = TransformedSubset(base_train, tr_idx, transform=train_tf)
    val_ds = TransformedSubset(base_train, va_idx, transform=eval_tf)
    test_indices = np.arange(len(base_test))
    test_ds = TransformedSubset(base_test, test_indices, transform=eval_tf)
    return DatasetBundle(
        name="fashionmnist",
        task_type="image",
        num_classes=10,
        input_shape=(1, image_size, image_size),
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        class_names=[str(i) for i in range(10)],
    )



def load_cifar10(root: str, image_size: int, download: bool, seed: int) -> DatasetBundle:
    train_tf, eval_tf = make_image_transforms("cifar10", image_size)
    base_train = CIFAR10(root=root, train=True, transform=None, download=download)
    base_test = CIFAR10(root=root, train=False, transform=None, download=download)
    y_train = np.asarray(base_train.targets)
    tr_idx, va_idx = stratified_split_indices(y_train, val_size=0.1, seed=seed)
    train_ds = TransformedSubset(base_train, tr_idx, transform=train_tf)
    val_ds = TransformedSubset(base_train, va_idx, transform=eval_tf)
    test_indices = np.arange(len(base_test))
    test_ds = TransformedSubset(base_test, test_indices, transform=eval_tf)
    return DatasetBundle(
        name="cifar10",
        task_type="image",
        num_classes=10,
        input_shape=(3, image_size, image_size),
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        class_names=[str(c) for c in range(10)],
    )



def load_pcam(root: str, image_size: int, download: bool) -> DatasetBundle:
    # PCAM requires torchvision.datasets.PCAM, h5py, and gdown for downloading.
    try:
        from torchvision.datasets import PCAM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PCAM support is unavailable in your torchvision build. "
            "Please upgrade torchvision."
        ) from exc
    try:
        import h5py  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PCAM requires h5py. Install it with: pip install h5py") from exc

    train_tf, eval_tf = make_image_transforms("pcam", image_size)
    train_ds = PCAM(root=root, split="train", transform=train_tf, download=download)
    val_ds = PCAM(root=root, split="val", transform=eval_tf, download=download)
    test_ds = PCAM(root=root, split="test", transform=eval_tf, download=download)
    return DatasetBundle(
        name="pcam",
        task_type="image",
        num_classes=2,
        input_shape=(3, image_size, image_size),
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        class_names=["normal", "metastatic"],
    )



def load_covtype(seed: int) -> DatasetBundle:
    data = fetch_covtype(return_X_y=True, shuffle=True, random_state=seed)
    X, y = data
    y = y.astype(np.int64) - 1  # convert classes 1..7 to 0..6
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=seed
    )
    return DatasetBundle(
        name="covtype",
        task_type="tabular",
        num_classes=7,
        input_shape=(X.shape[1],),
        train_data=WrappedDataset(X_train, y_train),
        val_data=WrappedDataset(X_val, y_val),
        test_data=WrappedDataset(X_test, y_test),
        class_names=[str(i) for i in range(7)],
    )



def load_breast_cancer_bundle(seed: int) -> DatasetBundle:
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=seed
    )
    return DatasetBundle(
        name="breast_cancer",
        task_type="tabular",
        num_classes=2,
        input_shape=(X.shape[1],),
        train_data=WrappedDataset(X_train, y_train),
        val_data=WrappedDataset(X_val, y_val),
        test_data=WrappedDataset(X_test, y_test),
        class_names=["benign", "malignant"],
    )



def load_dataset_bundle(name: str, root: str, image_size: int, download: bool, seed: int) -> DatasetBundle:
    lname = name.lower()
    if lname == "fashionmnist":
        return load_fashionmnist(root=root, image_size=image_size, download=download, seed=seed)
    if lname == "cifar10":
        return load_cifar10(root=root, image_size=image_size, download=download, seed=seed)
    if lname == "pcam":
        return load_pcam(root=root, image_size=image_size, download=download)
    if lname == "covtype":
        return load_covtype(seed=seed)
    if lname == "breast_cancer":
        return load_breast_cancer_bundle(seed=seed)
    raise ValueError(f"Unknown dataset: {name}")



def get_subset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, TransformedSubset):
        return infer_targets(dataset.base, dataset.indices)
    if isinstance(dataset, WrappedDataset):
        return dataset.ys.numpy().astype(int)
    return infer_targets(dataset)



def subset_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    if isinstance(dataset, TransformedSubset):
        new_indices = np.asarray(dataset.indices)[indices]
        return TransformedSubset(dataset.base, new_indices, transform=dataset.transform)
    return Subset(dataset, indices.tolist())


# =========================
# Model definitions
# =========================


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, variational: bool):
        super().__init__()
        self.variational = variational
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        mu = self.fc_mu(h)
        if self.variational:
            logvar = self.fc_logvar(h)
        else:
            logvar = torch.zeros_like(mu)
        return mu, logvar


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, variational: bool):
        super().__init__()
        self.variational = variational
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.fc_mu(h)
        if self.variational:
            logvar = self.fc_logvar(h)
        else:
            logvar = torch.zeros_like(mu)
        return mu, logvar


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentClassifier(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, latent_dim: int, variational: bool, task_type: str):
        super().__init__()
        self.task_type = task_type
        if task_type == "image":
            self.encoder = ConvEncoder(input_shape[0], latent_dim, variational)
        else:
            self.encoder = TabularEncoder(input_shape[0], latent_dim, variational)
        self.head = ClassifierHead(latent_dim, num_classes)
        self.variational = variational

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.variational:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.head(z)
        return {"logits": logits, "z": z, "mu": mu, "logvar": logvar}


# =========================
# Objective pieces
# =========================


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))



def participation_ratio(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    if zc.shape[0] < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    cov = zc.T @ zc / max(zc.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals.real, min=eps)
    return (eigvals.sum() ** 2) / (torch.sum(eigvals**2) + eps)



def jacobian_proxy(mu: torch.Tensor, x: torch.Tensor, probes: int = 1) -> torch.Tensor:
    # Hutchinson-style estimator for ||J_mu(x)||_F^2 using random output projections.
    total = 0.0
    for _ in range(probes):
        r = torch.randn_like(mu)
        scalar = torch.sum(mu * r)
        grad_x = torch.autograd.grad(scalar, x, create_graph=True, retain_graph=True)[0]
        total = total + torch.mean(torch.sum(grad_x.reshape(grad_x.shape[0], -1) ** 2, dim=1))
    return total / probes



def hessian_proxy(mu: torch.Tensor, x: torch.Tensor, probes: int = 1) -> torch.Tensor:
    # More expensive. Estimates a Hessian-based curvature proxy using nested HVPs.
    total = 0.0
    for _ in range(probes):
        r = torch.randn_like(mu)
        scalar = torch.sum(mu * r)
        grad_x = torch.autograd.grad(scalar, x, create_graph=True, retain_graph=True)[0]
        v = torch.randn_like(x)
        hvp = torch.autograd.grad(torch.sum(grad_x * v), x, create_graph=True, retain_graph=True)[0]
        total = total + torch.mean(torch.sum(hvp.reshape(hvp.shape[0], -1) ** 2, dim=1))
    return total / probes



def geometry_penalty(
    mu: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    curvature_mode: str,
    curvature_probes: int,
    latent_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if curvature_mode == "jacobian":
        curv = jacobian_proxy(mu, x, probes=curvature_probes)
    elif curvature_mode == "hessian":
        curv = hessian_proxy(mu, x, probes=curvature_probes)
    else:
        raise ValueError(f"Unsupported curvature mode: {curvature_mode}")
    dim_ratio = participation_ratio(z) / max(float(latent_dim), 1.0)
    return curv, dim_ratio



def empirical_label_entropy(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


# =========================
# Evaluation helpers
# =========================


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()



def compute_auroc(y_true: np.ndarray, probas: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probas[:, 1]))
        return float(roc_auc_score(y_true, probas, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


@torch.no_grad()

def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    ys_true: List[np.ndarray] = []
    ys_pred: List[np.ndarray] = []
    probas: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    mus: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        ys_true.append(to_numpy(yb))
        ys_pred.append(to_numpy(preds))
        probas.append(to_numpy(probs))
        zs.append(to_numpy(out["z"]))
        mus.append(to_numpy(out["mu"]))
    return {
        "y_true": np.concatenate(ys_true, axis=0),
        "y_pred": np.concatenate(ys_pred, axis=0),
        "probas": np.concatenate(probas, axis=0),
        "z": np.concatenate(zs, axis=0),
        "mu": np.concatenate(mus, axis=0),
    }



def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_entropy: float,
    latent_dim: int,
    curvature_mode: str,
    curvature_probes: int,
    beta_kl: float,
    beta_curv: float,
    gamma_dim: float,
    curvature_batches_eval: int,
) -> Dict[str, float]:
    preds = collect_predictions(model, loader, device)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    probas = preds["probas"]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auroc = compute_auroc(y_true, probas, num_classes)

    # Cross-entropy on saved probabilities.
    ce = -np.mean(np.log(np.clip(probas[np.arange(len(y_true)), y_true], 1e-12, 1.0)))

    # Geometry diagnostics computed on a limited number of batches.
    model.eval()
    curv_vals: List[float] = []
    dim_vals: List[float] = []
    kl_vals: List[float] = []

    num_done = 0
    for xb, _ in loader:
        if num_done >= curvature_batches_eval:
            break
        xb = xb.to(device)
        xb = xb.requires_grad_(True)
        out = model(xb)
        kl = kl_to_standard_normal(out["mu"], out["logvar"]) if model.variational else torch.tensor(0.0, device=device)
        curv, dim_ratio = geometry_penalty(
            out["mu"], out["z"], xb, curvature_mode=curvature_mode,
            curvature_probes=curvature_probes, latent_dim=latent_dim,
        )
        curv_vals.append(float(curv.detach().cpu()))
        dim_vals.append(float(dim_ratio.detach().cpu()))
        kl_vals.append(float(kl.detach().cpu()))
        num_done += 1

    avg_curv = float(np.mean(curv_vals)) if curv_vals else float("nan")
    avg_dim = float(np.mean(dim_vals)) if dim_vals else float("nan")
    avg_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
    mi_lower_bound = float(label_entropy - ce)
    utility_proxy = float(mi_lower_bound - beta_kl * avg_kl - beta_curv * avg_curv - gamma_dim * avg_dim)
    interpretive_efficiency = float(utility_proxy / max(len(y_true), 1))

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc),
        "cross_entropy": float(ce),
        "mi_lower_bound": float(mi_lower_bound),
        "kl": float(avg_kl),
        "curvature_proxy": float(avg_curv),
        "dim_ratio": float(avg_dim),
        "utility_proxy": float(utility_proxy),
        "interpretive_efficiency": float(interpretive_efficiency),
    }


# =========================
# Training loop
# =========================


def build_method_config(name: str, base_cfg: RunConfig) -> MethodConfig:
    lname = name.lower()
    if lname == "erm":
        return MethodConfig(name=lname, variational=False, beta_kl=0.0, beta_curv=0.0, gamma_dim=0.0)
    if lname == "vib":
        return MethodConfig(name=lname, variational=True, beta_kl=base_cfg.beta_kl, beta_curv=0.0, gamma_dim=0.0)
    if lname == "vgib":
        return MethodConfig(name=lname, variational=True, beta_kl=base_cfg.beta_kl, beta_curv=base_cfg.beta_curv, gamma_dim=base_cfg.gamma_dim)
    if lname == "vgib_no_curv":
        return MethodConfig(name=lname, variational=True, beta_kl=base_cfg.beta_kl, beta_curv=0.0, gamma_dim=base_cfg.gamma_dim)
    if lname == "vgib_no_dim":
        return MethodConfig(name=lname, variational=True, beta_kl=base_cfg.beta_kl, beta_curv=base_cfg.beta_curv, gamma_dim=0.0)
    raise ValueError(f"Unknown method: {name}")



def build_model(bundle: DatasetBundle, method_cfg: MethodConfig, latent_dim: int) -> LatentClassifier:
    return LatentClassifier(
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        latent_dim=latent_dim,
        variational=method_cfg.variational,
        task_type=bundle.task_type,
    )



def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)



def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )



def train_one_configuration(
    bundle: DatasetBundle,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    method_cfg: MethodConfig,
    run_cfg: RunConfig,
    seed: int,
    fraction: float,
    outdir: Path,
) -> Dict[str, float]:
    set_seed(seed)
    device = device_from_arg(run_cfg.device)
    model = build_model(bundle, method_cfg, run_cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(run_cfg.amp and device.type == "cuda"))

    train_loader = make_loader(train_dataset, run_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers)
    val_loader = make_loader(val_dataset, run_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers)
    test_loader = make_loader(test_dataset, run_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers)

    y_train = get_subset_targets(train_dataset)
    label_entropy = empirical_label_entropy(y_train)

    best_state = None
    best_val = -float("inf")
    patience = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, run_cfg.epochs + 1):
        model.train()
        t0 = time.time()
        loss_meter = []
        ce_meter = []
        kl_meter = []
        curv_meter = []
        dim_meter = []

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            if run_cfg.input_noise_std > 0:
                xb = xb + run_cfg.input_noise_std * torch.randn_like(xb)
            need_geom = (method_cfg.beta_curv > 0.0) or (method_cfg.gamma_dim > 0.0)
            xb = xb.requires_grad_(need_geom)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(run_cfg.amp and device.type == "cuda")):
                out = model(xb)
                logits = out["logits"]
                ce = F.cross_entropy(logits, yb)
                kl = kl_to_standard_normal(out["mu"], out["logvar"]) if method_cfg.variational else torch.tensor(0.0, device=device)
                if need_geom:
                    curv, dim_ratio = geometry_penalty(
                        out["mu"], out["z"], xb,
                        curvature_mode=run_cfg.curvature_mode,
                        curvature_probes=run_cfg.curvature_probes,
                        latent_dim=run_cfg.latent_dim,
                    )
                else:
                    curv = torch.tensor(0.0, device=device)
                    dim_ratio = torch.tensor(0.0, device=device)
                loss = ce + method_cfg.beta_kl * kl + method_cfg.beta_curv * curv + method_cfg.gamma_dim * dim_ratio

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.append(float(loss.detach().cpu()))
            ce_meter.append(float(ce.detach().cpu()))
            kl_meter.append(float(kl.detach().cpu()))
            curv_meter.append(float(curv.detach().cpu()))
            dim_meter.append(float(dim_ratio.detach().cpu()))

            if run_cfg.log_interval > 0 and step % run_cfg.log_interval == 0:
                print(
                    f"[{bundle.name}][{method_cfg.name}][frac={fraction:.2f}][seed={seed}] "
                    f"epoch {epoch:03d} step {step:04d} "
                    f"loss={np.mean(loss_meter):.4f} ce={np.mean(ce_meter):.4f} "
                    f"kl={np.mean(kl_meter):.4f} curv={np.mean(curv_meter):.4f} dim={np.mean(dim_meter):.4f}"
                )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=bundle.num_classes,
            label_entropy=label_entropy,
            latent_dim=run_cfg.latent_dim,
            curvature_mode=run_cfg.curvature_mode,
            curvature_probes=1,
            beta_kl=method_cfg.beta_kl,
            beta_curv=method_cfg.beta_curv,
            gamma_dim=method_cfg.gamma_dim,
            curvature_batches_eval=max(1, min(2, run_cfg.curvature_batches_eval)),
        )
        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(loss_meter)),
            "train_ce": float(np.mean(ce_meter)),
            "train_kl": float(np.mean(kl_meter)),
            "train_curvature": float(np.mean(curv_meter)),
            "train_dim_ratio": float(np.mean(dim_meter)),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_auroc": val_metrics["auroc"],
            "val_utility_proxy": val_metrics["utility_proxy"],
            "seconds": epoch_time,
        }
        history.append(row)
        print(
            f"[{bundle.name}][{method_cfg.name}][frac={fraction:.2f}][seed={seed}] "
            f"epoch {epoch:03d} done in {epoch_time:.1f}s | "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"val_auc={val_metrics['auroc']:.4f} val_u={val_metrics['utility_proxy']:.4f}"
        )

        # Select by validation utility first, then accuracy.
        selection_score = (val_metrics["utility_proxy"], val_metrics["accuracy"])
        current_best = (best_val, -float("inf"))
        if selection_score > current_best:
            best_val = selection_score[0]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= run_cfg.early_stop_patience:
                print(
                    f"[{bundle.name}][{method_cfg.name}][frac={fraction:.2f}][seed={seed}] "
                    f"early stopping at epoch {epoch}."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=bundle.num_classes,
        label_entropy=label_entropy,
        latent_dim=run_cfg.latent_dim,
        curvature_mode=run_cfg.curvature_mode,
        curvature_probes=run_cfg.curvature_probes,
        beta_kl=method_cfg.beta_kl,
        beta_curv=method_cfg.beta_curv,
        gamma_dim=method_cfg.gamma_dim,
        curvature_batches_eval=run_cfg.curvature_batches_eval,
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=bundle.num_classes,
        label_entropy=label_entropy,
        latent_dim=run_cfg.latent_dim,
        curvature_mode=run_cfg.curvature_mode,
        curvature_probes=run_cfg.curvature_probes,
        beta_kl=method_cfg.beta_kl,
        beta_curv=method_cfg.beta_curv,
        gamma_dim=method_cfg.gamma_dim,
        curvature_batches_eval=run_cfg.curvature_batches_eval,
    )

    # Save artifacts.
    run_name = f"{bundle.name}__{method_cfg.name}__frac_{fraction:.2f}__seed_{seed}"
    run_dir = ensure_dir(outdir / run_name)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": bundle.name,
                "method": asdict(method_cfg),
                "fraction": fraction,
                "seed": seed,
                "run_config": asdict(run_cfg),
                "input_shape": bundle.input_shape,
                "num_classes": bundle.num_classes,
            },
            f,
            indent=2,
        )

    result = {
        "dataset": bundle.name,
        "method": method_cfg.name,
        "fraction": fraction,
        "seed": seed,
        "n_train_labels": len(train_dataset),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    preds = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(preds["y_true"], preds["y_pred"])
    np.save(run_dir / "confusion_matrix.npy", cm)
    with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(preds["y_true"], preds["y_pred"], digits=4))

    if run_cfg.make_embeddings:
        save_embedding_plot(run_dir, preds["mu"], preds["y_true"], title=run_name)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# =========================
# Plotting
# =========================


def save_embedding_plot(run_dir: Path, features: np.ndarray, labels: np.ndarray, title: str) -> None:
    n = min(len(features), 2000)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(features), size=n, replace=False)
    X = features[idx]
    y = labels[idx]

    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=123).fit_transform(X)
    emb = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=123, perplexity=30).fit_transform(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], c=y, s=6, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(run_dir / "embedding_tsne.png", dpi=200)
    plt.close()



def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {"dataset", "method", "fraction", "seed"}]
    grouped = df.groupby(["dataset", "method", "fraction"], dropna=False)
    mean_df = grouped[numeric_cols].mean().reset_index()
    std_df = grouped[numeric_cols].std().reset_index()
    std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
    merged = mean_df.merge(std_df, on=["dataset", "method", "fraction"], how="left")
    return merged



def plot_metric_curves(summary_df: pd.DataFrame, outdir: Path, metric: str, ylabel: str) -> None:
    for dataset in sorted(summary_df["dataset"].unique()):
        sub = summary_df[summary_df["dataset"] == dataset].copy()
        plt.figure(figsize=(6, 4))
        for method in sorted(sub["method"].unique()):
            ss = sub[sub["method"] == method].sort_values("fraction")
            plt.plot(ss["fraction"], ss[metric], marker="o", label=method)
        plt.xlabel("Label fraction")
        plt.ylabel(ylabel)
        plt.title(f"{dataset}: {ylabel} vs label fraction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{dataset}__{metric}.png", dpi=200)
        plt.close()



def plot_scatter(summary_df: pd.DataFrame, outdir: Path, x: str, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    for method in sorted(summary_df["method"].unique()):
        ss = summary_df[summary_df["method"] == method]
        plt.scatter(ss[x], ss[y], label=method, alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=200)
    plt.close()



def write_markdown_summary(raw_df: pd.DataFrame, summary_df: pd.DataFrame, outdir: Path) -> None:
    lines = []
    lines.append("# V-GIB Real-Benchmark Summary")
    lines.append("")
    lines.append(f"Total runs: **{len(raw_df)}**")
    lines.append("")
    for dataset in sorted(summary_df["dataset"].unique()):
        lines.append(f"## {dataset}")
        sub = summary_df[summary_df["dataset"] == dataset].copy()
        best = sub.sort_values(["test_accuracy", "test_utility_proxy"], ascending=False).head(1)
        if len(best) > 0:
            row = best.iloc[0]
            lines.append(
                f"Best mean test accuracy: **{row['test_accuracy']:.4f}** "
                f"with **{row['method']}** at fraction **{row['fraction']:.2f}**."
            )
            lines.append(
                f"Mean test utility proxy: **{row['test_utility_proxy']:.4f}**; "
                f"curvature proxy: **{row['test_curvature_proxy']:.4f}**; "
                f"dim ratio: **{row['test_dim_ratio']:.4f}**."
            )
        lines.append("")
    with open(outdir / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# Orchestration
# =========================


def prepare_fraction_datasets(
    bundle: DatasetBundle,
    fraction: float,
    seed: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
) -> Tuple[Dataset, Dataset, Dataset]:
    y_train = get_subset_targets(bundle.train_data)
    base_train_idx = limit_training_indices(y_train, fraction=fraction, seed=seed)
    base_train_idx = maybe_cap_indices(base_train_idx, y_train, max_train_samples, seed=seed)
    train_ds = subset_dataset(bundle.train_data, base_train_idx)

    y_val = get_subset_targets(bundle.val_data)
    val_idx = np.arange(len(y_val))
    val_idx = maybe_cap_indices(val_idx, y_val, max_eval_samples, seed=seed)
    val_ds = subset_dataset(bundle.val_data, val_idx)

    y_test = get_subset_targets(bundle.test_data)
    test_idx = np.arange(len(y_test))
    test_idx = maybe_cap_indices(test_idx, y_test, max_eval_samples, seed=seed)
    test_ds = subset_dataset(bundle.test_data, test_idx)

    return train_ds, val_ds, test_ds



def run_experiments(run_cfg: RunConfig) -> None:
    root = ensure_dir(run_cfg.root)
    outdir = ensure_dir(run_cfg.outdir)

    all_results: List[Dict[str, float]] = []

    for dataset_name in run_cfg.datasets:
        print(f"\n=== Loading dataset: {dataset_name} ===")
        # Use the first seed for dataset split creation where relevant.
        bundle = load_dataset_bundle(
            name=dataset_name,
            root=str(root),
            image_size=run_cfg.image_size,
            download=run_cfg.download,
            seed=run_cfg.seeds[0],
        )

        for fraction in run_cfg.fractions:
            for seed in run_cfg.seeds:
                train_ds, val_ds, test_ds = prepare_fraction_datasets(
                    bundle=bundle,
                    fraction=fraction,
                    seed=seed,
                    max_train_samples=run_cfg.max_train_samples,
                    max_eval_samples=run_cfg.max_eval_samples,
                )

                for method_name in run_cfg.methods:
                    method_cfg = build_method_config(method_name, run_cfg)
                    print(
                        f"\n--- dataset={dataset_name} method={method_name} "
                        f"fraction={fraction:.2f} seed={seed} ---"
                    )
                    result = train_one_configuration(
                        bundle=bundle,
                        train_dataset=train_ds,
                        val_dataset=val_ds,
                        test_dataset=test_ds,
                        method_cfg=method_cfg,
                        run_cfg=run_cfg,
                        seed=seed,
                        fraction=fraction,
                        outdir=outdir,
                    )
                    all_results.append(result)

    if not all_results:
        raise RuntimeError("No results were produced.")

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(outdir / "all_runs.csv", index=False)
    summary_df = aggregate_results(raw_df)
    summary_df.to_csv(outdir / "summary_mean_std.csv", index=False)

    plot_metric_curves(summary_df, outdir, metric="test_accuracy", ylabel="Test accuracy")
    plot_metric_curves(summary_df, outdir, metric="test_macro_f1", ylabel="Test macro-F1")
    plot_metric_curves(summary_df, outdir, metric="test_auroc", ylabel="Test AUROC")
    plot_metric_curves(summary_df, outdir, metric="test_utility_proxy", ylabel="Test utility proxy")
    plot_metric_curves(summary_df, outdir, metric="test_curvature_proxy", ylabel="Curvature proxy")
    plot_metric_curves(summary_df, outdir, metric="test_dim_ratio", ylabel="Dimension ratio")

    plot_scatter(
        summary_df,
        outdir,
        x="test_curvature_proxy",
        y="test_accuracy",
        title="Accuracy vs curvature proxy",
        filename="scatter_accuracy_vs_curvature.png",
    )
    plot_scatter(
        summary_df,
        outdir,
        x="test_dim_ratio",
        y="test_accuracy",
        title="Accuracy vs dimension ratio",
        filename="scatter_accuracy_vs_dimratio.png",
    )
    plot_scatter(
        summary_df,
        outdir,
        x="test_utility_proxy",
        y="test_accuracy",
        title="Accuracy vs utility proxy",
        filename="scatter_accuracy_vs_utility.png",
    )

    write_markdown_summary(raw_df, summary_df, outdir)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(run_cfg), f, indent=2)

    print(f"\nDone. Results written to: {outdir}")


# =========================
# Argument parsing
# =========================


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Real-benchmark V-GIB validation suite")
    parser.add_argument("--root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--outdir", type=str, default="./runs/vgib_real", help="Output directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "pcam", "covtype"],
        choices=["fashionmnist", "cifar10", "pcam", "covtype", "breast_cancer"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["erm", "vib", "vgib", "vgib_no_curv", "vgib_no_dim"],
        choices=["erm", "vib", "vgib", "vgib_no_curv", "vgib_no_dim"],
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--download", action="store_true", help="Download datasets if needed")
    parser.add_argument("--make-embeddings", action="store_true", help="Save t-SNE embeddings")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--curvature-mode", type=str, default="jacobian", choices=["jacobian", "hessian"])
    parser.add_argument("--curvature-probes", type=int, default=1)
    parser.add_argument("--curvature-batches-eval", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision when possible")
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--beta-curv", type=float, default=1e-3)
    parser.add_argument("--gamma-dim", type=float, default=1e-2)
    parser.add_argument("--input-noise-std", type=float, default=0.0)

    args = parser.parse_args(argv)
    return RunConfig(
        root=args.root,
        outdir=args.outdir,
        datasets=list(args.datasets),
        methods=list(args.methods),
        fractions=list(args.fractions),
        seeds=list(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        download=args.download,
        make_embeddings=args.make_embeddings,
        image_size=args.image_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        curvature_mode=args.curvature_mode,
        curvature_probes=args.curvature_probes,
        curvature_batches_eval=args.curvature_batches_eval,
        log_interval=args.log_interval,
        early_stop_patience=args.early_stop_patience,
        amp=args.amp,
        beta_kl=args.beta_kl,
        beta_curv=args.beta_curv,
        gamma_dim=args.gamma_dim,
        input_noise_std=args.input_noise_std,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = parse_args(argv)
    run_experiments(cfg)


if __name__ == "__main__":
    main()
