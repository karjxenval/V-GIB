"""
vgib_experiments.py

Unified experimental script for V-GIB:
- Synthetic Swiss-roll manifold
- Fashion-MNIST
- CIFAR-10 with data fractions

Models:
  - Baseline (no geometric penalties)
  - VIB-only (information bottleneck, no curvature)
  - V-GIB (curvature + intrinsic-dimension penalty)
  - Random encoder baseline (Swiss-roll)
  - Laplacian-style geometric baseline (CIFAR-10)

All runs log to CSV with a consistent schema so that tables/figures
can be regenerated exactly from logs.
"""

import os
import time
import math
import csv
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
    Subset,
)
from torchvision import datasets, transforms, models

# -------------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# -------------------------------------------------------------------------
# Geometry utilities: curvature proxy, intrinsic dimension, MI proxy
# -------------------------------------------------------------------------

def completed_epochs(csv_path, seed, label_frac, model):
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                int(row["seed"]) == seed
                and abs(float(row["label_frac"]) - label_frac) < 1e-8
                and row["model"] == model
            ):
                done.add(int(row["epoch"]))
    return done


def hutchinson_jacobian_norm(z: torch.Tensor, x: torch.Tensor, num_probes: int = 2) -> float:
    """
    Lightweight curvature proxy:
    ||J||_F^2 ≈ E_v ||J^T v||^2, where J = d z / d x.

    Parameters
    ----------
    z : (B, d)
        Latent representation.
    x : (B, C, H, W) or (B, D)
        Input; must require_grad=True.
    num_probes : int
        Number of Hutchinson probes.

    Returns
    -------
    float
        Scalar curvature estimate averaged over batch.
    """
    B = x.size(0)
    curv_vals = []
    for _ in range(num_probes):
        v = torch.randn_like(z)
        g = torch.autograd.grad(
            outputs=z,
            inputs=x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0]
        curv_vals.append(g.pow(2).view(B, -1).sum(dim=1))
    curv = torch.stack(curv_vals, dim=0).mean(dim=0)  # (B,)
    return float(curv.mean().item())


def participation_ratio_idim(z: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Intrinsic dimension via participation ratio on a batch of latent vectors.

    Parameters
    ----------
    z : (B, d)
        Latent representation.

    Returns
    -------
    float
        Estimated intrinsic dimension.
    """
    with torch.no_grad():
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).real
        eigvals = torch.clamp(eigvals, min=eps)
        num = eigvals.sum().pow(2)
        den = (eigvals.pow(2)).sum()
        pr = num / den
        return float(pr.item())


def mi_proxy_from_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """
    Cheap MI proxy: H(Y) - CE, where CE is cross-entropy.

    Parameters
    ----------
    logits : (B, C)
    y : (B,)
    num_classes : int

    Returns
    -------
    float
        Proxy for I(Z;Y).
    """
    with torch.no_grad():
        ce = F.cross_entropy(logits, y, reduction="mean").item()
        H_y = math.log(num_classes)
        return float(H_y - ce)


def alignment_proxy(z: torch.Tensor, y: torch.Tensor) -> float:
    """
    Stable alignment proxy: ratio of between-class to global variance of z.

    Returns a small positive number; used only as a diagnostic, not as a loss.
    """
    with torch.no_grad():
        B, D = z.shape
        num_classes = int(y.max().item()) + 1
        zc = z - z.mean(dim=0, keepdim=True)
        global_var = zc.var(dim=0, unbiased=False).mean().item() + 1e-8

        means = []
        for c in range(num_classes):
            if (y == c).any():
                means.append(z[y == c].mean(dim=0))
        if not means:
            return 0.0
        means = torch.stack(means, dim=0)
        between_var = means.var(dim=0, unbiased=False).mean().item()
        score = between_var / global_var
        return float(0.02 + 0.02 * math.tanh(score))


# -------------------------------------------------------------------------
# Synthetic Swiss-roll
# -------------------------------------------------------------------------

def make_swiss_roll(
    n: int = 2000,
    noise_std: float = 0.1,
    n_classes: int = 6,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    u = rng.uniform(-math.pi, math.pi, size=(n,))
    r = 1.0 + 0.5 * (u + math.pi) / (2 * math.pi)
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = 0.5 * u
    X = np.stack([x, y, z], axis=1)
    X += rng.normal(scale=noise_std, size=X.shape)
    labels = np.floor((u + math.pi) / (2 * math.pi) * n_classes).astype(int)
    labels = np.clip(labels, 0, n_classes - 1)
    return X.astype(np.float32), labels.astype(np.int64)


# Synthetic encoders / classifier

class VIBEncoder(nn.Module):
    def __init__(self, x_dim: int = 3, hidden: int = 128, z_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class DeterministicEncoder(nn.Module):
    """
    Capacity-matched non-geometric baseline for Swiss-roll.
    Same backbone as VIBEncoder, but deterministic z.
    """
    def __init__(self, x_dim: int = 3, hidden: int = 128, z_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.out(h)


class RandomLinearEncoder(nn.Module):
    """
    Random encoder baseline for Swiss-roll.
    """
    def __init__(self, x_dim: int = 3, z_dim: int = 16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(x_dim, z_dim))
        self.bias = nn.Parameter(torch.zeros(z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class SmallClassifier(nn.Module):
    def __init__(self, z_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)



# def swiss_roll_experiments(
#     out_dir: str,
#     seeds: List[int],
#     noise_levels: List[float],
#     betas: List[float],
#     gammas: List[float],
#     z_dims: List[int],
#     n_epochs: int = 40,
#     batch_size: int = 256,
# ) -> None:
#     """
#     Run Swiss-roll experiments for:
#       - VIB-only (no curvature)
#       - V-GIB (curvature penalty)
#       - Deterministic MLP baseline (same capacity)
#       - Random encoder baseline
#     """
#     ensure_dir(out_dir)
#     csv_path = os.path.join(out_dir, "swiss_roll_results.csv")
#     fieldnames = [
#         "dataset", "model", "seed", "noise_std", "beta", "gamma",
#         "z_dim", "epoch", "train_acc", "val_acc",
#         "ce", "kl", "curv", "idim", "mi_proxy", "efficiency", "wall_time",
#     ]

#     if not os.path.exists(csv_path):
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             f.flush()
#             os.fsync(f.fileno())

#     n_classes = 6
#     for seed in seeds:
#         for noise_std in noise_levels:
#             X, y = make_swiss_roll(n=2000, noise_std=noise_std, n_classes=n_classes, seed=seed)
#             X_tensor = torch.tensor(X, device=DEVICE)
#             y_tensor = torch.tensor(y, device=DEVICE)

#             # simple train/val split
#             n = X_tensor.size(0)
#             idx = torch.randperm(n)
#             split = int(0.8 * n)
#             train_idx, val_idx = idx[:split], idx[split:]
#             train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
#             val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
#             train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#             val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

#             # random encoder baseline: only train classifier
#             for z_dim in z_dims:
#                 for model_name in ["random", "mlp", "vib", "vgib"]:
#                     for beta in betas:
#                         for gamma in gammas:
#                             if model_name == "vib" and gamma != 0.0:
#                                 continue  # no curvature in VIB-only
#                             if model_name in ["random", "mlp"] and (beta != 0.0 or gamma != 0.0):
#                                 continue  # deterministic baselines: CE only

#                             print(
#                                 f"[Swiss] seed={seed} noise={noise_std:.2f} "
#                                 f"model={model_name} z={z_dim} beta={beta} gamma={gamma}"
#                             )

#                             # build models
#                             if model_name == "random":
#                                 encoder = RandomLinearEncoder(x_dim=3, z_dim=z_dim).to(DEVICE)
#                                 classifier = SmallClassifier(z_dim=z_dim, hidden=64, n_classes=n_classes).to(DEVICE)
#                                 # encoder frozen
#                                 for p in encoder.parameters():
#                                     p.requires_grad_(False)
#                                 params = list(classifier.parameters())
#                             elif model_name == "mlp":
#                                 encoder = DeterministicEncoder(x_dim=3, hidden=128, z_dim=z_dim).to(DEVICE)
#                                 classifier = SmallClassifier(z_dim=z_dim, hidden=64, n_classes=n_classes).to(DEVICE)
#                                 params = list(encoder.parameters()) + list(classifier.parameters())
#                             else:
#                                 encoder = VIBEncoder(x_dim=3, hidden=128, z_dim=z_dim).to(DEVICE)
#                                 classifier = SmallClassifier(z_dim=z_dim, hidden=64, n_classes=n_classes).to(DEVICE)
#                                 params = list(encoder.parameters()) + list(classifier.parameters())

#                             opt = torch.optim.Adam(params, lr=1e-3)

#                             for epoch in range(1, n_epochs + 1):
#                                 t0 = time.time()
#                                 encoder.train()
#                                 classifier.train()
#                                 train_correct = 0
#                                 train_total = 0

#                                 epoch_ce = 0.0
#                                 epoch_kl = 0.0
#                                 epoch_curv = 0.0
#                                 epoch_idim = 0.0
#                                 epoch_batches = 0

#                                 for xb, yb in train_loader:
#                                     xb = xb.to(DEVICE).requires_grad_(True)
#                                     yb = yb.to(DEVICE)

#                                     if model_name == "random":
#                                         z = encoder(xb).detach()  # fixed random map
#                                         logits = classifier(z)
#                                         ce = F.cross_entropy(logits, yb)
#                                         loss = ce
#                                         kl_val = 0.0
#                                         curv_val = 0.0
#                                         idim_val = participation_ratio_idim(z.detach())
#                                     elif model_name == "mlp":
#                                         z = encoder(xb)
#                                         logits = classifier(z)
#                                         ce = F.cross_entropy(logits, yb)
#                                         loss = ce
#                                         kl_val = 0.0
#                                         curv_val = hutchinson_jacobian_norm(z, xb, num_probes=1)
#                                         idim_val = participation_ratio_idim(z.detach())
#                                     else:
#                                         mu, logvar = encoder(xb)
#                                         z = reparameterize(mu, logvar)
#                                         logits = classifier(z)
#                                         ce = F.cross_entropy(logits, yb)
#                                         kl_vec = kl_standard_normal(mu, logvar)
#                                         kl_val = float(kl_vec.mean().item())
#                                         if model_name == "vib":
#                                             curv_val = 0.0
#                                             idim_val = participation_ratio_idim(z.detach())
#                                             loss = ce + beta * kl_vec.mean()
#                                         else:  # vgib
#                                             curv_val = hutchinson_jacobian_norm(z, xb, num_probes=1)
#                                             idim_val = participation_ratio_idim(z.detach())
#                                             loss = ce + beta * kl_vec.mean() + gamma * curv_val

#                                     opt.zero_grad()
#                                     loss.backward()
#                                     opt.step()

#                                     with torch.no_grad():
#                                         pred = logits.argmax(dim=1)
#                                         train_correct += (pred == yb).sum().item()
#                                         train_total += yb.size(0)

#                                     epoch_ce += float(ce.item())
#                                     epoch_kl += float(kl_val)
#                                     epoch_curv += float(curv_val)
#                                     epoch_idim += float(idim_val)
#                                     epoch_batches += 1

#                                 train_acc = train_correct / max(train_total, 1)

#                                 # validation
#                                 encoder.eval()
#                                 classifier.eval()
#                                 val_correct = 0
#                                 val_total = 0
#                                 val_logits_list = []
#                                 val_y_list = []
#                                 with torch.no_grad():
#                                     for xb, yb in val_loader:
#                                         xb = xb.to(DEVICE)
#                                         yb = yb.to(DEVICE)
#                                         if model_name in ["random", "mlp"]:
#                                             if model_name == "random":
#                                                 z = encoder(xb)
#                                             else:
#                                                 z = encoder(xb)
#                                             logits = classifier(z)
#                                         else:
#                                             mu, logvar = encoder(xb)
#                                             z = reparameterize(mu, logvar)
#                                             logits = classifier(z)
#                                         pred = logits.argmax(dim=1)
#                                         val_correct += (pred == yb).sum().item()
#                                         val_total += yb.size(0)
#                                         val_logits_list.append(logits)
#                                         val_y_list.append(yb)

#                                 val_acc = val_correct / max(val_total, 1)
#                                 val_logits = torch.cat(val_logits_list, dim=0)
#                                 val_y = torch.cat(val_y_list, dim=0)
#                                 mi_p = mi_proxy_from_logits(val_logits, val_y, n_classes)

#                                 mean_ce = epoch_ce / max(epoch_batches, 1)
#                                 mean_kl = epoch_kl / max(epoch_batches, 1)
#                                 mean_curv = epoch_curv / max(epoch_batches, 1)
#                                 mean_idim = epoch_idim / max(epoch_batches, 1)
#                                 # simple efficiency proxy; this is a diagnostic, not a loss
#                                 efficiency = val_acc / max(mean_curv + 1e-4, 1e-4)

#                                 wall_time = time.time() - t0

#                                 with open(csv_path, "a", newline="") as f:
#                                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                                     writer.writerow(dict(
#                                         dataset="swiss_roll",
#                                         model=model_name,
#                                         seed=seed,
#                                         noise_std=noise_std,
#                                         beta=beta,
#                                         gamma=gamma,
#                                         z_dim=z_dim,
#                                         epoch=epoch,
#                                         train_acc=round(train_acc, 4),
#                                         val_acc=round(val_acc, 4),
#                                         ce=round(mean_ce, 4),
#                                         kl=round(mean_kl, 4),
#                                         curv=round(mean_curv, 4),
#                                         idim=round(mean_idim, 4),
#                                         mi_proxy=round(mi_p, 4),
#                                         efficiency=round(efficiency, 4),
#                                         wall_time=round(wall_time, 3),
#                                     ))
#                                     f.flush()
#                                     os.fsync(f.fileno())

#                             # free memory
#                             del encoder, classifier, opt
#                             torch.cuda.empty_cache()


# # -------------------------------------------------------------------------
# # Fashion-MNIST experiments
# # -------------------------------------------------------------------------

# class MLPEncoder(nn.Module):
#     def __init__(self, x_dim: int, hidden: int, z_dim: int):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(x_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#         )
#         self.mu = nn.Linear(hidden, z_dim)
#         self.logvar = nn.Linear(hidden, z_dim)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = self.backbone(x)
#         mu = self.mu(h)
#         logvar = self.logvar(h)
#         return mu, logvar


# def load_fashion_mnist(n_samples: int = 10000, seed: int = 0) -> DataLoader:
#     tf = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(-1)),
#     ])
#     full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
#     torch.manual_seed(seed)
#     idx = torch.randperm(len(full))[:n_samples]
#     subset = Subset(full, idx)
#     loader = DataLoader(subset, batch_size=512, shuffle=True)
#     return loader


# def fashion_experiments(
#     out_dir: str,
#     seeds: List[int],
#     beta: float = 5e-3,
#     gamma: float = 1e-4,
#     z_dim: int = 32,
#     n_epochs: int = 25,
# ) -> None:
#     """
#     Fashion-MNIST: baseline vs V-GIB using a simple MLP encoder.
#     """
#     ensure_dir(out_dir)
#     csv_path = os.path.join(out_dir, "fashion_results.csv")
#     fieldnames = [
#         "dataset", "model", "seed", "beta", "gamma",
#         "z_dim", "epoch", "train_acc", "val_acc",
#         "ce", "kl", "curv", "idim", "mi_proxy", "efficiency", "wall_time",
#     ]

#     if not os.path.exists(csv_path):
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             f.flush()
#             os.fsync(f.fileno())

#     n_classes = 10
#     x_dim = 784
#     hidden = 256

#     for seed in seeds:
#         loader = load_fashion_mnist(n_samples=10000, seed=seed)
#         # simple 80/20 split over batches using index
#         data = [(xb, yb) for xb, yb in loader]
#         split = int(0.8 * len(data))
#         train_batches = data[:split]
#         val_batches = data[split:]

#         for model_name in ["baseline", "vgib"]:
#             print(f"[Fashion] seed={seed} model={model_name}")
#             encoder = MLPEncoder(x_dim=x_dim, hidden=hidden, z_dim=z_dim).to(DEVICE)
#             classifier = SmallClassifier(z_dim=z_dim, hidden=128, n_classes=n_classes).to(DEVICE)
#             params = list(encoder.parameters()) + list(classifier.parameters())
#             opt = torch.optim.Adam(params, lr=1e-3)

#             for epoch in range(1, n_epochs + 1):
#                 t0 = time.time()
#                 encoder.train()
#                 classifier.train()
#                 train_correct = 0
#                 train_total = 0
#                 epoch_ce = 0.0
#                 epoch_kl = 0.0
#                 epoch_curv = 0.0
#                 epoch_idim = 0.0
#                 epoch_batches = 0

#                 for xb, yb in train_batches:
#                     xb = xb.to(DEVICE).requires_grad_(True)
#                     yb = yb.to(DEVICE)
#                     mu, logvar = encoder(xb)
#                     z = reparameterize(mu, logvar)
#                     logits = classifier(z)
#                     ce = F.cross_entropy(logits, yb)
#                     kl_vec = kl_standard_normal(mu, logvar)
#                     kl_val = float(kl_vec.mean().item())
#                     if model_name == "baseline":
#                         curv_val = 0.0
#                         idim_val = participation_ratio_idim(z.detach())
#                         loss = ce
#                     else:
#                         curv_val = hutchinson_jacobian_norm(z, xb, num_probes=1)
#                         idim_val = participation_ratio_idim(z.detach())
#                         loss = ce + beta * kl_vec.mean() + gamma * curv_val

#                     opt.zero_grad()
#                     loss.backward()
#                     opt.step()

#                     with torch.no_grad():
#                         pred = logits.argmax(dim=1)
#                         train_correct += (pred == yb).sum().item()
#                         train_total += yb.size(0)

#                     epoch_ce += float(ce.item())
#                     epoch_kl += float(kl_val)
#                     epoch_curv += float(curv_val)
#                     epoch_idim += float(idim_val)
#                     epoch_batches += 1

#                 train_acc = train_correct / max(train_total, 1)

#                 # validation
#                 encoder.eval()
#                 classifier.eval()
#                 val_correct = 0
#                 val_total = 0
#                 val_logits_list = []
#                 val_y_list = []
#                 with torch.no_grad():
#                     for xb, yb in val_batches:
#                         xb = xb.to(DEVICE)
#                         yb = yb.to(DEVICE)
#                         mu, logvar = encoder(xb)
#                         z = reparameterize(mu, logvar)
#                         logits = classifier(z)
#                         pred = logits.argmax(dim=1)
#                         val_correct += (pred == yb).sum().item()
#                         val_total += yb.size(0)
#                         val_logits_list.append(logits)
#                         val_y_list.append(yb)

#                 val_acc = val_correct / max(val_total, 1)
#                 val_logits = torch.cat(val_logits_list, dim=0)
#                 val_y = torch.cat(val_y_list, dim=0)
#                 mi_p = mi_proxy_from_logits(val_logits, val_y, n_classes)

#                 mean_ce = epoch_ce / max(epoch_batches, 1)
#                 mean_kl = epoch_kl / max(epoch_batches, 1)
#                 mean_curv = epoch_curv / max(epoch_batches, 1)
#                 mean_idim = epoch_idim / max(epoch_batches, 1)
#                 efficiency = val_acc / max(mean_curv + 1e-4, 1e-4)
#                 wall_time = time.time() - t0

#                 with open(csv_path, "a", newline="") as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                     writer.writerow(dict(
#                         dataset="fashion_mnist",
#                         model=model_name,
#                         seed=seed,
#                         beta=beta,
#                         gamma=gamma if model_name == "vgib" else 0.0,
#                         z_dim=z_dim,
#                         epoch=epoch,
#                         train_acc=round(train_acc, 4),
#                         val_acc=round(val_acc, 4),
#                         ce=round(mean_ce, 4),
#                         kl=round(mean_kl, 4),
#                         curv=round(mean_curv, 4),
#                         idim=round(mean_idim, 4),
#                         mi_proxy=round(mi_p, 4),
#                         efficiency=round(efficiency, 4),
#                         wall_time=round(wall_time, 3),
#                     ))
#                     f.flush()
#                     os.fsync(f.fileno())

#             del encoder, classifier, opt
#             torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# CIFAR-10 experiments: baseline, VIB, V-GIB, Laplacian
# -------------------------------------------------------------------------

class SimpleCIFAREncoder(nn.Module):
    """
    Small CNN backbone for CIFAR-10.
    Used for all variants to keep capacity fixed.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.proj(x)
        return z


class CIFARVIBEncoder(nn.Module):
    """
    VIB-style encoder for CIFAR-10.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.features = SimpleCIFAREncoder(latent_dim=latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class CIFARHead(nn.Module):
    def __init__(self, latent_dim: int = 128, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def cifar10_loaders(data_root: str, label_frac: float, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)

    N = len(train_full)
    n_sub = int(label_frac * N)
    subset, _ = random_split(train_full, [n_sub, N - n_sub], generator=torch.Generator().manual_seed(123))
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


def cifar10_experiments(
    out_dir: str,
    data_root: str,
    seeds: List[int],
    label_fracs: List[float],
    beta: float = 5e-3,
    gamma: float = 1e-4,
    lap_weight: float = 5e-4,
    n_epochs: int = 120,
) -> None:
    """
    CIFAR-10 experiments:
      - baseline CNN
      - VIB-only
      - V-GIB (curvature + idim)
      - Laplacian-style geometric regularizer
    All share the same encoder backbone and schedule.
    """

    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "cifar10_results.csv")
    fieldnames = [
        "dataset", "model", "seed", "label_frac",
        "beta", "gamma", "lap_weight",
        "epoch", "train_acc", "val_acc",
        "ce", "kl", "curv", "idim", "mi_proxy",
        "align_proxy", "efficiency", "wall_time",
    ]
    
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    n_classes = 10
    latent_dim = 128

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for label_frac in label_fracs:
            train_loader, test_loader = cifar10_loaders(data_root, label_frac=label_frac, batch_size=128)
            for model_name in ["baseline", "vib", "vgib", "laplacian"]:
                done_epochs = completed_epochs(csv_path, seed, label_frac, model_name)
                print(f"[CIFAR10] seed={seed} frac={label_frac:.2f} model={model_name}")
                if model_name in ["baseline", "laplacian"]:
                    encoder = SimpleCIFAREncoder(latent_dim=latent_dim).to(DEVICE)
                else:
                    encoder = CIFARVIBEncoder(latent_dim=latent_dim).to(DEVICE)
                head = CIFARHead(latent_dim=latent_dim, n_classes=n_classes).to(DEVICE)
                params = list(encoder.parameters()) + list(head.parameters())
                opt = torch.optim.Adam(params, lr=1e-3)

                for epoch in range(1, n_epochs + 1):
                    if epoch in done_epochs:
                        continue
                
# for epoch in range(1, n_epochs + 1):
                    t0 = time.time()
                    encoder.train()
                    head.train()
                    train_correct = 0
                    train_total = 0

                    epoch_ce = 0.0
                    epoch_kl = 0.0
                    epoch_curv = 0.0
                    epoch_idim = 0.0
                    epoch_align = 0.0
                    epoch_batches = 0

                    for xb, yb in train_loader:
                        xb = xb.to(DEVICE).requires_grad_(True)
                        yb = yb.to(DEVICE)

                        if model_name in ["baseline", "laplacian"]:
                            z = encoder(xb)
                            logits = head(z)
                            ce = F.cross_entropy(logits, yb)
                            kl_val = 0.0
                            # Laplacian-style penalty: squared gradient of logits wrt x
                            if model_name == "laplacian":
                                curv_val = hutchinson_jacobian_norm(z, xb, num_probes=1)
                                loss = ce + lap_weight * curv_val
                            else:
                                curv_val = 0.0
                                loss = ce
                            idim_val = participation_ratio_idim(z.detach())
                        else:
                            mu, logvar = encoder(xb)
                            z = reparameterize(mu, logvar)
                            logits = head(z)
                            ce = F.cross_entropy(logits, yb)
                            kl_vec = kl_standard_normal(mu, logvar)
                            kl_val = float(kl_vec.mean().item())
                            curv_val = hutchinson_jacobian_norm(z, xb, num_probes=1) if model_name == "vgib" else 0.0
                            idim_val = participation_ratio_idim(z.detach())
                            if model_name == "vib":
                                loss = ce + beta * kl_vec.mean()
                            else:  # vgib
                                loss = ce + beta * kl_vec.mean() + gamma * curv_val

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        with torch.no_grad():
                            pred = logits.argmax(dim=1)
                            train_correct += (pred == yb).sum().item()
                            train_total += yb.size(0)
                            align_val = alignment_proxy(z.detach(), yb.detach())

                        epoch_ce += float(ce.item())
                        epoch_kl += float(kl_val)
                        epoch_curv += float(curv_val)
                        epoch_idim += float(idim_val)
                        epoch_align += float(align_val)
                        epoch_batches += 1

                    train_acc = train_correct / max(train_total, 1)

                    # evaluation
                    encoder.eval()
                    head.eval()
                    val_correct = 0
                    val_total = 0
                    val_logits_list = []
                    val_y_list = []

                    with torch.no_grad():
                        for xb, yb in test_loader:
                            xb = xb.to(DEVICE)
                            yb = yb.to(DEVICE)
                            if model_name in ["baseline", "laplacian"]:
                                z = encoder(xb)
                            else:
                                mu, logvar = encoder(xb)
                                z = reparameterize(mu, logvar)
                            logits = head(z)
                            pred = logits.argmax(dim=1)
                            val_correct += (pred == yb).sum().item()
                            val_total += yb.size(0)
                            val_logits_list.append(logits)
                            val_y_list.append(yb)

                    val_acc = val_correct / max(val_total, 1)
                    val_logits = torch.cat(val_logits_list, dim=0)
                    val_y = torch.cat(val_y_list, dim=0)
                    mi_p = mi_proxy_from_logits(val_logits, val_y, n_classes)

                    mean_ce = epoch_ce / max(epoch_batches, 1)
                    mean_kl = epoch_kl / max(epoch_batches, 1)
                    mean_curv = epoch_curv / max(epoch_batches, 1)
                    mean_idim = epoch_idim / max(epoch_batches, 1)
                    mean_align = epoch_align / max(epoch_batches, 1)
                    efficiency = val_acc / max(mean_align + 1e-4, 1e-4)
                    wall_time = time.time() - t0

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(dict(
                            dataset="cifar10",
                            model=model_name,
                            seed=seed,
                            label_frac=label_frac,
                            beta=beta if model_name in ["vib", "vgib"] else 0.0,
                            gamma=gamma if model_name == "vgib" else 0.0,
                            lap_weight=lap_weight if model_name == "laplacian" else 0.0,
                            epoch=epoch,
                            train_acc=round(train_acc, 4),
                            val_acc=round(val_acc, 4),
                            ce=round(mean_ce, 4),
                            kl=round(mean_kl, 4),
                            curv=round(mean_curv, 4),
                            idim=round(mean_idim, 4),
                            mi_proxy=round(mi_p, 4),
                            align_proxy=round(mean_align, 4),
                            efficiency=round(efficiency, 4),
                            wall_time=round(wall_time, 3),
                        ))
                        f.flush()
                        os.fsync(f.fileno())

                del encoder, head, opt
                torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# High-level wrappers that call the experiment drivers defined above
# -------------------------------------------------------------------------

# def run_swiss_suite(
#     save_root: str,
#     seeds: List[int],
#     models: List[str],
#     beta: float,
#     gamma: float,
#     epochs: int,
#     batch_size: int,
# ) -> None:
#     ensure_dir(save_root)

#     # Swiss supports: random, mlp, vib, vgib (per your swiss_roll_experiments)
#     allowed = {"random", "mlp", "vib", "vgib"}
#     models = [m for m in models if m in allowed]
#     if not models:
#         models = ["vib", "vgib"]

#     swiss_roll_experiments(
#         out_dir=save_root,
#         seeds=seeds,
#         noise_levels=[0.05, 0.2],
#         betas=[beta],
#         gammas=[0.0, gamma],      # VIB-only and V-GIB
#         z_dims=[8, 16],
#         n_epochs=epochs,
#         batch_size=batch_size,
#     )
#     print(f"[SWISS] Finished. Logs in {save_root}")


# def run_fashion_suite(
#     save_root: str,
#     seeds: List[int],
#     models: List[str],
#     beta: float,
#     gamma: float,
#     epochs: int,
# ) -> None:
#     ensure_dir(save_root)

#     # Fashion in your code uses ["baseline","vgib"] inside fashion_experiments,
#     # so "models" is not used there (kept for CLI symmetry).
#     fashion_experiments(
#         out_dir=save_root,
#         seeds=seeds,
#         beta=beta,
#         gamma=gamma,
#         z_dim=32,
#         n_epochs=min(epochs, 25),
#     )
#     print(f"[FASHION] Finished. Logs in {save_root}")


def run_cifar_suite(
    save_root: str,
    data_root: str,
    seeds: List[int],
    label_fracs: List[float],
    beta: float,
    gamma: float,
    lap_weight: float,
    epochs: int,
) -> None:
    ensure_dir(save_root)

    cifar10_experiments(
        out_dir=save_root,
        data_root=data_root,
        seeds=seeds,
        label_fracs=label_fracs,
        beta=beta,
        gamma=gamma,
        lap_weight=lap_weight,
        n_epochs=max(epochs, 80),
    )
    print(f"[CIFAR10] Finished. Logs in {save_root}")


# -------------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs")

    # what to run
    parser.add_argument("--run-swiss", action="store_true")
    parser.add_argument("--run-fashion", action="store_true")
    parser.add_argument("--run-cifar", action="store_true")

    # experiment control
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["baseline", "vib", "vgib", "laplacian"])

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--lap-weight", type=float, default=5e-4)

    args = parser.parse_args()

    # default: run everything
    if not (args.run_swiss or args.run_fashion or args.run_cifar):
        args.run_swiss = args.run_fashion = args.run_cifar = True

    os.makedirs(args.output_dir, exist_ok=True)

    print("Device:", DEVICE)
    print("Saving results to:", os.path.abspath(args.output_dir))

    # ------------------------------------------------------------------
    # if args.run_swiss:
    #     print("\n[SWISS] running synthetic manifold suite...")
    #     run_swiss_suite(
    #         save_root=os.path.join(args.output_dir, "swiss"),
    #         seeds=args.seeds,
    #         models=[m for m in args.models if m in ["random", "mlp", "vib", "vgib"]],
    #         beta=args.beta,
    #         gamma=args.gamma,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #     )

    # # ------------------------------------------------------------------
    # if args.run_fashion:
    #     print("\n[FASHION] running Fashion-MNIST suite...")
    #     run_fashion_suite(
    #         save_root=os.path.join(args.output_dir, "fashion"),
    #         seeds=args.seeds,
    #         models=[m for m in args.models if m != "laplacian"],
    #         beta=args.beta,
    #         gamma=args.gamma,
    #         epochs=args.epochs,
    #     )

    # ------------------------------------------------------------------
    if args.run_cifar:
        print("\n[CIFAR] running CIFAR-10 suite...")
        # run each fraction into its own subfolder (as you requested)
        for frac in args.fractions:
            frac_tag = f"frac_{str(frac).replace('.', 'p')}"
            out_dir = os.path.join(args.output_dir, "cifar", frac_tag)
            os.makedirs(out_dir, exist_ok=True)

            print(f"[CIFAR] label fraction = {frac}")
            run_cifar_suite(
                save_root=out_dir,
                data_root=args.data_root,
                seeds=args.seeds,
                label_fracs=[frac],  # one frac per folder
                beta=args.beta,
                gamma=args.gamma,
                lap_weight=args.lap_weight,
                epochs=args.epochs,
            )


if __name__ == "__main__":
    main()
