"""Reusable geometry and information diagnostic utilities for V-GIB experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) for diagonal Gaussian q."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterized Gaussian latent sample."""
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def participation_ratio_idim(z: torch.Tensor, eps: float = 1e-8) -> float:
    """Estimate intrinsic dimension using the participation ratio of latent covariance."""
    with torch.no_grad():
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).real.clamp(min=eps)
        return float(eigvals.sum().pow(2).div(eigvals.pow(2).sum()).item())


def hutchinson_jacobian_norm(z: torch.Tensor, x: torch.Tensor, num_probes: int = 2) -> float:
    """Estimate ||dz/dx||_F^2 with Hutchinson probes."""
    batch = x.size(0)
    values = []
    for _ in range(num_probes):
        v = torch.randn_like(z)
        grad = torch.autograd.grad(
            outputs=z,
            inputs=x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0]
        values.append(grad.pow(2).view(batch, -1).sum(dim=1))
    return float(torch.stack(values, dim=0).mean().item())


def mi_proxy_from_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    """Empirical proxy H(Y) - CE(logits, y)."""
    with torch.no_grad():
        ce = F.cross_entropy(logits, y, reduction="mean").item()
        return float(math.log(float(num_classes)) - ce)
