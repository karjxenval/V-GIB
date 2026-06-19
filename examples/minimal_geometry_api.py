"""Minimal example of the reusable V-GIB geometry API."""

from __future__ import annotations

import torch

from vgib.geometry import kl_standard_normal, mi_proxy_from_logits, participation_ratio_idim
from vgib.reproducibility import set_global_seed


def main() -> None:
    set_global_seed(13)
    mu = torch.zeros(16, 8)
    logvar = torch.zeros(16, 8)
    kl = kl_standard_normal(mu, logvar)
    z = torch.randn(64, 8)
    logits = torch.randn(64, 3)
    y = torch.randint(0, 3, (64,))

    print("KL shape:", tuple(kl.shape))
    print("Intrinsic-dimension proxy:", participation_ratio_idim(z))
    print("MI proxy:", mi_proxy_from_logits(logits, y, num_classes=3))


if __name__ == "__main__":
    main()
