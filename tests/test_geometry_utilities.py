from __future__ import annotations

import torch

from vgib.geometry import kl_standard_normal, mi_proxy_from_logits, participation_ratio_idim, reparameterize


def test_kl_standard_normal_zero_for_standard_normal_parameters():
    mu = torch.zeros(5, 3)
    logvar = torch.zeros(5, 3)
    kl = kl_standard_normal(mu, logvar)
    assert kl.shape == (5,)
    assert torch.allclose(kl, torch.zeros_like(kl))


def test_reparameterize_shape():
    mu = torch.zeros(7, 4)
    logvar = torch.zeros(7, 4)
    z = reparameterize(mu, logvar)
    assert z.shape == mu.shape


def test_participation_ratio_range():
    z = torch.randn(64, 6)
    value = participation_ratio_idim(z)
    assert value > 0
    assert value <= 6.0 + 1e-5


def test_mi_proxy_is_float():
    logits = torch.randn(20, 3)
    y = torch.randint(0, 3, (20,))
    value = mi_proxy_from_logits(logits, y, num_classes=3)
    assert isinstance(value, float)
