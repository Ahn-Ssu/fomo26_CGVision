"""Signed GeM (Generalized-Mean) pooling for 3D feature maps.

Spec: "07. Task 1 Head Architecture 실험 -- Multi-stage GeM + Late Fusion",
Sec 3.3. Our ResEnc UNet backbone (FOMO26/networks/student.py) uses
LeakyReLU, unlike the ReLU-based backbone the original (unsigned,
clamp-only) GeM formulation assumes -- LeakyReLU passes negative
activations through, so `clamp(min=eps).pow(p)` on raw features would
silently zero out every negative activation before pooling. The signed
variant (`sign(x) * |x|^p`) preserves that information instead.

`p` is parameterized as `1 + softplus(theta)` (not a raw learnable scalar)
so it can never leave the domain `pow()` needs (`p > 0`, in fact `p > 1`
here, biased toward max-pooling-like behavior at large p and average-pooling-
like behavior as p approaches 1) regardless of what gradient descent does to
`theta`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_gem3d(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = x.sign()
    y = s * x.abs().clamp(min=eps).pow(p)
    y = F.adaptive_avg_pool3d(y, (1, 1, 1))
    # clamp again before the inverse power: without it, a pooled value that
    # lands exactly on 0 (e.g. a channel whose activations cancel perfectly)
    # gives pow(1/p) a zero-gradient singularity at the origin.
    return y.sign() * y.abs().clamp(min=eps).pow(1.0 / p)


class SignedGeM3d(nn.Module):
    """Stage-independent instance: each stage in MultiStageGeMHead owns one
    of these, so `p` (and therefore max-like vs average-like behavior) can
    differ per stage -- see MultiStageGeMHead.learned_p_values() for the
    diagnostic this is meant to enable (spec Sec 6, Sec 8 REPORT.md item 4)."""

    def __init__(self, theta_init: float = 1.85):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return signed_gem3d(x, self.p)
