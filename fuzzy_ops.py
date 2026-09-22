"""Yager T-/S-norm family and composite aggregation (Section 3.3.2).

T_p(a, b) = max(0, 1 - ((1-a)^p + (1-b)^p)^{1/p})          [eq. yager_tnorm]
S_p(a, b) = 1 - T_p(1-a, 1-b) = min(1, (a^p + b^p)^{1/p})  [eq. yager_snorm]

p = 1 recovers the Lukasiewicz T-norm; p -> inf recovers min.
"""
from __future__ import annotations

import math

from torch import Tensor, clamp, minimum


def yager_tnorm(a: Tensor, b: Tensor, p: float) -> Tensor:
    """Yager T-norm; `p = math.inf` yields the minimum T-norm."""
    if p < 1.0:
        raise ValueError("Yager parameter p must be >= 1 (or +inf).")
    if math.isinf(p):
        return minimum(a, b)
    return clamp(1.0 - ((1.0 - a) ** p + (1.0 - b) ** p) ** (1.0 / p), min=0.0)


def yager_snorm(a: Tensor, b: Tensor, p: float) -> Tensor:
    """Dual S-norm via De Morgan's law."""
    return 1.0 - yager_tnorm(1.0 - a, 1.0 - b, p)


def composite_membership(mu1: Tensor, mu2: Tensor, mu3: Tensor, p: float) -> Tensor:
    """S_p(T_p(mu1, mu2), mu3): 'IF R1 AND R2 OR R3' (eq. composite_membership)."""
    return yager_snorm(yager_tnorm(mu1, mu2, p), mu3, p)