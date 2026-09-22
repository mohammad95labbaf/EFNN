"""Paired statistical inference: JKM test, Holm correction, power analysis."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

TRADING_DAYS = 252


def jkm_paired(r1: np.ndarray, r2: np.ndarray) -> tuple[float, float]:
    """Jobson-Korkie-Memmel paired Sharpe test (two-sided).

    Per-period (daily) Sharpe ratios enter the statistic; the z-score is
    invariant to consistent annualisation. Variance of Delta-SR (Memmel 2003):
        V = [ 2(1-rho) + (SR1^2 + SR2^2)/2 - rho^2 SR1 SR2 ] / n.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    if r1.shape != r2.shape or r1.size < 30:
        raise ValueError("JKM requires equal-length series with n >= 30")
    s1, s2 = float(np.std(r1, ddof=1)), float(np.std(r2, ddof=1))
    if s1 <= 0 or s2 <= 0:
        return float("nan"), float("nan")
    sr1 = float(np.mean(r1)) / s1
    sr2 = float(np.mean(r2)) / s2
    rho = float(np.corrcoef(r1, r2)[0, 1])
    n = r1.size
    var = (2.0 * (1.0 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2) - rho ** 2 * sr1 * sr2) / n
    z = (sr1 - sr2) / math.sqrt(max(var, 1e-18))
    p = 2.0 * float(sps.norm.sf(abs(z)))
    return z, p


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm step-down adjustment with monotonicity enforcement."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    reject = [bool(adj <= alpha) for adj in adjusted]
    return reject, [float(a) for a in adjusted]


def required_paths(delta_sr: float, sigma_d: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Corrected paired-design power analysis (eq. power_corrected):

        n >= (z_{1-a/2} + z_{1-b})^2 sigma_d^2 / Delta^2.
    """
    if delta_sr <= 0 or sigma_d <= 0:
        raise ValueError("delta_sr and sigma_d must be positive")
    z = sps.norm.ppf(1.0 - alpha / 2.0) + sps.norm.ppf(power)
    return int(math.ceil(z ** 2 * sigma_d ** 2 / delta_sr ** 2))


def paired_t(metric_a: np.ndarray, metric_b: np.ndarray) -> tuple[float, float]:
    """Paired t-test on per-path metrics (the design the power analysis sizes)."""
    t, p = sps.ttest_rel(metric_a, metric_b)
    return float(t), float(p)


def stouffer_combine(zs: np.ndarray) -> tuple[float, float]:
    """Combine per-path z-statistics (independent paths)."""
    z = float(np.sum(zs) / np.sqrt(zs.size))
    return z, 2.0 * float(sps.norm.sf(abs(z)))