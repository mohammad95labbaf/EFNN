"""Split-conformal wrappers of Theorem `fuzzy_prob_duality` (parts i and ii)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _quantile_order_statistic(scores: np.ndarray, alpha: float) -> float:
    """q-hat = k-th smallest score with k = ceil((n+1)(1-alpha)); +inf fallback."""
    n = scores.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k < 1:
        return float(np.min(scores))
    if k > n:
        return math.inf  # conservative: sets/interval become trivially wide
    return float(np.sort(scores)[k - 1])


@dataclass(frozen=True)
class EventConformalModel:
    """Conformalised event sets, eq. conformal_set."""

    qhat: float
    alpha: float
    n: int

    def predict(self, mu_max: np.ndarray) -> list[set[int]]:
        out: list[set[int]] = []
        for mu in np.atleast_1d(mu_max):
            s: set[int] = set()
            if math.isinf(self.qhat) or mu >= 1.0 - self.qhat:
                s.add(1)
            if math.isinf(self.qhat) or mu <= self.qhat:
                s.add(0)
            if not s:  # qhat < 0 cannot occur, but keep sets non-empty by safety
                s = {0, 1}
            out.append(s)
        return out


def fit_event_conformal(
    mu_max: np.ndarray, z_event: np.ndarray, alpha: float
) -> EventConformalModel:
    """Fit on an exchangeable calibration set; score per eq. conformal_score."""
    mu = np.asarray(mu_max, dtype=float)
    ze = np.asarray(z_event, dtype=float)
    if mu.shape != ze.shape:
        raise ValueError("mu_max and z_event must have equal shape")
    if mu.size < 1:
        raise ValueError("calibration set must be non-empty")
    scores = (1.0 - mu) * ze + mu * (1.0 - ze)
    return EventConformalModel(qhat=_quantile_order_statistic(scores, alpha),
                               alpha=alpha, n=mu.size)


@dataclass(frozen=True)
class RegressionConformalModel:
    """Split-conformal prediction intervals for the continuous target."""

    qhat: float
    alpha: float
    n: int

    def predict(self, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_hat, dtype=float)
        return y - self.qhat, y + self.qhat


def fit_regression_conformal(
    y_hat_cal: np.ndarray, z_cal: np.ndarray, alpha: float
) -> RegressionConformalModel:
    """Nonconformity s_reg = |z - y_hat| (part ii of the theorem)."""
    resid = np.abs(np.asarray(z_cal, dtype=float) - np.asarray(y_hat_cal, dtype=float))
    return RegressionConformalModel(qhat=_quantile_order_statistic(resid, alpha),
                                    alpha=alpha, n=resid.size)