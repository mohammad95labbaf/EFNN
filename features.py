"""Causal, market-observable feature construction.

IMPORTANT: every feature is a function of the price path up to day t ONLY. The
latent variance V_t of the simulator and any oracle jump probability are never
used -- they are unavailable to a real strategy. The short/long realized-vol
ratio serves as the market-observable proxy for the paper's risk-off rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252

FEATURE_NAMES = (
    "ret_1", "mom_20", "mom_60", "rv_20", "rv_ratio", "vol_of_vol", "jump_ewma",
)


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray            # (T, d) standardised features
    names: tuple[str, ...]
    valid_start: int         # first index with fully defined, standardised rows


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.shape[0], np.nan)
    out[w - 1 :] = c[w:] - c[:-w]
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    if x.size < w:
        return out
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
    out[w - 1 :] = np.sqrt(var)
    return out


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = np.empty_like(x, dtype=np.float64)
    acc = 0.0
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _causal_zscore(x: np.ndarray, min_periods: int, clip: float = 5.0) -> np.ndarray:
    n = x.shape[0]
    out = np.full(n, np.nan)
    c = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    for t in range(min_periods, n):
        k = t + 1
        mean = c[k] / k
        var = max(c2[k] / k - mean ** 2, 1e-12)
        out[t] = np.clip((x[t] - mean) / np.sqrt(var), -clip, clip)
    return out


def build_features(prices: np.ndarray) -> FeatureSet:
    """Construct the standardised feature matrix; strictly causal (past-only)."""
    log_r = np.diff(np.log(prices))  # (T,) return of day t uses close_t/close_{t-1}
    T = log_r.shape[0]

    ret_1 = log_r
    mom_20 = _rolling_sum(log_r, 20)
    mom_60 = _rolling_sum(log_r, 60)
    rv_20 = _rolling_std(log_r, 20) * np.sqrt(TRADING_DAYS)
    rv_5 = _rolling_std(log_r, 5) * np.sqrt(TRADING_DAYS)
    vol_of_vol = _rolling_std(np.abs(log_r), 60)
    sigma_60 = _rolling_std(log_r, 60)
    jump_indicator = np.where(
        np.isnan(sigma_60), 0.0, (np.abs(log_r) > 3.0 * np.nan_to_num(sigma_60)).astype(float)
    )
    jump_ewma = _ewma(jump_indicator, halflife=5.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv_ratio = np.clip(rv_5 / np.where(rv_20 > 0, rv_20, np.nan), 0.25, 4.0)

    raw = np.stack(
        [ret_1, mom_20, mom_60, rv_20, rv_ratio, vol_of_vol, jump_ewma], axis=1
    )
    X = np.full_like(raw, np.nan)
    for j in range(raw.shape[1]):
        X[:, j] = _causal_zscore(np.nan_to_num(raw[:, j], nan=0.0), min_periods=60)
    valid_start = 66
    return FeatureSet(X=X, names=FEATURE_NAMES, valid_start=valid_start)


def forward_targets(
    prices: np.ndarray, horizon: int, z_max: float, vol_window: int = 20
) -> np.ndarray:
    """Volatility-scaled h-day forward returns, clipped to +/- Z_max.

    z[t] = (P[t+h] / P[t] - 1) / (sigma_daily[t] * sqrt(h)), NaN for the last h
    days (label not yet realised).
    """
    log_r = np.diff(np.log(prices))
    sigma = _rolling_std(log_r, vol_window)
    T = prices.shape[0] - 1
    z = np.full(T, np.nan)
    for t in range(T - horizon):
        denom = max(sigma[t], 1e-4) * np.sqrt(horizon)
        z[t] = np.clip((prices[t + horizon] / prices[t] - 1.0) / denom, -z_max, z_max)
    return z