"""Position policy, execution accounting, and baseline strategies."""
from __future__ import annotations

from typing import Protocol

import numpy as np


# --------------------------------------------------------------------------- #
#  Policy and execution                                                        #
# --------------------------------------------------------------------------- #
def position_policy(y_hat: np.ndarray, mu_max: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """pi(x) = clip(y_hat(x) * rho(mu_max(x)), -1, 1), rho(u) = u (eq. position_policy)."""
    return np.clip(y_hat * mu_max, -cap, cap)


def strategy_returns(
    positions: np.ndarray, asset_returns: np.ndarray, cost_bps: float
) -> np.ndarray:
    """Daily accounting: pi[t-1] earns r[t]; turnover charged at cost_bps.

    positions[t] is decided at the close of day t and held into day t+1.
    """
    T = asset_returns.shape[0]
    out = np.zeros(T)
    turnover = np.abs(np.diff(positions, prepend=0.0))
    out[1:] = positions[:-1] * asset_returns[1:] - (cost_bps * 1e-4) * turnover[1:]
    return out


# --------------------------------------------------------------------------- #
#  Baselines                                                                   #
# --------------------------------------------------------------------------- #
class Baseline(Protocol):
    name: str

    def positions(self, prices: np.ndarray) -> np.ndarray: ...


class BuyAndHold:
    name = "Buy & Hold"

    def positions(self, prices: np.ndarray) -> np.ndarray:
        return np.ones(prices.shape[0] - 1)


class Portfolio6040:
    """60% market / 40% risk-free rebalance-free portfolio."""

    name = "60/40"

    def __init__(self, rf_annual: float) -> None:
        self.rf_daily = rf_annual / 252.0

    def positions(self, prices: np.ndarray) -> np.ndarray:
        return np.full(prices.shape[0] - 1, 0.6)

    def overlay(self, positions: np.ndarray) -> np.ndarray:
        """Cash-leg return stream to be added to the market leg."""
        return (1.0 - np.abs(positions)) * self.rf_daily


def _ewm(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


class MACDBaseline:
    """MACD (12, 26, 9) with volatility-scaled tanh signal mapping."""

    name = "MACD"

    def positions(self, prices: np.ndarray) -> np.ndarray:
        close = prices[:-1]  # position[t] decided from data up to close_t
        macd = _ewm(close, 12) - _ewm(close, 26)
        signal = _ewm(macd, 9)
        raw = macd - signal
        T = close.shape[0]
        pos = np.zeros(T)
        for t in range(26, T):
            hist = raw[max(t - 252, 0) : t + 1]
            scale = np.std(hist) + 1e-12
            pos[t] = np.clip(np.tanh(raw[t] / scale), -1.0, 1.0)
        return pos