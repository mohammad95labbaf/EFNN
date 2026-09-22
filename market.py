"""Synthetic market generator: Heston + Merton jumps + injected crash.

All time quantities are TRADING DAYS (Remark `time_indexing`); the simulator
runs 10 internal sub-steps per day for order-book interaction, fills, and
impact, while decisions remain daily.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MarketConfig

TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketResult:
    prices: np.ndarray        # (T+1,) price path, P[0] = P0
    log_returns: np.ndarray   # (T,) daily log returns
    variance: np.ndarray      # (T+1,) latent instantaneous variance (diagnostics only)
    jump_days: np.ndarray      # days with at least one jump arrival
    crash_day: int
    seed: int


def simulate_market(cfg: MarketConfig, seed: int, T_days: int) -> MarketResult:
    """Full-truncation Euler scheme at sub-daily granularity.

    Feller condition 2 kappa theta = 0.24 vs sigma_v^2 = 0.25 is mildly violated
    (as in the paper's parameterisation); full truncation keeps the scheme
    well-defined with variance clamped at zero inside the coefficients.
    """
    rng = np.random.default_rng(seed)
    sub = cfg.substeps
    dt = 1.0 / (TRADING_DAYS * sub)
    chol = np.array(
        [[1.0, 0.0], [cfg.rho, math_sqrt(1.0 - cfg.rho ** 2)]]
        if False else [[1.0, cfg.rho], [0.0, np.sqrt(1.0 - cfg.rho ** 2)]]
    )  # dW = chol @ (z1, z2): dW_S = z1, dW_V = rho z1 + sqrt(1-rho^2) z2

    n_sub = T_days * sub
    var = np.empty(n_sub + 1)
    log_s = np.empty(n_sub + 1)
    jump_flags = np.zeros(n_sub, dtype=bool)

    var[0] = cfg.theta
    log_s[0] = 0.0
    jump_prob = cfg.jump_intensity * dt
    sigma_daily = np.sqrt(cfg.theta / TRADING_DAYS)
    crash_extra = cfg.crash_drift_sigma * sigma_daily / sub  # per sub-step

    for i in range(n_sub):
        v_pos = max(var[i], 0.0)
        day = i // sub
        v = var[i]
        if day == cfg.crash_day:
            v = v * cfg.crash_vol_mult  # variance doubles at crash onset
            v_pos = v
        dw = chol @ rng.standard_normal(2)
        step = (cfg.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dw[0]
        if cfg.crash_day <= day < cfg.crash_day + cfg.crash_len:
            step += crash_extra
        if rng.random() < jump_prob:
            step += rng.normal(cfg.jump_mean, cfg.jump_std)
            jump_flags[i] = True
        log_s[i + 1] = log_s[i] + step
        var[i + 1] = v + cfg.kappa * (cfg.theta - v_pos) * dt + cfg.sigma_v * np.sqrt(v_pos) * dw[1]

    log_returns = log_s[1:].reshape(T_days, sub).sum(axis=1)
    prices = np.exp(log_s.reshape(-1, sub)[:, 0].sum(axis=1).cumsum())
    prices = np.concatenate([[1.0], prices])  # P0 = 1 (per-unit normalisation)
    daily_var = var.reshape(-1, sub)[:, 0]
    jump_days = np.flatnonzero(jump_flags.reshape(T_days, sub).any(axis=1))
    return MarketResult(
        prices=prices, log_returns=log_returns,
        variance=np.concatenate([[cfg.theta], daily_var]),
        jump_days=jump_days, crash_day=cfg.crash_day, seed=seed,
    )


def math_sqrt(x: float) -> float:  # helper kept tiny; chol uses np.sqrt anyway
    return float(np.sqrt(x))