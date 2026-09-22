"""Financial performance metrics and the backtest consistency audit."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class PerfSummary:
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float        # negative fraction, e.g. -0.124
    cum_return: float
    vol: float
    win_rate: float
    calmar: float
    n_days: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(min(0.0, dd.min()))


def summarize(returns: np.ndarray, rf_annual: float = 0.02) -> PerfSummary:
    """All metrics over the supplied (live-window) daily returns."""
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    if n < 2:
        raise ValueError("need at least two return observations")
    rf_d = rf_annual / TRADING_DAYS
    excess = r - rf_d

    sd = float(np.std(r, ddof=1))
    sharpe = float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    downside = np.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    sortino = float(excess.mean() / downside * TRADING_DAYS) if downside > 0 else 0.0

    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    mdd = max_drawdown(equity)
    cum = float(equity[-1] - 1.0)
    years = n / TRADING_DAYS
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    vol = sd * np.sqrt(TRADING_DAYS)
    win = float(np.mean(r > 0.0))
    calmar = cagr / abs(mdd) if mdd < 0 else float("inf")

    return PerfSummary(
        sharpe=sharpe, sortino=sortino, cagr=cagr, max_drawdown=mdd,
        cum_return=cum, vol=vol, win_rate=win, calmar=calmar, n_days=n,
    )


def jensen_alpha(port_ret: np.ndarray, bench_ret: np.ndarray, rf_daily: float) -> tuple[float, float]:
    """CAPM alpha and beta via OLS on excess returns."""
    y = port_ret - rf_daily
    x = bench_ret - rf_daily
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    var = float(np.var(x, ddof=1))
    beta = cov / var if var > 0 else 0.0
    alpha = float(np.mean(y) - beta * np.mean(x))
    return alpha * TRADING_DAYS, beta


def consistency_checks(summary: PerfSummary, rf_annual: float, tol: float = 0.25) -> dict[str, object]:
    """Audit identities exposed by the paper's own consistency-check revision.

    (1) Sharpe x Vol must match CAGR - rf up to the second-order (log-normal)
    correction; a large mismatch indicates window or basis mixing.
    """
    lhs = summary.sharpe * summary.vol
    rhs = summary.cagr - rf_annual + 0.5 * summary.vol ** 2
    ok = abs(lhs - rhs) <= tol * max(abs(rhs), 1e-9)
    return {"sharpe_cagr_identity": ok, "lhs": lhs, "rhs": rhs}