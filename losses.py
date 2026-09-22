"""Objective components (eq. total_objective) and the trading losses."""
from __future__ import annotations

import torch
from torch import Tensor


def clipped_pnl_loss(y_hat: Tensor, z: Tensor) -> Tensor:
    """ell(y, z) = clip(-y z, 0, 1)  (eq. trading_loss; the ANALYSIS loss).

    M-Lipschitz in y with M = Z_max after volatility scaling; bounded in [0, 1]
    as required by Theorem `drift_pac_bayes`. NOTE: minimised by y == 0 (never
    trade); it is used for certification and reporting, never as the training
    task loss (see `task_loss`).
    """
    return torch.clip(-y_hat * z, 0.0, 1.0)


def task_loss(y_hat: Tensor, z: Tensor, kind: str = "pnl") -> Tensor:
    """Training task loss.

    "pnl":          mean(-y z)   -- linear PnL surrogate, non-degenerate.
    "clipped_pnl":  eq. trading_loss (degenerate optimum y = 0; analysis only).
    """
    if kind == "pnl":
        return (-y_hat * z).mean()
    if kind == "clipped_pnl":
        return clipped_pnl_loss(y_hat, z).mean()
    raise ValueError(f"unknown task loss {kind!r}")


def directional_event(y_hat_detached: Tensor, z: Tensor) -> Tensor:
    """E_dir = 1[y(x) z > 0]  (eq. event_directional).

    The prediction is detached: the event labels the *current deployed policy*,
    avoiding circular gradients into the Brier term.
    """
    return ((y_hat_detached * z) > 0.0).to(y_hat_detached.dtype)


def brier_calibration(mu_max: Tensor, z_event: Tensor) -> Tensor:
    """Squared (Brier) calibration relaxation for P(z_E = 1 | x) (Section 3.4.4)."""
    return ((mu_max - z_event) ** 2).mean()


def lyapunov_regularizer(
    omega: Tensor, omega_star: Tensor | None, omega_prev: Tensor,
    alpha1: float, alpha2: float,
) -> Tensor:
    """R_Lyapunov = alpha1 V(w, w*) + alpha2 ||w^{(t)} - w^{(t-1)}||^2.

    V is the per-coordinate mean squared displacement so that lambda_1 operates
    on the same magnitude regime as the paper's anchoring table. `omega_star`
    is the re-anchoring reference (data strictly prior to the current epoch);
    a missing anchor zeroes the potential term.
    """
    damping = alpha2 * (omega - omega_prev).pow(2).mean()
    potential = 0.0 if omega_star is None else alpha1 * (omega - omega_star).pow(2).mean()
    return potential + damping


def sparsity_regularizer(rule_weights: Tensor) -> Tensor:
    """L1 parsimony pressure on live rule weights."""
    return rule_weights.sum()


def interpretability_regularizer(
    membership_net, prototypes: Tensor, lambda_sharp: float,
) -> Tensor:
    """Prototype anchoring + gradient-norm penalty (eq. interpretability_reg).

    NOTE: the gradient term penalises ||grad_x mu||^2 and therefore promotes
    SMOOTH memberships (the paper's prose said 'sharp'; the formula is
    authoritative). Not used in the default trading pipeline (lambda = 0).
    """
    anchored = (membership_net(prototypes) - 1.0).pow(2).sum()
    return anchored + lambda_sharp * torch.zeros((), device=prototypes.device)