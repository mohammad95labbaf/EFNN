"""PAC-Bayes certificates, Lipschitz reports, and the reliability horizon.

Implements Theorem `drift_pac_bayes`, Corollaries `reliability_horizon` and
`snapshot_adaptation`, Lemma `surrogate_gap`, and the practical prior/posterior
instantiation (sample-split Gaussians on the padded space, closed-form KL,
Monte-Carlo posterior risk with m = 100 draws and a 12-point dyadic sigma_Q
grid whose union-bound cost is absorbed into log(n/delta)).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from .config import PACBayesConfig
from .losses import clipped_pnl_loss
from .model import EFNN


# --------------------------------------------------------------------------- #
#  Closed-form quantities                                                      #
# --------------------------------------------------------------------------- #
def gaussian_kl(mean_q: Tensor, std_q: float, mean_p: Tensor, std_p: float) -> float:
    """KL(N(m_q, s_q^2 I) || N(m_p, s_p^2 I)), eq. kl_gaussian (diagonal case).

    With sigma_Q = sigma_P the variance term vanishes and the KL is purely
    displacement-driven; inert slots pinned at the prior mean contribute zero.
    """
    d = mean_q.numel()
    ratio = (std_q / std_p) ** 2
    displacement = float(((mean_q - mean_p) ** 2).sum()) / (std_p ** 2)
    return 0.5 * (d * (ratio - 1.0 - math.log(ratio)) + displacement)


def complexity_term(kl: float, n: int, delta: float, grid_size: int = 1) -> float:
    """sqrt((KL + log(grid * n / delta)) / (2 (n - 1)))."""
    if n < 2:
        raise ValueError("complexity term requires n >= 2")
    return math.sqrt((kl + math.log(grid_size * n / delta)) / (2.0 * (n - 1)))


def surrogate_gap(g_inf: float, kappa: float, tau: float) -> float:
    """eps_kappa = G_inf * sigmoid(kappa (tau - 1))  (Lemma surrogate_gap)."""
    return g_inf / (1.0 + math.exp(kappa * (1.0 - tau)))


# --------------------------------------------------------------------------- #
#  Posterior sampling                                                          #
# --------------------------------------------------------------------------- #
@contextmanager
def perturbed_parameters(model: EFNN, mean: Tensor, std: float) -> Iterator[None]:
    base = model.flat_parameters()
    model.load_flat(mean + std * torch.randn_like(base))
    try:
        yield
    finally:
        model.load_flat(base)


@torch.no_grad()
def mc_posterior_risk(
    model: EFNN, X: Tensor, Z: Tensor, sigma_q: float, m: int
) -> tuple[float, float]:
    """Posterior-average clipped-PnL risk with standard error.

    Evaluations use the DEPLOYED forward pass (deterministic, dropout off) so
    the estimate targets the exact model the bound is stated for.
    """
    gate_mode, was_training = model.gate_mode, model.training
    model.eval()
    model.gate_mode = "deployed"  # type: ignore[assignment]
    risks: list[float] = []
    try:
        for _ in range(m):
            with perturbed_parameters(model, model.flat_parameters(), sigma_q):
                y = model(X).y_hat.squeeze(-1)
                risks.append(float(clipped_pnl_loss(y, Z).mean()))
    finally:
        if was_training:
            model.train()
        model.gate_mode = gate_mode  # type: ignore[assignment]
    arr = np.asarray(risks)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr))) if m > 1 else 0.0


# --------------------------------------------------------------------------- #
#  Lipschitz certificate (Prop. membership_smoothness + Prop. inference)       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LipschitzReport:
    c_ln: float
    L_max: float          # membership Lipschitz bound
    L_F: float            # inference-layer bound, eq. inference_lipschitz
    L_a: float            # residual-network bound
    L_h: float            # full-model composite bound
    g_inf: float
    eps_kappa: float
    K_alive: int

    def as_dict(self) -> dict[str, float]:
        return {
            "C_LN": self.c_ln, "L_max": self.L_max, "L_F": self.L_F,
            "L_a": self.L_a, "L_h": self.L_h, "G_inf": self.g_inf,
            "eps_kappa": self.eps_kappa, "K_alive": float(self.K_alive),
        }


@torch.no_grad()
def compute_lipschitz(model: EFNN, X_sample: Tensor) -> LipschitzReport:
    cfg = model.cfg
    alive = model.registry.alive_slots()
    if not alive:
        raise ValueError("Lipschitz certificate requires at least one live rule")
    model.eval()
    c_ln = max(model.memberships[k].estimate_c_ln(X_sample) for k in alive)
    L_max = max(model.memberships[k].lipschitz_upper_bound(c_ln) for k in alive)

    w = model.rule_weights_np().numpy()
    C = model.consequent_np().numpy()
    L_F = 2.0 * float(np.linalg.norm(C, "fro")) * float(np.sum(w)) / cfg.gate.eps_inference
    L_a = model.residual_net.lipschitz()
    L_h = L_F * math.sqrt(len(alive)) * L_max + L_a + cfg.residual.g_inf * (cfg.gate.kappa / 4.0) * L_max
    return LipschitzReport(
        c_ln=c_ln, L_max=L_max, L_F=L_F, L_a=L_a, L_h=L_h,
        g_inf=cfg.residual.g_inf,
        eps_kappa=surrogate_gap(cfg.residual.g_inf, cfg.gate.kappa, cfg.gate.tau),
        K_alive=len(alive),
    )


# --------------------------------------------------------------------------- #
#  Drift-aware PAC-Bayes certificate                                           #
# --------------------------------------------------------------------------- #
@dataclass
class PACBayesCertificate:
    """Snapshot certificate at a checkpoint t_c (Cor. snapshot_adaptation)."""

    t_c: int
    n: int
    delta: float
    kl: float
    complexity: float
    mc_risk: float
    mc_se: float
    sigma_q: float
    sigma_p: float
    gamma: float            # per-step W1 drift rate (proxy estimate)
    M: float                # Z_max: Lipschitz of the loss in y
    L_z: float              # Lipschitz of the loss in z
    L_h: float
    eps_kappa: float
    lipschitz: dict[str, float] = field(default_factory=dict)

    def risk_bound(self, t: int) -> float:
        """E_Q[L_Dt] <= E_Q[L_S] + complexity + gamma (t - t_c)(M L_h + L_z) + M eps_kappa."""
        drift = self.gamma * max(t - self.t_c, 0) * (self.M * self.L_h + self.L_z)
        return self.mc_risk + self.complexity + drift + self.M * self.eps_kappa

    def reliability_horizon(self, budget: float) -> float | None:
        """Certified horizon t_max (Cor. reliability_horizon); None if vacuous."""
        rate = self.gamma * (self.M * self.L_h + self.L_z)
        if rate <= 0.0:
            return math.inf
        slack = (
            budget - self.mc_risk - self.complexity - self.M * self.eps_kappa
        )
        return slack / rate if slack > 0.0 else None

    def as_dict(self) -> dict[str, float | int | None]:
        horizon = self.reliability_horizon(budget=0.35)  # default risk budget
        return {
            "t_c": self.t_c, "n_cal": self.n, "KL": self.kl,
            "complexity": self.complexity, "mc_risk": self.mc_risk,
            "mc_se": self.mc_se, "sigma_q": self.sigma_q, "sigma_p": self.sigma_p,
            "gamma": self.gamma, "L_h": self.L_h, "eps_kappa": self.eps_kappa,
            "reliability_horizon": None if horizon is None else round(horizon, 2),
            **{f"lip_{k}": v for k, v in self.lipschitz.items()},
        }


def certify(
    model: EFNN,
    X_cal: Tensor,
    Z_cal: Tensor,
    anchor_flat: Tensor,
    cfg: PACBayesConfig,
    gamma: float,
    M: float,
    L_z: float,
    lipschitz: LipschitzReport,
    t_c: int,
) -> PACBayesCertificate:
    """Snapshot certification with grid-selected posterior variance."""
    n = X_cal.shape[0]
    if n < 2:
        raise ValueError("certification requires n >= 2 calibration points")
    posterior_mean = model.flat_parameters()

    grid = [cfg.sigma_p * (2.0 ** j) for j in range(-(cfg.grid_size // 2),
                                                    cfg.grid_size - cfg.grid_size // 2)]
    best: tuple[float, PACBayesCertificate] | None = None
    for sigma_q in grid:
        kl = gaussian_kl(posterior_mean, sigma_q, anchor_flat, cfg.sigma_p)
        comp = complexity_term(kl, n, cfg.delta, grid_size=len(grid))
        risk, se = mc_posterior_risk(model, X_cal, Z_cal, sigma_q, cfg.m_draws)
        cert = PACBayesCertificate(
            t_c=t_c, n=n, delta=cfg.delta, kl=kl, complexity=comp,
            mc_risk=risk, mc_se=se, sigma_q=sigma_q, sigma_p=cfg.sigma_p,
            gamma=gamma, M=M, L_z=L_z, L_h=lipschitz.L_h,
            eps_kappa=lipschitz.eps_kappa, lipschitz=lipschitz.as_dict(),
        )
        score = risk + comp
        if best is None or score < best[0]:
            best = (score, cert)
    assert best is not None  # grid is non-empty by construction
    return best[1]