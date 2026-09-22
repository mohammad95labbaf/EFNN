"""EFNN model: Algorithm 1 (deployed forward pass) and the smooth-gate surrogate.

Invariants enforced here (all justified in the paper's corrected layer):
  * The confidence gate is evaluated EXACTLY ONCE per forward pass; the
    conditional of step 4 of Algorithm 1 *is* the definition of g_phi
    (Definition `confidence_gate`). No further gate factor multiplies the
    residual. `gate_evaluations` exposes the count; `tests/test_gate_once.py`
    asserts the count and bit-exact agreement with a single-gate reference.
  * Optimisation proceeds through the smooth surrogate (Remark
    `nonsmooth_operators`): set `gate_mode = "surrogate"`.
  * The parameter vector lives on the fixed padded space of dimension
    dim(Omega_full with K := K_max) regardless of the number of live rules
    (padded-space embedding), so PAC-Bayes operates on a fixed dimension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .registry import RuleRegistry

GateMode = Literal["deployed", "surrogate"]


# --------------------------------------------------------------------------- #
#  Normalised fuzzy inference (eq. fuzzy_inference)                            #
# --------------------------------------------------------------------------- #
def fuzzy_inference(mu: Tensor, C: Tensor, w: Tensor, eps: float) -> Tensor:
    """F(mu; C, w) = sum_k w_k mu_k c_k / (sum_k w_k mu_k + eps).

    Args:
        mu: (B, K) membership vector.
        C:  (d_y, K) consequent matrix.
        w:  (K,) non-negative rule weights.
    Returns:
        (B, d_y) rule-based prediction.
    """
    wmu = w * mu                                    # (B, K)
    num = torch.einsum("bk,yk->by", wmu, C)         # (B, d_y)
    den = wmu.sum(dim=-1, keepdim=True) + eps       # (B, 1)
    return num / den


@dataclass(frozen=True)
class EFNNOutput:
    y_hat: Tensor    # (B, d_y) fused prediction, eq. hypothesis_class
    mu_max: Tensor   # (B,)   active-rule maximum membership
    mu: Tensor       # (B, K_max) membership vector
    y_rule: Tensor   # (B, d_y) rule subsystem output
    residual: Tensor  # (B, d_y) gated residual contribution


# --------------------------------------------------------------------------- #
#  Weight-normalised building blocks (Assumption Lipschitz / B_W bound)       #
# --------------------------------------------------------------------------- #
class SpectralBoundedLinear(nn.Module):
    """Linear layer with operator norm bounded by `bound` (weight normalisation).

    The rescaling shrinks the weight only when its spectral norm exceeds the
    bound, so unconstrained initialisations are preserved when already valid.
    """

    def __init__(self, in_features: int, out_features: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def effective_weight(self) -> Tensor:
        sigma = torch.linalg.matrix_norm(self.weight, ord=2)
        scale = torch.clamp(self.bound / (sigma + 1e-12), max=1.0)
        return self.weight * scale

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class MembershipLayer(nn.Module):
    """f^{(l)}(h) = alpha * Dropout(LN(W h + b)) + h  (eq. membership_layer)."""

    def __init__(self, width: int, alpha: float, dropout: float, eps_ln: float) -> None:
        super().__init__()
        self.lin = SpectralBoundedLinear(width, width, bound=1.0)
        self.ln = nn.LayerNorm(width, eps=eps_ln)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

    def forward(self, h: Tensor) -> Tensor:
        z = self.ln(self.lin(h))
        if self.training and self.dropout > 0.0:
            z = F.dropout(z, self.dropout, self.training)
        return h + self.alpha * z


class MembershipNetwork(nn.Module):
    """Per-rule membership network (eq. membership_network).

    Input projection -> residual blocks -> sigmoid head. Bounded output [0, 1]
    by construction; C^inf in evaluation mode thanks to the eps_LN-regularised
    LayerNorm (Proposition `membership_smoothness`).
    """

    def __init__(self, d_input: int, cfg: ModelConfig) -> None:
        super().__init__()
        m = cfg.membership
        self.input = SpectralBoundedLinear(d_input, m.width, bound=m.bound)
        self.blocks = nn.ModuleList(
            [
                MembershipLayer(m.width, alpha, p, m.eps_ln)
                for alpha, p in zip(m.alphas, m.dropouts, strict=True)
            ]
        )
        self.head = SpectralBoundedLinear(m.width, 1, bound=m.bound)
        self._bound = m.bound

    def forward(self, x: Tensor) -> Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return torch.sigmoid(self.head(h)).squeeze(-1)  # (B,)

    def estimate_c_ln(self, x: Tensor) -> float:
        """Empirical sup of the LayerNorm Jacobian gain 1/sqrt(var(h) + eps_LN)."""
        eps = self.blocks[0].ln.eps if self.blocks else 1e-5
        gains: list[float] = []

        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            h = inputs[0].detach()
            var = h.var(dim=-1, unbiased=False)
            gains.append(float((1.0 / torch.sqrt(var + eps)).max()))

        handles = [
            block.ln.register_forward_pre_hook(hook)
            for block in self.blocks
        ]
        try:
            with torch.no_grad():
                self(x)
        finally:
            for handle in handles:
                handle.remove()
        return max(gains) if gains else 1.0

    def lipschitz_upper_bound(self, c_ln: float) -> float:
        """L_k <= (C_LN^L / 4) * prod_l (1 + alpha_l * B_W)  (eq. lipschitz_bound).

        The extra factor B_W^2 covers the un-normalised input projection and
        sigmoid head, making the bound strictly more conservative than the
        proposition's expression.
        """
        prod = self._bound ** 2
        for block in self.blocks:
            prod *= 1.0 + block.alpha * self._bound
        return (c_ln ** len(self.blocks)) * prod / 4.0


class ResidualNetwork(nn.Module):
    """psi(MLP_phi(x)) with hard output clamp to +/- G_infty."""

    def __init__(self, d_input: int, d_output: int, cfg: ModelConfig) -> None:
        super().__init__()
        r = cfg.residual
        layers: list[nn.Module] = []
        in_dim = d_input
        for h in r.hidden:
            layers.append(SpectralBoundedLinear(in_dim, h, bound=r.bound))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(SpectralBoundedLinear(in_dim, d_output, bound=r.bound))
        self.net = nn.Sequential(*layers)
        self.g_inf = float(r.g_inf)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).clamp(-self.g_inf, self.g_inf)

    def lipschitz(self) -> float:
        prod = 1.0
        for module in self.net:
            if isinstance(module, SpectralBoundedLinear):
                prod *= module.bound  # ReLU and the clamp are 1-Lipschitz
        return prod


# --------------------------------------------------------------------------- #
#  The EFNN (Algorithm 1)                                                      #
# --------------------------------------------------------------------------- #
class EFNN(nn.Module):
    """Epistemic Fuzzy Neural Network, eq. hypothesis_class.

    h_omega(x) = F_inference(mu(x); C, w) + g_phi(x),  with
    g_phi(x) = psi(MLP_phi(x)) * Gamma(x; mu(x), tau)     [Definition confidence_gate]
    """

    def __init__(self, d_input: int, cfg: ModelConfig, registry: RuleRegistry) -> None:
        super().__init__()
        self.cfg = cfg
        self.registry = registry
        K = cfg.K_max
        self.memberships = nn.ModuleList(
            [MembershipNetwork(d_input, cfg) for _ in range(K)]
        )
        self.rule_weight_raw = nn.Parameter(torch.zeros(K))       # sigmoid -> [0,1]
        self.consequent_raw = nn.Parameter(torch.zeros(1, K))      # tanh*bound
        self.residual_net = ResidualNetwork(d_input, 1, cfg)
        self.gate_mode: GateMode = "deployed"
        self.disable_gate = cfg.disable_gate
        self._gate_evals = 0

    # ------------------------------------------------------------- gate
    @property
    def tau(self) -> float:
        return self.cfg.gate.tau

    @property
    def kappa(self) -> float:
        return self.cfg.gate.kappa

    def _gate(self, mu_max: Tensor) -> Tensor:
        """Evaluate the confidence gate exactly once per forward pass."""
        self._gate_evals += 1
        if self.disable_gate:                       # ablation: Gamma == 1
            return torch.ones_like(mu_max)
        smooth = torch.sigmoid(self.kappa * (self.tau - mu_max))   # eq. smooth_gate
        if self.gate_mode == "surrogate":
            return smooth
        return torch.where(                         # eq. confidence_gate
            mu_max >= self.tau, torch.zeros_like(mu_max), smooth
        )

    @property
    def gate_evaluations(self) -> int:
        return self._gate_evals

    def reset_gate_counter(self) -> None:
        self._gate_evals = 0

    # ----------------------------------------------------------- forward
    def forward(self, x: Tensor) -> EFNNOutput:
        mu = torch.stack([net(x) for net in self.memberships], dim=-1)  # (B, K)
        device = x.device
        alive = self.registry.alive_mask_torch(device)
        active = self.registry.active_mask_torch(device)

        w = torch.sigmoid(self.rule_weight_raw) * alive              # inert slots pinned
        C = self.cfg.y_bound * torch.tanh(self.consequent_raw)       # (1, K)
        y_rule = fuzzy_inference(mu, C, w, self.cfg.gate.eps_inference)

        if bool(active.any()):
            mu_max = mu.masked_fill(~active, float("-inf")).amax(dim=-1)
        else:  # empty active set: max over empty set is defined as 0
            mu_max = torch.zeros(x.shape[0], device=device, dtype=mu.dtype)

        residual = self.residual_net(x) * self._gate(mu_max).unsqueeze(-1)
        return EFNNOutput(
            y_hat=y_rule + residual, mu_max=mu_max, mu=mu,
            y_rule=y_rule, residual=residual,
        )

    @torch.no_grad()
    def predict(self, x: Tensor) -> EFNNOutput:
        """Deterministic deployment forward (dropout disabled, Remark scheduling)."""
        was_training = self.training
        self.eval()
        try:
            return self(x)
        finally:
            if was_training:
                self.train()

    # ------------------------------------------- padded-space parameter API
    def _ordered_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield from sorted(self.named_parameters(), key=lambda kv: kv[0])

    def flat_parameters(self) -> Tensor:
        """Fixed-dimension flat parameter vector on the padded space Omega_full."""
        return torch.cat([p.detach().reshape(-1) for _, p in self._ordered_parameters()])

    def load_flat(self, vector: Tensor) -> None:
        with torch.no_grad():
            offset = 0
            for _, p in self._ordered_parameters():
                n = p.numel()
                p.copy_(vector[offset : offset + n].view_as(p))
                offset += n

    def rule_weights_np(self) -> "torch.Tensor":  # numpy-free accessor for cert.
        alive = torch.as_tensor(self.registry.alive_mask())
        return (torch.sigmoid(self.rule_weight_raw) * alive).detach()

    def consequents_np(self) -> Tensor:
        return (self.cfg.y_bound * torch.tanh(self.consequent_raw)).detach()

    def set_rule_weight(self, slot: int, weight: float) -> None:
        """Soft get/set on the sigmoid-parameterised rule weight."""
        weight = min(max(weight, 1e-4), 1.0 - 1e-4)
        raw = math.log(weight / (1.0 - weight))
        with torch.no_grad():
            self.rule_weight_raw[slot] = float(raw)


class MLPOnly(nn.Module):
    """Ablation variant: rule layer replaced by a plain MLP (Section 4.4.1).

    The gate/calibration/anchoring machinery is retained where meaningful, but
    with no rule subsystem the epistemic certificate is undefined; we report
    mu_max = 1 (no confidence scaling), mirroring interpretability = 0.
    """

    def __init__(self, d_input: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = ResidualNetwork(d_input, 1, cfg)

    def forward(self, x: Tensor) -> EFNNOutput:
        y = self.net(x)
        mu_max = torch.ones(x.shape[0], device=x.device)
        return EFNNOutput(y_hat=y, mu_max=mu_max, mu=y.new_zeros((x.shape[0], 0)),
                          y_rule=y.new_zeros_like(y), residual=y)

    predict = EFNN.predict  # type: ignore[assignment]


def build_model(d_input: int, cfg: ModelConfig, registry: RuleRegistry) -> nn.Module:
    """Factory honouring the `mlp_only` ablation flag."""
    if cfg.use_rules:
        return EFNN(d_input, cfg, registry)
    return MLPOnly(d_input, cfg)


class EpistemicMonitor:
    """Optional shadow MC-dropout dispersion of rule confidences.

    Runs T stochastic forward passes to estimate the epistemic dispersion of
    mu_max. Per the paper's scheduling remark it NEVER gates an order: it is a
    monitoring diagnostic executed off the decision path.
    """

    def __init__(self, model: nn.Module, T: int = 16) -> None:
        self.model = model
        self.T = int(T)

    @torch.no_grad()
    def dispersion(self, x: Tensor) -> Tensor:
        model = self.model
        gate_mode = getattr(model, "gate_mode", "deployed")
        was_training = model.training
        model.train()  # activates dropout only (no batch-norm in the architecture)
        try:
            samples = torch.stack([model(x).mu_max for _ in range(self.T)])
        finally:
            model.eval()
            if was_training:
                model.train()
            if hasattr(model, "gate_mode"):
                model.gate_mode = gate_mode  # type: ignore[assignment]
        return samples.std(dim=0)