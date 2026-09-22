"""Protocol and model configuration.

Single source of truth for every constant of the framework. Values mirror the
paper: gate/protocol constants (Section 3), anchoring defaults (Section 4,
Table `anchor_sensitivity`: lambda_1 = 0.10, alpha_1 = 1, alpha_2 = 0.5,
refresh at regime breaks with a 63-day cool-down), and the synthetic-market
protocol (Section 4.1). All crash/time quantities are in *trading days*
(Remark `time_indexing`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class GateConfig:
    """Confidence-gate constants (Definitions `confidence_gate`, `smooth_gate`)."""

    tau: float = 0.8          # confidence threshold, in (0, 1)
    kappa: float = 20.0       # gate sharpness, > 0
    eps_inference: float = 1e-8  # denominator stabiliser, eq. `fuzzy_inference`

    def __post_init__(self) -> None:
        _check(0.0 < self.tau < 1.0, "tau must lie in (0, 1)")
        _check(self.kappa > 0.0, "kappa must be positive")
        _check(self.eps_inference > 0.0, "eps_inference must be positive")


@dataclass(frozen=True)
class MembershipConfig:
    """Neural membership networks, eqs. `membership_network` / `membership_layer`."""

    width: int = 32
    alphas: tuple[float, ...] = (1.0, 0.5)   # residual skip scales, one per block
    dropouts: tuple[float, ...] = (0.1, 0.1)  # MC-dropout rates (training only)
    bound: float = 1.0                        # B_W: spectral bound on layer weights
    eps_ln: float = 1e-5                      # LayerNorm regularisation (paper: 1e-5)

    def __post_init__(self) -> None:
        _check(self.width >= 1, "width must be >= 1")
        _check(len(self.alphas) == len(self.dropouts), "alphas/dropouts length mismatch")
        _check(all(0.0 <= p < 1.0 for p in self.dropouts), "dropout rates in [0, 1)")
        _check(self.bound > 0.0, "bound must be positive")


@dataclass(frozen=True)
class ResidualConfig:
    """Confidence-gated residual network, eq. `residual_network`."""

    hidden: tuple[int, ...] = (64, 64)
    bound: float = 1.0    # spectral weight bound (feeds L_a)
    g_inf: float = 1.0    # G_infty: hard clamp of the residual output
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _check(all(h >= 1 for h in self.hidden), "hidden widths must be >= 1")
        _check(self.g_inf > 0.0, "g_inf must be positive")


@dataclass(frozen=True)
class ModelConfig:
    K_max: int = 16          # padded-space slot count (Theorem `complexity_bound`)
    K0: int = 4              # initial (seed) rules
    y_bound: float = 1.0     # Y = [-y_bound, y_bound]
    gate: GateConfig = GateConfig()
    membership: MembershipConfig = MembershipConfig()
    residual: ResidualConfig = ResidualConfig()
    disable_gate: bool = False  # ablation: Gamma == 1
    use_rules: bool = True      # ablation: MLP-only variant when False

    def __post_init__(self) -> None:
        _check(1 <= self.K0 <= self.K_max, "need 1 <= K0 <= K_max")


@dataclass(frozen=True)
class EvolutionConfig:
    """Rule-evolution protocol constants (Section 3.5)."""

    eta_birth: float = 0.35    # sustained low confidence triggers birth
    tau_merge: float = 0.90    # membership-correlation merge threshold
    eps_prune: float = 0.02    # rule-weight prune threshold
    eps_act: float = 0.05      # mean-activation liveness threshold
    warmup_days: int = 21      # newborn rules enter A only after warm-up
    evolve_every: int = 21      # evolution cadence (trading days)
    min_rules: int = 2          # never prune below this many alive rules
    ema_beta: float = 0.98      # EMA for activation statistics
    mu_window: int = 256        # rolling window for membership correlation

    def __post_init__(self) -> None:
        _check(0.0 < self.tau_merge <= 1.0, "tau_merge in (0, 1]")
        _check(self.warmup_days >= 1, "warmup_days must be >= 1")


@dataclass(frozen=True)
class TrainingConfig:
    """Objective weights (eq. `total_objective`) and optimisation protocol."""

    lr: float = 3e-4
    batch: int = 64
    steps_per_observation: int = 2
    replay_cap: int = 2048
    pretrain_epochs: int = 3
    grad_clip: float = 5.0
    task: str = "pnl"           # "pnl" (training) | "clipped_pnl" (analysis)
    z_max: float = 3.0          # Z_max: label clipping / Lipschitz constant M
    lambda1: float = 0.10       # Lyapunov (anchoring + damping)
    lambda2: float = 0.05       # calibration (Brier)
    lambda3: float = 0.01       # sparsity
    alpha1: float = 1.0         # anchoring potential weight
    alpha2: float = 0.5         # damping weight
    anchor_cooldown: int = 63   # re-anchoring cool-down (days)

    def __post_init__(self) -> None:
        _check(self.task in ("pnl", "clipped_pnl"), "unknown task loss")
        for name in ("lambda1", "lambda2", "lambda3"):
            _check(getattr(self, name) >= 0.0, f"{name} must be >= 0")


@dataclass(frozen=True)
class PACBayesConfig:
    """Practical instantiation of the prior/posterior (Theorem `drift_pac_bayes`)."""

    sigma_p: float = 0.05      # prior std (data-independent scale)
    grid_size: int = 12         # dyadic sigma_Q grid; union bound absorbed
    m_draws: int = 100         # Monte-Carlo posterior draws
    delta: float = 0.05        # confidence level
    risk_budget: float = 0.35   # B in Cor. `reliability_horizon`
    n_cal: int = 84            # calibration-split cap (paper: n_c <= 84)

    def __post_init__(self) -> None:
        _check(0.0 < self.delta < 1.0, "delta in (0, 1)")
        _check(self.m_draws >= 1 and self.grid_size >= 1, "non-positive grid/draws")


@dataclass(frozen=True)
class MarketConfig:
    """Heston + jump DGP (Definition 15) and crash injection, all in day units."""

    mu: float = 0.08             # annualised drift
    kappa: float = 3.0           # variance mean-reversion speed
    theta: float = 0.04          # long-run variance (vol ~ 20%)
    sigma_v: float = 0.5         # volatility of volatility
    rho: float = -0.7            # price/variance shock correlation (leverage)
    jump_mean: float = -0.02     # log jump amplitude mean
    jump_std: float = 0.03       # log jump amplitude std
    jump_intensity: float = 0.5  # Poisson intensity per year
    substeps: int = 10           # intraday execution checkpoints per day
    crash_day: int = 500         # crash start (trading days; Remark time_indexing)
    crash_len: int = 10          # crash duration (days)
    crash_vol_mult: float = 2.0  # instantaneous variance multiplier at crash
    crash_drift_sigma: float = -2.0  # extra daily drift, in units of sigma_daily

    def __post_init__(self) -> None:
        _check(-1.0 < self.rho < 1.0, "rho must lie in (-1, 1)")
        _check(self.substeps >= 1, "substeps must be >= 1")
        _check(self.crash_day >= 0, "crash_day must be >= 0")


@dataclass(frozen=True)
class ExecutionConfig:
    v0: float = 1_000_000.0
    cost_bps: float = 5.0       # linear slippage per unit turnover
    leverage: float = 1.5        # gross cap |pi| <= leverage
    rf_annual: float = 0.02


@dataclass(frozen=True)
class DriftConfig:
    ref_window: int = 126
    test_window: int = 21
    threshold: float = 0.4       # dimensionless W1 score


@dataclass(frozen=True)
class BacktestConfig:
    T_days: int = 1008
    warmup: int = 252
    horizon: int = 5            # h-day forward label
    n_paths: int = 50
    seed0: int = 42             # seeds = seed0 .. seed0 + n_paths - 1
    monitor_T: int = 16         # shadow MC-dropout passes (never gates orders)
    calib_modulo: int = 2       # matured labels alternate train/calibration split


@dataclass(frozen=True)
class EFNNConfig:
    """Root aggregate configuration."""

    model: ModelConfig = ModelConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    training: TrainingConfig = TrainingConfig()
    pacbayes: PACBayesConfig = PACBayesConfig()
    market: MarketConfig = MarketConfig()
    execution: ExecutionConfig = ExecutionConfig()
    drift: DriftConfig = DriftConfig()
    backtest: BacktestConfig = BacktestConfig()

    def with_ablation(self, name: str) -> "EFNNConfig":
        """One-factor-at-a-time ablation variants (Section 4.4.1)."""
        if name == "full":
            return self
        if name == "no_lyapunov":
            return replace(self, training=replace(self.training, lambda1=0.0))
        if name == "no_calibration":
            return replace(self, training=replace(self.training, lambda2=0.0))
        if name == "no_sparsity":
            return replace(self, training=replace(self.training, lambda3=0.0))
        if name == "no_gate":
            return replace(self, model=replace(self.model, disable_gate=True))
        if name == "mlp_only":
            return replace(self, model=replace(self.model, use_rules=False))
        raise ValueError(f"unknown ablation: {name!r}")