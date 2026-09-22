"""Online trainer with replay buffer, anchoring, and the total objective."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .config import TrainingConfig
from .losses import (
    brier_calibration,
    directional_event,
    lyapunov_regularizer,
    sparsity_regularizer,
    task_loss,
)
from .model import EFNN


class ReplayBuffer:
    """Uniform replay buffer over (features, volatility-scaled label) pairs."""

    def __init__(self, capacity: int, d_input: int, seed: int) -> None:
        self.capacity = capacity
        self._x = np.zeros((capacity, d_input), dtype=np.float32)
        self._z = np.zeros(capacity, dtype=np.float32)
        self._n = 0
        self._rng = random.Random(seed)

    def add(self, x: np.ndarray, z: float) -> None:
        idx = self._n % self.capacity
        self._x[idx] = x
        self._z[idx] = z
        self._n += 1

    def __len__(self) -> int:
        return min(self._n, self.capacity)

    def sample(self, batch: int) -> tuple[Tensor, Tensor]:
        n = len(self)
        if n == 0:
            raise RuntimeError("replay buffer is empty")
        idx = np.array(
            [self._rng.randrange(n) for _ in range(min(batch, n))], dtype=np.int64
        )
        return (
            torch.from_numpy(self._x[idx]),
            torch.from_numpy(self._z[idx]),
        )


@dataclass
class StepLosses:
    total: float
    task: float
    lyapunov: float
    calibration: float
    sparsity: float


class OnlineTrainer:
    """Performs gradient steps on the stability-constrained objective.

    Optimisation proceeds through the smooth-gate surrogate (Remark
    `nonsmooth_operators`); deployment always evaluates the deployed gate.
    """

    def __init__(self, model: EFNN, cfg: TrainingConfig, d_input: int, seed: int):
        self.model = model
        self.cfg = cfg
        self.buffer = ReplayBuffer(cfg.replay_cap, d_input, seed)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.anchor: Tensor | None = None
        self._prev = model.flat_parameters().clone()
        self._last_losses: StepLosses | None = None

    # ------------------------------------------------------------------ API
    def set_anchor(self, flat: Tensor) -> None:
        self.anchor = flat.detach().clone()

    def observe(self, x: np.ndarray, z: float) -> None:
        self.buffer.add(x, z)
        for _ in range(self.cfg.steps_per_observation):
            if len(self.buffer) < 2:
                break
            xb, zb = self.buffer.sample(self.cfg.batch)
            self._step(xb, zb)

    def pretrain(self, epochs: int) -> None:
        for _ in range(epochs):
            for _ in range(max(1, len(self.buffer) // self.cfg.batch)):
                xb, zb = self.buffer.sample(self.cfg.batch)
                self._step(xb, zb)

    @property
    def last_losses(self) -> StepLosses | None:
        return self._last_losses

    # -------------------------------------------------------------- internals
    def _step(self, xb: Tensor, zb: Tensor) -> None:
        cfg = self.cfg
        model = self.model
        model.train()
        model.gate_mode = "surrogate"  # optimise the smooth surrogate

        out = model(xb)
        y = out.y_hat.squeeze(-1)

        loss_task = task_loss(y, zb, cfg.task)
        z_event = directional_event(y.detach(), zb)
        loss_calib = brier_calibration(out.mu_max, z_event)

        flat = model.flat_parameters()
        loss_lyap = lyapunov_regularizer(
            flat, self.anchor, self._prev, cfg.alpha1, cfg.alpha2
        )
        alive = model.registry.alive_mask_torch(xb.device)
        rule_w = torch.sigmoid(model.rule_weight_raw) * alive
        loss_sparse = sparsity_regularizer(rule_w)

        total = (
            loss_task
            + cfg.lambda1 * loss_lyap
            + cfg.lambda2 * loss_calib
            + cfg.lambda3 * loss_sparse
        )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self._prev = model.flat_parameters().clone()
        self._last_losses = StepLosses(
            total=float(total.detach()),
            task=float(loss_task.detach()),
            lyapunov=float(loss_lyap.detach()),
            calibration=float(loss_calib.detach()),
            sparsity=float(loss_sparse.detach()),
        )


class AnchorManager:
    """Re-anchoring with regime-break triggering and a hard cool-down.

    The anchor stores parameters trained strictly prior to the current epoch,
    so it is a valid data-dependent prior mean for the calibration split
    (Corollary `snapshot_adaptation`).
    """

    def __init__(self, cooldown_days: int) -> None:
        self.cooldown_days = cooldown_days
        self._last_refresh = -10**9
        self.anchor: Tensor | None = None

    def force(self, model: EFNN, day: int) -> None:
        self.anchor = model.flat_parameters().clone()
        self._last_refresh = day

    def maybe_refresh(self, model: EFNN, day: int, drift_flag: bool) -> bool:
        if not drift_flag:
            return False
        if day - self._last_refresh < self.cooldown_days:
            return False
        self.force(model, day)
        return True