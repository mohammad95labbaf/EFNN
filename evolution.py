"""Rule-evolution protocol: birth, merge, prune (Section 3.5)."""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import EvolutionConfig
from .model import EFNN
from .registry import RuleRegistry

BirthNoiseScale = 0.05


@dataclass(frozen=True)
class EvolutionEvent:
    day: int
    kind: str            # "birth" | "merge" | "prune"
    slots: tuple[int, ...]
    detail: str = ""


class RuleEvolver:
    """One-factor rule-lifecycle controller driven by membership statistics."""

    def __init__(self, cfg: EvolutionConfig, registry: RuleRegistry) -> None:
        self.cfg = cfg
        self.registry = registry
        self.ema_activation = np.zeros(registry.K_max, dtype=np.float64)
        self.ema_confidence = 1.0
        self._mu_window: deque[np.ndarray] = deque(maxlen=cfg.mu_window)
        self._last_birth = -10**9

    # ---------------------------------------------------------------- input
    def push(self, mu_row: np.ndarray) -> None:
        """Feed one day's membership vector (K_max,) and update statistics."""
        beta = self.cfg.ema_beta
        alive = self.registry.alive_mask()
        self.ema_activation = np.where(
            alive,
            beta * self.ema_activation + (1.0 - beta) * mu_row,
            0.0,
        )
        active = self.registry.active_mask()
        conf = float(mu_row[active].max()) if active.any() else 0.0
        self.ema_confidence = beta * self.ema_confidence + (1.0 - beta) * conf
        self._mu_window.append(mu_row.copy())

    # ---------------------------------------------------------------- engine
    def maybe_evolve(self, day: int, model: EFNN) -> list[EvolutionEvent]:
        events: list[EvolutionEvent] = []
        events += self._prune(day)
        events += self._merge(day, model)
        events += self._birth(day, model)
        return events

    # --------------------------------------------------------------- actions
    def _prune(self, day: int) -> list[EvolutionEvent]:
        cfg = self.cfg
        events: list[EvolutionEvent] = []
        alive = self.registry.alive_slots()
        if len(alive) <= cfg.min_rules:
            return events
        weights = torch.sigmoid(model.rule_weight_raw).detach().numpy()
        for slot in alive:
            dead_weight = weights[slot] < cfg.eps_prune
            dead_activation = self.ema_activation[slot] < cfg.eps_act
            if (dead_weight or dead_activation) and len(self.registry.alive_slots()) > cfg.min_rules:
                self.registry.kill(slot)
                events.append(
                    EvolutionEvent(day, "prune", (slot,),
                                   f"w={weights[slot]:.3f}, act={self.ema_activation[slot]:.3f}")
                )
        return events

    def _merge(self, day: int, model: EFNN) -> list[EvolutionEvent]:
        cfg = self.cfg
        alive = self.registry.alive_slots()
        if len(self._mu_window) < 8 or len(alive) < 2:
            return []
        window = np.stack(self._mu_window)[:, list(alive)]
        if window.std(axis=0).min() < 1e-6:
            return []
        corr = np.corrcoef(window, rowvar=False)
        events: list[EvolutionEvent] = []
        merged: set[int] = set()
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                si, sj = alive[i], alive[j]
                if si in merged or sj in merged or not self.registry.is_alive(sj):
                    continue
                if corr[i, j] > cfg.tau_merge:
                    self._merge_pair(model, si, sj)
                    merged.add(sj)
                    events.append(
                        EvolutionEvent(day, "merge", (si, sj), f"corr={corr[i, j]:.3f}")
                    )
        return events

    def _merge_pair(self, model: EFNN, keep: int, drop: int) -> None:
        with torch.no_grad():
            w = torch.sigmoid(model.rule_weight_raw)
            c = torch.tanh(model.consequent_raw)
            w_keep, w_drop = float(w[keep]), float(w[drop])
            total = w_keep + w_drop
            if total > 1e-8:
                model.consequent_raw[0, keep] = float(
                    torch.atanh(
                        torch.clamp((w_keep * c[0, keep] + w_drop * c[0, drop]) / total,
                                    -0.999, 0.999)
                    )
                )
            model.set_rule_weight(keep, min(0.95, total))
        self.registry.kill(drop)

    def _birth(self, day: int, model: EFNN) -> list[EvolutionEvent]:
        cfg = self.cfg
        if self.ema_confidence >= cfg.eta_birth:
            return []
        if len(self.registry.alive_slots()) >= self.registry.K_max:
            return []
        if day - self._last_birth < cfg.warmup_days:
            return []

        template_slot = int(np.argmax(self.ema_activation))
        free = next(
            (s for s in range(self.registry.K_max) if not self.registry.is_alive(s)), None
        )
        if free is None:
            return []

        self._clone_membership(model, template_slot, free)
        with torch.no_grad():
            model.consequent_raw[0, free] = 0.0
        model.set_rule_weight(free, 0.10)
        self.registry.birth(free, day)
        self._last_birth = day
        return [EvolutionEvent(day, "birth", (free,), f"template={template_slot}")]

    @staticmethod
    def _clone_membership(model: EFNN, src: int, dst: int) -> None:
        state = copy.deepcopy(model.memberships[src].state_dict())
        with torch.no_grad():
            for key, tensor in state.items():
                noise = torch.randn_like(tensor) * BirthNoiseScale
                state[key] = tensor + noise * (tensor.std() + 1e-8)
        model.memberships[dst].load_state_dict(state)