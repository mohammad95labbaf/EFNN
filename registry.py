"""Active-rule registry.

Implements the active set A of Definition `active_registry`: newborn rules
participate in fuzzy inference immediately but enter the confidence certificate
mu_max = max_{k in A} mu_{R_k} only after the warm-up period. Dead slots are
pinned to zero rule weight (the padded-space embedding).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RuleRecord:
    slot: int
    birth_day: int
    merged_from: tuple[int, ...] = field(default_factory=tuple)


class RuleRegistry:
    """Structural bookkeeping of alive rules and the active set A."""

    def __init__(self, K_max: int, warmup_days: int, initial_rules: int) -> None:
        if not 1 <= initial_rules <= K_max:
            raise ValueError("initial_rules must satisfy 1 <= K0 <= K_max")
        self.K_max = K_max
        self.warmup_days = warmup_days
        self._records: dict[int, RuleRecord] = {
            k: RuleRecord(slot=k, birth_day=0) for k in range(initial_rules)
        }
        self._day = 0

    # ---------------------------------------------------------------- state
    def set_day(self, day: int) -> None:
        self._day = int(day)

    @property
    def day(self) -> int:
        return self._day

    def birth(self, slot: int, day: int) -> RuleRecord:
        if slot in self._records:
            raise ValueError(f"slot {slot} is already alive")
        record = RuleRecord(slot=slot, birth_day=int(day))
        self._records[slot] = record
        return record

    def kill(self, slot: int) -> None:
        self._records.pop(slot, None)

    def record(self, slot: int) -> RuleRecord:
        return self._records[slot]

    # ---------------------------------------------------------------- masks
    def alive_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._records))

    def is_alive(self, slot: int) -> bool:
        return slot in self._records

    def _active(self, slot: int) -> bool:
        rec = self._records.get(slot)
        return rec is not None and (self._day - rec.birth_day) >= self.warmup_days

    def alive_mask(self) -> np.ndarray:
        mask = np.zeros(self.K_max, dtype=bool)
        for slot in self.alive_slots():
            mask[slot] = True
        return mask

    def active_mask(self) -> np.ndarray:
        mask = np.zeros(self.K_max, dtype=bool)
        for slot in self.alive_slots():
            if self._active(slot):
                mask[slot] = True
        return mask

    # ------------------------------------------------------- torch adapters
    def alive_mask_torch(self, device: torch.device) -> Tensor:
        return torch.as_tensor(self.alive_mask(), dtype=torch.bool, device=device)

    def active_mask_torch(self, device: torch.device) -> Tensor:
        return torch.as_tensor(self.active_mask(), dtype=torch.bool, device=device)