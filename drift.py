"""Empirical W1 drift detector on the feature stream (regime-break triggers)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import DriftConfig


def wasserstein_1d(a: np.ndarray, b: np.ndarray, grid: int = 256) -> float:
    """1-D Wasserstein distance via W1 = int_0^1 |Q_a(u) - Q_b(u)| du."""
    q = np.linspace(0.0, 1.0, grid)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


class WassersteinDriftDetector:
    """Dimension-normalised W1 score between reference and test windows.

    A flag is raised when the mean per-feature W1 (normalised by the reference
    feature scale) exceeds `threshold`. The detector is deliberately simple and
    post-hoc: unannounced Poisson jumps cannot be anticipated, only observed
    (the paper reports a median detection lag of 3 days).
    """

    def __init__(self, cfg: DriftConfig) -> None:
        self.cfg = cfg
        self._history: deque[np.ndarray] = deque(maxlen=cfg.ref_window + cfg.test_window)
        self._score = 0.0
        self._scores: list[float] = []

    @property
    def score(self) -> float:
        return self._score

    @property
    def flagged(self) -> bool:
        return self._score > self.cfg.threshold

    def median_score(self) -> float:
        """Empirical proxy for the drift rate gamma (per-step W1)."""
        return float(np.median(self._scores)) if self._scores else 0.0

    def update(self, x_row: np.ndarray) -> float:
        self._history.append(np.asarray(x_row, dtype=np.float64))
        need = self.cfg.ref_window + self.cfg.test_window
        if len(self._history) < need:
            self._score = 0.0
            return 0.0
        window = np.stack(self._history)
        ref = window[: self.cfg.ref_window]
        test = window[self.cfg.ref_window :]
        scales = ref.std(axis=0) + 1e-12
        self._score = float(
            np.mean(
                [
                    wasserstein_1d(ref[:, j], test[:, j]) / scales[j]
                    for j in range(ref.shape[1])
                ]
            )
        )
        self._scores.append(self._score)
        return self._score

    def refresh(self) -> None:
        """Accept the new regime: the test window becomes the reference."""
        need = self.cfg.ref_window + self.cfg.test_window
        if len(self._history) < need:
            return
        accepted = list(self._history)[self.cfg.test_window :]
        self._history = deque(accepted, maxlen=need)