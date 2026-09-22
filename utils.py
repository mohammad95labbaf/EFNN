"""Deterministic seeding across all RNG streams."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed python/numpy/torch (CPU and CUDA) identically.

    `deterministic` switches torch into deterministic algorithms (warn-only so
    exotic ops degrade gracefully) and sets the CUBLAS workspace env var before
    the first CUDA context is created.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")