"""EFNN: Epistemic Fuzzy Neural Networks for bounded-regression trading."""
from __future__ import annotations

from .certificates import (
    LipschitzReport,
    PACBayesCertificate,
    certify,
    complexity_term,
    compute_lipschitz,
    gaussian_kl,
    surrogate_gap,
)
from .config import EFNNConfig
from .conformal import fit_event_conformal, fit_regression_conformal
from .model import EFNN, EFNNOutput, MLPOnly, build_model, fuzzy_inference
from .registry import RuleRegistry

__version__ = "0.1.0"

__all__ = [
    "EFNN", "EFNNConfig", "EFNNOutput", "MLPOnly", "RuleRegistry",
    "build_model", "fuzzy_inference",
    "PACBayesCertificate", "LipschitzReport", "certify", "compute_lipschitz",
    "complexity_term", "gaussian_kl", "surrogate_gap",
    "fit_event_conformal", "fit_regression_conformal",
]