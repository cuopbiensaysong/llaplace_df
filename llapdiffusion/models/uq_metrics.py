"""Calibration metrics for the Theorem-C analytic (Gaussian) predictive law.

All functions are pure and operate on aligned tensors of observations and predicted
per-element Gaussian parameters (mean, variance). Masks select the elements that
carry supervision (observed entries).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

_SQRT2 = math.sqrt(2.0)


def gaussian_pit(
    y: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    min_var: float = 1e-6,
) -> torch.Tensor:
    """Probability integral transform u = Phi((y - mean)/std), flattened over mask.

    A perfectly calibrated predictive law gives u ~ Uniform(0, 1).
    """
    std = variance.clamp_min(min_var).sqrt()
    u = 0.5 * (1.0 + torch.erf((y - mean) / (std * _SQRT2)))
    if mask is not None:
        u = u[torch.as_tensor(mask, device=u.device, dtype=torch.bool)]
    return u.reshape(-1)


def pit_calibration_error(u: torch.Tensor, *, num_bins: int = 20) -> float:
    """Mean absolute deviation between the empirical CDF of the PIT values and the
    uniform CDF, evaluated at bin edges (0 for perfect calibration)."""
    if u.numel() == 0:
        raise ValueError("PIT calibration error needs at least one PIT value.")
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=u.device, dtype=u.dtype)[1:]
    empirical = (u.reshape(-1, 1) <= edges.reshape(1, -1)).float().mean(dim=0)
    return float((empirical - edges).abs().mean().item())


def reliability_curve(
    u: torch.Tensor,
    *,
    levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> Dict[float, float]:
    """Empirical coverage of central predictive intervals at the nominal levels.

    For a central interval of nominal mass p, the PIT value falls inside
    [(1-p)/2, (1+p)/2]; calibrated forecasts give coverage == p.
    """
    if u.numel() == 0:
        raise ValueError("Reliability curve needs at least one PIT value.")
    out: Dict[float, float] = {}
    for level in levels:
        p = float(level)
        lo, hi = (1.0 - p) / 2.0, (1.0 + p) / 2.0
        out[p] = float(((u >= lo) & (u <= hi)).float().mean().item())
    return out


def gaussian_nll(
    y: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    min_var: float = 1e-6,
) -> float:
    """Mean per-element Gaussian negative log-likelihood (with the constant term)."""
    var = variance.clamp_min(min_var)
    nll = 0.5 * (torch.log(2.0 * math.pi * var) + (y - mean).pow(2) / var)
    if mask is not None:
        nll = nll[torch.as_tensor(mask, device=nll.device, dtype=torch.bool)]
    if nll.numel() == 0:
        raise ValueError("Gaussian NLL needs at least one observed element.")
    return float(nll.mean().item())


__all__ = [
    "gaussian_nll",
    "gaussian_pit",
    "pit_calibration_error",
    "reliability_curve",
]
