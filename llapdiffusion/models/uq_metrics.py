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


def ensemble_pit(
    draws: torch.Tensor,
    y: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """PIT values for a SAMPLED predictive distribution, given as an ensemble.

    ``draws``: [S, ...] samples; ``y``: [...] observations, broadcastable against a draw.

    Table 3's A1 arm has no closed-form law, which is why the old pre-registration excluded
    calibration metrics for it -- leaving A1-vs-A2 to rest on CRPS alone, the metric that is
    blind to a 3 % -> 87 % coverage swing (recovery plan, Phase 3). A closed-form law is not
    required: ``pit_calibration_error`` and ``reliability_curve`` take PIT values and are
    law-agnostic; only ``gaussian_pit`` assumes a Gaussian. This supplies the missing
    estimator so **every arm is scored by the same one**, rather than by two lookalikes.

    Uses the **randomized rank** form ``(rank + V) / (S + 1)`` with ``V ~ U(0,1)``, which is
    *exactly* uniform at any ensemble size when the observation is exchangeable with the draws.

    🔴 The obvious form ``(below + 0.5·ties) / S`` is NOT, and the error is large at the size
    this pipeline runs. It places the PIT on the S+1 atoms {0, 1/S, …, 1}, which is
    over-dispersed against U(0,1). Measured on a perfectly calibrated synthetic ensemble
    (N = 400 000, y and draws both ~ N(0,1)) — every column should read 0 error:

    | S | PIT-ECE | coverage @ 0.9 |
    |---|---|---|
    | 5 (campaign val profile) | 0.0553 | 0.666 |
    | **25 (`NUM_EVAL_SAMPLES`, what U3 runs)** | **0.0132** | **0.846** |
    | 100 | 0.0056 | 0.901 |
    | `gaussian_pit` (A2/A3's estimator) | 0.0009 | 0.900 |
    | **this form, S = 25** | **0.0009** | **0.899** |

    At S = 25 the naive form would have reported A1 as **5.4 pt under-covered at nominal 0.9
    with ~15× the PIT-ECE of A2/A3, on forecasts that are identically and perfectly
    calibrated** — manufacturing exactly the arm difference Table 3 exists to measure.
    (Found by Claude C in review, 2026-08-08; reproduced here before the change.)

    ``generator`` seeds the randomization so a reported table is reproducible; the campaign's
    CRPS estimator already carries RNG, but a calibration number should not move between runs
    of the same command.
    """
    if draws.dim() < 1:
        raise ValueError("draws must have a leading sample axis [S, ...]")
    S = draws.shape[0]
    if S < 2:
        raise ValueError(f"ensemble PIT needs at least 2 draws, got {S}")
    y_b = y.unsqueeze(0)
    # Ties get a uniform share of their own rank interval, which keeps the transform exact for
    # a discrete predictive law too (decoder output is continuous, so ties are ~impossible here).
    below = (draws < y_b).sum(dim=0)
    ties = (draws == y_b).sum(dim=0)
    v = torch.rand(below.shape, device=draws.device, dtype=torch.float32, generator=generator)
    u = (below + v * (ties + 1)) / float(S + 1)
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
    "ensemble_pit",
    "gaussian_nll",
    "gaussian_pit",
    "pit_calibration_error",
    "reliability_curve",
]
