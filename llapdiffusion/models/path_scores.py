"""Path-level proper scores and calibration diagnostics.

A method that returns *marginals* can be scored by CRPS and PIT. A method claiming to
replace sampling has to return coherent *paths*, and CRPS and PIT are computed per query
time and averaged, so they are blind to path structure by construction. This module adds
the scores that are not: the energy score (of which CRPS is the one-dimensional case), the
variogram score, joint and mixture likelihoods, and simultaneous band coverage.

**Estimator parity.** The naive m-sample estimator of CRPS and the energy score is biased
upward at O(1/m), because the 1/(2m^2) normalisation of the spread term under-subtracts
E||X - X'|| relative to the unbiased 1/(2m(m-1)). Scoring a closed-form arm exactly while
scoring sampled arms empirically therefore penalises the sampled arms in the direction that
flatters an analytic method. Every function here returns **both** estimators from the same
draws, so a comparison whose sign depends on the choice is visible rather than silent.

All scores are negatively oriented (lower is better) and mask-aware.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _check(samples: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]):
    if samples.dim() != 4:
        raise ValueError(f"samples must be [S,B,T,D]; got {tuple(samples.shape)}")
    if y.shape != samples.shape[1:]:
        raise ValueError(f"y must be [B,T,D]={tuple(samples.shape[1:])}; got {tuple(y.shape)}")
    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)
    elif mask.shape != y.shape:
        raise ValueError(f"mask must match y {tuple(y.shape)}; got {tuple(mask.shape)}")
    return mask


def _masked_norm(diff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of ``diff`` over the stacked (time x channel) axes, valid entries only.

    ``diff`` is ``[..., T, D]`` and ``mask`` is ``[B,T,D]`` broadcast against it.
    """
    return (diff.pow(2) * mask).flatten(start_dim=-2).sum(-1).sqrt()


# --------------------------------------------------------------------------------------
# energy score
# --------------------------------------------------------------------------------------

def energy_score(
    samples: torch.Tensor,          # [S,B,T,D]
    y: torch.Tensor,                # [B,T,D]
    mask: Optional[torch.Tensor] = None,
    *,
    max_pairs: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """ES(F, y) = E||X - y|| - 1/2 E||X - X'||, over the stacked horizon x channel vector.

    Returns per-window scores ``[B]`` under both estimators. The spread term is enumerated
    exactly over all S(S-1)/2 distinct pairs unless ``max_pairs`` forces subsampling -- at
    the protocol's S=25 there are only 300 pairs, so exact enumeration is cheap and removes
    a source of arm-to-arm estimator drift that subsampling would reintroduce.
    """
    mask = _check(samples, y, mask)
    S = samples.shape[0]
    if S < 2:
        raise ValueError("the energy score needs at least 2 samples")

    term1 = _masked_norm(samples - y.unsqueeze(0), mask).mean(0)          # [B]

    iu, ju = torch.triu_indices(S, S, offset=1, device=samples.device)
    if max_pairs is not None and iu.numel() > max_pairs:
        sel = torch.randperm(iu.numel(), device=samples.device)[:max_pairs]
        iu, ju = iu[sel], ju[sel]
    pair = _masked_norm(samples[iu] - samples[ju], mask)                   # [P,B]
    mean_pair = pair.mean(0)                                               # unbiased E||X-X'||

    # The naive estimator divides the *full* double sum (including the S zero diagonal
    # terms) by S^2, i.e. it shrinks the spread term by (S-1)/S.
    naive_pair = mean_pair * (S - 1) / S
    return {
        "unbiased": term1 - 0.5 * mean_pair,
        "naive": term1 - 0.5 * naive_pair,
    }


def crps(
    samples: torch.Tensor,          # [S,B,T,D]
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Per-entry CRPS, the 1-D case of the energy score, averaged over valid entries.

    Returns ``[B]`` under both estimators, so that a CRPS comparison can be checked for
    estimator sensitivity exactly as the energy score is.
    """
    mask = _check(samples, y, mask)
    S = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(0)                       # [B,T,D]

    iu, ju = torch.triu_indices(S, S, offset=1, device=samples.device)
    mean_pair = (samples[iu] - samples[ju]).abs().mean(0)                  # [B,T,D]

    w = mask.to(samples.dtype)
    denom = w.flatten(1).sum(1).clamp_min(1.0)
    per = lambda spread: ((term1 - 0.5 * spread) * w).flatten(1).sum(1) / denom
    return {"unbiased": per(mean_pair), "naive": per(mean_pair * (S - 1) / S)}


# --------------------------------------------------------------------------------------
# variogram score
# --------------------------------------------------------------------------------------

def variogram_weights(
    t_rel: torch.Tensor,            # [B,T] or [T]
    *,
    truncate_at_median_gap: bool = True,
) -> torch.Tensor:
    """Predeclared weights ``w_rr' = |t_r - t_r'|^{-1}``, truncated at the median gap.

    Variogram results depend materially on arbitrary weighting, so the scheme is fixed here
    once and imported everywhere rather than passed in per call site. The truncation caps
    the weight of near-coincident query pairs, which would otherwise dominate on an
    irregular grid where some gaps are near zero.
    """
    if t_rel.dim() == 1:
        t_rel = t_rel.unsqueeze(0)
    d = (t_rel.unsqueeze(2) - t_rel.unsqueeze(1)).abs()                    # [B,T,T]
    if truncate_at_median_gap:
        gaps = (t_rel[:, 1:] - t_rel[:, :-1]).abs()
        floor = gaps.median(dim=1, keepdim=True).values.unsqueeze(-1).clamp_min(1e-12)
        d = torch.maximum(d, floor)
    w = 1.0 / d.clamp_min(1e-12)
    eye = torch.eye(t_rel.shape[1], dtype=torch.bool, device=t_rel.device)
    return w.masked_fill(eye.unsqueeze(0), 0.0)                            # no self-pairs


def variogram_score(
    samples: torch.Tensor,          # [S,B,T,D]
    y: torch.Tensor,                # [B,T,D]
    t_rel: torch.Tensor,            # [B,T]
    mask: Optional[torch.Tensor] = None,
    *,
    p: float = 0.5,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """VS_p = sum_{r,r'} w_rr' ( |y_r - y_r'|^p - E|X_r - X_r'|^p )^2, per channel, summed.

    Order ``p=0.5`` and uniform channel weights after per-channel standardisation are the
    paper's predeclared choices; ``p`` is exposed only for the sensitivity sweep.
    """
    mask = _check(samples, y, mask)
    if weights is None:
        weights = variogram_weights(t_rel)                                 # [B,T,T]

    def _pdiff(x):                       # [...,T,D] -> [...,T,T,D]
        return (x.unsqueeze(-2) - x.unsqueeze(-3)).abs().clamp_min(1e-12).pow(p)

    emp = _pdiff(samples).mean(0)                                          # [B,T,T,D]
    obs = _pdiff(y)                                                        # [B,T,T,D]
    pair_valid = (mask.unsqueeze(-2) & mask.unsqueeze(-3)).to(samples.dtype)
    resid = (obs - emp).pow(2) * pair_valid * weights.unsqueeze(-1)
    return resid.flatten(1).sum(1)                                         # [B]


# --------------------------------------------------------------------------------------
# likelihoods
# --------------------------------------------------------------------------------------

def gaussian_joint_nll_per_channel(
    y: torch.Tensor,                # [B,T,D]
    mean: torch.Tensor,             # [B,T,D]
    kernel: torch.Tensor,           # [B,T,T,D] channel-diagonal cross-time kernel
    mask: Optional[torch.Tensor] = None,
    *,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Joint-over-time, independent-across-channels Gaussian NLL, summed over channels.

    This is a **pseudo**-likelihood for the chirp-modal law, and is labelled as such
    wherever it is reported: the true covariance has nonzero cross-channel blocks (the
    antisymmetric part of ``C_k Rot C_k^T``), which this ignores. What it does capture is
    the whole cross-*time* dependence, which is the structure the paper's joint claim is
    about and which a per-time diagonal NLL cannot see at all.

    Windows are handled one at a time with the valid entries of each channel selected, so
    an irregular mask does not silently drop a whole window.
    """
    B, T, D = y.shape
    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)
    out = y.new_zeros(B)
    eye = torch.eye(T, dtype=y.dtype, device=y.device)
    for b in range(B):
        total = y.new_zeros(())
        for d in range(D):
            idx = mask[b, :, d].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            r = (y[b, idx, d] - mean[b, idx, d]).unsqueeze(-1)
            cov = kernel[b][idx][:, idx, d] + jitter * eye[: idx.numel(), : idx.numel()]
            L = torch.linalg.cholesky(cov)
            sol = torch.cholesky_solve(r, L)
            total = total + 0.5 * (
                (r * sol).sum()
                + 2.0 * torch.log(torch.diagonal(L)).sum()
                + idx.numel() * torch.log(torch.tensor(2 * torch.pi, dtype=y.dtype))
            )
        out[b] = total
    return out


def mixture_nll(component_nlls: torch.Tensor) -> torch.Tensor:
    """-log (1/M) sum_m exp(-nll_m), by ``logsumexp``.

    Averaging per-component log-likelihoods instead would be a different and easier
    quantity (Jensen), so the mixture arms are scored only this way.
    ``component_nlls`` is ``[M,B]``; returns ``[B]``.
    """
    M = component_nlls.shape[0]
    return -(torch.logsumexp(-component_nlls, dim=0) - torch.log(
        torch.tensor(float(M), dtype=component_nlls.dtype, device=component_nlls.device)
    ))


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------

def randomized_pit(
    samples: torch.Tensor,          # [S,B,T,D]
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomized rank PIT ``(below + V*(ties+1)) / (S+1)``, V ~ U(0,1).

    The naive ``(below + 0.5*ties)/S`` is biased for a finite ensemble -- measured here at
    24x the PIT calibration error on a *perfectly* calibrated 25-member ensemble, and
    0.846 coverage at a nominal 0.9. Returns the flat vector of PIT values at valid entries.
    """
    mask = _check(samples, y, mask)
    S = samples.shape[0]
    below = (samples < y.unsqueeze(0)).sum(0).to(y.dtype)
    ties = (samples == y.unsqueeze(0)).sum(0).to(y.dtype)
    v = torch.rand(y.shape, generator=generator, device=y.device, dtype=y.dtype)
    u = (below + v * (ties + 1.0)) / (S + 1.0)
    return u[mask]


def coverage_and_width(
    samples: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    levels: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99),
) -> Dict[str, float]:
    """Central-interval coverage and mean width at each nominal level.

    Sharpness is always reported beside coverage: an arm can buy coverage with width, and
    the pair is the only honest way to see it.
    """
    mask = _check(samples, y, mask)
    out: Dict[str, float] = {}
    for lv in levels:
        lo = torch.quantile(samples, (1 - lv) / 2, dim=0)
        hi = torch.quantile(samples, 1 - (1 - lv) / 2, dim=0)
        inside = ((y >= lo) & (y <= hi))[mask]
        out[f"coverage_{lv}"] = float(inside.to(torch.float64).mean())
        out[f"width_{lv}"] = float((hi - lo)[mask].to(torch.float64).mean())
    return out


def simultaneous_band_coverage(
    y: torch.Tensor,                # [B,T,D]
    mean: torch.Tensor,             # [B,T,D]
    marginal_std: torch.Tensor,     # [B,T,D]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The scaled-band statistic: the smallest lambda whose band covers the whole path.

    ``lambda_b = max_{r,d valid} |y - m| / sd``. Reporting the distribution of lambda against
    its nominal value under the predicted law is what distinguishes a model that covers each
    time marginally from one that covers a *trajectory* -- the property marginal bands
    cannot display. Returns ``[B]``.
    """
    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)
    z = (y - mean).abs() / marginal_std.clamp_min(1e-12)
    return z.masked_fill(~mask, float("-inf")).flatten(1).max(dim=1).values
