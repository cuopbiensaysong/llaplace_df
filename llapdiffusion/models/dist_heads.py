"""Predictive heads on a frozen history summary, sharing one interface.

Everything here maps ``E`` -- the frozen gap-aware summary of the history -- and a set of
query times to a predictive law over the latent trajectory, in **one pass**, with no
diffusion loop. That is Option A of the paper for the chirp-modal head, and read R10 (the
paper's primary novelty control) for the rest: they sit at exactly the same cost, so a tie
answers "is it the chirp-modal structure, or just an NLL head?".

Because the control matters more than the flagship if it wins, the heads are deliberately
matched: same frozen encoder, same query embedding, same trunk width, same objective, same
scoring. The only thing that differs is how the trunk's output becomes a law.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.models.chirp_gp import (
    marginal_law,
    modal_increments,
    modal_lambda_projective,
)
from llapdiffusion.models.laptrans import ChirpModalField

SIGMA2_MIN = 1e-4


def _trunk(cond_dim: int, hidden: int, depth: int = 2) -> nn.Sequential:
    layers = [nn.Linear(cond_dim, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    return nn.Sequential(*layers)


class _TimeEmbed(nn.Module):
    """Fourier features of the query offset, shared by every non-modal head.

    The chirp head needs no such thing -- its time dependence *is* the modal synthesis --
    so giving the controls an explicit time embedding is what keeps the comparison about
    the dynamical structure rather than about who can see the clock.
    """

    def __init__(self, dim: int = 32, max_period: float = 512.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_period), half) * -1.0)
        self.register_buffer("freqs", freqs)
        self.dim = 2 * half

    def forward(self, t_rel: torch.Tensor) -> torch.Tensor:      # [B,T] -> [B,T,dim]
        a = t_rel.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class ChirpGPHead(nn.Module):
    """Option A: ``E -> {rho_k(.), omega_k(.), c_k, b_k, q_k, p0_k, sigma^2}`` in one pass.

    The dynamical core is reused verbatim from the diffusion arm -- same ``ChirpModalField``,
    same closed-form synthesis, same variance quadrature -- so a CMD-GP-vs-CMD-D difference
    is attributable to the conditioning path and not to a reimplementation of the algebra.

    ``anchor_nodes > 0`` computes the variance against a fixed, query-independent grid, which
    is what makes the law projective in its *kernel* as well as its mean (see
    ``modal_lambda_projective``). It is on by default here because Option A's entire selling
    point is being a genuine process.
    """

    def __init__(
        self,
        cond_dim: int,
        d_z: int,
        *,
        k: int = 32,
        hidden: int = 256,
        num_basis: int = 4,
        horizon: float = 168.0,
        depth: int = 2,
        nugget: bool = True,
        anchor_nodes: int = 128,
        uq_init_var: float = 1.0,
        parameterization: str = "p_exact",
    ) -> None:
        super().__init__()
        self.k, self.d_z, self.horizon = int(k), int(d_z), float(horizon)
        self.anchor_nodes = int(anchor_nodes)
        self.trunk = _trunk(cond_dim, hidden, depth)
        self.field = ChirpModalField(
            k=self.k, cond_dim=hidden, num_basis=num_basis, uq_head=True,
            uq_init_var=uq_init_var, parameterization=parameterization,
            time_scale=self.horizon, rho_init_horizon=self.horizon,
            rho_basis="centered", omega_basis="centered", basis="half_integer",
        )
        self.to_theta = nn.Linear(hidden, 2 * self.k * self.d_z)
        nn.init.normal_(self.to_theta.weight, std=1e-2)
        nn.init.zeros_(self.to_theta.bias)
        self.to_nugget = nn.Linear(hidden, self.d_z) if nugget else None
        if self.to_nugget is not None:
            nn.init.zeros_(self.to_nugget.weight)
            nn.init.constant_(self.to_nugget.bias, math.log(math.expm1(0.1)))

    def forward(self, cond: torch.Tensor, t_rel: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(cond)
        t = t_rel.unsqueeze(-1) if t_rel.dim() == 2 else t_rel
        rho_bar, omega_bar = self.field.integrated(h, t)
        p0, q = self.field.uq_params(h)
        theta = self.to_theta(h).view(-1, 2 * self.k, self.d_z)

        if self.anchor_nodes > 0:
            t_anchor = torch.linspace(
                0.0, self.horizon, self.anchor_nodes + 1, dtype=t.dtype, device=t.device
            ).view(1, -1, 1).expand(t.shape[0], -1, 1).contiguous()
            rho_bar_a, _ = self.field.integrated(h, t_anchor)
            lam = modal_lambda_projective(rho_bar, t, rho_bar_a, t_anchor, q, p0)
        else:
            lam = modal_increments(rho_bar, t, q, p0)["lam"]

        sigma2 = None
        if self.to_nugget is not None:
            sigma2 = F.softplus(self.to_nugget(h)) + SIGMA2_MIN
        mean, var = marginal_law(theta, rho_bar, omega_bar, lam, sigma2=sigma2)
        return {"mean": mean, "var": var, "theta": theta, "rho_bar": rho_bar,
                "omega_bar": omega_bar, "lam": lam, "p0": p0, "q": q, "sigma2": sigma2}


class DiagonalGaussianHead(nn.Module):
    """R10's critical control: a plain per-(time, channel) Gaussian on the same encoder.

    No dynamics, no modal structure -- just an NLL head with a query-time embedding. If this
    ties the chirp head, the paper's own gate says the contribution is a cost result rather
    than a method result, so this arm has to exist and has to be given a fair chance.
    """

    def __init__(self, cond_dim: int, d_z: int, *, hidden: int = 256, depth: int = 2,
                 time_dim: int = 32) -> None:
        super().__init__()
        self.d_z = int(d_z)
        self.time = _TimeEmbed(time_dim)
        self.trunk = _trunk(cond_dim, hidden, depth)
        self.mix = nn.Sequential(
            nn.Linear(hidden + self.time.dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * self.d_z),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, cond: torch.Tensor, t_rel: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(cond)                                     # [B,H]
        te = self.time(t_rel)                                    # [B,T,dim]
        z = torch.cat([h.unsqueeze(1).expand(-1, te.shape[1], -1), te], dim=-1)
        out = self.mix(z)
        mean, raw = out[..., : self.d_z], out[..., self.d_z:]
        return {"mean": mean, "var": F.softplus(raw) + SIGMA2_MIN}


class LowRankGaussianHead(nn.Module):
    """Low-rank-plus-diagonal in the CHANNEL axis: ``Sigma_t = U_t U_t^T + diag``.

    The nearest non-dynamical relative of the chirp law, which is also low rank (rank <= 2K
    from ``sum_k C_k C_k^T``). Included so a chirp-vs-diagonal gap cannot be attributed to
    "the chirp head has correlated channels and the control does not".
    """

    def __init__(self, cond_dim: int, d_z: int, *, rank: int = 4, hidden: int = 256,
                 depth: int = 2, time_dim: int = 32) -> None:
        super().__init__()
        self.d_z, self.rank = int(d_z), int(rank)
        self.time = _TimeEmbed(time_dim)
        self.trunk = _trunk(cond_dim, hidden, depth)
        self.mix = nn.Sequential(
            nn.Linear(hidden + self.time.dim, hidden), nn.SiLU(),
            nn.Linear(hidden, self.d_z * (2 + self.rank)),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, cond: torch.Tensor, t_rel: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(cond)
        te = self.time(t_rel)
        z = torch.cat([h.unsqueeze(1).expand(-1, te.shape[1], -1), te], dim=-1)
        out = self.mix(z)
        d = self.d_z
        mean, raw, u = out[..., :d], out[..., d:2 * d], out[..., 2 * d:]
        u = u.view(*u.shape[:-1], d, self.rank)
        diag = F.softplus(raw) + SIGMA2_MIN
        return {"mean": mean, "var": diag + u.pow(2).sum(-1), "factor": u, "diag": diag}


class StudentTHead(DiagonalGaussianHead):
    """Heavier tails on the same encoder, for the same reason as the low-rank head."""

    def __init__(self, *args, df: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_df = nn.Parameter(torch.tensor(math.log(max(df - 2.0, 1e-3))))

    def forward(self, cond: torch.Tensor, t_rel: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(cond, t_rel)
        df = F.softplus(self.log_df) + 2.0 + 1e-3
        # Report the VARIANCE, so every arm's `var` means the same thing: for Student-t with
        # scale s, Var = s^2 df/(df-2).
        out["df"] = df
        out["scale2"] = out["var"]
        out["var"] = out["var"] * df / (df - 2.0)
        return out


HEADS = {
    "chirp_gp": ChirpGPHead,
    "diag_gaussian": DiagonalGaussianHead,
    "lowrank_gaussian": LowRankGaussianHead,
    "student_t": StudentTHead,
}


def gaussian_nll_per_time(y: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean per-entry Gaussian NLL. Constrains marginals only -- the cross-time structure is
    imposed by the dynamics and never compared with the residuals, which is exactly the gap
    a path-level score exposes."""
    v = var.clamp_min(SIGMA2_MIN)
    nll = 0.5 * (torch.log(v) + (y - mean).pow(2) / v + math.log(2 * math.pi))
    if mask is None:
        return nll.mean()
    w = mask.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def student_t_nll_per_time(y: torch.Tensor, mean: torch.Tensor, scale2: torch.Tensor,
                           df: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    s2 = scale2.clamp_min(SIGMA2_MIN)
    z2 = (y - mean).pow(2) / s2
    nll = (
        0.5 * torch.log(s2)
        + 0.5 * torch.log(df * math.pi)
        + torch.lgamma(df / 2)
        - torch.lgamma((df + 1) / 2)
        + (df + 1) / 2 * torch.log1p(z2 / df)
    )
    if mask is None:
        return nll.mean()
    w = mask.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0)
