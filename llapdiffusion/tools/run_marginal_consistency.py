"""Read R11: is the model a Gaussian *process*, or only a Gaussian on the grid you asked for?

Proposition C.1 says the finite-dimensional laws of CMD-GP are projective -- predicting a
subset of query times returns the marginal of predicting the full set -- under three
hypotheses, and names exactly what breaks when each is dropped:

  (i)   the parameters depend on the noisy query tensor ``z_tau`` (Option B),
  (ii)  the origin is re-anchored to the earliest *requested* query,
  (iii) (P-grid), whose cumulative-trapezoid integration is grid-dependent by construction.

This tool measures the discrepancy rather than asserting it is small, including for the arms
that should be exactly consistent -- an implementation can violate a property the
mathematics grants. For (P-exact)/(P-mono) under the shipped origin this is a **correctness
gate** with a declared tolerance of 1e-6 relative, not a result.

The measurement is made on the modal field with a *fixed* conditioning vector, which is what
isolates the question: with the parameters held fixed, any discrepancy is attributable to
the time map and the pole integration alone, which is precisely what (ii) and (iii) are
about. Failure mode (i) is a property of a trained denoiser and is measured separately by
passing ``--checkpoint``.

Subsets are formed both by dropping points and by *inserting* new query times before, between
and after the existing ones, because only insertion before the first point exercises the
origin at all.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch

from llapdiffusion.models.chirp_gp import (
    cross_time_kernel,
    marginal_law,
    modal_increments,
    modal_lambda_projective,
)
from llapdiffusion.models.laptrans import ChirpModalField

PARAMETERIZATIONS = ("p_exact", "p_mono", "p_grid")
ORIGINS = ("last_history", "query_first")
GATE_TOLERANCE = 1e-6


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative discrepancy in the Frobenius/2-norm sense, scale-free."""
    denom = b.abs().pow(2).sum().sqrt().clamp_min(1e-12)
    return float((a - b).abs().pow(2).sum().sqrt() / denom)


@torch.no_grad()
def _law_on(
    field: ChirpModalField,
    cond: torch.Tensor,
    theta: torch.Tensor,
    times: torch.Tensor,          # [B,T]
    p0: torch.Tensor,
    q: torch.Tensor,
    *,
    origin: str,
    anchor_nodes: int = 0,
    horizon: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Mean and cross-time kernel at the given query times, under the given origin.

    ``anchor_nodes > 0`` computes the variance on a fixed, query-independent grid of that
    many nodes instead of on the requested query grid -- the repair that makes the *kernel*
    projective, not just the mean. ``horizon`` must be a property of the WINDOW, never
    ``t.max()``: deriving it from the requested queries would smuggle the query set back into
    the anchor grid and undo the repair for exactly the insert-after case.
    """
    t = times.unsqueeze(-1)
    if origin == "query_first":
        t = t - t[:, :1]                       # the anchoring Section 3 removes
    rho_bar, omega_bar = field.integrated(cond, t)
    if anchor_nodes > 0:
        if horizon is None:
            raise ValueError("a fixed anchor grid needs an explicit, query-independent horizon")
        t_anchor = torch.linspace(
            0.0, float(horizon), int(anchor_nodes) + 1, dtype=t.dtype, device=t.device
        ).view(1, -1, 1).expand(t.shape[0], -1, 1).contiguous()
        rho_bar_anchor, _ = field.integrated(cond, t_anchor)
        lam = modal_lambda_projective(rho_bar, t, rho_bar_anchor, t_anchor, q, p0)
    else:
        lam = modal_increments(rho_bar, t, q, p0)["lam"]
    mean, _ = marginal_law(theta, rho_bar, omega_bar, lam)
    ker = cross_time_kernel(theta, rho_bar, omega_bar, lam)
    return {"mean": mean, "kernel": ker}


def _subset_cases(h: int, times: torch.Tensor, generator: torch.Generator) -> List[Dict]:
    """Subsets by dropping, and supersets by inserting before / between / after."""
    cases: List[Dict] = []
    for frac in (0.25, 0.5, 0.75):
        k = max(2, int(round(frac * h)))
        idx = torch.randperm(h, generator=generator)[:k].sort().values
        cases.append({"kind": f"drop_to_{frac:g}h", "insertion": "n/a", "index": idx})

    # Insertions: build the SUPERSET, then check the original points keep their law.
    gaps = times[0, 1:] - times[0, :-1]
    step = float(gaps.median())
    for where, new_t in (
        ("before", times[0, 0] - step * 0.5),
        ("between", (times[0, h // 2] + times[0, h // 2 + 1]) / 2),
        ("after", times[0, -1] + step * 0.5),
    ):
        merged = torch.cat([times[0], new_t.view(1)]).sort()
        keep = (merged.indices < h).nonzero(as_tuple=True)[0]
        cases.append({
            "kind": "insert", "insertion": where,
            "superset_times": merged.values.unsqueeze(0), "index": keep,
        })
    return cases


@torch.no_grad()
def measure(
    *,
    parameterization: str,
    origin: str,
    h: int = 24,
    k_modes: int = 8,
    d_z: int = 4,
    num_basis: int = 4,
    seed: int = 0,
    anchor_nodes: int = 0,
) -> Dict[str, object]:
    """Discrepancy between predicting a subset directly and marginalising the superset."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    field = ChirpModalField(
        k=k_modes, cond_dim=16, num_basis=num_basis, parameterization=parameterization,
        uq_head=True, time_scale=float(h), rho_init_horizon=float(h),
    )
    # A non-trivial conditioning vector: at the zero-initialised heads the pole coefficients
    # are degenerate, and a degenerate field would pass a consistency test vacuously.
    for p in field.parameters():
        p.data = p.data + 0.1 * torch.randn(p.shape, generator=g)
    cond = torch.randn(1, 16, generator=g)
    theta = torch.randn(1, 2 * k_modes, d_z, generator=g)
    p0, q = field.uq_params(cond)

    # Irregular query times with a genuine lead gap, as the loaders produce.
    gaps = torch.rand(h, generator=g) * 1.5 + 0.5
    times = torch.cumsum(gaps, 0).unsqueeze(0)

    # A window property, fixed before any query set is seen -- generous enough to cover the
    # inserted point beyond the last query.
    horizon = float(times.max()) * 1.25

    rows = []
    for case in _subset_cases(h, times, g):
        full_times = case.get("superset_times", times)
        idx = case["index"]
        kw = dict(origin=origin, anchor_nodes=anchor_nodes, horizon=horizon)
        full = _law_on(field, cond, theta, full_times, p0, q, **kw)
        sub_times = full_times[:, idx] if "superset_times" in case else times[:, idx]
        sub = _law_on(field, cond, theta, sub_times, p0, q, **kw)

        marg_mean = full["mean"][:, idx]
        marg_ker = full["kernel"][:, idx][:, :, idx]
        rows.append({
            "case": case["kind"],
            "insertion": case["insertion"],
            "n_query": int(idx.numel()),
            "rel_delta_mean": _relative(sub["mean"], marg_mean),
            "rel_delta_kernel": _relative(sub["kernel"], marg_ker),
        })

    worst_mean = max(r["rel_delta_mean"] for r in rows)
    worst_ker = max(r["rel_delta_kernel"] for r in rows)
    exact_expected = parameterization in ("p_exact", "p_mono") and origin == "last_history"
    return {
        "parameterization": parameterization,
        "origin": origin,
        "anchor_nodes": int(anchor_nodes),
        "rows": rows,
        "worst_rel_delta_mean": worst_mean,
        "worst_rel_delta_kernel": worst_ker,
        "consistency_expected": bool(exact_expected),
        "gate_tolerance": GATE_TOLERANCE,
        "gate_pass": bool(max(worst_mean, worst_ker) <= GATE_TOLERANCE) if exact_expected else None,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--num-basis", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--anchor-nodes", type=int, nargs="*", default=[0, 256],
                   help="0 = the shipped query-grid quadrature; N > 0 = the repair, a "
                        "fixed N-node grid that makes Lambda a function of time alone.")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main() -> Dict[str, object]:
    args = _parse_args()
    results = []
    for anchor in args.anchor_nodes:
      for param in PARAMETERIZATIONS:
        for origin in ORIGINS:
            per_seed = [
                measure(
                    parameterization=param, origin=origin, h=int(args.horizon),
                    k_modes=int(args.modes), d_z=int(args.latent_dim),
                    num_basis=int(args.num_basis), seed=int(s),
                    anchor_nodes=int(anchor),
                )
                for s in args.seeds
            ]
            agg = {
                "parameterization": param,
                "origin": origin,
                "anchor_nodes": int(anchor),
                "seeds": list(args.seeds),
                "worst_rel_delta_mean": max(r["worst_rel_delta_mean"] for r in per_seed),
                "worst_rel_delta_kernel": max(r["worst_rel_delta_kernel"] for r in per_seed),
                "consistency_expected": per_seed[0]["consistency_expected"],
                "gate_pass": (
                    all(r["gate_pass"] for r in per_seed)
                    if per_seed[0]["consistency_expected"] else None
                ),
                "by_case": per_seed[0]["rows"],
            }
            results.append(agg)

    report = {
        "read": "R11",
        "gate_tolerance": GATE_TOLERANCE,
        "note": (
            "Rows with consistency_expected=true are a CORRECTNESS GATE, not a result: "
            "(P-exact)/(P-mono) under the shipped last_history origin must be exactly "
            "projective. The other rows are the deliberate positive controls of "
            "Proposition C.1 and their magnitude is the reported quantity."
        ),
        "arms": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            fh.write(text + "\n")
    return report


if __name__ == "__main__":
    main()
