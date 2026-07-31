"""T1: Theorem-D pole-invariance stress test across sampling-gap regimes.

Evaluates ONE trained chirp checkpoint under increasing induced context missingness
(``--coverages``, the gap-regime knob) and reports, per regime:

- the learned pole *trajectories* rho_k(t), omega_k(t) on the same test windows, and
  their relative distance to the baseline regime (Theorem D predicts ~invariance);
- the observed context-gap moments E[dt], Var(dt), E[dt^2];
- the implied event-domain multiplier s_bar per Eq. (8),
  Re s_bar = -rho E[dt] + 1/2 (rho^2 - omega^2) Var(dt) - 1/2 rho' E[dt^2],
  Im s_bar =  omega E[dt] - rho omega Var(dt) + 1/2 omega' E[dt^2],
  which SHOULD shift with the gap law even while the pole functions stay put.

Run:  llapdiff-t1-poles --dataset-key physionet --pred 12 --checkpoint <chirp ckpt>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
from llapdiffusion.tools.run_synthetic_regime_shift import _write_rows
from llapdiffusion.trainers import train_val_llapdiff as tv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1 pole-invariance across gap regimes.")
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coverages", nargs="+", type=float, default=(0.0, 0.2, 0.4, 0.6, 0.8))
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--top-modes", type=int, default=4)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-root", type=str,
                        default=str(Path.cwd() / "ldt" / "results" / "t1_pole_invariance"))
    return parser.parse_args()


@torch.no_grad()
def _collect_regime(cfg, checkpoint: str, args, *, device) -> Dict[str, np.ndarray]:
    """Pole trajectories [W,T,K] + observed context-gap samples for one coverage."""
    loaders, stack = prepare_eval_stack(cfg, checkpoint, device=device)
    _, val_dl, test_dl, _ = loaders
    loader = val_dl if args.split == "val" else test_dl
    diff_model, _, summarizer, _, _ = stack
    chirp_field = diff_model.model.chirp_field
    if chirp_field is None:
        raise ValueError("T1 requires a chirp checkpoint (pole trajectories).")

    trajectories_rho: List[np.ndarray] = []
    trajectories_omega: List[np.ndarray] = []
    gaps: List[np.ndarray] = []
    timesteps = int(diff_model.scheduler.timesteps)
    for xb, yb, meta in loader:
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=meta.get("delta_t"), x_obs_mask=meta.get("x_obs_mask"),
        )
        t = torch.full((cond_summary.shape[0],), timesteps // 2, device=device, dtype=torch.long)
        t_vec = diff_model._time_embed(t).to(cond_summary.dtype)
        cond_vec = diff_model.model.make_pole_cond(
            t_vec, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw
        )
        t_grid = torch.arange(1, int(cfg.PRED) + 1, dtype=cond_vec.dtype, device=device)
        t_rel = t_grid.view(1, -1, 1).expand(cond_vec.shape[0], -1, 1).contiguous()
        rho, omega = chirp_field.instantaneous(cond_vec, t_rel)  # [B,T,K]

        # Observed context gaps per (row, entity): cumulate delta_t, diff over observed.
        delta_t = meta["delta_t"].cpu().numpy()  # [B,N,Kctx]
        obs = meta.get("x_obs_mask")
        obs = obs.cpu().numpy() if obs is not None else np.ones(delta_t.shape, dtype=bool)
        if obs.ndim == 4:
            obs = obs.any(axis=-1)
        valid = mask_bn.detach().cpu().numpy().astype(bool)  # [B,N]
        times = np.cumsum(delta_t, axis=-1)
        for row in range(delta_t.shape[0]):
            for ent in np.nonzero(valid[row])[0]:
                obs_times = times[row, ent][obs[row, ent].astype(bool)]
                if obs_times.size >= 2:
                    gaps.append(np.diff(obs_times))

        for row in range(rho.shape[0]):
            if len(trajectories_rho) >= int(args.num_windows):
                break
            trajectories_rho.append(rho[row].detach().cpu().numpy())
            trajectories_omega.append(omega[row].detach().cpu().numpy())
        if len(trajectories_rho) >= int(args.num_windows):
            break

    if not trajectories_rho:
        raise RuntimeError("No valid windows found for the selected split/coverage.")
    return {
        "rho": np.stack(trajectories_rho),  # [W,T,K]
        "omega": np.stack(trajectories_omega),
        "gaps": np.concatenate(gaps) if gaps else np.array([1.0]),
    }


def _top_mode_indices(base: Dict[str, np.ndarray], top_modes: int) -> np.ndarray:
    """Per-window top modes by baseline trajectory variability (fixed across regimes)."""
    variability = base["rho"].std(axis=1) + base["omega"].std(axis=1)  # [W,K]
    return np.argsort(-variability, axis=1)[:, :top_modes]  # [W,top]


def _gather(traj: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[W,T,K], [W,top] -> [W,T,top]."""
    return np.take_along_axis(traj, idx[:, None, :], axis=2)


def _eq8_multiplier(rho: np.ndarray, omega: np.ndarray, gaps: np.ndarray) -> Dict[str, float]:
    """Time-mean implied event-domain log-pole per Eq. (8), over [W,T,top] trajectories."""
    e1 = float(gaps.mean())
    var = float(gaps.var())
    e2 = float((gaps**2).mean())
    rho_p = np.gradient(rho, axis=1)
    omega_p = np.gradient(omega, axis=1)
    re = -rho * e1 + 0.5 * (rho**2 - omega**2) * var - 0.5 * rho_p * e2
    im = omega * e1 - rho * omega * var + 0.5 * omega_p * e2
    return {"re_sbar_mean": float(re.mean()), "im_sbar_mean": float(im.mean()),
            "gap_mean": e1, "gap_var": var, "gap_second_moment": e2}


def main() -> None:
    args = _parse_args()
    result_root = Path(args.output_root).resolve()
    coverages = [float(c) for c in args.coverages]

    regimes: Dict[float, Dict[str, np.ndarray]] = {}
    for coverage in coverages:
        cfg = build_eval_config(args.dataset_key, int(args.pred), coverage=coverage)
        device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False)
        regimes[coverage] = _collect_regime(cfg, args.checkpoint, args, device=device)
        print(f"coverage={coverage}: {regimes[coverage]['rho'].shape[0]} windows, "
              f"{regimes[coverage]['gaps'].size} observed gaps")

    base_cov = coverages[0]
    idx = _top_mode_indices(regimes[base_cov], int(args.top_modes))
    base_rho = _gather(regimes[base_cov]["rho"], idx)
    base_omega = _gather(regimes[base_cov]["omega"], idx)

    rows: List[Dict[str, object]] = []
    for coverage in coverages:
        rho = _gather(regimes[coverage]["rho"], idx)
        omega = _gather(regimes[coverage]["omega"], idx)
        row: Dict[str, object] = {
            "coverage": coverage,
            "rho_reldist_vs_base": float(
                np.linalg.norm(rho - base_rho) / max(np.linalg.norm(base_rho), 1e-9)
            ),
            "omega_reldist_vs_base": float(
                np.linalg.norm(omega - base_omega) / max(np.linalg.norm(base_omega), 1e-9)
            ),
        }
        row.update(_eq8_multiplier(rho, omega, regimes[coverage]["gaps"]))
        rows.append(row)

    tag = f"{args.dataset_key}_h{int(args.pred)}_{Path(args.checkpoint).stem}"
    _write_rows(rows, result_root / f"{tag}.csv", result_root / f"{tag}.json")

    # Overlay figure: window 0, top mode, across regimes.
    t = np.arange(1, base_rho.shape[1] + 1)
    fig, (ax_r, ax_o) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, coverage in enumerate(coverages):
        color = cmap(i / max(len(coverages) - 1, 1))
        ax_r.plot(t, _gather(regimes[coverage]["rho"], idx)[0, :, 0], color=color,
                  label=f"coverage {coverage:g}")
        ax_o.plot(t, _gather(regimes[coverage]["omega"], idx)[0, :, 0], color=color)
    ax_r.set_xlabel("t̃ (steps)"); ax_r.set_ylabel("ρ(t̃)"); ax_r.legend(fontsize=8)
    ax_o.set_xlabel("t̃ (steps)"); ax_o.set_ylabel("ω(t̃) [rad/step]")
    fig.suptitle(f"T1 pole invariance across gap regimes ({tag})")
    fig.tight_layout()
    fig_path = result_root / f"{tag}_trajectories.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)

    (result_root / f"{tag}_meta.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "coverages": coverages,
        "num_windows": int(args.num_windows),
        "top_modes": int(args.top_modes),
        "read": "Theorem D: rho/omega_reldist_vs_base ~ flat; re/im_sbar shift with coverage.",
    }, indent=2))
    print(f"completed: {len(rows)} regimes -> {result_root / (tag + '.csv')}")


if __name__ == "__main__":
    main()
