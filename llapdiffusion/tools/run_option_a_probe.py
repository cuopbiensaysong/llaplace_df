"""Option A (CMD-GP) against read R10's same-encoder controls, in one matched harness.

Two questions are answered by the same experiment, deliberately:

  * **Is a history-only head viable at all?** Everything upstream is frozen, so this asks
    whether ``E`` carries enough about the target to beat predicting the training mean. Three
    prior measurements say it might not: the Phase-1b gate failed on both seeds (the trained
    denoiser reached 6.5 % variance explained against a 5.6 % ridge ceiling), the analytic
    law's heteroscedasticity was uninformative (corr(predicted var, squared residual) =
    +0.016), and the existing one-shot arm was +14 % CRPS and the worst calibrated.
  * **Read R10: is it the chirp-modal structure, or just an NLL head?** The controls sit on
    the identical frozen encoder at the identical cost, so a tie reframes the contribution as
    a cost result -- which the paper's own gate pre-commits to reporting.

Everything is matched: same frozen VAE and summarizer, same windows, same trunk width and
depth, same objective, same schedule, same scoring. Parameter counts differ and are reported
rather than equalised, because forcing equal counts would change the modal budget ``K``,
which is the thing under test.

No diffusion is involved anywhere in this tool.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from llapdiffusion.configs.config_utils import refresh_artifact_paths
from llapdiffusion.models.chirp_gp import sample_joint_markov
from llapdiffusion.models.dist_heads import (
    HEADS,
    gaussian_nll_per_time,
    student_t_nll_per_time,
)
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.models.path_scores import (
    coverage_and_width,
    crps,
    energy_score,
    randomized_pit,
    variogram_score,
)
from llapdiffusion.models.uq_metrics import pit_calibration_error
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import build_eval_loaders
from llapdiffusion.tools.run_ridge_probe import _build_frozen_stack
from llapdiffusion.trainers import train_val_llapdiff as tv


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

@torch.no_grad()
def collect(dl, stack, device, *, summary_tokens: int = 16) -> Dict[str, torch.Tensor]:
    """(summary, latent target, query offsets, mask) per window, on CPU.

    The summary is taken UNNORMALISED (``norm=False``) and standardised against train by the
    caller, so the probe does not inherit a denoiser checkpoint's conditioning statistics --
    under ``COND_NORM_MODE='sample'`` with mean pooling those are identically zero (B23),
    which would make a summary-only head structurally unable to learn anything.

    🔴 **Do not mean-pool the summary tokens.** ``SUM_CONTEXT_LEN`` is 336 on this cell, not
    1, so the summary is ``[B, 336, 256]``. Collapsing it to a single 256-vector destroys
    almost all of the conditioning: measured here, a mean-pooled head reached only 1.8-2.9 %
    variance explained, *below* the 5.6 % a linear ridge attains on the same summary, which is
    the signature of an input problem rather than a model one. The project's own trap list
    records the same effect at -0.2 % against 14.2 % for a strided view. We therefore keep a
    **strided** set of ``summary_tokens`` tokens, exactly as the ridge probe does, and flatten
    them.
    """
    diff_model, vae, summarizer, mu_mean, mu_std = stack
    E, MU, DT, MASK = [], [], [], []
    for xb, yb, meta in dl:
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        _, raw = tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=meta.get("delta_t"), x_obs_mask=meta.get("x_obs_mask"), norm=False,
        )
        mu, obs = tv._latent_targets_for_batch(vae, yb, mask_bn, meta, device, mu_mean, mu_std)
        if mu is None or obs is None or not obs.any():
            continue
        dt_b = tv._flatten_dt(meta, mask_bn, device, key="delta_t_y")
        dt_b = tv._match_dt_to_horizon(dt_b, mu.size(1))
        m = torch.as_tensor(obs, device=device, dtype=torch.bool)
        while m.dim() < mu.dim():
            m = m.unsqueeze(-1)
        step = max(1, raw.shape[1] // int(summary_tokens))
        strided = raw[:, ::step][:, : int(summary_tokens)]           # [B, ntok, Hc]
        E.append(strided.reshape(strided.shape[0], -1).float().cpu())
        MU.append(mu.float().cpu())
        DT.append(dt_b.reshape(dt_b.shape[0], -1).float().cpu())
        MASK.append(m.expand_as(mu).cpu())
    if not E:
        raise RuntimeError("no eligible windows on this split")
    return {"E": torch.cat(E), "mu": torch.cat(MU), "dt": torch.cat(DT),
            "mask": torch.cat(MASK)}


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------

def _forward_loss(head, name, batch, *, objective: str) -> torch.Tensor:
    out = head(batch["E"], batch["dt"])
    y, m = batch["mu"], batch["mask"]
    if objective == "mse":
        w = m.to(y.dtype)
        return ((out["mean"] - y).pow(2) * w).sum() / w.sum().clamp_min(1.0)
    if name == "student_t":
        return student_t_nll_per_time(y, out["mean"], out["scale2"], out["df"], m)
    return gaussian_nll_per_time(y, out["mean"], out["var"], m)


def train_head(
    name: str, head, train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor],
    *, device, epochs_mse: int, epochs_nll: int, batch_size: int, lr: float,
    log_every: int = 5, train_mean: Optional[float] = None, patience: int = 8,
    weight_decay: float = 1e-4,
) -> Dict[str, object]:
    """MSE warm-up, then NLL -- each with early stopping and best-checkpoint restore.

    Training the Gaussian NLL from scratch is a documented failure here: the mean gradient
    is ``(pred - target)/sigma^2``, so a variance initialised far below the residual scale
    destroys the mean before the variance can adapt (measured at 2415x inflation elsewhere in
    this project). The warm-up is not a tuning nicety -- without it the arm is unusable.

    Early stopping is not optional either, and the arms need it *unequally*, which is exactly
    why it must be automatic rather than tuned per arm. The chirp head carries ~480k
    parameters against 16k highly-overlapping sliding windows over 4 entities, so its
    effective sample size is far below its nominal one: measured, it reaches its best val MSE
    early and then degrades while train MSE keeps falling (train 0.42 -> 0.35 while val
    1.22 -> 1.38). Reporting a final-epoch number would score overfitting, not capability, and
    would do so asymmetrically across arms of different capacity.

    Selection is on val MSE during the MSE phase and on the val objective during the NLL
    phase, and the best state is restored before returning.
    """
    head = head.to(device)
    n = train["E"].shape[0]
    history: List[Dict[str, float]] = []
    summary: Dict[str, object] = {}
    t0 = time.time()

    for phase, n_epochs in (("mse", epochs_mse), ("nll", epochs_nll)):
        if n_epochs <= 0:
            continue
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs))
        best = float("inf")
        bad_total = 0
        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        best_ep, stale = 0, 0
        for ep in range(n_epochs):
            head.train()
            perm = torch.randperm(n)
            tot, seen, n_bad = 0.0, 0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                batch = {k: v[idx].to(device, non_blocking=True) for k, v in train.items()}
                loss = _forward_loss(head, name, batch, objective=phase)
                # A nugget-free law can drive the variance toward zero, and then the NLL's
                # quadratic term r^2/sigma^2 explodes and poisons every later step with NaN.
                # That is the paper's own prediction about OPT{N} rather than a defect, so the
                # step is skipped and counted instead of being allowed to destroy the run --
                # an arm that cannot be trained must report that, not a NaN.
                if not torch.isfinite(loss):
                    n_bad += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                if not torch.isfinite(gn):
                    n_bad += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                tot += float(loss.detach()) * idx.numel()
                seen += idx.numel()
            sched.step()
            bad_total += n_bad

            head.eval()
            with torch.no_grad():
                vt, vn = 0.0, 0
                for i in range(0, val["E"].shape[0], 512):
                    b = {k: v[i:i + 512].to(device) for k, v in val.items()}
                    vt += float(_forward_loss(head, name, b, objective=phase)) * b["E"].shape[0]
                    vn += b["E"].shape[0]
            val_obj = vt / max(vn, 1)

            if val_obj < best - 1e-6:
                best, best_ep, stale = val_obj, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1

            if (ep + 1) % log_every == 0 or ep == n_epochs - 1 or stale >= patience:
                vm = evaluate_head(head, name, val, device=device, quick=True,
                                   train_mean=train_mean)
                history.append({"phase": phase, "epoch": ep + 1,
                                "train_loss": tot / max(seen, 1),
                                "val_obj": val_obj,
                                **{k: vm[k] for k in ("val_mse", "val_corr")}})
                print(f"[option-a] {name:16s} {phase} ep{ep + 1:3d} "
                      f"train {tot / max(seen, 1):8.4f}  val_obj {val_obj:8.4f}  "
                      f"val_mse {vm['val_mse']:.4f} corr {vm['val_corr']:+.4f}"
                      f"{'  *best' if best_ep == ep + 1 else ''}", flush=True)
            if stale >= patience:
                print(f"[option-a] {name:16s} {phase}: early stop at ep{ep + 1}, "
                      f"best ep{best_ep} (val_obj {best:.4f})", flush=True)
                break

        head.load_state_dict(best_state)      # report the selected model, not the last one
        summary[f"{phase}_skipped_nonfinite_steps"] = bad_total
        summary[f"{phase}_best_epoch"] = best_ep
        summary[f"{phase}_best_val_objective"] = best
    return {"history": history, "train_seconds": time.time() - t0, **summary}


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_head(head, name, data, *, device, quick: bool = False,
                  num_samples: int = 25, chunk: int = 128,
                  generator: Optional[torch.Generator] = None,
                  train_mean: Optional[float] = None) -> Dict[str, float]:
    """Latent-space metrics, plus path-level scores when ``quick=False``.

    The chirp head draws **exact joint paths** by the Markov recursion; the controls have no
    cross-time structure, so their draws are independent across query times. That difference
    is the thing the variogram score exists to see, and it is a property of the models, not
    an artefact of how they are sampled.
    """
    head.eval()
    n = data["E"].shape[0]
    se = sq = cnt = 0.0
    ys, ms = [], []
    acc: Dict[str, List[torch.Tensor]] = {}
    pit_chunks: List[torch.Tensor] = []

    for i in range(0, n, chunk):
        b = {k: v[i:i + chunk].to(device) for k, v in data.items()}
        out = head(b["E"], b["dt"])
        y, m = b["mu"], b["mask"]
        w = m.to(y.dtype)
        se += float(((out["mean"] - y).pow(2) * w).sum())
        cnt += float(w.sum())
        ys.append(y[m].cpu())
        ms.append(out["mean"][m].cpu())
        if quick:
            continue

        if name == "chirp_gp":
            from llapdiffusion.models.chirp_gp import modal_increments
            inc = modal_increments(out["rho_bar"], b["dt"].unsqueeze(-1), out["q"], out["p0"])
            s = sample_joint_markov(
                out["theta"], out["rho_bar"], out["omega_bar"], inc["incr"], inc["d_rho"],
                out["p0"], num_samples=num_samples, sigma2=out["sigma2"], generator=generator,
            )
        else:
            sd = out["var"].clamp_min(1e-12).sqrt()
            s = out["mean"].unsqueeze(0) + sd.unsqueeze(0) * torch.randn(
                (num_samples, *out["mean"].shape), device=device, generator=generator
            )
        es, cr = energy_score(s, y, m), crps(s, y, m)
        for key, val in (("ES_unbiased", es["unbiased"]), ("ES_naive", es["naive"]),
                         ("CRPS_unbiased", cr["unbiased"]), ("CRPS_naive", cr["naive"]),
                         ("VS_p0.5", variogram_score(s, y, b["dt"], m))):
            acc.setdefault(key, []).append(val.cpu())
        cw = coverage_and_width(s, y, m)
        acc.setdefault("_cw", []).append(torch.tensor([[cw[k] for k in sorted(cw)]]))
        acc.setdefault("_cw_keys", [sorted(cw)])
        acc.setdefault("_nw", []).append(torch.tensor([float(y.shape[0])]))
        pit_chunks.append(randomized_pit(s, y, m, generator=generator).cpu())

    y_all, m_all = torch.cat(ys), torch.cat(ms)
    yc, mc = y_all - y_all.mean(), m_all - m_all.mean()
    corr = float((yc * mc).sum() / (yc.norm() * mc.norm()).clamp_min(1e-12))
    res: Dict[str, float] = {
        "val_mse": se / max(cnt, 1.0),
        "val_rmse": math.sqrt(se / max(cnt, 1.0)),
        "val_corr": corr,
        # ⚠️ This baseline is computed on the EVAL split, so it knows that split's level and
        # is systematically stronger than anything a forecaster could have known -- measured
        # 2.3x stronger than an honest train-mean baseline elsewhere in this project. It is
        # kept for continuity with the existing gate reads and is NOT the headline.
        "baseline_mse_predict_mean_insplit": float(((y_all - y_all.mean()) ** 2).mean()),
        "baseline_mse_predict_zero": float((y_all ** 2).mean()),
        "n_elements": cnt,
    }
    res["variance_explained_vs_insplit_mean"] = 1.0 - res["val_mse"] / max(
        res["baseline_mse_predict_mean_insplit"], 1e-12)
    if train_mean is not None:
        # The honest baseline: a constant fitted on TRAIN, applied to val.
        res["baseline_mse_predict_train_mean"] = float(((y_all - train_mean) ** 2).mean())
        res["variance_explained_vs_train_mean"] = 1.0 - res["val_mse"] / max(
            res["baseline_mse_predict_train_mean"], 1e-12)
    if quick:
        return res

    nw = torch.cat(acc["_nw"])
    for k, v in acc.items():
        if k.startswith("_"):
            continue
        res[k] = float(torch.cat(v).mean())
    cw = torch.cat(acc["_cw"])
    for j, key in enumerate(acc["_cw_keys"][0]):
        res[key] = float((cw[:, j] * nw).sum() / nw.sum())
    pit = torch.cat(pit_chunks)
    res["PIT_ECE"] = float(pit_calibration_error(pit))
    return res


# --------------------------------------------------------------------------------------

@torch.no_grad()
def lag_covariance(head, name, data, *, device, max_lag: int = 24, chunk: int = 128,
                   max_windows: int = 512) -> Dict[str, List[float]]:
    """Read R16: predicted against empirical lag covariance of the residuals.

    The variogram score is not scale-free, so a path-score advantage could in principle be a
    better-scaled *marginal* rather than better *dependence*. This diagnostic separates them:
    it compares the model's own ``Cov(z(t), z(t+l))`` against the empirical covariance of its
    residuals at the same lag. A per-time head predicts exactly zero at every nonzero lag by
    construction, whatever its marginal width, so the comparison is decisive about structure.

    Returns the empirical curve, the predicted curve, and the lag axis, in units of query
    steps (the grid is near-regular on this cell, so index lag is proportional to time lag).
    """
    from llapdiffusion.models.chirp_gp import cross_time_kernel

    n = min(int(max_windows), data["E"].shape[0])
    emp = torch.zeros(max_lag + 1, dtype=torch.float64)
    pred = torch.zeros(max_lag + 1, dtype=torch.float64)
    cnt = torch.zeros(max_lag + 1, dtype=torch.float64)

    for i in range(0, n, chunk):
        b = {k: v[i:i + chunk].to(device) for k, v in data.items()}
        out = head(b["E"], b["dt"])
        r = (b["mu"] - out["mean"]) * b["mask"].to(b["mu"].dtype)      # [B,T,D]
        if name == "chirp_gp":
            ker = cross_time_kernel(out["theta"], out["rho_bar"], out["omega_bar"],
                                    out["lam"], sigma2=out["sigma2"])  # [B,T,T,D]
        else:
            ker = None                       # per-time heads: zero at every nonzero lag
        T = r.shape[1]
        for l in range(max_lag + 1):
            e = (r[:, : T - l] * r[:, l:]).mean()
            emp[l] += float(e) * r.shape[0]
            if ker is not None:
                idx = torch.arange(T - l, device=device)
                p = ker[:, idx, idx + l].mean()
                pred[l] += float(p) * r.shape[0]
            elif l == 0:
                pred[l] += float(out["var"].mean()) * r.shape[0]
            cnt[l] += r.shape[0]

    return {
        "lag": list(range(max_lag + 1)),
        "empirical": (emp / cnt.clamp_min(1)).tolist(),
        "predicted": (pred / cnt.clamp_min(1)).tolist(),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-chunk", type=int, default=128,
                   help="Windows per evaluation chunk. The variogram score materialises a "
                        "[S,B,T,T,D] pairwise tensor, so its footprint scales with d_z: at "
                        "B=128, S=25, T=168 that is 5.8 GB at d_z=16 (noaa_uk) but 8.7 GB at "
                        "d_z=24 (bms_air), which OOMs a 24 GB card. Default 128 keeps every "
                        "existing noaa_uk number bit-identical -- coverage and width are "
                        "pooled per chunk, so changing this changes them slightly. Lower it "
                        "only for wider latents.")
    p.add_argument("--dataset-key", default="noaa_uk")
    p.add_argument("--pred", type=int, default=168)
    p.add_argument("--heads", nargs="*", default=["chirp_gp", "diag_gaussian"],
                   choices=sorted(HEADS))
    p.add_argument("--modes", type=int, default=32, help="K for the chirp head")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--num-basis", type=int, default=4)
    p.add_argument("--anchor-nodes", type=int, default=128,
                   help="0 = the shipped query-grid variance quadrature (a GRID law); >0 = "
                        "the query-independent anchor that makes the kernel projective too.")
    p.add_argument("--no-nugget", action="store_true",
                   help="read R15 / the paper's OPT{N}: drop the per-coordinate nugget.")
    p.add_argument("--parameterization", default="p_exact",
                   choices=("p_exact", "p_mono", "p_grid"),
                   help="appendix A3 pole-function ablation.")
    p.add_argument("--tag", default=None,
                   help="label recorded in the report, for ablation bookkeeping.")
    p.add_argument("--epochs-mse", type=int, default=60)
    p.add_argument("--epochs-nll", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-grid", type=float, nargs="*", default=None,
                   help="identical LR grid for EVERY arm, selected on validation "
                        "(the symmetric-budget rule). Overrides --lr.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8,
                   help="early-stop patience in epochs, per phase")
    p.add_argument("--num-samples", type=int, default=25)
    p.add_argument("--max-lag", type=int, default=24,
                   help="lags for the R16 predicted-vs-empirical covariance curve")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-tokens", type=int, default=16,
                   help="strided summary tokens kept per window. NEVER 1: the "
                        "summary is [B,336,256] on this cell and mean-pooling it "
                        "drops it below the linear ridge ceiling.")
    p.add_argument("--cache", default="ldt/cache/option_a")
    p.add_argument("--vae-entity-encode", action="store_true", default=True)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main() -> Dict[str, object]:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    if args.vae_entity_encode:
        cfg.VAE_ENTITY_ENCODE = True
    refresh_artifact_paths(cfg)
    device = set_torch(seed=int(args.seed), deterministic=False,
                       allow_tf32=bool(getattr(cfg, "ALLOW_TF32", True)))

    # Token count is part of the cache identity: a cache built at a different
    # striding is a different experiment, not a reusable one.
    cache_dir = Path(args.cache) / f"{args.dataset_key}_h{args.pred}_tok{args.summary_tokens}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "splits.pt"
    if cache_file.exists():
        print(f"[option-a] loading cached windows from {cache_file}")
        blob = torch.load(cache_file, map_location="cpu")
        train, val = blob["train"], blob["val"]
    else:
        loaders = build_eval_loaders(cfg)
        stack = _build_frozen_stack(cfg, loaders[0], device)
        print("[option-a] collecting train windows ...", flush=True)
        train = collect(loaders[0], stack, device,
                        summary_tokens=int(args.summary_tokens))
        print("[option-a] collecting val windows ...", flush=True)
        val = collect(loaders[1], stack, device,
                      summary_tokens=int(args.summary_tokens))
        torch.save({"train": train, "val": val}, cache_file)
        print(f"[option-a] cached {train['E'].shape[0]} train / {val['E'].shape[0]} val "
              f"windows to {cache_file}")

    # Standardise the summary against TRAIN only.
    mu_e, sd_e = train["E"].mean(0, keepdim=True), train["E"].std(0, keepdim=True).clamp_min(1e-6)
    for d in (train, val):
        d["E"] = (d["E"] - mu_e) / sd_e

    print(f"[option-a] train {train['E'].shape[0]} windows, val {val['E'].shape[0]}; "
          f"E dim {train['E'].shape[1]}, horizon {train['mu'].shape[1]}, "
          f"d_z {train['mu'].shape[2]}")

    # The honest constant baseline: fitted on TRAIN, applied to val.
    train_mean = float(train["mu"][train["mask"]].mean())
    d_z = int(train["mu"].shape[2])
    horizon = float(train["dt"].max())
    gen = torch.Generator(device=device).manual_seed(int(args.seed))

    results: Dict[str, object] = {}
    lr_grid = [float(x) for x in args.lr_grid] if args.lr_grid else [float(args.lr)]
    for name in args.heads:
        kw = dict(hidden=int(args.hidden), depth=int(args.depth))
        if name == "chirp_gp":
            kw.update(k=int(args.modes), num_basis=int(args.num_basis),
                      horizon=horizon, anchor_nodes=int(args.anchor_nodes),
                      nugget=not bool(args.no_nugget),
                      parameterization=str(args.parameterization))

        # Identical grid and trial count for every arm, selected on validation -- the
        # symmetric-budget rule. It is load-bearing here, not bookkeeping: the arms have
        # genuinely different optima, and a single shared learning rate measures the choice
        # rather than the models. Measured on the mean-only phase, the diagonal head goes
        # 4.4% -> 11.0% variance explained between lr=3e-4 and 3e-5 while the chirp head
        # moves 8.8% -> 9.2%, which is enough to flip the ordering.
        best_lr, best_obj, best_pack = None, float("inf"), None
        for lr in lr_grid:
            torch.manual_seed(int(args.seed))
            head = HEADS[name](int(train["E"].shape[1]), d_z, **kw)
            n_params = sum(p.numel() for p in head.parameters())
            print(f"\n[option-a] === {name} ({n_params:,} params) lr={lr:g} ===", flush=True)
            tr = train_head(name, head, train, val, device=device,
                            epochs_mse=int(args.epochs_mse), epochs_nll=int(args.epochs_nll),
                            batch_size=int(args.batch_size), lr=lr,
                            train_mean=train_mean, patience=int(args.patience),
                            weight_decay=float(args.weight_decay))
            # Select on the final phase's val objective, i.e. the objective the arm is
            # reported under.
            key = "nll_best_val_objective" if int(args.epochs_nll) > 0 else "mse_best_val_objective"
            obj = float(tr[key])
            print(f"[option-a] {name} lr={lr:g}: selection objective {obj:.4f}", flush=True)
            if math.isfinite(obj) and obj < best_obj:
                best_lr, best_obj, best_pack = lr, obj, (head, tr, n_params)

        if best_pack is None:
            # Every learning rate produced a non-finite objective. For the
            # nugget-free arm this is the registered R15 outcome, not a crash:
            # report it and carry on with the other arms.
            print(f"[option-a] {name}: NOT TRAINABLE -- every lr in {lr_grid} gave a "
                  f"non-finite objective", flush=True)
            results[name] = {"params": n_params, "selected_lr": None,
                             "lr_grid": lr_grid, "trainable": False,
                             "failure": "non-finite objective at every lr"}
            continue
        head, tr, n_params = best_pack
        print(f"[option-a] {name}: SELECTED lr={best_lr:g} (val objective {best_obj:.4f})",
              flush=True)
        metrics = evaluate_head(head, name, val, device=device, quick=False,
                                num_samples=int(args.num_samples), generator=gen,
                                chunk=int(args.eval_chunk), train_mean=train_mean)
        lagcov = lag_covariance(head, name, val, device=device,
                                max_lag=int(args.max_lag))
        results[name] = {"params": n_params, "selected_lr": best_lr,
                         "lr_grid": lr_grid, "lag_covariance": lagcov, **tr, **metrics}
        # The NLL phase optimises the LAW, so it may trade mean accuracy for
        # calibration. Record the mean-only best separately so a reader can see
        # which of the two a difference came from.
        results[name]["mean_only_best_val_mse"] = tr.get("mse_best_val_objective")
        print(f"[option-a] {name}: val_mse {metrics['val_mse']:.4f}  "
              f"VE(train-mean) {metrics['variance_explained_vs_train_mean']*100:+.2f}%  "
              f"corr {metrics['val_corr']:+.4f}  ES {metrics['ES_unbiased']:.3f}  "
              f"VS {metrics['VS_p0.5']:.1f}  PIT-ECE {metrics['PIT_ECE']:.4f}  "
              f"cov80 {metrics['coverage_0.8']:.3f}", flush=True)

    report = {
        "read": "R10 + Option-A feasibility",
        "tag": args.tag,
        "variant": {"anchor_nodes": int(args.anchor_nodes),
                    "nugget": not bool(args.no_nugget),
                    "parameterization": str(args.parameterization)},
        "dataset_key": args.dataset_key, "pred": int(args.pred), "seed": int(args.seed),
        "split": "val",
        "config": {k: getattr(args, k) for k in
                   ("modes", "hidden", "depth", "num_basis", "anchor_nodes",
                    "epochs_mse", "epochs_nll", "batch_size", "lr", "num_samples",
                    "weight_decay", "patience", "summary_tokens",
                    "anchor_nodes", "parameterization")},
        "n_train_windows": int(train["E"].shape[0]),
        "n_val_windows": int(val["E"].shape[0]),
        "heads": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n")
    print(text)
    return report


if __name__ == "__main__":
    main()
