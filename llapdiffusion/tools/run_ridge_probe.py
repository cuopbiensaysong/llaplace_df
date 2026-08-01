"""Phase-1a positive control: what is forecastable, and where does it get lost.

`CMD_UQ_RECOVERY_PLAN.md` §1a used to fit ONE ridge — frozen summarizer tokens -> latent —
and branch on the result. That cannot distinguish "this dataset is unforecastable" from
"the conditioning pipeline loses the forecastable part", and on noaa_uk h=168 it produced
the wrong answer: it said move datasets, while the raw history was forecastable to 28.6 %
and the loss was entirely in the conditioning (B21). This tool fits the whole triple so the
branch is decidable:

    A  raw history   -> raw target     "is this dataset forecastable at all?"
    B  cond_summary  -> raw target     "does the conditioning preserve it?"
    C  cond_summary  -> latent         "can the 1b gate see anything?"

Three lessons are baked in, each of which produced a wrong number when absent:

* **Strided tokens, never mean-pooling.** Averaging the 336 summary tokens into one vector
  scores -0.2 % where the strided view scores 14.2 %: the plan's original "pooled E_ti"
  wording manufactures a FAIL out of a working conditioning.
* **Alpha on a chronological train holdout, never on val.** Selecting on val inflated the
  headline from 6.2 % to 9.8 % -- right across the 10 % threshold the branch turns on.
* **Report both targets.** The latent is the least linearly decodable object in the pipeline
  (B21); an intervention that measurably improves the forecast can lower the latent readout.

Run (no denoiser needed -- this gate precedes 1b):

    llapdiff-ridge-probe --dataset-key noaa_uk --pred 168
    llapdiff-ridge-probe --dataset-key noaa_uk --pred 168 --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from llapdiffusion.models.llapdiff_utils import (
    compute_latent_stats,
    infer_target_dim_from_loader,
    set_torch,
    vae_io_dims_for_target_dim,
)
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import build_eval_loaders, prepare_eval_stack
from llapdiffusion.tools.run_stage1_roundtrip import _build_vae
from llapdiffusion.trainers import train_val_llapdiff as tv


def _build_frozen_stack(cfg, train_dl, device):
    """The frozen stack WITHOUT a diffusion checkpoint, for a pre-1b probe.

    Mirrors ``_load_stack``'s upstream half but takes the denoiser from the live config
    (nothing is trained yet at 1a) and derives the latent normalisation from train the same
    way the trainer does, rather than reading it out of a checkpoint payload.
    """
    _, num_entities, window_size, feat_dim = tv._summarize_dataset(train_dl, None, verbose=False)
    vae = _build_vae(cfg, train_dl, device)
    summarizer = LaplaceAE(
        num_entities=num_entities, feat_dim=feat_dim, window_size=window_size,
        mix_dim=int(getattr(cfg, "SUM_MIX_DIM", 64)), tv_hidden=cfg.SUM_TV_HIDDEN,
        out_len=cfg.SUM_CONTEXT_LEN, context_dim=cfg.SUM_CONTEXT_DIM, n_heads=cfg.NUM_HEADS,
        dropout=cfg.SUM_DROPOUT, time2vec_dim=int(getattr(cfg, "SUM_TIME2VEC_DIM", 9)),
        irreg_pooling=str(getattr(cfg, "SUM_IRREG_POOLING", "none")),
        irreg_hidden=int(getattr(cfg, "SUM_IRREG_HIDDEN", 32)),
        irreg_residual_scale=float(getattr(cfg, "SUM_IRREG_RES_SCALE", 0.1)),
        t_token_mode=str(getattr(cfg, "SUM_T_TOKEN_MODE", "none")),
        t_token_scale=float(getattr(cfg, "SUM_T_TOKEN_SCALE", 0.1)),
        pos_encoding=str(getattr(cfg, "SUM_POS_ENCODING", "learned_abs")),
        rope_base=float(getattr(cfg, "SUM_ROPE_BASE", 10000.0)),
        channel_balanced_x_loss=bool(getattr(cfg, "SUM_CHANNEL_BALANCED_X_LOSS", False)),
    ).to(device)
    sum_state = torch.load(cfg.SUM_CKPT, map_location=device)
    tv._load_module_state(
        summarizer,
        sum_state["model"] if isinstance(sum_state, dict) and "model" in sum_state else sum_state,
        strict=True,
    )
    summarizer.eval()
    diff_model = tv.build_llapdiff_model(cfg, device)
    diff_model.eval()
    mu_mean, mu_std = compute_latent_stats(
        vae, train_dl, device, mode=str(getattr(cfg, "LATENT_NORM_MODE", "global")))
    return diff_model, vae, summarizer, mu_mean, mu_std

THRESHOLD = 10.0  # percent relative RMSE reduction, per the plan's pre-registered rule


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-1a paired ridge positive control.")
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--pred", type=int, required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Optional. Supplies the frozen stack and its conditioning settings "
                        "(cond_norm_mode + statistics). Omit to build from the live config, "
                        "which is the normal case: 1a runs BEFORE 1b trains anything.")
    p.add_argument("--split", choices=("val", "test"), default="val",
                   help="Evaluation split. Keep 'val' -- gates run on val.")
    p.add_argument("--tokens", type=int, nargs="+", default=(8, 32),
                   help="Summary tokens kept per feature set (strided over the token axis).")
    p.add_argument("--history-steps", type=int, default=32,
                   help="History timesteps kept for the raw-history probe.")
    p.add_argument("--entities", type=int, default=None,
                   help="Score only windows with exactly this many observed entities, so "
                        "every row of the raw-history/raw-target design has one width. "
                        "Default: the modal count on the split.")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


@torch.no_grad()
def _collect(dl, stack, device, *, history_steps: int, entities: Optional[int]):
    """-> (raw summaries, raw history, latent target, raw target) per eligible window."""
    diff_model, vae, summarizer, mu_mean, mu_std = stack
    bags: Dict[int, List[List[torch.Tensor]]] = {}
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
        y = tv.targets_to_bhnc(yb, mask_bn, device=device)
        hist = torch.nan_to_num(V, nan=0.0)
        step = max(1, hist.shape[2] // history_steps)
        full = obs.all(dim=1)
        ent_mask = mask_bn.bool()
        for i in torch.nonzero(full).flatten().tolist():
            ents = torch.nonzero(ent_mask[i]).flatten()
            n = int(ents.numel())
            if n < 1 or (entities is not None and n != entities):
                continue
            target = y[i, :, ents, 0]              # torch advanced indexing -> [H, N]
            if not torch.isfinite(target).all():
                continue
            h = hist[i, ents][:, ::step][:, :history_steps, 0]
            bag = bags.setdefault(n, [[], [], [], []])
            bag[0].append(raw[i].float().cpu())
            bag[1].append(h.reshape(-1).float().cpu())
            bag[2].append(mu[i].reshape(-1).float().cpu())
            bag[3].append(target.reshape(-1).float().cpu())
    if not bags:
        raise RuntimeError("no eligible windows found on this split")
    keep = entities if entities is not None else max(bags, key=lambda k: len(bags[k][0]))
    return keep, tuple(torch.stack(z) for z in bags[keep])


def blocked_purged_split(n: int, *, n_blocks: int = 15, holdout=(3, 8, 13), purge: int = 0):
    """Fit/holdout indices for selection INSIDE train, without a regime confound.

    The obvious choice — the chronological last 20 % of train — is pathological on a
    seasonal, chronologically split series. Measured on noaa_uk h=168: that band has grand
    mean −0.911 against the fit portion's +0.183, and **ridge scores −8.3 % there while
    scoring +28.8 % on val from the same fit**. A selector that anti-correlates with the
    target metric is not a selector. Ridge mostly survives it (a common offset cancels when
    *ranking* alphas) but it still cost ~1.3 pt: it picked alpha=1e4 (val 27.28 %) over
    alpha=1e3 (val 28.61 %).

    Instead hold out several contiguous blocks spread across the whole train span, so the
    selector sees the same mixture of regimes the fit does, and **purge** ``purge`` windows
    either side so no fit window shares its context with a holdout window — the same idea as
    the loaders' own ``global_purged_horizon`` policy. Pass ``purge=WINDOW``.

    Credit: diagnosed by a second agent while probing B21 (`w_docs/results_nonlinear_probe.md` §2).
    """
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    hold = np.zeros(n, dtype=bool)
    for b in holdout:
        if 0 <= b < n_blocks:
            hold[edges[b]:edges[b + 1]] = True
    fit = ~hold
    if purge > 0:
        banned = np.zeros(n, dtype=bool)
        for b in holdout:
            if not (0 <= b < n_blocks):
                continue
            lo, hi = edges[b], edges[b + 1]
            banned[max(0, lo - purge):lo] = True
            banned[hi:min(n, hi + purge)] = True
        fit &= ~banned
    return np.flatnonzero(fit), np.flatnonzero(hold)


def ridge_reduction(Xtr, ytr, Xva, yva, *, purge: int = 0) -> float:
    """Percent RMSE reduction vs predict-the-mean, alpha chosen on a train holdout.

    The holdout is blocked-and-purged rather than the trailing 20 % — see
    ``blocked_purged_split`` for why the obvious choice is unusable here.
    """
    Xtr, Xva = np.asarray(Xtr, np.float64), np.asarray(Xva, np.float64)
    ytr, yva = np.asarray(ytr, np.float64), np.asarray(yva, np.float64)
    fit_idx, hold_idx = blocked_purged_split(len(Xtr), purge=purge)
    grid = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)

    def fit(X, y):
        xm, ym = X.mean(0, keepdims=True), y.mean(0, keepdims=True)
        Xc = X - xm
        lam, V = np.linalg.eigh(Xc.T @ Xc)
        return xm, ym, lam, V, V.T @ (Xc.T @ (y - ym))

    xm0, ym0, lam0, V0, RV0 = fit(Xtr[fit_idx], ytr[fit_idx])
    best = (None, float("inf"))
    for a in grid:
        W = V0 @ (RV0 / (lam0 + a)[:, None])
        r = float(np.sqrt((((Xtr[hold_idx] - xm0) @ W + ym0 - ytr[hold_idx]) ** 2).mean()))
        if r < best[1]:
            best = (a, r)
    xm, ym, lam, V, RV = fit(Xtr, ytr)
    W = V @ (RV / (lam + best[0])[:, None])
    pred = (Xva - xm) @ W + ym
    rmse = float(np.sqrt(((pred - yva) ** 2).mean()))
    base = float(np.sqrt(((yva - yva.mean()) ** 2).mean()))
    return 100.0 * (1.0 - rmse / base)


def _verdict(a: float, b: float, c: float) -> Tuple[str, str]:
    if a < THRESHOLD:
        return ("MOVE DATASET", "the raw history is not forecastable either, so this cell "
                                "genuinely cannot answer the gate")
    if c >= THRESHOLD and b >= 0.7 * a:
        return ("PROCEED to 1b", "the signal survives to the latent; the gate can discriminate")
    if b < 0.7 * a:
        return ("FIX THE CONDITIONING (B21)",
                "the dataset is forecastable but the summarizer does not deliver it. Moving "
                "datasets would carry the same conditioning along and reproduce this")
    return ("FIX STAGE 1 / RE-TARGET 1b",
            "the conditioning preserves the signal but the latent does not expose it, so 1b "
            "reads a quantity the pipeline cannot predict")


def main() -> None:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False,
                       allow_tf32=bool(getattr(cfg, "ALLOW_TF32", False)))

    if args.checkpoint:
        loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    else:
        # 1a precedes 1b, so there is normally no denoiser yet: build the frozen stack from
        # the config-derived artifacts instead.
        loaders = build_eval_loaders(cfg)
        stack = _build_frozen_stack(cfg, loaders[0], device)
    train_dl, val_dl, test_dl, _ = loaders
    eval_dl = val_dl if args.split == "val" else test_dl
    diff_model = stack[0]
    mode = tv._resolve_cond_norm_mode(diff_model)
    if mode == "global" and tv._resolve_cond_norm_stats(diff_model) is None:
        diff_model.cond_norm_stats = tv.compute_global_cond_stats(
            stack[2], diff_model, train_dl, device, verbose=True)
    print(f"[1a] {args.dataset_key} h={args.pred} split={args.split} "
          f"cond_norm_mode={mode} LAPLACE_K={cfg.LAPLACE_K}")

    n_ent, (RAWtr, HTtr, Ltr, Rtr) = _collect(
        train_dl, stack, device, history_steps=args.history_steps, entities=args.entities)
    _, (RAWva, HTva, Lva, Rva) = _collect(
        eval_dl, stack, device, history_steps=args.history_steps, entities=n_ent)
    print(f"[1a] {len(RAWtr)} train / {len(RAWva)} {args.split} windows at {n_ent} entities\n")

    from llapdiffusion.models.llapdiff_utils import normalize_cond_per_batch
    stats = tv._resolve_cond_norm_stats(diff_model)

    def summary_feats(raw, ntok):
        cs = normalize_cond_per_batch(raw, mode=mode, stats=stats)
        step = max(1, cs.shape[1] // ntok)
        return cs[:, ::step][:, :ntok].reshape(cs.shape[0], -1).numpy()

    # purge = WINDOW: no fitting window may share its context with a selection window.
    purge = int(getattr(cfg, "WINDOW", 0))
    a_raw = ridge_reduction(HTtr.numpy(), Rtr.numpy(), HTva.numpy(), Rva.numpy(), purge=purge)
    a_lat = ridge_reduction(HTtr.numpy(), Ltr.numpy(), HTva.numpy(), Lva.numpy(), purge=purge)
    rows: List[Dict[str, object]] = []
    print(f"{'probe':<34}{'dims':>7}{'-> raw target':>15}{'-> latent':>11}")
    print(f"{'A  raw history':<34}{HTtr.shape[1]:>7}{a_raw:>14.1f}%{a_lat:>10.1f}%")
    best_b = best_c = -float("inf")
    for ntok in args.tokens:
        Xtr, Xva = summary_feats(RAWtr, ntok), summary_feats(RAWva, ntok)
        b = ridge_reduction(Xtr, Rtr.numpy(), Xva, Rva.numpy(), purge=purge)
        c = ridge_reduction(Xtr, Ltr.numpy(), Xva, Lva.numpy(), purge=purge)
        best_b, best_c = max(best_b, b), max(best_c, c)
        print(f"{'B/C cond_summary (' + str(ntok) + ' tokens)':<34}{Xtr.shape[1]:>7}"
              f"{b:>14.1f}%{c:>10.1f}%")
        rows.append({"tokens": int(ntok), "dims": int(Xtr.shape[1]),
                     "summary_to_raw": b, "summary_to_latent": c})

    verdict, why = _verdict(a_raw, best_b, best_c)
    print(f"\n  A = {a_raw:.1f}%  B = {best_b:.1f}%  C = {best_c:.1f}%   (threshold {THRESHOLD:.0f}%)")
    print(f"  VERDICT: {verdict}\n           {why}")

    if args.out_json:
        payload = {
            "dataset_key": args.dataset_key, "pred": int(args.pred), "split": args.split,
            "cond_norm_mode": mode, "laplace_k": int(cfg.LAPLACE_K), "entities": int(n_ent),
            "n_train": int(len(RAWtr)), "n_eval": int(len(RAWva)),
            "threshold_pct": THRESHOLD,
            "A_history_to_raw": a_raw, "A_history_to_latent": a_lat,
            "B_summary_to_raw_best": best_b, "C_summary_to_latent_best": best_c,
            "rows": rows, "verdict": verdict, "rationale": why,
        }
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
