"""Read R9: the variance decomposition that decides Option A against Option B.

Under Option B the parameters are read off a reverse-diffusion trajectory, so the object
Theorem C describes is the law *conditional on the terminal draw* ``z_T``, and the forecast
distribution is the mixture over ``z_T``. Uncertainty then decomposes exactly by the law of
total covariance:

    Cov(z_0 | H) = E_{z_T}[K_{z_T}]        (within-component, the SDE head)
                 + Cov_{z_T}(m_{z_T})      (between-component, what the loop buys)

and the between-to-total share ``R`` says whether the reverse loop is doing any work the
SDE head does not already do. Pre-registered reading:

    R < 0.10   the loop is idle          -> Option A is the model
    R > 0.25   a single GP is not the law -> Option B is required
    otherwise  inconclusive, carry both

This needs **no training**: it runs M reverse trajectories from an existing NLL-trained
checkpoint and decomposes the predictive variance. Nothing is scored against targets, so no
split is consumed in the sense the pre-registration cares about -- but the default is still
``val``, because the decision it feeds is a modelling decision.

Per-component parameters are captured at the **final** reverse step, which is what
Algorithm 2 says supplies the returned component. As a correctness check the reconstructed
component mean is compared against the tensor ``generate`` actually returned: with
``predict_type='x0'``, no output head and thresholding off, the two are the same object, so
a nonzero gap means the capture is not describing what the model returned.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import torch

from llapdiffusion.configs.config_utils import refresh_artifact_paths
from llapdiffusion.models.chirp_gp import marginal_law, modal_increments, sample_joint_markov
from llapdiffusion.models.path_scores import (
    coverage_and_width,
    crps,
    energy_score,
    randomized_pit,
    variogram_score,
)
from llapdiffusion.models.uq_metrics import pit_calibration_error
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools import llapdiff_checkpoint_eval as ce
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
from llapdiffusion.trainers import train_val_llapdiff as tv

# The M values the paper reports convergence over.
DEFAULT_LADDER = (1, 2, 4, 8, 16, 25)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-key", required=True)
    p.add_argument("--pred", type=int, required=True)
    p.add_argument("--checkpoint", required=True,
                   help="an NLL-trained checkpoint WITH the Theorem-C UQ head (stage s2)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--components", type=int, default=25,
                   help="M, the number of reverse trajectories (terminal draws).")
    p.add_argument("--ladder", type=int, nargs="*", default=list(DEFAULT_LADDER),
                   help="M values to report the decomposition at; each uses the first M draws.")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap the number of eval batches -- use for smoke runs.")
    p.add_argument("--score", action="store_true",
                   help="also score every arm against the latent targets, giving the "
                        "M-convergence frontier from the same trajectories.")
    p.add_argument("--within-at-timestep", type=int, default=None,
                   help="Sensitivity read: take the WITHIN-component variance from a "
                        "forward at this diffusion timestep on a re-noised component "
                        "mean, instead of from the final reverse step. Algorithm 2 "
                        "specifies the final step (the default); T-1 is the earlier A2 "
                        "protocol and inflates the within term ~3x. Use it to check "
                        "whether the gate verdict survives the read protocol.")
    p.add_argument("--score-samples", type=int, default=25,
                   help="ensemble size for the mixture arm; matched to the sampled arm at "
                        "M=25 so estimator parity holds.")
    p.add_argument("--weights", default="ema", choices=("raw", "ema"),
                   help="EMA by default: _load_stack reads payload['model'], which holds RAW "
                        "weights even in *_best_ema.pt (B16), and every other phase of this "
                        "project reports EMA numbers.")
    p.add_argument("--gen-steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None,
                   help="frozen at 1.0 for this cell: guidance measured monotone-harmful.")
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--vae-entity-encode", action="store_true")
    p.add_argument("--sum-ckpt", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


@torch.no_grad()
def _component_laws(
    diff_model, cfg, *, mu_shape, cond_summary, cond_summary_raw, dt_model,
    components: int, device: torch.device, generator: Optional[torch.Generator],
    keep_params: bool = False, within_at_timestep: Optional[int] = None,
) -> Dict[str, object]:
    """Run M reverse trajectories and return each component's mean and marginal variance.

    Returns ``means`` and ``vars`` of shape ``[M,B,T,D]``, the M returned DDIM draws (which
    *are* the conventional sampled arm's ensemble -- the same trajectories serve both arms,
    so no comparison is confounded by having been generated separately), and the max
    absolute gap between the reconstructed mean and the tensor ``generate`` returned.

    ``keep_params`` additionally retains everything ``sample_joint_markov`` needs, so the
    mixture arm can draw joint paths from the same components rather than a second run.
    """
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    means, variances, gaps, draws, params = [], [], [], [], []
    for _ in range(int(components)):
        cap: Dict[str, torch.Tensor] = {}
        sample = diff_model.generate(
            shape=tuple(mu_shape),
            steps=int(sampling["steps"]),
            guidance_strength=sampling["guidance_strength"],
            guidance_power=float(sampling["guidance_power"]),
            eta=0.0,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            dt=dt_model,
            dynamic_thresh_p=float(sampling.get("dynamic_thresh_p", 0.0)),
            generator=generator,
            modal_capture=cap,
        )
        missing = {"theta", "rho_bar", "omega_bar", "t_rel", "p0", "q"} - set(cap)
        if missing:
            raise RuntimeError(
                f"the final-step modal capture is missing {sorted(missing)}; this tool needs a "
                f"chirp core with the Theorem-C UQ head (stage s2_diffusion_uq)."
            )
        alpha = cap.get("alpha", torch.tensor(1.0, device=device))
        inc = modal_increments(cap["rho_bar"], cap["t_rel"], cap["q"], cap["p0"])
        m, v = marginal_law(
            cap["theta"], cap["rho_bar"], cap["omega_bar"], inc["lam"], alpha=alpha
        )
        if within_at_timestep is not None:
            # Sensitivity of the gate to the read protocol. Algorithm 2 says the FINAL reverse
            # step supplies the component, which is what `v` above is. The earlier A2 protocol
            # instead read the variance at a pure-noise input, where `energy(theta)` is the
            # modal decomposition of noise rather than of signal and the spread is ~3x larger.
            # That choice moves the within-component term by more than any arm difference, so
            # whether the gate's verdict survives it is worth measuring rather than inferring.
            t_read = torch.full((mu_shape[0],), int(within_at_timestep),
                                device=device, dtype=torch.long)
            noise = torch.randn(m.shape, device=device, dtype=m.dtype, generator=generator)
            x_read, _ = diff_model.scheduler.q_sample(m, t_read, noise)
            _, v = diff_model(
                x_read, t_read, cond_summary=cond_summary,
                cond_summary_raw=cond_summary_raw, dt=dt_model, return_variance=True,
            )
        means.append(m)
        variances.append(v)
        draws.append(sample)
        gaps.append((m - sample).abs().max())
        if keep_params:
            params.append({
                "theta": cap["theta"], "rho_bar": cap["rho_bar"],
                "omega_bar": cap["omega_bar"], "incr": inc["incr"],
                "d_rho": inc["d_rho"], "p0": cap["p0"], "alpha": alpha,
            })
    out: Dict[str, object] = {
        "means": torch.stack(means),
        "vars": torch.stack(variances),
        "ddim_draws": torch.stack(draws),
        "mean_reconstruction_gap": torch.stack(gaps).max(),
    }
    if keep_params:
        out["params"] = params
    return out


def _mixture_samples(params: List[Dict[str, torch.Tensor]], total: int,
                     generator: Optional[torch.Generator]) -> torch.Tensor:
    """Draw ``total`` joint paths from the M-component mixture, evenly across components.

    Each component contributes ``ceil(total/M)`` exact joint draws by the O(Kh) Markov
    recursion; the pool is trimmed to ``total`` so every arm is scored with the identical
    ensemble size, which is what estimator parity requires.
    """
    M = len(params)
    per = -(-total // M)
    pool = [
        sample_joint_markov(
            p["theta"], p["rho_bar"], p["omega_bar"], p["incr"], p["d_rho"], p["p0"],
            num_samples=per, alpha=p["alpha"], generator=generator,
        )
        for p in params
    ]
    return torch.cat(pool, dim=0)[:total]


@torch.no_grad()
def _score_batch(acc, out, y, mask, dt_model, ladder, *, ensemble: int, generator) -> None:
    """Score every arm on this batch from the SAME M trajectories.

    Two arm families come out of one pass, which is the point:
      * ``sampled(M)`` -- the conventional diffusion ensemble, i.e. the M returned DDIM
        draws scored empirically. At M=25 this is the paper's ``CMD-D + 25 samples`` row.
      * ``mixture(M)`` -- the CMD-D(M) predictive law: the analytic component per trajectory,
        sampled jointly and pooled.
    Sharing the trajectories means a sampled-vs-mixture gap cannot be a generation artefact.
    """
    t_rel = dt_model.reshape(dt_model.shape[0], -1)
    for m in ladder:
        if m > out["means"].shape[0]:
            continue
        arms = {
            f"sampled_M{m}": out["ddim_draws"][:m],
            f"mixture_M{m}": _mixture_samples(out["params"][:m], ensemble, generator),
        }
        for name, s in arms.items():
            if s.shape[0] < 2:
                continue                      # ES/CRPS need >= 2 members
            es = energy_score(s, y, mask)
            cr = crps(s, y, mask)
            row = acc.setdefault(name, {})
            for key, val in (
                ("ES_unbiased", es["unbiased"]), ("ES_naive", es["naive"]),
                ("CRPS_unbiased", cr["unbiased"]), ("CRPS_naive", cr["naive"]),
                ("VS_p0.5", variogram_score(s, y, t_rel, mask)),
            ):
                row.setdefault(key, []).append(val.detach().cpu())
            cw = coverage_and_width(s, y, mask)
            for key, val in cw.items():
                row.setdefault(key, []).append(torch.tensor([val]))
            # PIT values are kept RAW and pooled across batches, because the calibration error
            # is a functional of the whole empirical CDF -- averaging per-batch ECEs is not the
            # ECE of the pooled sample and is biased low, since each batch's CDF is closer to
            # uniform by chance than the pooled one. The leading underscore keeps this column
            # out of the weighted-average loop that finalises the scalar columns.
            row.setdefault("_pit", []).append(
                randomized_pit(s, y, mask, generator=generator).detach().cpu())
            row.setdefault("n_windows", []).append(torch.tensor([float(y.shape[0])]))


def _decompose(means: torch.Tensor, variances: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Law of total covariance, traces only.

    ``R`` needs the marginal variances alone -- the trace of a covariance is the sum of its
    diagonal -- so the O(K h^2) kernels are not materialised here.
    """
    M = means.shape[0]
    w = mask.to(means.dtype)
    n = w.sum().clamp_min(1.0)
    within = float(((variances.mean(0)) * w).sum() / n)
    if M > 1:
        # Unbiased across-component variance of the component means.
        between = float((means.var(dim=0, unbiased=True) * w).sum() / n)
    else:
        between = 0.0
    total = within + between
    return {
        "M": int(M),
        "within_E_K": within,
        "between_Cov_mu": between,
        "total": total,
        "R": (between / total) if total > 0 else float("nan"),
    }


def main() -> Dict[str, object]:
    args = _parse_args()

    cfg = build_eval_config(args.dataset_key, int(args.pred))
    # Stage-1/2 architecture flags first: the artifact names derive from them.
    if args.vae_entity_encode:
        cfg.VAE_ENTITY_ENCODE = True
    refresh_artifact_paths(cfg)
    if args.vae_ckpt:
        cfg.VAE_CKPT = str(args.vae_ckpt)
    if args.sum_ckpt:
        cfg.SUM_CKPT = str(args.sum_ckpt)
    if args.seed is not None:
        cfg.SEED = int(args.seed)
    if args.gen_steps is not None:
        cfg.TEST_STEPS = int(args.gen_steps)
    # Guidance is frozen at 1.0 for this cell (measured monotone-harmful), and at w=1 the
    # unconditional pass is skipped, so a component costs `steps` forwards rather than 2x.
    cfg.TEST_GUIDANCE = 1.0 if args.guidance is None else float(args.guidance)
    cfg.TEST_GUIDANCE_POWER = 1.0

    device = set_torch(
        seed=int(getattr(cfg, "SEED", 42)),
        deterministic=False,
        allow_tf32=bool(getattr(cfg, "ALLOW_TF32", False)),
    )

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    diff_model, vae, summarizer, mu_mean, mu_std = stack
    if not bool(getattr(diff_model.model, "chirp_uq_head", False)):
        raise ValueError(
            "This checkpoint has no Theorem-C UQ head, so it defines no component law and "
            "the decomposition is undefined. Use a stage s2_diffusion_uq checkpoint."
        )
    untrained = diff_model.model.chirp_field.uq_head_untrained_signature()
    if untrained is not None:
        raise ValueError(
            f"The UQ head is bit-identical to its initialization ({untrained}), so the "
            f"within-component term would just restate CHIRP_UQ_INIT_VAR. Train with "
            f"DIFF_LOSS_MODE='gaussian_nll' first."
        )
    if args.weights == "ema":
        n = ce._apply_ema_weights(diff_model, torch.load(args.checkpoint, map_location="cpu"))
        print(f"[mixture-uq] applied the EMA shadow to {n} parameters")

    train_dl, val_dl, test_dl, _ = loaders
    loader = {"train": train_dl, "val": val_dl, "test": test_dl}[args.split]

    gen = torch.Generator(device=device)
    gen.manual_seed(int(getattr(cfg, "SEED", 42)))

    all_means: List[torch.Tensor] = []
    all_vars: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    scores: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    ladder_requested = sorted(set(int(x) for x in args.ladder))
    max_gap, batches = 0.0, 0
    for xb, yb, meta in loader:
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=meta.get("delta_t"), x_obs_mask=meta.get("x_obs_mask"),
        )
        dt_b = tv._flatten_dt(meta, mask_bn, device, key="delta_t_y")
        mu_norm, obs_any = tv._latent_targets_for_batch(
            vae, yb, mask_bn, meta, device, mu_mean, mu_std
        )
        if mu_norm is None or obs_any is None or not obs_any.any():
            continue
        dt_model = tv._match_dt_to_horizon(dt_b, mu_norm.size(1))

        out = _component_laws(
            diff_model, cfg, mu_shape=mu_norm.shape, cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw, dt_model=dt_model,
            components=int(args.components), device=device, generator=gen,
            keep_params=bool(args.score),
            within_at_timestep=args.within_at_timestep,
        )
        mask = torch.as_tensor(obs_any, device=device, dtype=torch.bool)
        while mask.dim() < mu_norm.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(mu_norm)

        all_means.append(out["means"].cpu())
        all_vars.append(out["vars"].cpu())
        all_masks.append(mask.cpu())
        max_gap = max(max_gap, float(out["mean_reconstruction_gap"]))

        if args.score:
            _score_batch(
                scores, out, mu_norm, mask, dt_model, ladder_requested,
                ensemble=int(args.score_samples), generator=gen,
            )

        batches += 1
        print(f"[mixture-uq] batch {batches}  M={args.components}  "
              f"mean-reconstruction gap {max_gap:.3e}", flush=True)
        if args.max_batches is not None and batches >= int(args.max_batches):
            break

    if not all_means:
        raise RuntimeError("No observed latent targets on the selected split.")

    means = torch.cat(all_means, dim=1)
    variances = torch.cat(all_vars, dim=1)
    mask = torch.cat(all_masks, dim=0)

    ladder = [m for m in sorted(set(int(x) for x in args.ladder)) if 1 <= m <= means.shape[0]]
    decomposition = [_decompose(means[:m], variances[:m], mask) for m in ladder]

    r_full = decomposition[-1]["R"]
    verdict = (
        "OPTION_A: the reverse loop is idle (R < 0.10)" if r_full < 0.10 else
        "OPTION_B: a single GP is not the forecast law (R > 0.25)" if r_full > 0.25 else
        "INCONCLUSIVE: carry both arms (0.10 <= R <= 0.25)"
    )

    # Per-window columns (ES/VS/CRPS) average over windows; per-batch columns
    # (coverage/width, computed over pooled entries) average over batches weighted by the
    # window count, so a short trailing batch does not count as a full one.
    arm_scores: Dict[str, Dict[str, float]] = {}
    for arm, cols in scores.items():
        n_per_batch = torch.cat(cols["n_windows"])
        row: Dict[str, float] = {"n_windows": float(n_per_batch.sum())}
        for key, vals in cols.items():
            if key == "n_windows" or key.startswith("_"):
                continue
            stacked = torch.cat(vals)
            if stacked.numel() == n_per_batch.numel():        # one scalar per batch
                row[key] = float((stacked * n_per_batch).sum() / n_per_batch.sum())
            else:                                              # one value per window
                row[key] = float(stacked.mean())
        if cols.get("_pit"):
            pit = torch.cat(cols["_pit"])
            row["PIT_ECE"] = float(pit_calibration_error(pit))
            row["PIT_n"] = float(pit.numel())
        arm_scores[arm] = row

    report = {
        "read": "R9",
        "arm_scores": arm_scores,
        "arm_scores_note": (
            "mixture_M<m> draws --score-samples members from m components (sampling a "
            "Gaussian is free, which is the paper's efficiency claim: structured components "
            "substitute for outer samples). sampled_M<m> necessarily has exactly m members, "
            "so only sampled_M25 is a protocol arm ('CMD-D + 25 samples'); smaller m are "
            "diagnostics whose ensemble size differs and whose estimator bias therefore "
            "differs -- read the naive/unbiased pair before comparing them."
        ),
        "dataset_key": args.dataset_key,
        "pred": int(args.pred),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "components": int(args.components),
        "within_at_timestep": args.within_at_timestep,
        "within_read": ("final_reverse_step (Algorithm 2)"
                        if args.within_at_timestep is None
                        else f"renoised at t={args.within_at_timestep} (sensitivity)"),
        "batches": batches,
        "num_elements": int(mask.sum()),
        "decomposition": decomposition,
        "R_at_max_M": r_full,
        "verdict": verdict,
        "mean_reconstruction_gap": max_gap,
        "resolved": {
            "steps": int(tv._sampling_kwargs(cfg, prefix="TEST")["steps"]),
            "guidance_strength": float(cfg.TEST_GUIDANCE),
            "vae_ckpt": str(cfg.VAE_CKPT),
            "sum_ckpt": str(cfg.SUM_CKPT),
            # Read off the REBUILT MODEL, not the config: the model is reconstructed from the
            # checkpoint's own metadata, so a config default of "inherit" would misreport a
            # checkpoint trained under "global". Under "inherit" x mean-pooling the pole and
            # UQ conditioning is identically zero (B23) and the law is structurally
            # homoscedastic, so this field decides whether the row means anything at all.
            "summary_pool_norm_mode": str(
                getattr(diff_model.model, "summary_pool_norm_mode", "unknown")
            ),
            "laplace_k": int(getattr(diff_model.model, "k", -1)),
            "predict_type": str(getattr(diff_model, "predict_type", "?")),
            "output_head_active": bool(getattr(diff_model.model, "_use_output_head", False)),
            "seed": int(getattr(cfg, "SEED", 42)),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            fh.write(text + "\n")
    return report


if __name__ == "__main__":
    main()
