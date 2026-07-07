"""Calibration eval for the Theorem-C analytic UQ head (U2/U3).

Loads a chirp+UQ checkpoint and reports:

1. **Latent space** (the space where the law is exactly Gaussian): PIT calibration
   error, reliability (central-interval coverage), Gaussian NLL, mean RMSE — from
   one analytic read of N(mean, Var) at the test (or val) queries.
2. **Data space** (default; disable with ``--latent-only``): CRPS/MAE/MSE of the
   analytic law propagated through the decoder — an ensemble of latent Gaussian
   draws, one decoder pass per draw, scored by the UNCHANGED
   ``evaluate_regression`` machinery (same masking, same CRPS estimator, same
   sample count as the sampled baseline) — plus the sampled-diffusion baseline on
   the same split with wall-clock for both, i.e. the plan's "analytic vs sampled
   CRPS at matched wall-clock" comparison.

Two mean sources:

- ``oneshot`` (default): a single forward at the final diffusion step with
  information-free noise input — the U3 "no-diffusion" read of the model.
- ``ddim``: the deterministic DDIM x0 as the mean, with the variance read from a
  forward at t=1 around that mean.

Run:  llapdiff-uq-eval --dataset-key physionet --pred 12 --checkpoint <chirp-uq ckpt>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.models.uq_metrics import (
    gaussian_nll,
    gaussian_pit,
    pit_calibration_error,
    reliability_curve,
)
from llapdiffusion.tools import llapdiff_checkpoint_eval as ce
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.trainers import train_val_llapdiff as tv


def prepare_eval_stack(cfg, ckpt_path, *, device: torch.device):
    """Loaders + frozen stack for a checkpoint, mirroring evaluate_checkpoint's
    preamble (predict-type + target metadata synced from the checkpoint, loaders
    built with the resolved target columns). Returns (loaders, stack)."""
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    ce._apply_checkpoint_predict_type(cfg, payload, explicit_predict_type=None)
    setattr(
        cfg,
        "CHECKPOINT_TARGET_METADATA_APPLIED",
        ce._apply_checkpoint_target_metadata_if_unrequested(cfg, payload),
    )
    run_experiment = ce.resolve_run_experiment(cfg.DATA_DIR)
    batch_size = int(getattr(cfg, "BATCH_SIZE", getattr(cfg, "DATES_PER_BATCH", 1)))
    loaders = run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=batch_size,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        batch_size=batch_size,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=bool(getattr(cfg, "exact_timestamp_batches", True)),
        target_col=None if getattr(cfg, "TARGET_COLS", None) else getattr(cfg, "TARGET_COL", None),
        target_cols=getattr(cfg, "TARGET_COLS", None),
    )
    ce.sync_target_artifact_config(cfg, ce._target_policy(cfg), update_output_dirs=False)
    stack = ce._load_stack(cfg, ckpt_path, device, loaders[0])
    return loaders, stack


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic (Theorem-C) UQ calibration eval.")
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--mean-source", choices=("oneshot", "ddim"), default="oneshot")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap for the latent-space pass only; data space runs the full split.")
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--latent-only", action="store_true",
                        help="Skip the data-space (decoder-propagated) evaluation.")
    parser.add_argument("--skip-sampled", action="store_true",
                        help="Data space: skip the (expensive) sampled-diffusion baseline.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Ensemble size for BOTH data-space arms (default config NUM_EVAL_SAMPLES=25).")
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


class AnalyticLawSampler:
    """Duck-typed stand-in for LLapDiff inside ``evaluate_regression``.

    ``generate`` returns draws from the Theorem-C analytic Gaussian law
    N(mean, Var) instead of running reverse diffusion, so the unchanged
    evaluate_regression machinery scores the analytic predictive law with the
    exact same masking, decoding, and CRPS estimator as the sampled baseline.
    (mean, Var) are computed once per batch (cached on the conditioning tensors);
    each subsequent sample costs one Gaussian draw + one decoder pass downstream.
    Everything else (``eval()``, ``cond_adapter``, ``scheduler``, ...) delegates
    to the wrapped model.
    """

    def __init__(self, model, cfg, *, mean_source: str, device: torch.device) -> None:
        self._model = model
        self._cfg = cfg
        self._mean_source = str(mean_source)
        self._device = device
        self._cache_key = None
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def __getattr__(self, name):
        return getattr(self._model, name)

    @torch.no_grad()
    def generate(
        self,
        shape,
        *,
        cond_summary=None,
        cond_summary_raw=None,
        dt=None,
        generator: Optional[torch.Generator] = None,
        **_ignored,
    ) -> torch.Tensor:
        key = (id(cond_summary), id(cond_summary_raw), id(dt), tuple(shape))
        if key != self._cache_key:
            mean, var = _predict_mean_var(
                self._model,
                self._cfg,
                mu_shape=tuple(shape),
                cond_summary=cond_summary,
                cond_summary_raw=cond_summary_raw,
                dt_model=dt,
                mean_source=self._mean_source,
                device=self._device,
            )
            self._cache_key = key
            self._mean = mean
            self._std = var.clamp_min(1e-6).sqrt()
        if generator is not None:
            noise = torch.randn(
                self._mean.shape, device=self._mean.device,
                dtype=self._mean.dtype, generator=generator,
            )
        else:
            noise = torch.randn_like(self._mean)
        return self._mean + self._std * noise


@torch.no_grad()
def _predict_mean_var(
    diff_model,
    cfg,
    *,
    mu_shape,
    cond_summary,
    cond_summary_raw,
    dt_model,
    mean_source: str,
    device: torch.device,
):
    timesteps = int(diff_model.scheduler.timesteps)
    B = mu_shape[0]
    if mean_source == "oneshot":
        t = torch.full((B,), timesteps - 1, device=device, dtype=torch.long)
        x_t = torch.randn(mu_shape, device=device)
        return diff_model(
            x_t, t, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
            dt=dt_model, return_variance=True,
        )
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    mean = diff_model.generate(
        shape=tuple(mu_shape),
        steps=int(sampling["steps"]),
        guidance_strength=sampling["guidance_strength"],
        guidance_power=float(sampling["guidance_power"]),
        eta=0.0,
        cond_summary=cond_summary,
        cond_summary_raw=cond_summary_raw,
        dt=dt_model,
        dynamic_thresh_p=float(sampling.get("dynamic_thresh_p", 0.0)),
    )
    t1 = torch.ones(B, device=device, dtype=torch.long)
    x_t1, _ = diff_model.scheduler.q_sample(mean, t1, torch.randn_like(mean))
    _, variance = diff_model(
        x_t1, t1, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
        dt=dt_model, return_variance=True,
    )
    return mean, variance


def main() -> None:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False)

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    _, val_dl, test_dl, _ = loaders
    loader = val_dl if args.split == "val" else test_dl
    diff_model, vae, summarizer, mu_mean, mu_std = stack
    if not bool(getattr(diff_model.model, "chirp_uq_head", False)):
        raise ValueError(
            "Checkpoint was not trained with CHIRP_UQ_HEAD=True; the analytic "
            "Gaussian law is unavailable."
        )

    ys: List[torch.Tensor] = []
    means: List[torch.Tensor] = []
    variances: List[torch.Tensor] = []
    batches = 0
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

        mean, variance = _predict_mean_var(
            diff_model, cfg,
            mu_shape=mu_norm.shape,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            dt_model=dt_model,
            mean_source=str(args.mean_source),
            device=device,
        )

        mask = torch.as_tensor(obs_any, device=device, dtype=torch.bool)
        while mask.dim() < mu_norm.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(mu_norm)
        ys.append(mu_norm[mask].detach().cpu())
        means.append(mean[mask].detach().cpu())
        variances.append(variance[mask].detach().cpu())

        batches += 1
        if args.max_batches is not None and batches >= int(args.max_batches):
            break

    if not ys:
        raise RuntimeError("No observed latent targets found on the selected split.")
    y = torch.cat(ys)
    mean = torch.cat(means)
    var = torch.cat(variances)

    u = gaussian_pit(y, mean, var)
    report: Dict[str, object] = {
        "dataset_key": args.dataset_key,
        "pred": int(args.pred),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mean_source": args.mean_source,
        "num_elements": int(y.numel()),
        "latent_rmse": float((mean - y).pow(2).mean().sqrt().item()),
        "latent_gaussian_nll": gaussian_nll(y, mean, var),
        "pit_calibration_error": pit_calibration_error(u, num_bins=int(args.num_bins)),
        "reliability": reliability_curve(u),
        "mean_predicted_std": float(var.clamp_min(1e-6).sqrt().mean().item()),
    }

    if not args.latent_only:
        # Data-space comparison: the analytic law propagated through the decoder
        # (Gaussian latent draws -> decode) vs the sampled-diffusion baseline, scored
        # by the SAME evaluate_regression code with the SAME ensemble size and seed.
        if args.num_samples is not None:
            cfg.NUM_EVAL_SAMPLES = int(args.num_samples)
        num_samples = int(getattr(cfg, "NUM_EVAL_SAMPLES", 25))
        sampling = tv._sampling_kwargs(cfg, prefix="TEST")
        common = dict(
            device=device, mu_mean=mu_mean, mu_std=mu_std, config=cfg, ema=None,
            self_cond=bool(getattr(cfg, "SELF_COND", False)),
            disable_conditioning=False, verbose=False,
            generator_seed=int(getattr(cfg, "SEED", 42)),
        )

        analytic_model = AnalyticLawSampler(
            diff_model, cfg, mean_source=str(args.mean_source), device=device
        )
        start = time.perf_counter()
        analytic = tv.evaluate_regression(
            analytic_model, vae, summarizer, loader, **common, **sampling
        )
        analytic_wall = time.perf_counter() - start
        report["data_space_analytic"] = {
            "crps": analytic.get("crps"),
            "mae": analytic.get("mae"),
            "mse": analytic.get("mse"),
            "num_samples": num_samples,
            "mean_source": str(args.mean_source),
            "wall_seconds": analytic_wall,
        }

        if not args.skip_sampled:
            start = time.perf_counter()
            sampled = tv.evaluate_regression(
                diff_model, vae, summarizer, loader, **common, **sampling
            )
            sampled_wall = time.perf_counter() - start
            report["data_space_sampled"] = {
                "crps": sampled.get("crps"),
                "mae": sampled.get("mae"),
                "mse": sampled.get("mse"),
                "num_samples": num_samples,
                "ddim_steps": int(sampling["steps"]),
                "wall_seconds": sampled_wall,
            }
            report["analytic_speedup_x"] = (
                sampled_wall / analytic_wall if analytic_wall > 0 else None
            )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
