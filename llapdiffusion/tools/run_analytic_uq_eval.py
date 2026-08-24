"""Calibration eval for the Theorem-C analytic UQ head (U2/U3).

Loads a chirp+UQ checkpoint and reports:

1. **Latent space** (the space where the law is exactly Gaussian): PIT calibration
   error, reliability (central-interval coverage), Gaussian NLL, mean RMSE — from
   one analytic read of N(mean, Var) at the test (or val) queries.

   With ``--latent-only`` this also accepts a checkpoint **without** the UQ head: the
   mean is still readable, so the mean-only metrics (``latent_rmse``, ``latent_mse``,
   ``latent_corr`` against ``baseline_rmse_predict_mean`` / ``latent_target_std``) are
   reported and everything requiring a variance is omitted. That is the read the
   Phase-1 signal gate of ``w_docs/CMD_UQ_RECOVERY_PLAN.md`` needs, and it is taken on
   a plain-MSE S1 arm, which by construction has no UQ head.
2. **Data space** (default; disable with ``--latent-only``): CRPS/MAE/MSE of the
   analytic law propagated through the decoder — an ensemble of latent Gaussian
   draws, one decoder pass per draw, scored by the UNCHANGED
   ``evaluate_regression`` machinery (same masking, same CRPS estimator, same
   sample count as the sampled baseline) — plus the sampled-diffusion baseline on
   the same split with wall-clock for both, i.e. the plan's "analytic vs sampled
   CRPS at matched wall-clock" comparison.

Three mean sources. They differ ONLY in how the law's location is obtained, so choosing
between them is a protocol decision, not a modelling one — and it is recorded in the report's
``resolved`` block for exactly that reason:

- ``oneshot`` (default): a single forward at the final diffusion step with
  information-free noise input — the U3 "no-diffusion" read of the model (arm A3).
- ``ddim``: ONE reverse trajectory as the mean. Kept reachable because it is the
  deterministic-mean cell of the 2×2, but see the warning below before reporting it.
- ``ensemble`` (fix 1d): the mean of ``--mean-ensemble-size`` trajectories, defaulting to the
  eval ensemble size. **This is the one that makes A1-vs-A2 a calibration comparison.**

⚠️ ``generate(eta=0)`` is deterministic given ``x_T``, but ``x_T`` is random — so ``ddim``
returns ONE DRAW from the predictive distribution, not its mean, while the sampled arm's point
forecast is the mean of ``num_samples`` draws. Scoring the two against each other therefore
compares different LOCATIONS before any UQ question is asked. Measured on a live S2 checkpoint:
MSE 0.394 for ``ddim`` against 0.093 sampled, on the same weights.

The cost consequence is deliberate and is the recovery plan's own note (§1d): with the means
matched the analytic arm pays ``N*steps + 1`` denoiser passes against the sampled arm's
``N*steps``, so **A1-vs-A2 has no speedup left to report** — it becomes "calibration at matched
cost", and the method-level speedup claim belongs to A1-vs-A3.

⚠️ On a checkpoint trained with ``TRAIN_T_SAMPLER="max_only"`` (the U3 one-shot
arm), the sampled-diffusion baseline is meaningless — reverse DDIM would walk
timesteps the model never trained on. Pass ``--skip-sampled`` for that arm and
take ``data_space_sampled`` from the diffusion-trained checkpoint instead.

Run:  llapdiff-uq-eval --dataset-key physionet --pred 12 --checkpoint <chirp-uq ckpt> \
        --weights ema
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from llapdiffusion.configs.config_utils import dataloader_kwargs
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.models.uq_metrics import (
    gaussian_nll,
    gaussian_pit,
    pit_calibration_error,
    reliability_curve,
)
from llapdiffusion.tools import llapdiff_checkpoint_eval as ce
from llapdiffusion.configs.config_utils import refresh_artifact_paths
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.trainers import train_val_llapdiff as tv


def build_eval_loaders(cfg):
    """Train/val/test loaders for an eval config, with the target artifact config synced.

    Split out of ``prepare_eval_stack`` so a tool that needs the *data* but has no
    diffusion checkpoint can reuse the same construction — ``llapdiff-stage1-roundtrip``
    scores the frozen stage-1 VAE alone (Gate 0), which exists before any denoiser does.
    Callers that do have a checkpoint must apply its predict-type/target metadata to
    ``cfg`` BEFORE calling this, since the target columns feed the loaders.
    """
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
        **dataloader_kwargs(cfg),
    )
    ce.sync_target_artifact_config(cfg, ce._target_policy(cfg), update_output_dirs=False)
    return loaders


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
    loaders = build_eval_loaders(cfg)
    stack = ce._load_stack(cfg, ckpt_path, device, loaders[0])
    return loaders, stack


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic (Theorem-C) UQ calibration eval.")
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test",
        help="`train` is a DIAGNOSTIC read, never a reported number: it is the only way to "
             "tell an over-confident variance head that generalises badly (calibrated on "
             "train, sharp on val) from one whose objective or optimisation is wrong (sharp "
             "on both). The split is recorded in the report so a train number cannot be "
             "mistaken for a result.",
    )
    parser.add_argument("--mean-source", choices=("oneshot", "ddim", "ensemble"),
                        default="oneshot")
    parser.add_argument("--mean-ensemble-size", type=int, default=None,
                        help="Draws averaged for --mean-source ensemble (default: the eval "
                             "ensemble size, which is what matches the sampled arm). Setting "
                             "it BELOW that size reintroduces the location difference this "
                             "mean source exists to remove.")
    parser.add_argument(
        "--weights", choices=("raw", "ema"), default="raw",
        help=(
            "Which weights to score. 'raw' (default) keeps the historical behaviour: "
            "_load_stack reads payload['model'], which holds RAW weights even in "
            "llapdiff_*_best_ema.pt (that file differs only in which val metric selected "
            "the epoch — bug B16). 'ema' transplants payload['ema'] onto the live "
            "parameters, matching the training/finetuning protocol (EMA decay 0.999) and "
            "the plan's parity checklist. Recorded in the report JSON."
        ),
    )
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap for the latent-space pass only; data space runs the full split.")
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--add-reconstruction-variance", action="store_true",
                        help="Add the stage-1 round-trip residual variance (estimated on TRAIN) "
                             "to every data-space arm. Eq. (7) is a law for z0, so its "
                             "pushforward describes decode(z) while the target is y = "
                             "decode(encode(y)) + r; without this, r is missing from every "
                             "data-space interval. Applied to the SAMPLED arm too -- it is a "
                             "property of the decode step, not of the analytic law, and "
                             "correcting only one arm would bias the comparison.")
    parser.add_argument("--recon-var-batches", type=int, default=32,
                        help="Train batches used to estimate the reconstruction variance.")
    parser.add_argument("--variance-draws", type=int, default=4,
                        help="Noise draws averaged when reading the analytic variance at T-1. "
                             "At T-1 the head's `energy` term is the modal decomposition of a "
                             "pure-noise input, so a single draw leaves ~14.5%% median "
                             "per-element noise in the variance -- and PIT/coverage are "
                             "per-element. Averaging marginalises the draw, which is what 'the "
                             "law given only the conditioning' means; `s_modal` is "
                             "draw-independent so the heteroscedasticity is untouched. Costs "
                             "n-1 extra forwards per batch, not n-1 trajectories. Ignored for "
                             "--mean-source oneshot, where the single pass IS the arm.")
    parser.add_argument("--variance-timestep", type=int, default=None,
                        help="Diffusion timestep the analytic law's variance is read at, for "
                             "the `ddim`/`ensemble` mean sources (default: T-1, matching the "
                             "`oneshot` arm). The spread scales with the modal energy of the "
                             "x_t handed to the head, so this CHANGES THE LAW: measured on a "
                             "live S2 checkpoint, coverage@0.9 is 0.690 at t=1 and 0.931 at "
                             "t=T-1 for the same weights. Reading at t=1 asks 'how uncertain "
                             "am I about x0 given a nearly-clean x_t built from my own "
                             "prediction', which is not the quantity Table 3 reports.")
    parser.add_argument(
        "--coverage-level", type=float, default=0.8,
        help="Nominal mass of the central predictive interval reported in data space "
             "(coverage and mean interval width). Table 3's column is 0.8.",
    )
    parser.add_argument("--latent-only", action="store_true",
                        help="Skip the data-space (decoder-propagated) evaluation. Also the "
                             "mode that accepts a checkpoint without the UQ head, where only "
                             "the mean-only latent metrics are defined (Phase-1 signal gate).")
    parser.add_argument("--allow-untrained-uq-head", action="store_true",
                        help="Report the analytic law even when the UQ head was never trained "
                             "(an x0-MSE checkpoint). Off by default: q_k and p0_k get no "
                             "gradient under MSE, so the intervals would just restate "
                             "CHIRP_UQ_INIT_VAR with no loss-curve symptom. Use only to "
                             "measure the untrained head deliberately, and label it as such.")
    parser.add_argument("--skip-sampled", action="store_true",
                        help="Data space: skip the (expensive) sampled-diffusion baseline. "
                             "REQUIRED for a TRAIN_T_SAMPLER='max_only' checkpoint, where "
                             "reverse DDIM walks timesteps the model never trained on.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Ensemble size for BOTH data-space arms (default config NUM_EVAL_SAMPLES=25).")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Classifier-free guidance strength for the REPORTED numbers "
                             "(writes cfg.TEST_GUIDANCE, which _sampling_kwargs prefers). "
                             "Sampling knobs are NOT stored in the checkpoint, so a run "
                             "trained at one guidance is scored at the config default unless "
                             "this is passed -- and w > 1 sharpens the predictive law, which "
                             "is indistinguishable from miscalibration. Also the knob Phase 3's "
                             "w x GEN_STEPS sweep needs.")
    parser.add_argument("--guidance-power", type=float, default=None,
                        help="Ramp shape for --guidance (writes cfg.TEST_GUIDANCE_POWER).")
    parser.add_argument("--gen-steps", type=int, default=None,
                        help="Reverse-diffusion steps for the reported numbers "
                             "(writes cfg.TEST_STEPS).")
    parser.add_argument("--vae-ckpt", default=None,
                        help="Override the stage-1 artifact (default: the preset's VAE_CKPT).")
    parser.add_argument("--vae-entity-encode", action="store_true",
                        help="Score against the VAE trained with the encoder-side entity "
                             "embedding (B20). Loads are strict, so this must match the "
                             "artifact the checkpoint was TRAINED against.")
    parser.add_argument("--sum-ckpt", default=None,
                        help="Override the stage-2 artifact (default: the preset's SUM_CKPT).")
    parser.add_argument("--sum-decode-mode", default=None,
                        help="Architecture of --sum-ckpt (B21). Must match the artifact.")
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def estimate_reconstruction_variance(vae, train_dl, device, mu_mean, mu_std, *,
                                     max_batches: Optional[int] = 32):
    """Per-(horizon, channel) variance of the stage-1 round-trip residual, on TRAIN.

    The Theorem-C law (Eq. 7) is a law for ``z_0``: pushing it through the decoder gives the
    predictive law for ``decode(z)``, but the target is ``y``, and

        y = decode(encode(y)) + r

    so ``r`` -- everything stage 1 cannot reconstruct -- is a variance component no data-space
    arm accounts for. That is measurable rather than assumed: the latent metrics score against
    ``encode(y)`` and the data-space metrics against ``y``, so the gap between them IS this term.

    Estimated on **train** only; using the eval split would leak it into the number it is meant
    to correct. Returned per (H, C) because reconstruction error is not flat over the horizon,
    with the scalar mean reported alongside for the record.

    ⚠️ This is a MARGINAL correction. ``r`` is structured (correlated across time and entities);
    per-element PIT, coverage, CRPS and interval width depend only on its marginal variance, so
    it is the right correction for those. A joint/trajectory-level statement would need the full
    covariance and this would understate it.
    """
    sq = None
    cnt = None
    batches = 0
    for xb, yb, meta in train_dl:
        if max_batches is not None and batches >= int(max_batches):
            break
        (V, T), yb_s, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        x_tok, entity_pad, obs = tv.pack_targets_tokens(
            yb_s, mask_bn, device, y_obs_mask=meta.get("y_obs_mask")
        )
        if x_tok is None or not obs.any():
            continue
        mu_norm, _ = tv._latent_targets_for_batch(vae, yb_s, mask_bn, meta, device, mu_mean, mu_std)
        if mu_norm is None:
            continue
        y_true = torch.nan_to_num(
            tv.targets_to_bhnc(yb_s, mask_bn, device=device), nan=0.0, posinf=0.0, neginf=0.0
        )
        y_rec = tv.decode_latents_with_vae(
            vae, mu_norm, entity_pad=entity_pad, mu_mean=mu_mean, mu_std=mu_std
        )
        valid = (obs.unsqueeze(-1) if obs.dim() == 3 else obs).expand_as(y_true).to(y_true.dtype)
        res2 = ((y_rec - y_true) ** 2 * valid).sum(dim=(0, 2))       # [H, C]
        n = valid.sum(dim=(0, 2))
        sq = res2 if sq is None else sq + res2
        cnt = n if cnt is None else cnt + n
        batches += 1
    if sq is None or float(cnt.sum()) <= 0:
        raise RuntimeError(
            "could not estimate the reconstruction variance: the train loader produced no "
            "usable batches."
        )
    var = sq / cnt.clamp_min(1.0)
    # Horizon steps with no observations anywhere get the pooled value rather than 0, which
    # would silently switch the correction off for them.
    pooled = float(sq.sum() / cnt.sum())
    var = torch.where(cnt > 0, var, torch.full_like(var, pooled))
    return var, pooled, batches


class ReconstructionNoiseVAE:
    """The stage-1 decoder, plus the round-trip residual the analytic law never modelled.

    Wraps rather than patches: ``decode_latents_with_vae`` dispatches on ``vae.decode_mu``, so
    every arm keeps the unchanged ``evaluate_regression`` path, the same masking and the same
    CRPS/PIT estimators -- only the decoded draws gain the missing variance component.

    The noise is drawn independently PER DRAW, because it is genuine predictive uncertainty
    rather than a fixed offset: the ensemble then samples the corrected law instead of a
    shifted copy of the old one.
    """

    def __init__(self, vae, std: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        self._vae = vae
        self._std = std          # [H, C]
        self._generator = generator

    def __getattr__(self, name):
        return getattr(self._vae, name)

    def decode_mu(self, mu_est, entity_pad):
        out = self._vae.decode_mu(mu_est, entity_pad)
        std = self._std
        if std.dim() == 2 and std.shape[0] == out.shape[1] and std.shape[-1] == out.shape[-1]:
            std = std.view(1, out.shape[1], 1, out.shape[-1])
        else:  # horizon/channel layout did not match -- fall back to the pooled scalar
            std = std.mean()
        noise = torch.randn(
            out.shape, device=out.device, dtype=out.dtype, generator=self._generator
        )
        return out + std.to(device=out.device, dtype=out.dtype) * noise


def _mean_source_label(mean_source: str, ensemble_size: int) -> str:
    """The mean source as it should appear in a table, with its size baked in.

    ``"ensemble25"`` rather than ``"ensemble"``: the arm's location depends on how many draws
    were averaged, and every bug this campaign chased in the last week was a protocol that was
    not written down next to its number.
    """
    if str(mean_source) == "ensemble":
        return f"ensemble{int(ensemble_size)}"
    return str(mean_source)


def _denoiser_passes(mean_source: str, *, steps: int, ensemble_size: int,
                     with_variance: bool = True, variance_draws: int = 1) -> int:
    """Denoiser forwards per batch for the analytic arm — the cost that is not wall-clock.

    Wall-clock mixes in the decoder and the dataloader, both of which the arms share; this is
    the part that is actually attributable to the method. Note what it says about fix 1d: at
    ``ensemble`` with N = num_samples the analytic arm costs N*steps + 1 against the sampled
    arm's N*steps, so **the A1-vs-A2 speedup is gone by construction**. That is the intended
    outcome (recovery plan §1d) — A1 vs A2 becomes "calibration at matched cost", and the
    method-level speedup claim moves to A1 vs A3, where ``oneshot`` costs 1.
    """
    if str(mean_source) == "oneshot":
        return 1
    draws = 1 if str(mean_source) == "ddim" else max(1, int(ensemble_size))
    return draws * int(steps) + (max(1, int(variance_draws)) if with_variance else 0)


def _calibration_block(metrics: Dict[str, object]) -> Dict[str, object]:
    """Flatten ``evaluate_regression``'s calibration payload into a report arm.

    Kept flat and prefix-free so a Table 3 row reads one dict, and returned empty rather than
    with ``None`` placeholders when calibration was not requested -- an absent key is honest,
    a null one invites a table cell that says "0.0".
    """
    block = (metrics or {}).get("calibration")
    if not isinstance(block, dict):
        return {}
    return {
        "pit_calibration_error": block.get("pit_calibration_error"),
        "coverage": block.get("coverage"),
        "coverage_level": block.get("coverage_level"),
        "mean_interval_width": block.get("mean_interval_width"),
        "reliability": block.get("reliability"),
        "calibration_estimator": block.get("estimator"),
        "calibration_space": block.get("space"),
        "num_pit_values": block.get("num_pit_values"),
    }


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

    def __init__(self, model, cfg, *, mean_source: str, device: torch.device,
                 ensemble_size: int = 1, mean_seed: Optional[int] = None,
                 variance_timestep: Optional[int] = None,
                 variance_draws: int = 1) -> None:
        self._model = model
        self._cfg = cfg
        self._mean_source = str(mean_source)
        self._device = device
        self._ensemble_size = int(ensemble_size)
        self._variance_timestep = variance_timestep
        self._variance_draws = int(variance_draws)
        self._cache_key = None
        self._key_refs: tuple = ()
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        # The law's OWN stream, separate from the draw noise evaluate_regression supplies.
        # Sharing one would make the fitted mean depend on how many draws had been taken
        # before the cache missed, i.e. on the ensemble size -- so changing `--num-samples`
        # would move the point forecast as well as the spread, and the two effects would be
        # inseparable in the table.
        self._mean_generator: Optional[torch.Generator] = None
        if mean_seed is not None:
            self._mean_generator = torch.Generator(device=device)
            self._mean_generator.manual_seed(int(mean_seed))

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
        # The key is identity-based, so the keyed tensors must be kept alive: once a
        # batch's cond/dt are freed, CPython can hand their id()s to the next batch's
        # tensors and a stale law would be scored with no error. self._key_refs pins them.
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
                ensemble_size=self._ensemble_size,
                variance_timestep=self._variance_timestep,
                variance_draws=self._variance_draws,
                generator=self._mean_generator,
            )
            self._cache_key = key
            self._key_refs = (cond_summary, cond_summary_raw, dt)
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
    with_variance: bool = True,
    ensemble_size: int = 1,
    variance_timestep: Optional[int] = None,
    variance_draws: int = 1,
    generator: Optional[torch.Generator] = None,
):
    """(mean, variance) under the requested mean source.

    ``with_variance=False`` reads the mean alone, for a checkpoint built without the
    Theorem-C UQ head (``LapFormer.forward`` raises on ``return_variance=True`` there).
    The returned variance is then ``None`` and the caller must skip every
    law-dependent metric. It also saves the extra t=1 forward in the ``ddim`` path.

    ``ensemble`` averages ``ensemble_size`` reverse-diffusion draws instead of taking one
    (fix 1d). ``ddim`` is that same path at N = 1 and is kept reachable deliberately: the two
    together isolate the location contribution from the spread contribution, which a single
    matched arm cannot.

    Why N = 1 was a defect and not just a choice: ``generate(eta=0)`` is deterministic given
    ``x_T``, but ``x_T`` is random, so one call is ONE DRAW from the predictive distribution --
    not its mean. The sampled arm's point forecast is the mean of ``num_samples`` such draws,
    so at N = 1 the two arms differ in LOCATION before any UQ question is asked, and a CRPS or
    MSE gap between them reads as a method effect. Measured on a live S2 checkpoint: MSE 0.394
    (N=1) against the sampled arm's 0.093 on the same weights -- a 4.2x gap that is entirely
    location. ``CMD_CAMPAIGN_STATE.md`` §1b records the same trap costing ~50 points elsewhere.

    ⚠️ The match is distributional, not draw-for-draw: these draws are generated here rather
    than shared with ``evaluate_regression``'s sampled arm, and the decoder is non-linear, so
    ``mean_j decode(z_j)`` and ``decode(mean_j z_j)`` differ by a Jensen gap. Both arms'
    ``mse`` are reported so the size of that residual is visible rather than assumed.
    """
    timesteps = int(diff_model.scheduler.timesteps)
    B = mu_shape[0]
    if mean_source == "oneshot":
        t = torch.full((B,), timesteps - 1, device=device, dtype=torch.long)
        # A3's mean is a forward at t=T-1 on information-free noise, so an UNSEEDED draw
        # makes the reported arm irreproducible run to run -- a "± std over seeds" would then
        # carry estimator noise that no seed controls. Seeded from the config by default.
        x_t = torch.randn(mu_shape, device=device, generator=generator)
        if not with_variance:
            mean = diff_model(
                x_t, t, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
                dt=dt_model,
            )
            return mean, None
        return diff_model(
            x_t, t, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
            dt=dt_model, return_variance=True,
        )
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    # `ddim` IS the N = 1 case. Sharing one code path rather than two keeps the only
    # difference between the arms the number of draws averaged -- pinned by a test.
    draws = 1 if mean_source == "ddim" else max(1, int(ensemble_size))
    total = None
    for _ in range(draws):
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
        )
        total = sample if total is None else total + sample
    mean = total / float(draws)
    if not with_variance:
        return mean, None
    # 🔴 The timestep this is read at CHANGES THE LAW, and by more than any arm difference in
    # Table 3. `LapFormer` computes Var = einsum(s_modal(cond), energy(theta)) where `theta` is
    # the modal state of the x_t it is handed -- so the spread scales with the modal energy of
    # that input. Only `p0`/`q` come from the conditioning.
    #
    # Measured, seed-1 S2, val, matched ensemble25 mean, 322 560 latent elements
    # (realised residual variance 0.444):
    #     t=1    E[var] 0.190   ratio 2.34   coverage@0.9 0.690   <- the historical default
    #     t=100  E[var] 0.196   ratio 2.26   coverage@0.9 0.698
    #     t=400  E[var] 0.237   ratio 1.88   coverage@0.9 0.747
    #     t=T-1  E[var] 0.681   ratio 0.65   coverage@0.9 0.931   <- where `oneshot` reads
    #
    # At t=1 the model is handed a nearly-clean x_t built from its OWN predicted mean, so it
    # reports the residual uncertainty about x0 GIVEN that x_t -- a true statement about the
    # denoising problem and the wrong quantity for Table 3, which reports uncertainty about y
    # given only the history. At t=T-1 the state carries no information about the target, which
    # is the regime the `oneshot` arm reads in and the one the predictive law is defined for.
    #
    # The default is now t=T-1 (was a hardcoded 1). This is a PARITY fix, not a modelling
    # preference: A2 read at 1 while A3 read at T-1, so the A2-vs-A3 row reported a ~3.6x
    # spread difference that was the read protocol, not the arms. T-1 is the only value that
    # makes them agree without redefining A3, whose mean IS the t=T-1 forward. Override with
    # --variance-timestep; the value used is recorded in the report either way.
    t_read = int(timesteps - 1 if variance_timestep is None else variance_timestep)
    t_read = max(0, min(t_read, timesteps - 1))
    t_vec = torch.full((B,), t_read, device=device, dtype=torch.long)
    # At T-1 the `energy` term is the modal decomposition of an essentially pure-noise x_read,
    # so a single draw makes the PER-ELEMENT variance ride that draw -- measured 14.5 % median
    # relative spread between two seeds, against a 2 % aggregate. Seeding makes that
    # reproducible, not right: PIT and coverage are per-element, so the noise lands directly in
    # A2's reported calibration. Averaging marginalises the draw, which is what "the law given
    # only the conditioning" means. `s_modal` is draw-independent and the readout is linear in
    # `energy`, so averaging the variance is exactly averaging the energy -- the heteroscedasticity
    # that lives in `s_modal` is untouched. (Refinement proposed by Claude A, 2026-08-08.)
    n_draws = max(1, int(variance_draws))
    total_var = None
    for _ in range(n_draws):
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        x_read, _ = diff_model.scheduler.q_sample(mean, t_vec, noise)
        _, v = diff_model(
            x_read, t_vec, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
            dt=dt_model, return_variance=True,
        )
        total_var = v if total_var is None else total_var + v
    return mean, total_var / float(n_draws)


def main() -> None:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    # Stage-1/2 architecture flags first: the artifact names derive from them.
    if getattr(args, "vae_entity_encode", False):
        cfg.VAE_ENTITY_ENCODE = True
    if getattr(args, "sum_decode_mode", None):
        cfg.SUM_DECODE_MODE = str(args.sum_decode_mode)
    refresh_artifact_paths(cfg)
    if getattr(args, "vae_ckpt", None):
        cfg.VAE_CKPT = str(args.vae_ckpt)
    if getattr(args, "sum_ckpt", None):
        cfg.SUM_CKPT = str(args.sum_ckpt)
    # Sampling protocol for the REPORTED numbers. _sampling_kwargs reads TEST_* first.
    if getattr(args, "guidance", None) is not None:
        cfg.TEST_GUIDANCE = float(args.guidance)
    if getattr(args, "guidance_power", None) is not None:
        cfg.TEST_GUIDANCE_POWER = float(args.guidance_power)
    if getattr(args, "gen_steps", None) is not None:
        cfg.TEST_STEPS = int(args.gen_steps)
    if getattr(args, "num_samples", None) is not None:
        # Both keys: _sampling_kwargs prefers TEST_NUM_SAMPLES, so writing only
        # NUM_EVAL_SAMPLES would let a config that sets TEST_NUM_SAMPLES silently win while
        # the report claimed the requested size. Applied here rather than in the data-space
        # block because the mean ensemble size is derived from it, and the latent pass needs
        # that before the data-space block runs.
        cfg.NUM_EVAL_SAMPLES = int(args.num_samples)
        cfg.TEST_NUM_SAMPLES = int(args.num_samples)

    # One resolved protocol for the whole run: the latent pass, the analytic arm's mean and
    # the sampled arm must not each re-derive their own.
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    eval_ensemble = int(sampling["num_samples"])
    mean_ensemble_size = int(args.mean_ensemble_size or eval_ensemble)
    if mean_ensemble_size < 1:
        raise ValueError(f"--mean-ensemble-size must be >= 1, got {mean_ensemble_size}")
    if args.mean_source == "ensemble" and mean_ensemble_size != eval_ensemble:
        print(
            f"[uq-eval] ⚠️  mean ensemble {mean_ensemble_size} != eval ensemble "
            f"{eval_ensemble}: the analytic arm's LOCATION no longer matches the sampled "
            f"arm's, so an A1-vs-A2 gap is partly a mean difference again (fix 1d)."
        )
    mean_label = _mean_source_label(args.mean_source, mean_ensemble_size)

    device = set_torch(
        seed=int(getattr(cfg, "SEED", 42)),
        deterministic=False,
        allow_tf32=bool(getattr(cfg, "ALLOW_TF32", False)),
    )

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    train_dl, val_dl, test_dl, _ = loaders
    loader = {"train": train_dl, "val": val_dl, "test": test_dl}[args.split]
    if args.split == "train":
        print("[uq-eval] ⚠️  --split train: DIAGNOSTIC ONLY (the model fit these windows). "
              "Use it to compare against val, never as a reported number.")
    diff_model, vae, summarizer, mu_mean, mu_std = stack

    # A checkpoint without the UQ head carries no predictive law, but its MEAN is still
    # readable -- and the mean-only latent metrics (latent_rmse/latent_corr against
    # baseline_rmse_predict_mean) are exactly the Phase-1 signal gate of
    # w_docs/CMD_UQ_RECOVERY_PLAN.md, which is read on a plain-MSE S1 arm that by
    # construction has no UQ head. Those metrics live only in this tool, so refusing
    # the checkpoint outright made the gate unmeasurable. Everything that needs the
    # variance is dropped rather than approximated, and the data-space arms -- which
    # propagate the law through the decoder -- are still refused.
    # Resolved once the scheduler is known, and recorded: the analytic law's spread scales with
    # the modal energy of the x_t it is read at, so this is part of the law's definition, not a
    # detail of the read. `oneshot` takes mean and variance from one forward at T-1 and is
    # unaffected by the flag.
    _timesteps = int(diff_model.scheduler.timesteps)
    variance_timestep_used = (
        _timesteps - 1 if args.mean_source == "oneshot" or args.variance_timestep is None
        else max(0, min(int(args.variance_timestep), _timesteps - 1))
    )
    if variance_timestep_used != _timesteps - 1:
        print(
            f"[uq-eval] ⚠️  reading the analytic variance at t={variance_timestep_used}, not "
            f"t={_timesteps - 1}. The `oneshot` arm reads at T-1, so an A2-vs-A3 spread "
            f"comparison across this setting is confounded by the read protocol."
        )

    has_uq_head = bool(getattr(diff_model.model, "chirp_uq_head", False))
    if not has_uq_head and not args.latent_only:
        raise ValueError(
            "Checkpoint was not trained with CHIRP_UQ_HEAD=True, so the analytic "
            "Gaussian law is unavailable and the data-space analytic arm cannot be "
            "computed. Pass --latent-only for the mean-only latent metrics "
            "(latent_rmse, latent_mse, latent_corr, baseline_rmse_predict_mean, "
            "latent_target_std), or use llapdiff-checkpoint-eval for plain sampled "
            "forecast metrics."
        )
    if not has_uq_head:
        print(
            "[uq-eval] checkpoint has no UQ head: reporting mean-only latent metrics; "
            "latent_gaussian_nll / pit_calibration_error / reliability / "
            "mean_predicted_std are omitted."
        )

    # ---- Correctness gate (e): an analytic interval must come from an NLL-trained head ----
    # q_k and p0_k do not enter the mean, so under DIFF_LOSS_MODE="mse" they receive exactly
    # zero gradient and the returned "law" is CHIRP_UQ_INIT_VAR wearing a checkpoint. This
    # fails silently -- the loss curve is fine and CRPS barely moves -- so it is refused here
    # rather than reported. Two independent detectors, because checkpoints written before the
    # objective was persisted carry no record of it.
    uq_trained_report: Dict[str, object] = {"gate": "analytic_law_requires_nll_training"}
    if has_uq_head:
        recorded_mode = tv.diff_loss_mode_from_checkpoint(
            torch.load(args.checkpoint, map_location="cpu")
        )
        untrained = diff_model.model.chirp_field.uq_head_untrained_signature()
        uq_trained_report.update(
            recorded_diff_loss_mode=recorded_mode,
            untrained_head_signature=untrained,
        )
        offence = None
        if recorded_mode is not None and recorded_mode != "gaussian_nll":
            offence = (
                f"the checkpoint records DIFF_LOSS_MODE='{recorded_mode}', so the UQ head "
                f"never received a gradient"
            )
        elif untrained is not None:
            offence = (
                "the UQ head is bit-identical to its initialization "
                f"({untrained}), so it was never trained"
            )
        if offence is not None and not args.allow_untrained_uq_head:
            raise ValueError(
                f"Refusing to report an analytic law: {offence}. Every number it produced "
                f"would be a restatement of CHIRP_UQ_INIT_VAR, not a measurement. Train with "
                f"DIFF_LOSS_MODE='gaussian_nll' (warm-started from the MSE mean checkpoint via "
                f"DIFF_INIT_CKPT), or pass --allow-untrained-uq-head if you are deliberately "
                f"measuring the untrained head and will label it as such."
            )
        if offence is not None:
            print(f"[uq-eval] ⚠️  --allow-untrained-uq-head: {offence}. Results are NOT a law.")
            uq_trained_report["override"] = offence

    # `oneshot` returns the network's RAW output, which is the x0 estimate only under
    # predict_type='x0'; under 'v'/'eps' it is a different quantity entirely and
    # latent_rmse would silently score it against x0 targets. This could not fire while
    # the tool was UQ-only (chirp_uq_head forces x0), so it guards the path just opened.
    # `ddim` is safe either way -- generate() converts through scheduler.to_x0.
    checkpoint_predict_type = str(getattr(diff_model, "predict_type", "x0"))
    if args.mean_source == "oneshot" and checkpoint_predict_type != "x0":
        raise ValueError(
            f"--mean-source oneshot reads the raw network output, which is an x0 "
            f"estimate only for predict_type='x0'; this checkpoint is "
            f"'{checkpoint_predict_type}', so the latent metrics would compare "
            f"different quantities. Use --mean-source ddim (converts to x0 for any "
            f"parameterization), or evaluate an x0 checkpoint."
        )

    # _load_stack (shared with u1/t1/t4/eval_sampling) deliberately loads payload["model"],
    # i.e. RAW weights. Apply the EMA shadow here, in this tool only, reusing the
    # alias-proof named_parameters() transplant -- LapFormer aliases analysis.* as
    # synthesis.encoder.*, so a key-wise state_dict merge would leave the alias raw.
    if args.weights == "ema":
        replaced = ce._apply_ema_weights(diff_model, torch.load(args.checkpoint, map_location="cpu"))
        print(f"[uq-eval] applied the EMA shadow to {replaced} parameters")

    ys: List[torch.Tensor] = []
    means: List[torch.Tensor] = []
    variances: List[torch.Tensor] = []
    batches = 0
    # Seeded so the latent read is reproducible: `oneshot` draws an information-free x_T and
    # `ensemble`/`ddim` draw one x_T per averaged sample, none of which any seed controlled.
    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(int(getattr(cfg, "SEED", 42)))
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
            with_variance=has_uq_head,
            ensemble_size=mean_ensemble_size,
            variance_timestep=args.variance_timestep,
            variance_draws=int(args.variance_draws),
            generator=latent_generator,
        )

        mask = torch.as_tensor(obs_any, device=device, dtype=torch.bool)
        while mask.dim() < mu_norm.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(mu_norm)
        ys.append(mu_norm[mask].detach().cpu())
        means.append(mean[mask].detach().cpu())
        if variance is not None:
            variances.append(variance[mask].detach().cpu())

        batches += 1
        if args.max_batches is not None and batches >= int(args.max_batches):
            break

    if not ys:
        raise RuntimeError("No observed latent targets found on the selected split.")
    y = torch.cat(ys)
    mean = torch.cat(means)
    var = torch.cat(variances) if variances else None

    report: Dict[str, object] = {
        "dataset_key": args.dataset_key,
        "pred": int(args.pred),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mean_source": mean_label,
        "weights": args.weights,
        # False => this is a mean-only read and every law-dependent key below is absent.
        "has_uq_head": has_uq_head,
        "num_elements": int(y.numel()),
        # Trivial-predictor baselines, so latent_rmse is interpretable without assuming
        # the latents are unit-scale (they are not: std ~1.46 on physionet h=12).
        "latent_target_std": float(y.std().item()),
        "baseline_rmse_predict_zero": float(y.pow(2).mean().sqrt().item()),
        "baseline_rmse_predict_mean": float((y - y.mean()).pow(2).mean().sqrt().item()),
        "latent_rmse": float((mean - y).pow(2).mean().sqrt().item()),
        # Squared forms, reported alongside the RMSEs because the two quantities the
        # plan reads them for are squared: the Phase-1 gate's "latent x0 MSE", and the
        # Phase-3 residual scale that CHIRP_UQ_INIT_VAR must be initialised near (a
        # VARIANCE -- confusing it with a std is the failure that phase exists to fix).
        "latent_mse": float((mean - y).pow(2).mean().item()),
        "baseline_mse_predict_mean": float((y - y.mean()).pow(2).mean().item()),
        "latent_corr": (
            float(torch.corrcoef(torch.stack([mean.reshape(-1), y.reshape(-1)]))[0, 1].item())
            if y.numel() > 1 else None
        ),
    }
    if var is not None:
        u = gaussian_pit(y, mean, var)
        report.update({
            "latent_gaussian_nll": gaussian_nll(y, mean, var),
            "pit_calibration_error": pit_calibration_error(u, num_bins=int(args.num_bins)),
            "reliability": reliability_curve(u),
            "mean_predicted_std": float(var.clamp_min(1e-6).sqrt().mean().item()),
            # ⚠️ `mean_predicted_std` is E[sigma]; `latent_rmse` is sqrt(E[e^2]). Comparing
            # them is off by Jensen whenever the law is heteroscedastic -- E[sigma] <=
            # sqrt(E[sigma^2]) -- and the bias always points toward "over-confident", which is
            # the verdict this pair is used to reach. These two are the matched-units form:
            # both are variances, so their ratio is the sharpness factor with no Jensen term.
            # ratio > 1 == over-confident (predicted variance smaller than realised error).
            "mean_predicted_var": float(var.clamp_min(1e-6).mean().item()),
            "variance_calibration_ratio": float(
                (mean - y).pow(2).mean().item() / var.clamp_min(1e-6).mean().item()
            ),
        })

    recon_var_scalar = None
    if not args.latent_only:
        # Data-space comparison: the analytic law propagated through the decoder
        # (Gaussian latent draws -> decode) vs the sampled-diffusion baseline, scored
        # by the SAME evaluate_regression code with the SAME ensemble size and seed.
        num_samples = eval_ensemble          # resolved once, at the top of main()
        common = dict(
            device=device, mu_mean=mu_mean, mu_std=mu_std, config=cfg, ema=None,
            self_cond=bool(getattr(cfg, "SELF_COND", False)),
            disable_conditioning=False, verbose=False,
            generator_seed=int(getattr(cfg, "SEED", 42)),
            # Table 3's PIT-ECE / coverage / interval-width columns. Requested for BOTH arms
            # from the same `common`, so they are scored by one estimator in one space on the
            # same mask -- the analytic arm's latent closed-form numbers stay available above
            # as the exact-Gaussian cross-check, not as A2's entry in a column A1 cannot fill.
            calibration=True,
            calibration_level=float(args.coverage_level),
        )

        # The missing variance component (see estimate_reconstruction_variance). Estimated
        # ONCE, on train, and applied identically to both arms so the comparison stays matched.
        decode_vae = vae
        recon_var_scalar = None
        if args.add_reconstruction_variance:
            rv, recon_var_scalar, rv_batches = estimate_reconstruction_variance(
                vae, train_dl, device, mu_mean, mu_std,
                max_batches=int(args.recon_var_batches),
            )
            print(f"[uq-eval] stage-1 reconstruction variance from {rv_batches} TRAIN batches: "
                  f"pooled {recon_var_scalar:.5f} (std {recon_var_scalar ** 0.5:.5f}), "
                  f"per-horizon range [{float(rv.min()):.5f}, {float(rv.max()):.5f}]")
            rv_gen = torch.Generator(device=device)
            rv_gen.manual_seed(int(getattr(cfg, "SEED", 42)) + 51_000_000)
            decode_vae = ReconstructionNoiseVAE(vae, rv.sqrt(), generator=rv_gen)

        analytic_model = AnalyticLawSampler(
            diff_model, cfg, mean_source=str(args.mean_source), device=device,
            ensemble_size=mean_ensemble_size,
            mean_seed=int(getattr(cfg, "SEED", 42)),
            variance_timestep=args.variance_timestep,
            variance_draws=int(args.variance_draws),
        )
        start = time.perf_counter()
        analytic = tv.evaluate_regression(
            analytic_model, decode_vae, summarizer, loader, **common, **sampling
        )
        analytic_wall = time.perf_counter() - start
        report["data_space_analytic"] = {
            "crps": analytic.get("crps"),
            "mae": analytic.get("mae"),
            "mse": analytic.get("mse"),
            "num_samples": num_samples,
            "mean_source": mean_label,
            "mean_ensemble_size": (
                mean_ensemble_size if str(args.mean_source) == "ensemble" else 1
            ),
            "wall_seconds": analytic_wall,
            # The law is read ONCE per batch and cached; every further draw is a Gaussian
            # sample plus a decoder pass, neither of which touches the denoiser. See
            # _denoiser_passes for what this says about the A1-vs-A2 speedup after fix 1d.
            "denoiser_passes_per_batch": _denoiser_passes(
                str(args.mean_source), steps=int(sampling["steps"]),
                ensemble_size=mean_ensemble_size,
                variance_draws=int(args.variance_draws),
            ),
            "variance_timestep": variance_timestep_used,
            "variance_draws": (1 if args.mean_source == "oneshot" else int(args.variance_draws)),
            **_calibration_block(analytic),
        }

        if not args.skip_sampled:
            start = time.perf_counter()
            sampled = tv.evaluate_regression(
                diff_model, decode_vae, summarizer, loader, **common, **sampling
            )
            sampled_wall = time.perf_counter() - start
            report["data_space_sampled"] = {
                "crps": sampled.get("crps"),
                "mae": sampled.get("mae"),
                "mse": sampled.get("mse"),
                "num_samples": num_samples,
                "ddim_steps": int(sampling["steps"]),
                # This arm's point forecast IS the mean of its draws, so it carries the same
                # label the analytic arm computes. Stated here rather than reconstructed by
                # the driver, so a standalone report JSON is self-describing -- and so two
                # identical labels in a table are the visible proof that fix 1d is in force.
                "mean_source": f"ensemble{num_samples}",
                "wall_seconds": sampled_wall,
                # One full reverse trajectory per draw: this is the cost the analytic arm
                # exists to avoid, and the ratio against the analytic count is the method-level
                # speedup that survives matching the means.
                "denoiser_passes_per_batch": num_samples * int(sampling["steps"]),
                **_calibration_block(sampled),
            }
            report["analytic_speedup_x"] = (
                sampled_wall / analytic_wall if analytic_wall > 0 else None
            )
            # The wall ratio flatters the analytic arm: both arms pay the same decoder MC and
            # dataloader, so the shared cost dilutes the difference in BOTH directions
            # depending on the batch. This is the attributable one.
            analytic_passes = report["data_space_analytic"]["denoiser_passes_per_batch"]
            report["analytic_denoiser_pass_ratio_x"] = (
                report["data_space_sampled"]["denoiser_passes_per_batch"] / analytic_passes
                if analytic_passes else None
            )
            if str(args.mean_source) == "ensemble":
                report["speedup_scope"] = (
                    "A1-vs-A2 is calibration at MATCHED COST: with the means matched (fix 1d) "
                    "the analytic arm pays num_samples*steps + 1 denoiser passes against the "
                    "sampled arm's num_samples*steps, so there is no A1/A2 speedup to claim. "
                    "The method-level speedup belongs to A1-vs-A3 (oneshot, 1 pass)."
                )

    # Provenance: a reported number must carry the stack and sampling protocol that produced
    # it. Without this, a legacy-VAE eval and an entity-encoded one are indistinguishable in
    # the JSON, which is how D1 stayed invisible.
    # The gate that licenses reading a law off this checkpoint at all (correctness gate (e)).
    report["uq_head_training"] = uq_trained_report
    report["resolved"] = {
        "vae_ckpt": str(cfg.VAE_CKPT),
        "sum_ckpt": str(cfg.SUM_CKPT),
        "vae_entity_encode": bool(getattr(cfg, "VAE_ENTITY_ENCODE", False)),
        "sum_decode_mode": str(getattr(cfg, "SUM_DECODE_MODE", "mean")),
        # The mean source sits HERE, beside guidance, because it is a protocol choice that
        # moves the arm's location as much as guidance moves its spread -- and an arm named
        # only "A2" carries neither.
        "mean_source": mean_label,
        "mean_ensemble_size": (
            mean_ensemble_size if str(args.mean_source) == "ensemble" else 1
        ),
        "variance_timestep": variance_timestep_used,
        "variance_draws": (1 if args.mean_source == "oneshot" else int(args.variance_draws)),
        "reconstruction_variance": (recon_var_scalar if not args.latent_only else None),
        "reconstruction_variance_source": ("train" if args.add_reconstruction_variance else None),
        "sampling": {k: v for k, v in tv._sampling_kwargs(cfg, prefix="TEST").items()
                     if k in ("steps", "num_samples", "guidance_strength", "guidance_power", "eta")},
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
