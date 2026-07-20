"""Ground-truth chirp benchmark (H2): lti-vs-chirp arms on synthetic time-varying poles.

For each synthetic task with known instantaneous pole functions (linear/quadratic
frequency chirps, damping ramps, growth-then-decay, and the piecewise freq shift as a
regime switch), this tool trains an ``lti`` and a ``chirp`` denoiser on the same cache
(shared VAE + summarizer per seed), evaluates forecast CRPS/MAE/MSE on the test split,
and — for the chirp arm — overlays the recovered pole trajectories rho_k(t), omega_k(t)
against the generator's ground truth (the identifiability figure).

Caches are generated with ``shared_poles=True`` so a joint date row has a single
well-defined ground-truth pole function.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from llapdiffusion.configs.config_utils import clone_config, make_jsonable
from llapdiffusion.logging_utils import apply_verbosity
from llapdiffusion.datasets.synthetic_regime_dataset import (
    CHIRP_TASKS,
    _SMOOTH_RAMP_TASKS,
    SyntheticRegimeCacheConfig,
    load_ground_truth_poles,
    prepare_synthetic_regime_cache,
    run_experiment as synthetic_run_experiment,
)
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools.llapdiff_checkpoint_eval import _load_stack
from llapdiffusion.tools.run_synthetic_regime_shift import (
    _stats,
    _train_or_reuse_stack,
    _write_rows,
)
from llapdiffusion.trainers import train_val_llapdiff as tv
from llapdiffusion.viz.plot_llapdiff_poles import modal_contributions


DEFAULT_WORK_ROOT = Path.cwd()
BENCHMARK_TASKS = CHIRP_TASKS + ("synthetic_freq_shift",)  # freq shift = regime switch
ARMS = ("lti", "chirp")
DATA_SEED = 20260501


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground-truth chirp benchmark: lti vs chirp on synthetic time-varying poles."
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(BENCHMARK_TASKS), default=list(CHIRP_TASKS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=48)
    # The purged ratio split needs the val band to hold a full target interval:
    # roughly val_ratio * (L - window - horizon + 1) > horizon (see _validate_geometry).
    # 288 (the regime tool's default) is structurally too short for 96/48 at 10% val.
    parser.add_argument("--series-length", type=int, default=768)
    # Renewal-gap sampling (the plan's H2 premise: irregular, tunable Var(Delta)).
    # Default gamma with shape 4 (Var = mean^2/4); use "regular" for the dense grid.
    # Keep gap-mean at 1.0 so the run's horizon (in samples) matches the chirp time
    # scale L = PRED in native units; if you change it, set CHIRP_TIME_SCALE ~
    # PRED * gap_mean in config.py.
    parser.add_argument("--gap-distribution", choices=("regular", "gamma"), default="gamma")
    parser.add_argument("--gap-mean", type=float, default=1.0)
    parser.add_argument("--gap-shape", type=float, default=4.0,
                        help="Gamma shape k: Var(Delta) = gap_mean^2 / k. Sweep for Var(Delta) regimes.")
    # Within-window pole excursion (fix-plan P5): the legacy series-long ramp gives a
    # ~6% sweep inside one horizon, so LTI is NOT structurally penalized within a
    # window. A triangle re-sweep of ~(window+horizon) steps puts the full excursion
    # inside every window (including the tail test windows the purged split uses).
    parser.add_argument(
        "--sweep-period", type=float, default=None,
        help="Triangle re-sweep period (native steps) for the smooth-ramp tasks "
             "(recommended ~window+horizon). Piecewise change-point tasks ignore it. "
             "Default None keeps the legacy slow series-long ramp.",
    )
    parser.add_argument("--change-point", type=int, default=None, help="Defaults to 3/4 of the series.")
    parser.add_argument("--num-entities", type=int, default=64)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "synthetic_data" / "chirp_benchmark"),
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "synthetic_artifacts" / "chirp_benchmark"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "results" / "chirp_benchmark"),
    )
    parser.add_argument("--num-recovery-windows", type=int, default=4)
    parser.add_argument("--recovery-top-modes", type=int, default=4)
    parser.add_argument(
        "--recovery-share-threshold", type=float, default=0.5,
        help="Minimum output-energy share the selected modes must explain; below it "
             "the top-N is escalated (x2 up to 16) and the window is flagged "
             "selection-invalid if still short (figure watermarked, not evidence).",
    )
    parser.add_argument("--recompute-artifacts", action="store_true")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short training/eval schedule for end-to-end smoke testing.",
    )
    return parser.parse_args()


def _resolve_change_point(args: argparse.Namespace) -> int:
    if args.change_point is not None:
        return int(args.change_point)
    return int(round(0.75 * int(args.series_length)))


def _validate_geometry(args: argparse.Namespace, *, val_ratio: float = 0.1) -> None:
    """The global purged split keeps a val window only if its entire target interval
    fits inside the val band, so the band must span more than one horizon."""
    L, K, H = int(args.series_length), int(args.window), int(args.horizon)
    unique_starts = L - (K + H) + 1
    val_band = int(unique_starts * float(val_ratio))
    if val_band <= H:
        raise ValueError(
            f"series_length={L} is too short for window={K}, horizon={H}: the val band "
            f"spans ~{val_band} start times but a val window needs {H}+ inside it. "
            f"Increase --series-length above ~{int((H + 1) / float(val_ratio)) + K + H}."
        )


def _gap_tag(args: argparse.Namespace) -> str:
    if str(args.gap_distribution) == "regular":
        return "gaps-regular"
    return f"gaps-gamma-m{float(args.gap_mean):g}-k{float(args.gap_shape):g}"


def _task_sweep_period(task: str, args: argparse.Namespace) -> Optional[float]:
    """--sweep-period applies only to the smooth-ramp tasks; the piecewise
    change-point tasks keep their design (the generator would reject it)."""
    period = getattr(args, "sweep_period", None)
    if period is None or task not in _SMOOTH_RAMP_TASKS:
        return None
    return float(period)


def _cache_dir(task: str, args: argparse.Namespace) -> Path:
    tag = (
        f"len-{int(args.series_length)}_cp-{_resolve_change_point(args)}_"
        f"entities-{int(args.num_entities)}_{_gap_tag(args)}"
    )
    period = _task_sweep_period(task, args)
    if period is not None:
        tag += f"_sweep-{period:g}"
    return (Path(args.data_root) / task / tag).resolve()


def _prepare_cache(task: str, args: argparse.Namespace) -> Mapping[str, object]:
    cfg = SyntheticRegimeCacheConfig(
        task=task,
        window=int(args.window),
        horizon=int(args.horizon),
        data_dir=str(_cache_dir(task, args)),
        num_entities=int(args.num_entities),
        series_length=int(args.series_length),
        change_point=_resolve_change_point(args),
        seed=DATA_SEED,
        shared_poles=True,
        gap_distribution=str(args.gap_distribution),
        gap_mean=float(args.gap_mean),
        gap_shape=float(args.gap_shape),
        sweep_period=_task_sweep_period(task, args),
        overwrite=bool(args.overwrite_data),
    )
    return prepare_synthetic_regime_cache(cfg)


def _configure(task: str, arm: str, seed: int, args: argparse.Namespace) -> SimpleNamespace:
    """Mirror run_synthetic_regime_shift._configure; VAE/summarizer are shared per
    (task, seed) so both arms condition on identical frozen upstream artifacts."""
    cfg = clone_config()
    base = (Path(args.artifact_root) / task / f"seed-{int(seed)}").resolve()
    cfg.DATASET_KEY = task
    cfg.MKT = task
    cfg.SEED = int(seed)
    cfg.DATA_DIR = str(_cache_dir(task, args))
    cfg.WINDOW = int(args.window)
    cfg.PRED = int(args.horizon)
    cfg.SUM_CONTEXT_LEN_FIXED = int(args.window)
    cfg.SUM_CONTEXT_LEN = int(args.window)
    cfg.COVERAGE = 0.0
    cfg.date_batching = True
    cfg.DATES_PER_BATCH = 16
    cfg.BATCH_SIZE = 16
    cfg.VAE_LATENT_CHANNELS = 24
    cfg.VAE_ENTITY_CONDITION = True
    cfg.VAE_NUM_ENTITIES = None
    cfg.PREDICT_TYPE = "v"
    cfg.LOSS_WEIGHT_SCHEME = "weighted_min_snr"
    cfg.MINSNR_GAMMA = 5.0
    cfg.BASE_LR = 1.5e-4
    cfg.LR_SCHEDULE = "warmup_constant"
    cfg.PRIMARY_EVAL_METRIC = "val_diag_mse_raw"
    cfg.TARGET_MASK_AUX_P = 0.0
    cfg.TARGET_MASK_AUX_START_EPOCH = 0
    cfg.TIMESTEPS = 1000
    cfg.SCHEDULE = "cosine"
    cfg.MODEL_WIDTH = 256
    cfg.NUM_LAYERS = 5
    cfg.NUM_HEADS = 4
    cfg.LAPLACE_K = 256
    cfg.EPOCHS = 80
    cfg.EARLY_STOP = 8
    cfg.EARLY_STOP_MIN_EPOCHS = 20
    cfg.EVAL_EVERY = 1
    cfg.VAL_DIAG_EVERY = 1
    cfg.DOWNSTREAM_EVAL_EVERY = 10
    cfg.IRREG_CHECK_EVERY = 0
    cfg.EMA_COMPARE_EVERY = 0
    cfg.NUM_EVAL_SAMPLES = 10
    cfg.EVAL_STEPS = 32
    cfg.TEST_STEPS = 64
    cfg.GEN_STEPS = 64
    cfg.GEN_ETA = 0.0
    cfg.DIFF_AMP = False

    # Arm selection (chirp keeps CHIRP_* base defaults; time scale resolves to PRED).
    cfg.DENOISER_MODAL_TYPE = str(arm)
    cfg.DENOISER_OUTPUT_HEAD = "auto"

    if args.smoke:
        cfg.EPOCHS = 1
        cfg.EARLY_STOP = 1
        cfg.EARLY_STOP_MIN_EPOCHS = 0
        cfg.SUM_EPOCHS = 1
        cfg.SUM_PATIENCE = 1
        cfg.VAE_WARMUP_EPOCHS = 0
        cfg.VAE_KL_ANNEAL_EPOCHS = 1
        cfg.VAE_MIN_EPOCHS = 1
        cfg.VAE_MAX_PATIENCE = 1
        cfg.NUM_EVAL_SAMPLES = 2
        cfg.EVAL_STEPS = 4
        cfg.TEST_STEPS = 4
        cfg.GEN_STEPS = 4
        cfg.DOWNSTREAM_EVAL_EVERY = 1

    cfg.VAE_DIR = str(base / "vae")
    cfg.SUM_DIR = str(base / "summarizer")
    cfg.CKPT_DIR = str(base / arm / "checkpoints")
    cfg.OUT_DIR = str(base / arm / "output")
    cfg.POLE_PLOT_DIR = str(base / arm / "output" / "pole_plots")
    cfg.VAE_CKPT = str(
        Path(cfg.VAE_DIR) / f"pred-{cfg.PRED}_ch-{cfg.VAE_LATENT_CHANNELS}_entity_elbo.pt"
    )
    cfg.SUM_CKPT = str(Path(cfg.SUM_DIR) / f"{cfg.PRED}-{cfg.VAE_LATENT_CHANNELS}-summarizer.pt")
    cfg.TARGET_COL = None
    cfg.TARGET_COLS = None
    apply_verbosity(cfg, verbose=bool(args.verbose), debug=bool(args.debug))
    return cfg


def _build_loaders(cfg: SimpleNamespace):
    batch_size = int(getattr(cfg, "BATCH_SIZE", 16))
    return synthetic_run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=True,
        dates_per_batch=batch_size,
        K=int(cfg.WINDOW),
        H=int(cfg.PRED),
        coverage=0.0,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        batch_size=batch_size,
        norm="train_only",
        per_asset=True,
        split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=True,
        shuffle_train=False,
    )


def _select_top_modes(
    share: np.ndarray, top_n: int, threshold: float, cap: int = 16
) -> Tuple[np.ndarray, float, bool, int]:
    """Top-contribution mode selection with escalation (fix-plan P3/P9).

    Picks the ``top_n`` modes with the largest output-energy share; if together
    they explain less than ``threshold`` of the output, doubles ``top_n`` (up to
    ``cap``) before giving up. Returns (indices, selected_share, valid, n_used).
    """
    k_total = int(share.shape[-1])
    order = np.argsort(share)[::-1]
    n = max(1, min(int(top_n), k_total))
    while True:
        sel_share = float(share[order[:n]].sum())
        if sel_share >= float(threshold) or n >= min(int(cap), k_total):
            break
        n = min(n * 2, int(cap), k_total)
    return order[:n].copy(), sel_share, sel_share >= float(threshold), n


def _stratified_pick(
    candidates: Sequence[Tuple[int, int, int, int]], n: int
) -> List[Tuple[int, int, int, int]]:
    """Spread ``n`` recovery windows evenly across the test span by window start
    (fix-plan P6). Candidates are (batch_idx, row, asset_id, window_start)."""
    cands = sorted(candidates, key=lambda c: (c[3], c[2], c[0], c[1]))
    if len(cands) <= int(n):
        return list(cands)
    idx = np.unique(np.round(np.linspace(0, len(cands) - 1, int(n))).astype(int))
    return [cands[i] for i in idx]


def _generate_kwargs(cfg: SimpleNamespace) -> Dict[str, object]:
    """The evaluation sampling settings, restricted to ``generate()``'s surface."""
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    keys = (
        "steps", "guidance_strength", "guidance_power", "eta",
        "dynamic_thresh_p", "dynamic_thresh_max", "rho",
    )
    return {k: sampling[k] for k in keys}


@torch.no_grad()
def _recover_pole_trajectories(
    cfg: SimpleNamespace,
    checkpoint: str,
    loaders,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Residue-weighted pole recovery from the ACTUAL generation, for either arm.

    A ``modal_capture`` hook on ``generate()`` records residues theta and pole
    trajectories at the final denoising step of the evaluated forecast; modes are
    ranked by their output contribution E_k = mean_t e^{-2 rho_bar_k(t)} ||theta_k||^2
    (never by coefficient variation — that criterion anti-selected unconstrained
    junk modes), and the primary recovered curve is the E_k-weighted effective
    trajectory over ALL modes, the identifiable object when many modes share one
    signal. Windows are stratified across the test span.

    Returns (metrics, figure windows). The lti arm goes through the identical
    path with constant poles, giving the structural-failure overlay a number.
    """
    train_dl, _, test_dl, _ = loaders
    diff_model, _, summarizer, mu_mean, _ = _load_stack(cfg, Path(checkpoint), device, train_dl)
    truth = load_ground_truth_poles(cfg.DATA_DIR)
    window = int(cfg.WINDOW)
    horizon = int(cfg.PRED)
    latent_dim = int(mu_mean.shape[-1])
    sampling = _generate_kwargs(cfg)
    threshold = float(getattr(args, "recovery_share_threshold", 0.5))

    # Pass 1 (meta only): enumerate candidate windows, then stratify by start.
    candidates: List[Tuple[int, int, int, int]] = []
    for b_idx, (xb, yb, meta) in enumerate(test_dl):
        _, _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        mask_np = mask_bn.detach().cpu().numpy().astype(bool)
        asset_ids = meta["cache_asset_ids"].cpu().numpy()
        starts = meta["cache_window_starts"].cpu().numpy()
        for row in range(mask_np.shape[0]):
            valid = np.nonzero(mask_np[row])[0]
            if valid.size == 0:
                continue
            n0 = int(valid[0])
            candidates.append((b_idx, row, int(asset_ids[row, n0]), int(starts[row, n0])))
    if not candidates:
        raise RuntimeError("No valid test windows found for pole recovery.")
    chosen: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for cand in _stratified_pick(candidates, int(args.num_recovery_windows)):
        chosen.setdefault(cand[0], []).append(cand)

    win_metrics: List[Dict[str, object]] = []
    fig_windows: List[Dict[str, object]] = []
    for b_idx, (xb, yb, meta) in enumerate(test_dl):
        if b_idx not in chosen:
            continue
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=meta.get("delta_t"), x_obs_mask=meta.get("x_obs_mask"),
        )
        dt_b = tv._flatten_dt(meta, mask_bn, device, key="delta_t_y")
        if dt_b is None:
            raise RuntimeError("Synthetic cache batches must carry delta_t_y for recovery.")
        dt_model = tv._match_dt_to_horizon(dt_b, horizon)

        for _, row, aid, start in chosen[b_idx]:
            capture: Dict[str, torch.Tensor] = {}
            gen = torch.Generator(device=device)
            gen.manual_seed(int(cfg.SEED) * 100003 + start * 31 + aid)
            diff_model.generate(
                shape=(1, horizon, latent_dim),
                cond_summary=cond_summary[row : row + 1],
                cond_summary_raw=cond_summary_raw[row : row + 1],
                dt=dt_model[row : row + 1],
                cfg_rescale=True,
                generator=gen,
                modal_capture=capture,
                **sampling,
            )
            con = modal_contributions(capture)
            share = con["energy_share"][0].numpy()
            sel, sel_share, sel_valid, n_used = _select_top_modes(
                share, int(args.recovery_top_modes), threshold
            )

            # Forecast steps start+window .. start+window+H-1 in absolute series time.
            lo = start + window
            hi = lo + horizon
            rho_true = truth[aid]["rho"][lo:hi]
            omega_true = truth[aid]["omega"][lo:hi]
            times = truth[aid].get("times")
            t_norm_span = None
            if times is not None and float(times[-1]) > 0:
                t_norm_span = (float(times[lo] / times[-1]), float(times[hi - 1] / times[-1]))

            rho_hat = con["rho"][0].numpy()  # [H,K] instantaneous
            omega_hat = con["omega"][0].numpy()
            rho_eff = con["rho_eff"][0].numpy()  # [H]
            omega_eff = con["omega_eff"][0].numpy()

            def _rmse(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.sqrt(((a - b) ** 2).mean()))

            omega_sel_rmse = [_rmse(omega_hat[:, k], omega_true) for k in sel]
            rho_sel_rmse = [_rmse(rho_hat[:, k], rho_true) for k in sel]

            win_metrics.append(
                {
                    "asset_id": aid,
                    "window_start": start,
                    "t_norm_span": t_norm_span,
                    "capture_t_idx": int(capture["t_idx"]),
                    "omega_eff_rmse": _rmse(omega_eff, omega_true),
                    "rho_eff_rmse": _rmse(rho_eff, rho_true),
                    "omega_best_rmse": float(min(omega_sel_rmse)),
                    "rho_best_rmse": float(min(rho_sel_rmse)),
                    "omega_true_mean": float(omega_true.mean()),
                    "rho_true_mean": float(rho_true.mean()),
                    "selected_share": sel_share,
                    "selection_valid": bool(sel_valid),
                    "selected_top_n": int(n_used),
                    "num_modes_above_1pct": int((share > 0.01).sum()),
                    "selected_modes": [
                        {
                            "mode": int(k),
                            "energy_share": float(share[k]),
                            "residue_norm2": float(con["residue_norm2"][0, k]),
                            "envelope_mass": float(con["envelope_mass"][0, k]),
                            "rho_mean": float(rho_hat[:, k].mean()),
                            "omega_mean": float(omega_hat[:, k].mean()),
                        }
                        for k in sel
                    ],
                }
            )
            fig_windows.append(
                {
                    "arm": str(cfg.DENOISER_MODAL_TYPE),
                    "asset_id": aid,
                    "window_start": start,
                    "t_norm_span": t_norm_span,
                    "t_grid": con["t_rel"][0].numpy(),
                    "mode_ids": sel,
                    "mode_shares": share[sel],
                    "rho_modes": rho_hat[:, sel],
                    "omega_modes": omega_hat[:, sel],
                    "rho_eff": rho_eff,
                    "omega_eff": omega_eff,
                    "rho_true": rho_true,
                    "omega_true": omega_true,
                    "selected_share": sel_share,
                    "selection_valid": bool(sel_valid),
                }
            )

    chirp_field = getattr(diff_model.model, "chirp_field", None)
    metrics: Dict[str, object] = {
        "arm": str(cfg.DENOISER_MODAL_TYPE),
        "num_windows": len(win_metrics),
        "share_threshold": threshold,
        "capture": "final DDIM step of the evaluated forecast, conditional branch",
        "laplace_k": int(diff_model.model.k),
        "chirp_num_basis": None if chirp_field is None else int(chirp_field.num_basis),
        "selection_valid": bool(all(w["selection_valid"] for w in win_metrics)),
        "min_selected_share": float(min(w["selected_share"] for w in win_metrics)),
        "metric_definitions": {
            "energy": "E_k = mean_t exp(-2 rho_bar_k(t)) * (||c_k||^2 + ||b_k||^2) from the "
                      "final-step modal capture of the evaluated generation",
            "omega_eff_rmse": "RMSE vs truth of the E_k-weighted effective omega trajectory "
                              "over ALL modes (the primary recovered curve)",
            "rho_eff_rmse": "same as omega_eff_rmse for rho",
            "omega_best_rmse": "min RMSE vs truth among the selected top-contribution modes",
            "rho_best_rmse": "same as omega_best_rmse for rho",
            "selected_share": "sum of E_k shares of the selected modes; below share_threshold "
                              "the window is selection-invalid (figure watermarked, not evidence)",
        },
        "windows": win_metrics,
    }
    for name in ("omega_eff_rmse", "rho_eff_rmse", "omega_best_rmse", "rho_best_rmse"):
        stat = _stats(w[name] for w in win_metrics)
        metrics[f"{name}_mean"] = stat["mean"]
        metrics[f"{name}_std"] = stat["std"]
    return metrics, fig_windows


def _assert_native_step_grid(t: np.ndarray) -> None:
    """The axis labels claim [rad/step] / [1/step]; catch unit drift before plotting."""
    d = np.diff(t)
    if t.shape[0] >= 2 and (not (d > 0).all() or not (0.05 <= float(np.median(d)) <= 20.0)):
        raise AssertionError(
            f"query grid does not look like native steps (median gap {float(np.median(d)):g}); "
            "[rad/step] axis labels would be wrong"
        )


_MODE_LINESTYLES = ("-", "--", "-.", ":")


def _plot_recovery(
    chirp_windows: List[Dict[str, object]],
    lti_windows: List[Dict[str, object]],
    save_path: Path,
    *,
    title: str,
) -> None:
    """Small-multiples pole-recovery figure: one row per stratified window,
    omega left, rho right. Every chirp line is legend-decodable (mode id +
    output-energy share); the SAME top-contribution mode is highlighted in both
    panels; the lti arm's constant recovered poles overlay as gray dashed; axes
    stay on the truth/contributing-mode scale with an omega_max=pi reference."""
    if not chirp_windows:
        return
    lti_by_key = {(w["asset_id"], w["window_start"]): w for w in lti_windows}
    n_rows = len(chirp_windows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.5, 3.5 * n_rows + 0.4), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for i, w in enumerate(chirp_windows):
        ax_omega, ax_rho = axes[i]
        t = np.asarray(w["t_grid"], dtype=np.float64)
        _assert_native_step_grid(t)
        lti = lti_by_key.get((w["asset_id"], w["window_start"]))

        for j in range(len(w["mode_ids"])):
            color = cmap(j % 10)
            ls = _MODE_LINESTYLES[j % len(_MODE_LINESTYLES)]
            lw, alpha = (2.4, 1.0) if j == 0 else (1.2, 0.85)
            label = f"chirp mode {int(w['mode_ids'][j])} ({100.0 * float(w['mode_shares'][j]):.0f}% E)"
            ax_omega.plot(t, w["omega_modes"][:, j], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
            ax_rho.plot(t, w["rho_modes"][:, j], color=color, ls=ls, lw=lw, alpha=alpha)
        ax_omega.plot(t, w["omega_eff"], color="tab:purple", lw=2.6, label="chirp ω_eff (E-weighted)")
        ax_rho.plot(t, w["rho_eff"], color="tab:purple", lw=2.6)
        if lti is not None:
            ax_omega.plot(
                lti["t_grid"], lti["omega_eff"], color="0.45", ls=(0, (5, 2)), lw=2.0,
                label="LTI ω_eff (const)",
            )
            ax_rho.plot(lti["t_grid"], lti["rho_eff"], color="0.45", ls=(0, (5, 2)), lw=2.0)
        ax_omega.plot(t, w["omega_true"], color="black", ls="--", lw=2.0, label="ground truth")
        ax_rho.plot(t, w["rho_true"], color="black", ls="--", lw=2.0)

        # Truth/selected-mode y-limits (junk modes can no longer dictate the axis);
        # the omega_max reference joins the axis only when the data comes near it.
        omega_data = [w["omega_modes"].ravel(), w["omega_eff"], np.asarray(w["omega_true"]).ravel()]
        rho_data = [w["rho_modes"].ravel(), w["rho_eff"], np.asarray(w["rho_true"]).ravel()]
        if lti is not None:
            omega_data.append(np.asarray(lti["omega_eff"]).ravel())
            rho_data.append(np.asarray(lti["rho_eff"]).ravel())
        omega_all = np.concatenate(omega_data)
        rho_all = np.concatenate(rho_data)
        omega_hi = float(omega_all.max())
        ax_omega.axhline(math.pi, color="0.6", ls=":", lw=0.9)
        if omega_hi >= 0.6 * math.pi:
            omega_hi = max(omega_hi, 1.05 * math.pi)
            ax_omega.text(t[-1], math.pi, " ω_max=π", va="bottom", ha="right", fontsize=7, color="0.4")
        else:
            ax_omega.text(
                0.99, 0.98, "ω_max=π (off-axis)", transform=ax_omega.transAxes,
                va="top", ha="right", fontsize=7, color="0.4",
            )
        pad_o = 0.08 * max(omega_hi - float(omega_all.min()), 1e-6)
        ax_omega.set_ylim(float(omega_all.min()) - pad_o, omega_hi + pad_o)
        pad_r = 0.08 * max(float(rho_all.max()) - float(rho_all.min()), 1e-6)
        ax_rho.set_ylim(float(rho_all.min()) - pad_r, float(rho_all.max()) + pad_r)

        span = w.get("t_norm_span")
        span_txt = "" if span is None else f" · t_norm {span[0]:.2f}–{span[1]:.2f}"
        ax_omega.set_title(
            f"entity {w['asset_id']} · start {w['window_start']}{span_txt} · "
            f"top-{len(w['mode_ids'])} share {100.0 * float(w['selected_share']):.0f}%",
            fontsize=8,
        )
        ax_omega.set_ylabel("ω(t̃) [rad/step]")
        ax_rho.set_ylabel("ρ(t̃) [1/step]")
        if i == n_rows - 1:
            ax_omega.set_xlabel("relative time t̃ (native steps)")
            ax_rho.set_xlabel("relative time t̃ (native steps)")
        ax_omega.legend(loc="best", fontsize=6.5)

    if any(not w["selection_valid"] for w in chirp_windows):
        fig.text(
            0.5, 0.5, "SELECTION INVALID", color="red", alpha=0.25, fontsize=46,
            ha="center", va="center", rotation=30,
        )
    fig.suptitle(f"{title}\n(ρ panel shares the ω panel's line encoding; poles captured at the "
                 f"final denoising step of the evaluated forecast)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _plot_cross_window(
    chirp_windows: List[Dict[str, object]],
    lti_windows: List[Dict[str, object]],
    data_dir: str,
    save_path: Path,
    *,
    title: str,
    window: int,
) -> None:
    """Stitched recovery (fix-plan P5's cross-window view): each window's effective
    trajectory placed at its absolute series position, over the generator's full
    truth curve — the history-conditioned poles should step along the slow sweep."""
    if not chirp_windows:
        return
    truth = load_ground_truth_poles(data_dir)
    aid = int(chirp_windows[0]["asset_id"])  # shared_poles: one truth per cache
    lti_by_key = {(w["asset_id"], w["window_start"]): w for w in lti_windows}

    fig, (ax_omega, ax_rho) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    spans = []
    for i, w in enumerate(chirp_windows):
        horizon = len(np.asarray(w["omega_eff"]))
        x = int(w["window_start"]) + int(window) + np.arange(horizon)
        spans.append((x[0], x[-1]))
        label_c = "chirp ω_eff per window" if i == 0 else None
        label_l = "LTI ω_eff per window" if i == 0 else None
        ax_omega.plot(x, w["omega_eff"], color="tab:purple", lw=2.0, label=label_c)
        ax_rho.plot(x, w["rho_eff"], color="tab:purple", lw=2.0)
        lti = lti_by_key.get((w["asset_id"], w["window_start"]))
        if lti is not None:
            ax_omega.plot(x, lti["omega_eff"], color="0.45", ls=(0, (5, 2)), lw=1.8, label=label_l)
            ax_rho.plot(x, lti["rho_eff"], color="0.45", ls=(0, (5, 2)), lw=1.8)
    lo = max(0, min(s[0] for s in spans) - int(window))
    hi = min(len(truth[aid]["omega"]), max(s[1] for s in spans) + 2)
    x_truth = np.arange(lo, hi)
    ax_omega.plot(x_truth, truth[aid]["omega"][lo:hi], color="black", ls="--", lw=1.6, label="ground truth")
    ax_rho.plot(x_truth, truth[aid]["rho"][lo:hi], color="black", ls="--", lw=1.6)
    ax_omega.set_ylabel("ω [rad/step]")
    ax_rho.set_ylabel("ρ [1/step]")
    ax_rho.set_xlabel("absolute series position (samples)")
    ax_omega.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _evaluate_forecast(
    cfg: SimpleNamespace, checkpoint: str, loaders, args: argparse.Namespace, *, device: torch.device
) -> Mapping[str, object]:
    train_dl, _, test_dl, _ = loaders
    diff_model, vae, summarizer, mu_mean, mu_std = _load_stack(cfg, Path(checkpoint), device, train_dl)
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    return tv.evaluate_regression(
        diff_model,
        vae,
        summarizer,
        test_dl,
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        config=cfg,
        ema=None,
        self_cond=bool(getattr(cfg, "SELF_COND", False)),
        disable_conditioning=False,
        verbose=bool(args.verbose or args.debug),
        **sampling,
    )


def _metric(payload: Mapping[str, object], name: str) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object, object], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (row.get("task"), row.get("arm"), row.get("gap_regime"))
        groups.setdefault(key, []).append(row)
    out = []
    for (task, arm, gap_regime), group in sorted(
        groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])
    ):
        entry: Dict[str, object] = {
            "task": task, "arm": arm, "gap_regime": gap_regime, "runs": len(group),
        }
        for metric in ("crps", "mae", "mse", "omega_eff_rmse_mean", "rho_eff_rmse_mean"):
            stat = _stats(r.get(metric) for r in group)
            entry[f"{metric}_mean"] = stat["mean"]
            entry[f"{metric}_std"] = stat["std"]
        out.append(entry)
    return out


def main() -> None:
    args = _parse_args()
    _validate_geometry(args)
    result_root = Path(args.output_root).resolve()
    rows: List[Dict[str, object]] = []

    for task in args.tasks:
        _prepare_cache(task, args)
        for seed in args.seeds:
            recoveries: Dict[str, Tuple[Dict[str, object], List[Dict[str, object]]]] = {}
            for arm in args.arms:
                cfg = _configure(task, str(arm), int(seed), args)
                device = set_torch(seed=int(seed), deterministic=False)
                loaders = _build_loaders(cfg)
                stage_payload = _train_or_reuse_stack(cfg, loaders, args)
                checkpoint = str(stage_payload["paths"]["llapdiff"])

                forecast = _evaluate_forecast(cfg, checkpoint, loaders, args, device=device)
                row: Dict[str, object] = {
                    "task": task,
                    "arm": str(arm),
                    "seed": int(seed),
                    "gap_regime": _gap_tag(args),
                    "checkpoint": checkpoint,
                    "crps": _metric(forecast, "crps"),
                    "mae": _metric(forecast, "mae"),
                    "mse": _metric(forecast, "mse"),
                }

                # Recovery runs for BOTH arms: the lti arm's constant recovered
                # poles are the structural-failure contrast (fix-plan P2).
                recovery, fig_windows = _recover_pole_trajectories(
                    cfg, checkpoint, loaders, args, device=device
                )
                row["omega_eff_rmse_mean"] = recovery["omega_eff_rmse_mean"]
                row["rho_eff_rmse_mean"] = recovery["rho_eff_rmse_mean"]
                row["omega_best_rmse_mean"] = recovery["omega_best_rmse_mean"]
                row["rho_best_rmse_mean"] = recovery["rho_best_rmse_mean"]
                row["recovery_selection_valid"] = recovery["selection_valid"]
                recoveries[str(arm)] = (recovery, fig_windows)
                rows.append(row)

            if recoveries:
                recovery_path = result_root / "recovery" / f"{task}_seed-{seed}.json"
                recovery_path.parent.mkdir(parents=True, exist_ok=True)
                recovery_path.write_text(
                    json.dumps(
                        make_jsonable({arm: rec for arm, (rec, _) in recoveries.items()}),
                        indent=2,
                        sort_keys=True,
                    )
                )
            if "chirp" in recoveries:
                _plot_recovery(
                    recoveries["chirp"][1],
                    recoveries.get("lti", (None, []))[1],
                    result_root / "figures" / f"{task}_seed-{seed}_pole_recovery.pdf",
                    title=f"Recovered vs ground-truth poles ({task}, seed={seed})",
                )
                _plot_cross_window(
                    recoveries["chirp"][1],
                    recoveries.get("lti", (None, []))[1],
                    str(_cache_dir(task, args)),
                    result_root / "figures" / f"{task}_seed-{seed}_pole_recovery_series.pdf",
                    title=f"Cross-window recovered poles along the series ({task}, seed={seed})",
                    window=int(args.window),
                )

    _write_rows(rows, result_root / "chirp_benchmark_raw.csv", result_root / "chirp_benchmark_raw.json")
    _write_rows(
        _summary_rows(rows),
        result_root / "chirp_benchmark_summary.csv",
        result_root / "chirp_benchmark_summary.json",
    )
    overall = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_rows": len(rows),
        "tasks": list(args.tasks),
        "arms": list(args.arms),
        "seeds": [int(s) for s in args.seeds],
        "result_root": str(result_root),
    }
    (result_root / "chirp_benchmark_overall.json").write_text(
        json.dumps(make_jsonable(overall), indent=2, sort_keys=True)
    )
    print(
        f"completed: rows={len(rows)} tasks={len(args.tasks)} arms={list(args.arms)} "
        f"result_root={result_root}"
    )


if __name__ == "__main__":
    main()
