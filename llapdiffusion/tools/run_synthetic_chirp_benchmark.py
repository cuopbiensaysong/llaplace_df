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
from llapdiffusion.viz.plot_llapdiff_poles import extract_chirp_pole_trajectories


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


def _cache_dir(task: str, args: argparse.Namespace) -> Path:
    return (
        Path(args.data_root)
        / task
        / f"len-{int(args.series_length)}_cp-{_resolve_change_point(args)}_entities-{int(args.num_entities)}"
    ).resolve()


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


@torch.no_grad()
def _recover_pole_trajectories(
    cfg: SimpleNamespace,
    checkpoint: str,
    loaders,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Extract chirp pole trajectories for a few test windows and score them
    against the generator's ground truth (best-matching mode among the top
    time-varying modes). Returns (metrics, figure payload)."""
    train_dl, _, test_dl, _ = loaders
    diff_model, _, summarizer, _, _ = _load_stack(cfg, Path(checkpoint), device, train_dl)
    truth = load_ground_truth_poles(cfg.DATA_DIR)
    window = int(cfg.WINDOW)

    rows: List[Dict[str, object]] = []
    fig_payload: Dict[str, np.ndarray] = {}
    collected = 0
    for xb, yb, meta in test_dl:
        (V, T), _, mask_bn = tv._sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=meta.get("delta_t"), x_obs_mask=meta.get("x_obs_mask"),
        )
        asset_ids = meta["cache_asset_ids"].cpu().numpy()  # [B,N]
        starts = meta["cache_window_starts"].cpu().numpy()  # [B,N]
        delta_t_y = meta["delta_t_y"].cpu().numpy()  # [B,N,H]
        mask_np = mask_bn.detach().cpu().numpy().astype(bool)

        for row in range(cond_summary.shape[0]):
            valid = np.nonzero(mask_np[row])[0]
            if valid.size == 0:
                continue
            n0 = int(valid[0])
            aid = int(asset_ids[row, n0])
            start = int(starts[row, n0])
            t_grid = torch.from_numpy(delta_t_y[row, n0].astype(np.float32))

            traj = extract_chirp_pole_trajectories(
                diff_model,
                t_idx=1,
                cond_summary=cond_summary[row : row + 1],
                cond_summary_raw=cond_summary_raw[row : row + 1],
                t_grid=t_grid,
                top_modes=int(args.recovery_top_modes),
            )
            # Forecast steps start+window .. start+window+H-1 in absolute series time.
            lo = start + window
            hi = lo + t_grid.shape[0]
            rho_true = truth[aid]["rho"][lo:hi]
            omega_true = truth[aid]["omega"][lo:hi]
            rho_hat = traj["rho"][0].numpy()  # [H, top]
            omega_hat = traj["omega"][0].numpy()

            omega_rmse = np.sqrt(((omega_hat - omega_true[:, None]) ** 2).mean(axis=0))
            rho_rmse = np.sqrt(((rho_hat - rho_true[:, None]) ** 2).mean(axis=0))
            best_omega = int(omega_rmse.argmin())
            rows.append(
                {
                    "asset_id": aid,
                    "window_start": start,
                    "omega_rmse_best": float(omega_rmse.min()),
                    "rho_rmse_best": float(rho_rmse.min()),
                    "omega_true_mean": float(omega_true.mean()),
                    "rho_true_mean": float(rho_true.mean()),
                    "best_mode_index": int(traj["mode_indices"][0, best_omega]),
                }
            )
            if not fig_payload:
                fig_payload = {
                    "t_grid": t_grid.numpy(),
                    "rho_hat": rho_hat,
                    "omega_hat": omega_hat,
                    "rho_true": rho_true,
                    "omega_true": omega_true,
                    "best_omega_mode": np.int64(best_omega),
                    "best_rho_mode": np.int64(rho_rmse.argmin()),
                }
            collected += 1
            if collected >= int(args.num_recovery_windows):
                break
        if collected >= int(args.num_recovery_windows):
            break

    if not rows:
        raise RuntimeError("No valid test windows found for pole recovery.")
    metrics = {
        "num_windows": len(rows),
        "omega_rmse_best_mean": float(np.mean([r["omega_rmse_best"] for r in rows])),
        "rho_rmse_best_mean": float(np.mean([r["rho_rmse_best"] for r in rows])),
        "windows": rows,
    }
    return metrics, fig_payload


def _plot_recovery(payload: Dict[str, np.ndarray], save_path: Path, *, title: str) -> None:
    t = payload["t_grid"]
    fig, (ax_omega, ax_rho) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in range(payload["omega_hat"].shape[1]):
        alpha = 0.95 if k == int(payload["best_omega_mode"]) else 0.25
        ax_omega.plot(t, payload["omega_hat"][:, k], color="tab:blue", alpha=alpha)
    ax_omega.plot(t, payload["omega_true"], color="black", linestyle="--", linewidth=2, label="ground truth")
    ax_omega.set_xlabel("relative time t̃ (steps)")
    ax_omega.set_ylabel("ω(t̃) [rad/step]")
    ax_omega.legend(loc="best", fontsize=8)
    for k in range(payload["rho_hat"].shape[1]):
        alpha = 0.95 if k == int(payload["best_rho_mode"]) else 0.25
        ax_rho.plot(t, payload["rho_hat"][:, k], color="tab:red", alpha=alpha)
    ax_rho.plot(t, payload["rho_true"], color="black", linestyle="--", linewidth=2, label="ground truth")
    ax_rho.set_xlabel("relative time t̃ (steps)")
    ax_rho.set_ylabel("ρ(t̃) [1/step]")
    ax_rho.legend(loc="best", fontsize=8)
    fig.suptitle(title)
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
    groups: Dict[Tuple[object, object], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((row.get("task"), row.get("arm")), []).append(row)
    out = []
    for (task, arm), group in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        entry: Dict[str, object] = {"task": task, "arm": arm, "runs": len(group)}
        for metric in ("crps", "mae", "mse", "omega_rmse_best_mean", "rho_rmse_best_mean"):
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
                    "checkpoint": checkpoint,
                    "crps": _metric(forecast, "crps"),
                    "mae": _metric(forecast, "mae"),
                    "mse": _metric(forecast, "mse"),
                }

                if arm == "chirp":
                    recovery, fig_payload = _recover_pole_trajectories(
                        cfg, checkpoint, loaders, args, device=device
                    )
                    row["omega_rmse_best_mean"] = recovery["omega_rmse_best_mean"]
                    row["rho_rmse_best_mean"] = recovery["rho_rmse_best_mean"]
                    recovery_path = result_root / "recovery" / f"{task}_seed-{seed}.json"
                    recovery_path.parent.mkdir(parents=True, exist_ok=True)
                    recovery_path.write_text(json.dumps(make_jsonable(recovery), indent=2, sort_keys=True))
                    _plot_recovery(
                        fig_payload,
                        result_root / "figures" / f"{task}_seed-{seed}_pole_recovery.pdf",
                        title=f"Recovered vs ground-truth poles ({task}, seed={seed})",
                    )
                rows.append(row)

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
