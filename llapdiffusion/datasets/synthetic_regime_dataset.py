"""Synthetic regime-shift dataset utilities for LLapDiff robustness checks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _validate_context_missingness_rate,
    rebuild_window_index_only,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
)


DATASET_NAME = "synthetic_regime"
TARGET_COLUMN = "signal"
DEFAULT_FREQ = "1h"
MAX_WINDOW = 96
MAX_HORIZON = 48

# Piecewise (regime-shift) tasks used by the boundary-crossing protocol.
SHIFT_TASKS = ("synthetic_freq_shift", "synthetic_decay_shift")
# Smoothly time-varying ground-truth pole tasks (the chirp benchmark, H2). The
# freq-shift task doubles as the piecewise-pole "regime switch" benchmark case.
CHIRP_TASKS = (
    "synthetic_linear_chirp",
    "synthetic_quadratic_chirp",
    "synthetic_ramp_damping_up",
    "synthetic_ramp_damping_down",
    "synthetic_growth_decay",
)
ALL_TASKS = SHIFT_TASKS + CHIRP_TASKS


@dataclass
class SyntheticRegimeCacheConfig:
    task: str = "synthetic_freq_shift"
    window: int = MAX_WINDOW
    horizon: int = MAX_HORIZON
    data_dir: PathLike = "./synthetic_regime_cache"
    num_entities: int = 64
    series_length: int = 288
    change_point: int = 216
    freq: str = DEFAULT_FREQ
    seed: int = 20260327
    normalize_per_entity: bool = True
    overwrite: bool = False
    amplitude_min: float = 0.8
    amplitude_max: float = 1.2
    baseline_min: float = -0.1
    baseline_max: float = 0.1
    frequency_min: float = 1.0 / 48.0
    frequency_max: float = 1.0 / 24.0
    decay_min: float = 0.002
    decay_max: float = 0.01
    freq_multiplier: float = 2.0
    decay_multiplier: float = 2.5
    noise_std: float = 0.05
    phase_min: float = 0.0
    phase_max: float = 2.0 * np.pi
    keep_time_meta: str = "end"
    # Total log-amplitude the envelope gains before change_point in the
    # growth-then-decay task (the Theorem-B' budget case; e.g. log 2 = c_g).
    growth_log_amplitude: float = float(np.log(2.0))
    # Share one (base_frequency, base_decay) draw across all entities (amplitude,
    # baseline, phase, and noise stay per-entity). The chirp benchmark uses this so
    # a joint date row has a single well-defined ground-truth pole function.
    shared_poles: bool = False


def _validate_task(task: str) -> str:
    value = str(task).strip().lower()
    if value not in set(ALL_TASKS):
        raise ValueError(
            f"Unsupported synthetic task '{task}'. Expected one of {sorted(ALL_TASKS)}."
        )
    return value


def _pole_profiles(
    cfg: SyntheticRegimeCacheConfig,
    *,
    base_frequency: float,
    base_decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step ground-truth pole profiles (frequency [cycles/step], decay [1/step]).

    Every task is a special case of the chirp-modal closed form
    ``amplitude * exp(-cumsum(decay)) * sin(phase0 + cumsum(2*pi*frequency))``:
    the task only decides how the instantaneous poles vary over the series.
    """
    T = int(cfg.series_length)
    ramp = np.linspace(0.0, 1.0, T, dtype=np.float64)  # u = t/(T-1)
    frequency = np.full((T,), float(base_frequency), dtype=np.float64)
    decay = np.full((T,), float(base_decay), dtype=np.float64)

    if cfg.task == "synthetic_freq_shift":
        frequency[cfg.change_point :] = float(cfg.freq_multiplier) * base_frequency
    elif cfg.task == "synthetic_decay_shift":
        decay[cfg.change_point :] = float(cfg.decay_multiplier) * base_decay
    elif cfg.task == "synthetic_linear_chirp":
        frequency = base_frequency * (1.0 + (float(cfg.freq_multiplier) - 1.0) * ramp)
    elif cfg.task == "synthetic_quadratic_chirp":
        frequency = base_frequency * (1.0 + (float(cfg.freq_multiplier) - 1.0) * ramp**2)
    elif cfg.task == "synthetic_ramp_damping_up":
        decay = base_decay * (1.0 + (float(cfg.decay_multiplier) - 1.0) * ramp)
    elif cfg.task == "synthetic_ramp_damping_down":
        decay = base_decay * (float(cfg.decay_multiplier) - (float(cfg.decay_multiplier) - 1.0) * ramp)
    elif cfg.task == "synthetic_growth_decay":
        # Envelope rises by exp(growth_log_amplitude) up to the change point
        # (instantaneous decay is negative there), then damps.
        decay = np.full((T,), float(cfg.decay_multiplier) * base_decay, dtype=np.float64)
        decay[: cfg.change_point] = -float(cfg.growth_log_amplitude) / float(cfg.change_point)
    else:  # pragma: no cover - guarded by _validate_task
        raise ValueError(f"Unhandled synthetic task '{cfg.task}'.")

    return frequency.astype(np.float32), decay.astype(np.float32)


def _generate_signal(
    cfg: SyntheticRegimeCacheConfig,
    rng: np.random.Generator,
    *,
    base_frequency: Optional[float] = None,
    base_decay: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signal, frequency, decay); the pole arrays are the ground truth.

    Explicit ``base_frequency``/``base_decay`` override the per-entity draw
    (used when ``cfg.shared_poles`` shares one pole function across entities).
    """
    amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
    baseline = float(rng.uniform(cfg.baseline_min, cfg.baseline_max))
    phase0 = float(rng.uniform(cfg.phase_min, cfg.phase_max))
    if base_frequency is None:
        base_frequency = float(rng.uniform(cfg.frequency_min, cfg.frequency_max))
    if base_decay is None:
        base_decay = float(rng.uniform(cfg.decay_min, cfg.decay_max))

    frequency, decay = _pole_profiles(
        cfg, base_frequency=base_frequency, base_decay=base_decay
    )

    phase_path = phase0 + np.cumsum((2.0 * np.pi * frequency).astype(np.float64))
    envelope = np.exp(-np.cumsum(decay.astype(np.float64)))
    noise = rng.normal(0.0, cfg.noise_std, size=(cfg.series_length,)).astype(np.float64)
    signal = baseline + amplitude * envelope * np.sin(phase_path) + noise
    return signal.astype(np.float32, copy=False), frequency, decay


def prepare_synthetic_regime_cache(cfg: SyntheticRegimeCacheConfig) -> Mapping[str, object]:
    cfg.task = _validate_task(cfg.task)
    if cfg.window > MAX_WINDOW or cfg.horizon > MAX_HORIZON:
        raise ValueError(
            f"Requested window/horizon ({cfg.window}, {cfg.horizon}) exceed "
            f"the supported synthetic maxima ({MAX_WINDOW}, {MAX_HORIZON})."
        )
    if cfg.series_length < (cfg.window + cfg.horizon):
        raise ValueError(
            "series_length must be at least window + horizon "
            f"({cfg.window + cfg.horizon})."
        )
    if not (0 < cfg.change_point < cfg.series_length):
        raise ValueError("change_point must lie strictly inside the generated series.")

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    rng = np.random.default_rng(int(cfg.seed))
    assets = [f"entity_{idx:03d}" for idx in range(int(cfg.num_entities))]
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}
    feature_cols = [TARGET_COLUMN]

    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=bool(cfg.normalize_per_entity),
    )

    start_time = np.datetime64("2000-01-01T00:00:00")
    pairs: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    generation_rows: List[Dict[str, object]] = []

    pole_truth_dir = paths.cache_root / "pole_truth"
    pole_truth_dir.mkdir(parents=True, exist_ok=True)

    shared_frequency = shared_decay = None
    if cfg.shared_poles:
        shared_frequency = float(rng.uniform(cfg.frequency_min, cfg.frequency_max))
        shared_decay = float(rng.uniform(cfg.decay_min, cfg.decay_max))

    for asset in assets:
        aid = asset_to_id[asset]
        signal, frequency, decay = _generate_signal(
            cfg, rng, base_frequency=shared_frequency, base_decay=shared_decay
        )
        features = signal.reshape(-1, 1).astype(np.float32, copy=False)
        targets = signal.astype(np.float32, copy=False)
        times = start_time + np.arange(cfg.series_length).astype("timedelta64[h]")
        obs_mask = np.ones_like(features, dtype=bool)
        fill_mask = np.ones_like(features, dtype=bool)

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16, copy=False))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16, copy=False))
        np.save(paths.times / f"{aid}.npy", times.astype("datetime64[ns]"))
        np.save(paths.obs_masks / f"{aid}.npy", obs_mask)
        np.save(paths.fill_masks / f"{aid}.npy", fill_mask)
        # Ground-truth instantaneous poles in model units (per native step):
        # rho = decay, omega = 2*pi*frequency [rad/step]. Consumed by the chirp
        # benchmark's recovery figure (load_ground_truth_poles).
        np.savez(
            pole_truth_dir / f"{aid}.npz",
            rho=decay.astype(np.float32),
            omega=(2.0 * np.pi * frequency).astype(np.float32),
        )
        norm_acc.update(aid, features, targets)

        total_rows = int(features.shape[0])
        max_start = total_rows - (cfg.window + cfg.horizon) + 1
        starts = np.arange(0, max_start, dtype=np.int32)
        end_times = times[starts + cfg.window - 1]
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        ends.append(end_times.astype("datetime64[ns]"))
        generation_rows.append(
            {
                "asset": asset,
                "asset_id": aid,
                "series_length": total_rows,
                "change_point": int(cfg.change_point),
            }
        )

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    end_times = np.concatenate(ends, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", end_times)

    with paths.norm_stats.open("w") as f:
        json.dump(norm_acc.finalize(assets), f)

    meta = {
        "dataset": DATASET_NAME,
        "task": cfg.task,
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "max_window": int(MAX_WINDOW),
        "max_horizon": int(MAX_HORIZON),
        "freq": str(cfg.freq),
        "keep_time_meta": str(cfg.keep_time_meta),
        "normalize_per_entity": bool(cfg.normalize_per_entity),
        "series_length": int(cfg.series_length),
        "change_point": int(cfg.change_point),
        "seed": int(cfg.seed),
        "generation_rows": generation_rows,
        "generation_config": {
            "amplitude_min": float(cfg.amplitude_min),
            "amplitude_max": float(cfg.amplitude_max),
            "baseline_min": float(cfg.baseline_min),
            "baseline_max": float(cfg.baseline_max),
            "frequency_min": float(cfg.frequency_min),
            "frequency_max": float(cfg.frequency_max),
            "decay_min": float(cfg.decay_min),
            "decay_max": float(cfg.decay_max),
            "freq_multiplier": float(cfg.freq_multiplier),
            "decay_multiplier": float(cfg.decay_multiplier),
            "noise_std": float(cfg.noise_std),
            "growth_log_amplitude": float(cfg.growth_log_amplitude),
            "shared_poles": bool(cfg.shared_poles),
        },
    }
    with paths.meta.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "data_dir": str(data_dir),
        "task": cfg.task,
        "num_entities": int(cfg.num_entities),
        "series_length": int(cfg.series_length),
        "window_count": int(global_pairs.shape[0]),
        "change_point": int(cfg.change_point),
    }


def load_ground_truth_poles(data_dir: PathLike) -> Dict[int, Dict[str, np.ndarray]]:
    """Load the per-entity ground-truth instantaneous poles saved by the generator.

    Returns ``{asset_id: {"rho": [T], "omega": [T]}}`` in model units (per native
    step; omega in rad/step). Raises if the cache predates pole-truth persistence.
    """
    paths = CachePaths.from_dir(data_dir)
    truth_dir = paths.cache_root / "pole_truth"
    if not truth_dir.exists():
        raise FileNotFoundError(
            f"No pole_truth/ directory under '{paths.cache_root}'. "
            "Regenerate the cache (overwrite=True) with the current generator."
        )
    truth: Dict[int, Dict[str, np.ndarray]] = {}
    for npz_path in sorted(truth_dir.glob("*.npz")):
        payload = np.load(npz_path)
        truth[int(npz_path.stem)] = {
            "rho": payload["rho"].astype(np.float32),
            "omega": payload["omega"].astype(np.float32),
        }
    if not truth:
        raise FileNotFoundError(f"pole_truth/ under '{paths.cache_root}' is empty.")
    return truth


def _validate_cache(paths: CachePaths) -> Dict[str, object]:
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}'. Did you run prepare_synthetic_regime_cache()?"
        )
    with paths.meta.open("r") as f:
        meta = json.load(f)
    if meta.get("dataset") != DATASET_NAME:
        raise ValueError(
            f"The cache at '{paths.cache_root}' does not correspond to the synthetic regime dataset."
        )
    return meta


def run_experiment(
    data_dir: PathLike,
    K: Optional[int] = None,
    H: Optional[int] = None,
    *,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.0,
    panel_coverage: float = 0.0,
    dates_per_batch: int = 16,
    batch_size: int = 16,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
):
    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    paths = CachePaths.from_dir(data_dir)
    meta = _validate_cache(paths)
    cached_window = int(meta.get("window", MAX_WINDOW))
    cached_horizon = int(meta.get("horizon", MAX_HORIZON))
    max_window = int(meta.get("max_window", MAX_WINDOW))
    max_horizon = int(meta.get("max_horizon", MAX_HORIZON))

    if K is None:
        K = cached_window
    if H is None:
        H = cached_horizon
    K = int(K)
    H = int(H)

    if K > max_window or H > max_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_synthetic_regime_cache with larger values first."
        )

    needs_reindex = K != cached_window or H != cached_horizon
    needs_target_reindex = target_col is not None or target_cols is not None

    if reindex and (needs_reindex or needs_target_reindex):
        rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=needs_reindex,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )

    train_dl, val_dl, test_dl, lengths = _load_fin_ratio_split(
        data_dir=str(data_dir),
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,
        per_asset=per_asset,
        norm_scope=norm,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coverage_per_window=panel_coverage,
        date_batching=date_batching,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        coverage=coverage,
    )
    return train_dl, val_dl, test_dl, lengths


def _resolve_pairs_and_window(dataset) -> Tuple[np.ndarray, int]:
    if isinstance(dataset, Subset):
        base_pairs, window = _resolve_pairs_and_window(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return np.asarray(base_pairs)[indices], int(window)
    if not hasattr(dataset, "pairs") or not hasattr(dataset, "window"):
        raise TypeError("dataset must expose 'pairs' and 'window'.")
    return np.asarray(dataset.pairs), int(dataset.window)


def build_context_end_eval_loader(
    test_loader: DataLoader,
    *,
    min_context_end: Optional[int] = None,
    max_context_end: Optional[int] = None,
) -> DataLoader:
    dataset = getattr(test_loader, "dataset", None)
    if dataset is None:
        raise TypeError("test_loader must expose a dataset.")

    pairs, window = _resolve_pairs_and_window(dataset)
    end_indices = pairs[:, 1].astype(np.int64) + int(window) - 1
    keep_mask = np.ones(end_indices.shape[0], dtype=bool)
    if min_context_end is not None:
        keep_mask &= end_indices >= int(min_context_end)
    if max_context_end is not None:
        keep_mask &= end_indices <= int(max_context_end)
    keep = np.nonzero(keep_mask)[0]
    if keep.size == 0:
        raise RuntimeError(
            "No evaluation windows found for the requested context-end slice "
            f"[{min_context_end}, {max_context_end}]."
        )

    subset = Subset(dataset, keep.tolist())
    return DataLoader(
        subset,
        batch_size=getattr(test_loader, "batch_size", None) or 1,
        shuffle=False,
        num_workers=int(getattr(test_loader, "num_workers", 0)),
        pin_memory=bool(getattr(test_loader, "pin_memory", False)),
        collate_fn=getattr(test_loader, "collate_fn", None),
        drop_last=False,
    )


def build_regime_eval_loader(
    test_loader: DataLoader,
    *,
    change_point: int,
    lookback_steps: int = 12,
) -> DataLoader:
    lo = int(change_point) - int(lookback_steps)
    hi = int(change_point) - 1
    return build_context_end_eval_loader(
        test_loader,
        min_context_end=lo,
        max_context_end=hi,
    )


__all__ = [
    "ALL_TASKS",
    "CHIRP_TASKS",
    "DATASET_NAME",
    "DEFAULT_FREQ",
    "MAX_HORIZON",
    "MAX_WINDOW",
    "SHIFT_TASKS",
    "SyntheticRegimeCacheConfig",
    "build_context_end_eval_loader",
    "build_regime_eval_loader",
    "load_ground_truth_poles",
    "prepare_synthetic_regime_cache",
    "run_experiment",
]
