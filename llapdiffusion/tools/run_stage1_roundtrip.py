"""Stage-1 round-trip check (Fig-2 "Gate 0").

Encodes the TRUE targets through the trained set-VAE and decodes them straight back
-- no forecasting -- and reports the correlation with the originals. This is the
upper bound on any denoiser built on that VAE: if stage 1 cannot reconstruct its own
targets, no downstream chirp-vs-LTI comparison is meaningful.

It exists because H2 silently sat under a 0.62 ceiling for exactly this reason: the
set-VAE mean-pools across entities while the generator drew an independent U(0, 2pi)
phase per entity, so the panel cancelled before the latent. ``--phase-spread`` fixes
the generator; this tool is the gate that confirms it on the run's own cache.

Two addressing modes:

**Synthetic tasks** (the original), mirroring the benchmark's cache/artifact addressing::

    llapdiff-stage1-roundtrip --tasks synthetic_linear_chirp --seeds 0 \
        --sweep-period 144 --phase-spread 0.393 \
        --artifact-root ldt/synthetic_artifacts/chirp_benchmark_phasefix

**A real dataset** (added 2026-08-01), against that preset's frozen stage-1 VAE::

    llapdiff-stage1-roundtrip --dataset-key noaa_uk --pred 168

The round-trip itself was always dataset-agnostic — only the config/loader construction
was bound to the synthetic benchmark, so Gate 0 of ``w_docs/CMD_UQ_RECOVERY_PLAN.md``
could not be run on the cell it gates. Dataset mode reads **val** by default, because the
plan's standing rules put every gate and diagnostic on val; task mode keeps its historical
**test** default so the recorded synthetic numbers (0.62 pre-fix, 0.857 post-fix) stay
reproducible. Both are overridable with ``--split``, and the resolved choice is printed
and recorded in the JSON.

Dataset mode needs no seed loop: the VAE is frozen, shared across arms and seeds (stage-1
paths are never seed-routed), and the round-trip reads the posterior MEAN rather than a
sample, so it is deterministic given the data. ``--seeds`` is ignored there.

Prints one PASS/FAIL row per cell against ``--threshold`` (default 0.85).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from llapdiffusion.latent_space.latent_vae import LatentVAE
from llapdiffusion.models.llapdiff_utils import (
    decode_latents_with_vae,
    infer_target_dim_from_loader,
    pack_targets_tokens,
    set_torch,
    vae_io_dims_for_target_dim,
)
from llapdiffusion.target_artifacts import unwrap_checkpoint_model
from llapdiffusion.tools import run_synthetic_chirp_benchmark as bench
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import build_eval_loaders
from llapdiffusion.trainers import train_val_llapdiff as tv


def _build_vae(cfg, train_dl, device: torch.device) -> LatentVAE:
    _, num_entities, _, _ = tv._summarize_dataset(train_dl, None, verbose=False)
    target_dim = infer_target_dim_from_loader(train_dl)
    cfg.TARGET_DIM = target_dim
    vae_in, vae_out = vae_io_dims_for_target_dim(cfg, target_dim)
    vae = LatentVAE(
        seq_len=cfg.PRED, latent_dim=cfg.VAE_LATENT_DIM,
        latent_channel=cfg.VAE_LATENT_CHANNELS,
        enc_layers=cfg.VAE_LAYERS, enc_heads=cfg.VAE_HEADS, enc_ff=cfg.VAE_FF,
        dec_layers=cfg.VAE_LAYERS, dec_heads=cfg.VAE_HEADS, dec_ff=cfg.VAE_FF,
        input_dim=vae_in, output_dim=vae_out, num_entities=num_entities,
        entity_conditioned=bool(getattr(cfg, "VAE_ENTITY_CONDITION", False)),
    ).to(device)
    payload = torch.load(cfg.VAE_CKPT, map_location=device, weights_only=False)
    tv._load_module_state(vae, unwrap_checkpoint_model(payload), strict=True)
    vae.eval()
    return vae


@torch.no_grad()
def roundtrip_correlation(
    cfg, loaders, device: torch.device, *, max_batches: int = 2, split: str = "test"
) -> Dict[str, float]:
    """Encode->decode the true targets and correlate with the originals.

    Returns per-entity-window mean ``corr`` and ``std_ratio`` (reconstructed std over
    true std), plus the sample count. Normalization uses the batch's own latent
    moments so the check is independent of stored mu stats.

    ``split`` selects which loader is read. The VAE is always built from the TRAIN
    loader, which only supplies shapes (entity count, target dim) -- it is loaded
    frozen from ``cfg.VAE_CKPT``, never fitted here.
    """
    train_dl, val_dl, test_dl, _ = loaders
    by_split = {"train": train_dl, "val": val_dl, "test": test_dl}
    if split not in by_split:
        raise ValueError(f"split must be one of {sorted(by_split)}, got {split!r}")
    vae = _build_vae(cfg, train_dl, device)
    corrs: List[float] = []
    ratios: List[float] = []
    degenerate = 0
    for b, (xb, yb, meta) in enumerate(by_split[split]):
        if b >= max_batches:
            break
        _, _, mask = tv._sanitize_batch(xb, yb, meta, device)
        x_tok, ent_pad, _ = pack_targets_tokens(yb, mask, device, y_obs_mask=meta.get("y_obs_mask"))
        if x_tok is None:
            continue
        y_true = tv.targets_to_bhnc(yb, mask, device=device).cpu().numpy()
        _, mu, _ = vae(x_tok, ent_pad)
        mm = mu.mean(dim=(0, 1))
        ss = mu.std(dim=(0, 1)).clamp_min(1e-6)
        recon = decode_latents_with_vae(vae, (mu - mm) / ss, entity_pad=ent_pad,
                                        mu_mean=mm, mu_std=ss).cpu().numpy()
        mnp = mask.cpu().numpy().astype(bool)
        for row in range(mnp.shape[0]):
            for e in np.nonzero(mnp[row])[0]:
                yt, yr = y_true[row, :, e, 0], recon[row, :, e, 0]
                if yt.std() < 1e-8:
                    degenerate += 1
                    continue
                # errstate: a constant series makes corrcoef divide by ~0. That case is
                # handled below on its own terms, so the warning is pure noise (physionet
                # would emit hundreds per run).
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = float(np.corrcoef(yr, yt)[0, 1])
                # A window can clear the 1e-8 floor and still be numerically constant
                # (physionet val holds targets with std ~1e-8, median std exactly 0 --
                # sparse clinical series that are carried forward). corrcoef is then
                # undefined, and ONE such window turned the whole mean into nan while
                # inflating std_ratio to 1e6. Synthetic targets are never constant, so
                # this only surfaced once real datasets were admitted. Drop and count.
                if not np.isfinite(corr):
                    degenerate += 1
                    continue
                corrs.append(corr)
                ratios.append(float(yr.std() / yt.std()))
    if not corrs:
        raise RuntimeError("No valid entity-windows found for the round-trip check.")
    return {
        "corr": float(np.mean(corrs)),
        "std_ratio": float(np.mean(ratios)),
        "n": len(corrs),
        # Windows excluded as constant/undefined. A large share means the gate is scored
        # on a minority of the split and the headline corr should not be read alone.
        "n_degenerate": int(degenerate),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 round-trip ceiling (Gate 0).")
    p.add_argument("--dataset-key", type=str, default=None,
                   help="Real dataset preset (noaa_uk, bms_air, physionet, ...). Mutually "
                        "exclusive with --tasks; requires --pred. Scores the preset's frozen "
                        "stage-1 VAE, so run llapdiff-artifact-prep first.")
    p.add_argument("--pred", type=int, default=None,
                   help="Horizon for --dataset-key (must be one of the preset's horizons).")
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="Which split to score. Default: 'val' for --dataset-key (the plan "
                        "puts gates and diagnostics on val), 'test' for --tasks (historical).")
    p.add_argument("--max-batches", type=int, default=2,
                   help="Batches read from the chosen split (default 2).")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Synthetic benchmark tasks (default: synthetic_linear_chirp).")
    p.add_argument("--seeds", nargs="+", type=int, default=(0,),
                   help="Task-mode seeds. Ignored with --dataset-key: the VAE is frozen and "
                        "shared across seeds, and the round-trip reads the posterior mean.")
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--series-length", type=int, default=768)
    p.add_argument("--num-entities", type=int, default=64)
    p.add_argument("--change-point", type=int, default=None)
    p.add_argument("--gap-distribution", choices=("regular", "gamma"), default="gamma")
    p.add_argument("--gap-mean", type=float, default=1.0)
    p.add_argument("--gap-shape", type=float, default=4.0)
    p.add_argument("--sweep-period", type=float, default=None)
    p.add_argument("--phase-spread", type=float, default=2.0 * np.pi)
    p.add_argument("--freq-multiplier", type=float, default=2.0)
    p.add_argument("--base-frequency", type=float, default=None)
    p.add_argument("--laplace-k", type=int, default=256)
    p.add_argument("--chirp-num-basis", type=int, default=None)
    p.add_argument(
        "--data-root", type=str,
        default=str(bench.DEFAULT_WORK_ROOT / "ldt" / "synthetic_data" / "chirp_benchmark"),
    )
    p.add_argument(
        "--artifact-root", type=str,
        default=str(bench.DEFAULT_WORK_ROOT / "ldt" / "synthetic_artifacts" / "chirp_benchmark"),
    )
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Round-trip corr below this fails the gate.")
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    # Every flag below --tasks configures the SYNTHETIC cache, i.e. it defines the data.
    # Silently accepting one in dataset mode would score a different cell than the caller
    # asked for -- the exact failure class this repo keeps recording (B4: artifacts reused
    # across incompatible caches). So they are rejected rather than ignored.
    synthetic_only = (
        "window", "horizon", "series_length", "num_entities", "change_point",
        "gap_distribution", "gap_mean", "gap_shape", "sweep_period", "phase_spread",
        "freq_multiplier", "base_frequency", "laplace_k", "chirp_num_basis",
        "data_root", "artifact_root",
    )
    if args.dataset_key:
        if args.pred is None:
            p.error("--dataset-key requires --pred")
        if args.tasks is not None:
            p.error("--dataset-key and --tasks are mutually exclusive")
        stray = [n for n in synthetic_only if getattr(args, n) != p.get_default(n)]
        if stray:
            p.error(
                "these flags configure the synthetic benchmark cache and have no meaning "
                "with --dataset-key: "
                + ", ".join("--" + n.replace("_", "-") for n in stray)
                + ". A real dataset's geometry comes from its preset."
            )
        if tuple(args.seeds) != (0,):
            print("[gate0] note: --seeds is ignored with --dataset-key (the stage-1 VAE is "
                  "frozen and shared across seeds; the round-trip reads the posterior mean).")
        args.split = args.split or "val"
    else:
        if args.pred is not None:
            p.error("--pred belongs to --dataset-key; synthetic tasks use --horizon")
        args.tasks = args.tasks or ["synthetic_linear_chirp"]
        args.split = args.split or "test"
    return args


def _dataset_cell(args, device: torch.device) -> Dict[str, object]:
    """Gate 0 for a real (dataset, horizon) against its preset's frozen stage-1 VAE."""
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    vae_ckpt = Path(cfg.VAE_CKPT)
    if not vae_ckpt.exists():
        raise SystemExit(
            f"stage-1 VAE not found at {vae_ckpt}\n"
            "Gate 0 scores an already-trained VAE. Build it first with\n"
            f"  llapdiff-artifact-prep --datasets {args.dataset_key}\n"
            "and run this from the repo root (ARTIFACT_ROOT './ldt' is CWD-relative)."
        )
    loaders = build_eval_loaders(cfg)
    stats = roundtrip_correlation(
        cfg, loaders, device, max_batches=int(args.max_batches), split=args.split
    )
    return {
        "task": f"{args.dataset_key}_h{int(args.pred)}",   # legacy label key
        "dataset_key": args.dataset_key,
        "pred": int(args.pred),
        "vae_checkpoint": str(vae_ckpt),
        "seed": None,
        **stats,
    }


def main() -> None:
    args = _parse_args()
    device = set_torch(seed=0, deterministic=False)
    rows: List[Dict[str, object]] = []
    print(f"[gate0] split={args.split}  max_batches={args.max_batches}  "
          f"threshold={args.threshold}")
    print(f"{'cell':<28}{'seed':>5}{'corr':>8}{'std_ratio':>11}{'gate':>7}")

    cells: List[Dict[str, object]] = []
    if args.dataset_key:
        cells.append(_dataset_cell(args, device))
    else:
        for task in args.tasks:
            for seed in args.seeds:
                cfg = bench._configure(task, "chirp", int(seed), args)
                loaders = bench._build_loaders(cfg)
                stats = roundtrip_correlation(
                    cfg, loaders, device,
                    max_batches=int(args.max_batches), split=args.split,
                )
                cells.append({"task": task, "seed": int(seed), **stats})

    for cell in cells:
        passed = cell["corr"] >= float(args.threshold)
        rows.append({**cell, "split": args.split,
                     "threshold": float(args.threshold), "pass": passed})
        seed_col = "-" if cell["seed"] is None else cell["seed"]
        skipped = int(cell.get("n_degenerate", 0))
        note = f"   (n={cell['n']}" + (f", {skipped} constant windows skipped)" if skipped else ")")
        print(f"{str(cell['task']):<28}{seed_col:>5}{cell['corr']:>8.3f}"
              f"{cell['std_ratio']:>11.2f}{'PASS' if passed else 'FAIL':>7}{note}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rows, indent=2, sort_keys=True))
    if not all(r["pass"] for r in rows):
        raise SystemExit(
            f"stage-1 round-trip below the {args.threshold} gate on "
            f"{sum(1 for r in rows if not r['pass'])}/{len(rows)} runs: the pipeline "
            "cannot reconstruct its own targets, so downstream Fig-2 metrics are not "
            "trustworthy. Regenerate the cache with a narrower --phase-spread."
        )


if __name__ == "__main__":
    main()
