"""Stage-1 round-trip check (Fig-2 "Gate 0").

Encodes the TRUE targets through the trained set-VAE and decodes them straight back
-- no forecasting -- and reports the correlation with the originals. This is the
upper bound on any denoiser built on that VAE: if stage 1 cannot reconstruct its own
targets, no downstream chirp-vs-LTI comparison is meaningful.

It exists because H2 silently sat under a 0.62 ceiling for exactly this reason: the
set-VAE mean-pools across entities while the generator drew an independent U(0, 2pi)
phase per entity, so the panel cancelled before the latent. ``--phase-spread`` fixes
the generator; this tool is the gate that confirms it on the run's own cache.

Usage mirrors the benchmark's cache/artifact addressing:

    llapdiff-stage1-roundtrip --tasks synthetic_linear_chirp --seeds 0 \
        --sweep-period 144 --phase-spread 0.393 \
        --artifact-root ldt/synthetic_artifacts/chirp_benchmark_phasefix

Prints one PASS/FAIL row per (task, seed) against ``--threshold`` (default 0.85).
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
    cfg, loaders, device: torch.device, *, max_batches: int = 2
) -> Dict[str, float]:
    """Encode->decode the true targets and correlate with the originals.

    Returns per-entity-window mean ``corr`` and ``std_ratio`` (reconstructed std over
    true std), plus the sample count. Normalization uses the batch's own latent
    moments so the check is independent of stored mu stats.
    """
    train_dl, _, test_dl, _ = loaders
    vae = _build_vae(cfg, train_dl, device)
    corrs: List[float] = []
    ratios: List[float] = []
    for b, (xb, yb, meta) in enumerate(test_dl):
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
                    continue
                corrs.append(float(np.corrcoef(yr, yt)[0, 1]))
                ratios.append(float(yr.std() / yt.std()))
    if not corrs:
        raise RuntimeError("No valid entity-windows found for the round-trip check.")
    return {
        "corr": float(np.mean(corrs)),
        "std_ratio": float(np.mean(ratios)),
        "n": len(corrs),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 round-trip ceiling (Fig-2 Gate 0).")
    p.add_argument("--tasks", nargs="+", default=["synthetic_linear_chirp"])
    p.add_argument("--seeds", nargs="+", type=int, default=(0,))
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = set_torch(seed=0, deterministic=False)
    rows: List[Dict[str, object]] = []
    print(f"{'task':<28}{'seed':>5}{'corr':>8}{'std_ratio':>11}{'gate':>7}")
    for task in args.tasks:
        for seed in args.seeds:
            cfg = bench._configure(task, "chirp", int(seed), args)
            loaders = bench._build_loaders(cfg)
            stats = roundtrip_correlation(cfg, loaders, device)
            passed = stats["corr"] >= float(args.threshold)
            rows.append({"task": task, "seed": int(seed), **stats,
                         "threshold": float(args.threshold), "pass": passed})
            print(f"{task:<28}{seed:>5}{stats['corr']:>8.3f}{stats['std_ratio']:>11.2f}"
                  f"{'PASS' if passed else 'FAIL':>7}   (n={stats['n']})")
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
