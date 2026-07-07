"""T4: sampling wall-clock table across horizons/checkpoints.

Times deterministic DDIM generation (the eval path) and a single denoiser forward
for each checkpoint, using real conditioning from one test batch. The paper target
is a wall-clock roughly flat in the horizon; chirp adds per-query basis evaluation,
so measure rather than assert.

Run:  llapdiff-t4-timing --dataset-key physionet --checkpoints ldt/output/.../llapdiff_pred-12_best.pt ...
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
from llapdiffusion.tools.run_synthetic_regime_shift import _write_rows
from llapdiffusion.trainers import train_val_llapdiff as tv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 sampling wall-clock across horizons.")
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=str, required=True,
                        help="Checkpoints (horizon inferred from 'pred-<H>' in the name).")
    parser.add_argument("--steps", type=int, default=None, help="DDIM steps; default cfg TEST/GEN_STEPS.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-root", type=str,
                        default=str(Path.cwd() / "ldt" / "results" / "t4_timing"))
    return parser.parse_args()


def _infer_pred(path: Path) -> int:
    match = re.search(r"pred-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer horizon from checkpoint name: {path.name}")
    return int(match.group(1))


@torch.no_grad()
def _time_checkpoint(dataset_key: str, ckpt: Path, args) -> Dict[str, object]:
    pred = _infer_pred(ckpt)
    cfg = build_eval_config(dataset_key, pred)
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False)
    loaders, stack = prepare_eval_stack(cfg, ckpt, device=device)
    _, _, test_dl, _ = loaders
    diff_model, vae, summarizer, mu_mean, mu_std = stack
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    steps = int(args.steps) if args.steps is not None else int(sampling["steps"])

    # Real conditioning + latent geometry from one test batch.
    for xb, yb, meta in test_dl:
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
        break
    else:
        raise RuntimeError("No valid test batch found.")

    def _gen():
        return diff_model.generate(
            shape=tuple(mu_norm.shape),
            steps=steps,
            guidance_strength=sampling["guidance_strength"],
            guidance_power=float(sampling["guidance_power"]),
            eta=0.0,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            dt=dt_model,
        )

    def _fwd():
        t1 = torch.ones(mu_norm.shape[0], device=device, dtype=torch.long)
        return diff_model(torch.randn_like(mu_norm), t1,
                          cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
                          dt=dt_model)

    def _bench(fn, repeats):
        fn()  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)
        arr = torch.tensor(times)
        return float(arr.mean()), float(arr.std())

    gen_mean, gen_std = _bench(_gen, int(args.repeats))
    fwd_mean, fwd_std = _bench(_fwd, int(args.repeats))
    modal = getattr(diff_model.model, "denoiser_modal_type", "lti")
    return {
        "checkpoint": str(ckpt),
        "pred": pred,
        "modal_type": modal,
        "batch": int(mu_norm.shape[0]),
        "latent_dim": int(mu_norm.shape[2]),
        "ddim_steps": steps,
        "generate_ms_mean": gen_mean,
        "generate_ms_std": gen_std,
        "forward_ms_mean": fwd_mean,
        "forward_ms_std": fwd_std,
        "device": str(device),
    }


def main() -> None:
    args = _parse_args()
    rows: List[Dict[str, object]] = []
    for ckpt in args.checkpoints:
        row = _time_checkpoint(args.dataset_key, Path(ckpt).resolve(), args)
        rows.append(row)
        print(f"pred={row['pred']} modal={row['modal_type']}: "
              f"generate {row['generate_ms_mean']:.1f}±{row['generate_ms_std']:.1f} ms, "
              f"forward {row['forward_ms_mean']:.1f} ms")

    result_root = Path(args.output_root).resolve()
    tag = f"{args.dataset_key}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _write_rows(rows, result_root / f"{tag}.csv", result_root / f"{tag}.json")
    print(f"completed: {len(rows)} checkpoints -> {result_root / (tag + '.csv')}")


if __name__ == "__main__":
    main()
