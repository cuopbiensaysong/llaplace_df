"""U1 calibration sweep: guidance strength x DDIM steps on the VALIDATION split.

Evaluates one trained checkpoint under a grid of sampling knobs (the Tier-1 levers
that reshape the predictive distribution without retraining) and logs the dynamic-
threshold clip rate per cell. Selection stays on val — test is touched once, later,
with the chosen configuration (pre-registration rule).

Run:
  llapdiff-u1-sweep --dataset-key physionet --pred 12 --checkpoint <ckpt> \
      --guidance 1.0 1.25 1.5 2.0 --steps 16 32 64
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
from llapdiffusion.tools.run_synthetic_regime_shift import _write_rows
from llapdiffusion.trainers import train_val_llapdiff as tv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guidance x DDIM-steps calibration sweep on the val split (U1)."
    )
    parser.add_argument("--dataset-key", type=str, required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--guidance", nargs="+", type=float, default=(1.0, 1.25, 1.5, 2.0))
    parser.add_argument("--steps", nargs="+", type=int, default=(16, 32, 64))
    parser.add_argument("--dynamic-thresh-p", type=float, default=None,
                        help="Override the dynamic-thresholding quantile for the whole sweep "
                        "(default: config, i.e. 0.0 = off). The plan's U1 clip check is "
                        "--dynamic-thresh-p 0.995: does the post-head-removal x0 brush the "
                        "threshold? Read clip_fraction_mean in the output rows.")
    parser.add_argument("--dynamic-thresh-max", type=float, default=None,
                        help="Override the thresholding clamp ceiling (default: config, 1.0).")
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Keep 'val' for tuning; 'test' only for the single final read.")
    parser.add_argument("--output-root", type=str,
                        default=str(Path.cwd() / "ldt" / "results" / "u1_sweep"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _metric(payload: Dict[str, object], name: str) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> None:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False)

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    _, val_dl, test_dl, _ = loaders
    loader = val_dl if args.split == "val" else test_dl
    diff_model, vae, summarizer, mu_mean, mu_std = stack

    base_sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    rows: List[Dict[str, object]] = []
    for w in args.guidance:
        for steps in args.steps:
            sampling = dict(base_sampling)
            sampling["guidance_strength"] = float(w)
            sampling["steps"] = int(steps)
            sampling["eta"] = 0.0
            if args.dynamic_thresh_p is not None:
                sampling["dynamic_thresh_p"] = float(args.dynamic_thresh_p)
            if args.dynamic_thresh_max is not None:
                sampling["dynamic_thresh_max"] = float(args.dynamic_thresh_max)
            clip_stats: Dict[str, float] = {}
            payload = tv.evaluate_regression(
                diff_model, vae, summarizer, loader,
                device=device, mu_mean=mu_mean, mu_std=mu_std, config=cfg,
                ema=None, self_cond=bool(getattr(cfg, "SELF_COND", False)),
                disable_conditioning=False, verbose=bool(args.verbose),
                clip_stats=clip_stats,
                **sampling,
            )
            n_clip = int(clip_stats.get("steps", 0))
            rows.append(
                {
                    "guidance": float(w),
                    "steps": int(steps),
                    "split": args.split,
                    "dynamic_thresh_p": float(sampling.get("dynamic_thresh_p", 0.0)),
                    "crps": _metric(payload, "crps"),
                    "mae": _metric(payload, "mae"),
                    "mse": _metric(payload, "mse"),
                    "clip_fraction_mean": (
                        float(clip_stats.get("clipped_fraction_sum", 0.0)) / n_clip
                        if n_clip > 0
                        else 0.0
                    ),
                }
            )
            print(f"guidance={w} steps={steps}: crps={rows[-1]['crps']} "
                  f"clip_frac={rows[-1]['clip_fraction_mean']:.4g}")

    result_root = Path(args.output_root).resolve()
    tag = f"{args.dataset_key}_h{int(args.pred)}_{Path(args.checkpoint).stem}_{args.split}"
    _write_rows(rows, result_root / f"{tag}.csv", result_root / f"{tag}.json")
    (result_root / f"{tag}_meta.json").write_text(
        __import__("json").dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(args.checkpoint),
                "grid": {"guidance": list(map(float, args.guidance)),
                         "steps": list(map(int, args.steps))},
            },
            indent=2,
        )
    )
    print(f"completed: {len(rows)} cells -> {result_root / (tag + '.csv')}")


if __name__ == "__main__":
    main()
