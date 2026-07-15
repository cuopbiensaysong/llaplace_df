"""Evaluate a trained checkpoint over a grid of sampling-time cells.

The harness's own version of ``llapdiff-u1-sweep``, extended with:
  * the config-default cell (guidance null keeps the (1.0, 2.0) ramp), so the
    incumbent protocol is always one of the compared cells;
  * a weights dimension raw|ema (Tier-1 "EMA on/off"): for "ema" the
    checkpoint's EMA shadow is swapped into the model parameters;
  * a fixed generator seed, so every cell sees identical sampling noise;
  * incremental JSON output (rewritten after each cell) so a crash keeps
    completed cells.

Selection stays on val (PREREG rule) — pass --split test only for a final read.

Cell spec (--cells-json): a JSON list of objects; every field optional, null =
keep the config default:
  {"guidance": 1.5, "steps": 32, "weights": "ema",
   "guidance_power": null, "dynamic_thresh_p": null}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampling-cell evaluation of one checkpoint.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cells-json", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override NUM_EVAL_SAMPLES (keep 25 for reported numbers).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    cells = json.loads(args.cells_json)

    import torch
    from llapdiffusion.models.llapdiff_utils import set_torch
    from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
    from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
    from llapdiffusion.trainers import train_val_llapdiff as tv

    cfg = build_eval_config(args.dataset_key, int(args.pred))
    if args.num_samples is not None:
        cfg.NUM_EVAL_SAMPLES = int(args.num_samples)
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False)

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    _, val_dl, test_dl, _ = loaders
    loader = val_dl if args.split == "val" else test_dl
    diff_model, vae, summarizer, mu_mean, mu_std = stack

    payload = torch.load(args.checkpoint, map_location="cpu")
    ema_shadow = payload.get("ema") or None
    raw_params = {n: p.detach().clone() for n, p in diff_model.named_parameters()}

    @torch.no_grad()
    def use_weights(source: str) -> None:
        for name, param in diff_model.named_parameters():
            if source == "ema" and name in ema_shadow:
                param.data.copy_(ema_shadow[name].to(param.device))
            else:
                param.data.copy_(raw_params[name])

    def metric(values: dict, name: str):
        value = values.get(name)
        if value is None:
            return None
        value = float(value)
        return value if value == value else None  # NaN guard

    base_sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def flush() -> None:
        out_path.write_text(json.dumps(
            {
                "dataset": args.dataset_key,
                "pred": int(args.pred),
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "num_samples": int(getattr(cfg, "NUM_EVAL_SAMPLES", 25)),
                "rows": rows,
            },
            indent=2,
        ))

    for cell in cells:
        weights = str(cell.get("weights") or "raw")
        row = {
            "guidance": cell.get("guidance"),
            "steps": cell.get("steps"),
            "weights": weights,
            "guidance_power": cell.get("guidance_power"),
            "dynamic_thresh_p": cell.get("dynamic_thresh_p"),
        }
        if weights == "ema" and not ema_shadow:
            rows.append({**row, "status": "skipped_no_ema"})
            flush()
            continue
        use_weights(weights)

        sampling = dict(base_sampling)
        guidance = cell.get("guidance")
        if guidance is not None:
            # A list/tuple is a scheduled (g_min, g_max) ramp; a number is a
            # constant weight. guidance_power below shapes the ramp only.
            sampling["guidance_strength"] = (
                tuple(float(g) for g in guidance)
                if isinstance(guidance, (list, tuple))
                else float(guidance)
            )
        if cell.get("steps") is not None:
            sampling["steps"] = int(cell["steps"])
        if cell.get("guidance_power") is not None:
            sampling["guidance_power"] = float(cell["guidance_power"])
        if cell.get("dynamic_thresh_p") is not None:
            sampling["dynamic_thresh_p"] = float(cell["dynamic_thresh_p"])
        sampling["eta"] = 0.0

        clip_stats: dict = {}
        values = tv.evaluate_regression(
            diff_model, vae, summarizer, loader,
            device=device, mu_mean=mu_mean, mu_std=mu_std, config=cfg,
            ema=None, self_cond=bool(getattr(cfg, "SELF_COND", False)),
            disable_conditioning=False, verbose=False,
            generator_seed=int(getattr(cfg, "SEED", 42)),
            clip_stats=clip_stats,
            **sampling,
        )
        n_clip = int(clip_stats.get("steps", 0))
        rows.append({
            **row,
            "status": "ok",
            "crps": metric(values, "crps"),
            "mae": metric(values, "mae"),
            "mse": metric(values, "mse"),
            "clip_fraction_mean": (
                float(clip_stats.get("clipped_fraction_sum", 0.0)) / n_clip if n_clip > 0 else 0.0
            ),
        })
        print(f"[eval_sampling] guidance={row['guidance']} steps={row['steps']} "
              f"weights={weights}: crps={rows[-1]['crps']}")
        flush()

    print(f"[eval_sampling] {len(rows)} cells -> {out_path}")


if __name__ == "__main__":
    main()
