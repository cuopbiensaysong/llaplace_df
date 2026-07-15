"""Final TEST evaluation of a chosen configuration — the touch-test-once step.

Stamps the chosen sampling knobs onto the config module (picked up by
``build_eval_config``'s clone) and runs the standard ``llapdiff-checkpoint-eval``
protocol (forecast + regular-keep + random-mask imputation) on the test split.
For an EMA-weights winner, a sibling checkpoint is materialized whose "model"
entries are replaced by the EMA shadow (checkpoint-eval always loads "model").

Sampling spec (--sampling-json), every field optional / null = config default:
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
    parser = argparse.ArgumentParser(description="Touch-test-once final evaluation.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampling-json", default="{}")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--imputation-random-mask-ratio", type=float, default=0.30)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny ensemble/steps plumbing check (never for reported numbers).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    sampling = json.loads(args.sampling_json)

    from llapdiffusion.configs import config

    guidance = sampling.get("guidance")
    if guidance is not None:
        # list/tuple = scheduled (g_min, g_max) ramp; number = constant weight.
        config.GUIDANCE_STRENGTH = (
            tuple(float(g) for g in guidance)
            if isinstance(guidance, (list, tuple))
            else float(guidance)
        )
    if sampling.get("steps") is not None:
        config.GEN_STEPS = int(sampling["steps"])
    if sampling.get("guidance_power") is not None:
        config.GUIDANCE_POWER = float(sampling["guidance_power"])
    if sampling.get("dynamic_thresh_p") is not None:
        config.DYNAMIC_THRESH_P = float(sampling["dynamic_thresh_p"])
    if args.smoke:
        config.NUM_EVAL_SAMPLES = 4
        config.GEN_STEPS = 8

    checkpoint = Path(args.checkpoint)
    if str(sampling.get("weights") or "raw") == "ema":
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        shadow = payload.get("ema") or {}
        if not shadow:
            raise ValueError(f"weights=ema requested but checkpoint has no EMA state: {checkpoint}")
        model_state = payload["model"]
        replaced = 0
        for name, tensor in shadow.items():
            if name in model_state:
                model_state[name] = tensor
                replaced += 1
        if replaced == 0:
            raise ValueError("EMA shadow keys did not match any model parameters.")
        checkpoint = checkpoint.with_name(checkpoint.stem + "_emaweights.pt")
        torch.save(payload, checkpoint)
        print(f"[final_eval] materialized EMA-weights checkpoint ({replaced} tensors): {checkpoint}")

    from llapdiffusion.tools import llapdiff_checkpoint_eval as checkpoint_eval

    sys.argv = [
        "llapdiff-checkpoint-eval",
        "--dataset-key", args.dataset_key,
        "--pred", str(args.pred),
        "--checkpoint", str(checkpoint),
        "--imputation-random-mask-ratio", str(args.imputation_random_mask_ratio),
        "--out-json", args.out_json,
    ]
    checkpoint_eval.main()


if __name__ == "__main__":
    main()
