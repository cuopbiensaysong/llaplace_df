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
    parser.add_argument(
        "--generator-seed", default="config",
        help="DDIM sampling-noise seed. 'config' (default) = config.SEED, which is what "
             "eval_sampling.py pins during the sweep -- so the final read is directly "
             "comparable with the selected sampling cell instead of differing by a fresh "
             "noise draw (measured ~0.012 CRPS on physionet h=12, larger than the whole "
             "tuning gain). 'none' restores the legacy unpinned checkpoint-eval behaviour; "
             "or pass an integer.",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny ensemble/steps plumbing check (never for reported numbers).")
    return parser.parse_args()


def _resolve_generator_seed(value: object, config) -> int | None:
    text = str(value).strip().lower()
    if text == "none":
        return None
    if text == "config":
        return int(getattr(config, "SEED", 42))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"--generator-seed must be 'config', 'none', or an integer; got {value!r}"
        ) from exc


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    sampling = json.loads(args.sampling_json)

    from llapdiffusion.configs import config

    generator_seed = _resolve_generator_seed(args.generator_seed, config)

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
    # The weight source is handed to checkpoint-eval, which applies the EMA shadow to
    # the live parameters. This used to materialise a sibling *_emaweights.pt by
    # rewriting the state_dict key-wise, which silently scored a raw/EMA hybrid:
    # model.synthesis.encoder.* aliases model.analysis.* and is absent from the
    # shadow, so the raw alias overwrote the EMA values on load (0.010 CRPS on
    # physionet h=12, larger than the whole tuning gain).
    weights = str(sampling.get("weights") or "raw").strip().lower()

    from llapdiffusion.tools import llapdiff_checkpoint_eval as checkpoint_eval

    sys.argv = [
        "llapdiff-checkpoint-eval",
        "--dataset-key", args.dataset_key,
        "--pred", str(args.pred),
        "--checkpoint", str(checkpoint),
        "--imputation-random-mask-ratio", str(args.imputation_random_mask_ratio),
        "--out-json", args.out_json,
        "--weights", weights,
    ]
    if generator_seed is not None:
        sys.argv += ["--generator-seed", str(generator_seed)]
        print(f"[final_eval] DDIM sampling noise pinned to generator seed {generator_seed}")
    checkpoint_eval.main()


if __name__ == "__main__":
    main()
