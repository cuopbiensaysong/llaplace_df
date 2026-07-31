"""Train ONE tuning trial (stage-3 LLapDiff) with hyperparameter overrides.

Runs ``llapdiffusion.pipeline.main()`` in-process with three adjustments:

1. Config overrides are applied both on the config module AND inside a wrapped
   ``apply_dataset_preset`` — the DEVELOPER_GUIDE §3 footgun: the preset is
   re-applied inside ``run_single_pred`` and would silently reset a plain
   runtime assignment (EPOCHS, MINSNR_GAMMA, MODEL_WIDTH, ...).
2. Stage-3 artifacts are routed under ``ldt/tuning/<run-tag>/...`` so trials
   never collide with each other or with the default (paper) runs. The shared
   frozen VAE/summarizer paths are left untouched: every trial reuses the same
   stage-1/2 artifacts (the parity requirement for fair comparisons).
3. ``FINAL_TEST_EVAL = "skip"``: tuning selection is on the validation split
   only (PREREG rule). The test split is spent once, later, by final_eval.py.
   This also saves the 25-sample test pass on every trial.

Normally invoked by tune.py; can be run standalone for debugging:

  python finetuning/run_trial.py --dataset-key physionet --pred 12 --arm d \
      --seed 0 --trial-id manual0 --run-tag debug \
      --overrides-json '{"config": {"MINSNR_GAMMA": 3.0}}' \
      --summary-json /tmp/manual0_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_FLAGS, REPO_ROOT, TUNING_ARTIFACT_ROOT  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CMD tuning trial.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--pred", type=int, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_FLAGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--overrides-json", default="{}",
                        help='JSON: {"cli": {"predict_type": "x0"}, "config": {"MINSNR_GAMMA": 3.0}}')
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="3-epoch plumbing check with tiny eval settings.")
    return parser.parse_args()


# The in-training validation protocol (DEVELOPER_GUIDE.md §7.6). These knobs make the
# trainer's val CRPS ~61x cheaper, and they are NOT global defaults on purpose: they change
# which epoch the trainer selects, i.e. the epoch this harness then scores. The cheap
# protocol was validated on noaa_uk h=168 only, so it is opted into per campaign rather than
# inherited. finetune_noaa_uk.sh exports these; every other run keeps the full protocol.
#
# Precedence: env profile < --smoke < per-trial --overrides-json.
_EVAL_PROFILE_ENV = {
    "LLAPDIFF_EVAL_STEPS": ("EVAL_STEPS", int),
    "LLAPDIFF_EVAL_NUM_SAMPLES": ("EVAL_NUM_SAMPLES", int),
    "LLAPDIFF_EVAL_MAX_BATCHES": ("EVAL_MAX_BATCHES", int),
    "LLAPDIFF_EVAL_SUBSET_MODE": ("EVAL_SUBSET_MODE", str),
    "LLAPDIFF_EVAL_SEED": ("EVAL_SEED", int),
    "LLAPDIFF_VAL_DIAG_EVERY": ("VAL_DIAG_EVERY", int),
}


def _apply_eval_profile_env(config) -> None:
    """Stamp any LLAPDIFF_EVAL_* / LLAPDIFF_VAL_DIAG_EVERY env overrides onto the config.

    These are base-config keys (not preset-stamped), so a plain setattr survives both
    applications of apply_dataset_preset -- see DEVELOPER_GUIDE.md §3.
    """
    applied = {}
    for env_name, (attr, cast) in _EVAL_PROFILE_ENV.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            value = cast(raw)
        except ValueError as exc:
            raise SystemExit(f"{env_name}={raw!r} is not a valid {cast.__name__}") from exc
        setattr(config, attr, value)
        applied[attr] = value
    if applied:
        print(f"[run_trial] validation-protocol profile from env: {applied}", flush=True)


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)  # ARTIFACT_ROOT ("./ldt") is CWD-relative

    overrides = json.loads(args.overrides_json)
    cli_overrides = dict(overrides.get("cli") or {})
    config_overrides = dict(overrides.get("config") or {})

    from llapdiffusion.configs import config
    import llapdiffusion.pipeline as pipeline

    _apply_eval_profile_env(config)

    if args.smoke:
        config_overrides.setdefault("EPOCHS", 3)
        config.NUM_EVAL_SAMPLES = 4
        config.GEN_STEPS = 8
        config.DOWNSTREAM_EVAL_EVERY = 1  # 3 epochs must still exercise the best-ckpt path
        # Drop the per-role validation overrides so the smoke actually uses the tiny
        # settings above (EVAL_* would otherwise win for the trainer's val evals).
        config.EVAL_STEPS = None
        config.EVAL_NUM_SAMPLES = None
        config.EVAL_MAX_BATCHES = 0
    # The harness scores the checkpoint itself (eval_sampling/final_eval), so the
    # trainer's own final test pass is redundant.
    config.FINAL_TEST_EVAL = "skip"
    for name, value in config_overrides.items():
        setattr(config, name, value)

    # The trainer's checkpoint selection and early stopping both read
    # `current_primary_metric`, which for PRIMARY_EVAL_METRIC="crps" is only populated
    # when the val CRPS eval actually runs (DOWNSTREAM_EVAL_EVERY > 0). With CRPS
    # selected but the eval disabled, no best checkpoint is ever saved and early stopping
    # never fires — the run silently trains all EPOCHS and hands back the last epoch.
    primary_metric = str(getattr(config, "PRIMARY_EVAL_METRIC", "crps")).strip().lower()
    if primary_metric == "crps" and int(getattr(config, "DOWNSTREAM_EVAL_EVERY", 0) or 0) <= 0:
        raise SystemExit(
            "Incoherent trainer config: PRIMARY_EVAL_METRIC='crps' requires "
            "DOWNSTREAM_EVAL_EVERY > 0 (otherwise val CRPS is never computed: no best "
            "checkpoint is saved, early stopping never fires, and the last epoch is "
            "scored). Fix in llapdiffusion/configs/config.py: set DOWNSTREAM_EVAL_EVERY=5, "
            "or set PRIMARY_EVAL_METRIC='val_diag_mse_raw'."
        )

    trial_root = (
        TUNING_ARTIFACT_ROOT / args.run_tag / f"{args.dataset_key}_h{args.pred}"
        / args.arm / args.trial_id
    )

    original_apply = pipeline.apply_dataset_preset

    def apply_preset_with_overrides(cfg, key, *, pred=None):
        out = original_apply(cfg, key, pred=pred)
        for name, value in config_overrides.items():
            setattr(cfg, name, value)
        # Re-root stage-3 artifacts only; VAE_/SUM_ paths stay shared.
        cfg.OUT_DIR = str(trial_root / "output")
        cfg.CKPT_DIR = str(trial_root / "checkpoints")
        cfg.POLE_PLOT_DIR = str(trial_root / "output" / "pole_plots")
        return out

    pipeline.apply_dataset_preset = apply_preset_with_overrides

    argv = [
        "llapdiff-train",
        "--dataset-key", args.dataset_key,
        "--preds", str(args.pred),
        *ARM_FLAGS[args.arm],
        "--seed", str(args.seed),
        "--summary-json", args.summary_json,
    ]
    predict_type = cli_overrides.pop("predict_type", None)
    if predict_type and str(predict_type) != "v":
        argv += ["--predict-type", str(predict_type)]
    if cli_overrides:
        raise ValueError(f"Unsupported cli overrides: {sorted(cli_overrides)}")

    sys.argv = argv
    pipeline.main()


if __name__ == "__main__":
    main()
