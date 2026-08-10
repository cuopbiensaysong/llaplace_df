"""Train stage-1 (VAE) and/or stage-2 (summarizer) for ONE dataset x horizon cell.

`llapdiff-artifact-prep` loops over every horizon in the preset and `llapdiff-train` always
continues into stage 3, so neither can retrain a single cell's conditioning artifacts -- which
is exactly what an architecture change to either stage needs (B20, B21).

Both stages are shared and frozen across arms and seeds, so an artifact must never be
overwritten silently: a stale file would be picked up by every later stage with no warning
(the loaders are strict=True on shape, not on provenance). This tool refuses to overwrite by
default, and the architecture flags below route new artifacts to their own filenames.

    llapdiff-stage-pretrain --dataset-key noaa_uk --pred 168 --stage summarizer \
        --decode-mode token --epochs 200

    llapdiff-stage-pretrain --dataset-key noaa_uk --pred 168 --stage vae --entity-encode
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llapdiffusion.configs.config_utils import (
    clone_config,
    make_jsonable,
    refresh_artifact_paths,
)
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset, dataset_keys
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.logging_utils import apply_verbosity
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.models.summarizer import DECODE_MODES
from llapdiffusion.trainers import train_val_latent, train_val_summarizer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain one cell's stage-1/stage-2 artifacts.")
    p.add_argument("--dataset-key", required=True, choices=sorted(dataset_keys()))
    p.add_argument("--pred", type=int, required=True)
    p.add_argument("--stage", choices=("vae", "summarizer", "both"), default="summarizer")
    p.add_argument("--decode-mode", choices=sorted(DECODE_MODES), default=None,
                   help="Stage-2 pretraining objective (B21 §6). 'mean' decodes from the token "
                        "mean and can supervise only that one direction; 'token' decodes per "
                        "time step over all S tokens. Default: the config's SUM_DECODE_MODE.")
    p.add_argument("--entity-encode", action="store_true",
                   help="Stage-1: add the encoder-side entity embedding, so the pooled latent "
                        "is not permutation-invariant in the entities (B20/B5).")
    p.add_argument("--latent-channels", type=int, default=None,
                   help="Override VAE_LATENT_CHANNELS. It is a per-dataset preset field sized "
                        "when the encoder could not tell entities apart; with --entity-encode "
                        "the width may no longer be the right one. NOTE it also appears in the "
                        "STAGE-2 filename even though the summarizer does not depend on it, so "
                        "probes on a re-widened latent need an explicit --sum-ckpt.")
    p.add_argument("--vae-beta", type=float, default=None,
                   help="Override VAE_BETA (KL weight). Raising it buys a smoother, "
                        "lower-capacity code, which may be more linearly predictable from "
                        "history -- the quantity that caps C.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Epoch cap for the stage being trained (SUM_EPOCHS for stage 2; EPOCHS "
                        "for stage 1, which is what its loop reads). Early stopping still "
                        "applies: SUM_PATIENCE / VAE_MAX_PATIENCE.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--artifact-tag", default=None,
                   help="Append '_<tag>' to the artifact filenames. Use it for a run that must "
                        "not claim the canonical name -- e.g. retraining the LEGACY objective "
                        "in this tree as a matched control for a new one, which is otherwise "
                        "the same filename as the shipped artifact.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing an artifact that already exists. Off by default: the "
                        "frozen artifacts are shared by every recorded result.")
    p.add_argument("--json-out", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _build_config(args: argparse.Namespace):
    cfg = clone_config()
    if args.decode_mode is not None:
        cfg.SUM_DECODE_MODE = str(args.decode_mode)
    if args.entity_encode:
        cfg.VAE_ENTITY_ENCODE = True
    if args.artifact_tag:
        # Must be set BEFORE the paths are derived: the stage-1 trainer rebuilds its own save
        # path from the config rather than reading VAE_CKPT, so a tag applied afterwards would
        # not reach it -- and an untagged legacy-architecture control run would then overwrite
        # the shipped artifact.
        cfg.ARTIFACT_TAG = str(args.artifact_tag)
    apply_dataset_preset(cfg, args.dataset_key, pred=int(args.pred))
    # After the preset: it sets VAE_LATENT_CHANNELS itself, and both artifact names derive
    # from it, so the override has to land before the paths are refreshed.
    if args.latent_channels is not None:
        cfg.VAE_LATENT_CHANNELS = int(args.latent_channels)
    if args.vae_beta is not None:
        cfg.VAE_BETA = float(args.vae_beta)
    refresh_artifact_paths(cfg)
    cfg.COVERAGE = 0.0
    cfg.TARGET_COL = None
    cfg.TARGET_COLS = None
    if args.seed is not None:
        cfg.SEED = int(args.seed)
    if args.epochs is not None:
        cfg.SUM_EPOCHS = int(args.epochs)
        cfg.EPOCHS = int(args.epochs)          # train_val_latent's loop reads EPOCHS
        cfg.VAE_MIN_EPOCHS = min(int(getattr(cfg, "VAE_MIN_EPOCHS", 40)), int(args.epochs))
    apply_verbosity(cfg, verbose=args.verbose, debug=args.debug)
    return cfg


def _build_loaders(cfg):
    run_experiment = resolve_run_experiment(cfg.DATA_DIR)
    batch_size = int(getattr(cfg, "BATCH_SIZE", getattr(cfg, "DATES_PER_BATCH", 1)))
    return run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=cfg.date_batching,
        dates_per_batch=batch_size,
        K=cfg.WINDOW,
        H=cfg.PRED,
        coverage=cfg.COVERAGE,
        batch_size=batch_size,
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=bool(getattr(cfg, "exact_timestamp_batches", True)),
        target_col=None,
        target_cols=None,
    )


def _guard_existing(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"{label} already exists: {path}\n"
            "Refusing to overwrite a frozen artifact. Pass --overwrite if that is what you "
            "want, or change the architecture flags so the new run lands on its own filename."
        )


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False,
                       allow_tf32=bool(getattr(cfg, "ALLOW_TF32", False)))

    want_vae = args.stage in ("vae", "both")
    want_sum = args.stage in ("summarizer", "both")
    if want_vae:
        _guard_existing(Path(cfg.VAE_CKPT), overwrite=args.overwrite, label="stage-1 artifact")
    if want_sum:
        _guard_existing(Path(cfg.SUM_CKPT), overwrite=args.overwrite, label="stage-2 artifact")

    print(f"[pretrain] {args.dataset_key} h={cfg.PRED} device={device} seed={getattr(cfg, 'SEED', 42)}")
    print(f"[pretrain] stage-1 {cfg.VAE_CKPT}  (entity_encode={bool(getattr(cfg, 'VAE_ENTITY_ENCODE', False))})")
    print(f"[pretrain] stage-2 {cfg.SUM_CKPT}  (decode_mode={getattr(cfg, 'SUM_DECODE_MODE', 'mean')}, "
          f"epochs={cfg.SUM_EPOCHS}, lr={cfg.SUM_LR}, amp={cfg.SUM_AMP})")

    train_dl, val_dl, test_dl, sizes = _build_loaders(cfg)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": args.dataset_key,
        "pred": int(cfg.PRED),
        "seed": int(getattr(cfg, "SEED", 42)),
        "sum_decode_mode": str(getattr(cfg, "SUM_DECODE_MODE", "mean")),
        "vae_entity_encode": bool(getattr(cfg, "VAE_ENTITY_ENCODE", False)),
        "vae_ckpt": str(cfg.VAE_CKPT),
        "sum_ckpt": str(cfg.SUM_CKPT),
        "sizes": list(sizes) if sizes is not None else None,
    }

    if want_vae:
        report["vae"] = train_val_latent.run(
            train_dl=train_dl, val_dl=val_dl, test_dl=test_dl, sizes=sizes, config=cfg,
        )
    if want_sum:
        report["summarizer"] = train_val_summarizer.run(
            train_loader=train_dl, val_loader=val_dl, test_loader=test_dl, sizes=sizes, config=cfg,
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(make_jsonable(report), indent=2), encoding="utf-8")
        print(f"[pretrain] wrote {out}")
    print(json.dumps(make_jsonable({k: v for k, v in report.items() if k != "sizes"}), indent=2)[:2000])


if __name__ == "__main__":
    main()
