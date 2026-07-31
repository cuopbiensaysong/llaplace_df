#!/bin/bash
# CMD hyperparameter tuning — NOAA-UK h=168, arm d (chirp - head).
#
# HOW TO RUN:  bash finetune_noaa_uk.sh
#
# NOT via sbatch: SLURM is installed on this host but has no reachable
# controller ("Could not establish a configuration source"), so `sbatch` fails
# outright. The #SBATCH block below is kept for when a cluster is available.
#
# ⚠️ READ THE COST NOTE BELOW BEFORE LAUNCHING. At the shipped evaluation
#    protocol this campaign does not finish in a usable time.
#
#SBATCH --job-name=cmd-tune-noaa-uk-h168
#SBATCH --output=slurm/noaa_uk_h168_%j.out
#SBATCH --error=slurm/noaa_uk_h168_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

set -euo pipefail

# --------------------------------------------------------------------------
# Environment. BOTH lines are load-bearing (CLAUDE.md).
#
# tune.py launches run_trial.py / eval_sampling.py as subprocesses, so their
# sys.path[0] is finetuning/ -- NOT the repo root. Without PYTHONPATH the
# editable install resolves `llapdiffusion` to the sibling checkout
#   /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace_df  @ 19a6a24
# which is 3 commits behind and contains NONE of the B1-B15 fixes (no
# half_integer chirp basis, no COND_NORM_MODE, no rho cap). It fails silently:
# the run trains and reports normally, on the wrong code.
# --------------------------------------------------------------------------
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
# Points at THIS checkout, which is where the speed work now lives after the merge into
# main. It used to point at .../speed_up/llaplace_df; running that copy now would silently
# execute the pre-merge branch. Override with LLAPDIFF_SRC=... to run a different tree.
export LLAPDIFF_SRC="${LLAPDIFF_SRC:-/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df}"
export PYTHONPATH="$LLAPDIFF_SRC"
cd "$LLAPDIFF_SRC"

# Fail before spending GPU if the wrong checkout would be imported.
python - <<'GUARD'
import os, pathlib, sys
sys.path[0] = str(pathlib.Path("finetuning").resolve())  # mimic a trial subprocess
import llapdiffusion
want, got = os.environ["LLAPDIFF_SRC"], llapdiffusion.__file__
if not got.startswith(want):
    raise SystemExit(f"[guard] WRONG CHECKOUT: {got}\n[guard] expected under {want}")
print(f"[guard] llapdiffusion -> {got}")
GUARD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --------------------------------------------------------------------------
# In-training validation protocol (DEVELOPER_GUIDE.md §7.6).
#
# These are NOT config.py defaults: they change which epoch the trainer selects
# -- the epoch the harness then scores -- and they were measured and validated
# on noaa_uk h=168 ONLY. Enabling them globally would move checkpoint selection
# on every dataset, including physionet h=12, whose val split is five windows
# (B19). run_trial.py reads them from the environment, so they apply to this
# campaign and nothing else.
#
# Effect here: 146 batches x 25 samples x 64 steps x 2 forwards (CFG runs
# conditional AND unconditional) = ~110 min per val eval  ->  ~61x cheaper.
# Reported numbers are untouched: they all resolve TEST_*, not EVAL_*.
#
# 🔴 Before reusing this profile on a NEW (dataset, horizon): score >=8
#    checkpoints under both protocols on val and require Spearman rho >= 0.9,
#    or you have made the campaign select the wrong hyperparameters, faster.
# --------------------------------------------------------------------------
export LLAPDIFF_EVAL_STEPS="${LLAPDIFF_EVAL_STEPS:-16}"
export LLAPDIFF_EVAL_NUM_SAMPLES="${LLAPDIFF_EVAL_NUM_SAMPLES:-5}"
export LLAPDIFF_EVAL_MAX_BATCHES="${LLAPDIFF_EVAL_MAX_BATCHES:-48}"
export LLAPDIFF_EVAL_SUBSET_MODE="${LLAPDIFF_EVAL_SUBSET_MODE:-stride}"
export LLAPDIFF_EVAL_SEED="${LLAPDIFF_EVAL_SEED:-4242}"
export LLAPDIFF_VAL_DIAG_EVERY="${LLAPDIFF_VAL_DIAG_EVERY:-5}"

# A fresh tag: this campaign selects on val (below), so it must not land in the
# same results tree as the earlier test-selected runs.
RUN_TAG="${RUN_TAG:-speed_up}"
ARMS="${ARMS:-d}"
# tune.py defaults to --select-split test, which costs 314 test batches per trial
# AND makes the reported test CRPS selection-biased (see the banner in
# finetuning/README.md). val is 2.15x cheaper and is what cmd_plan_v2.md §1
# pre-registers.
SELECT_SPLIT="${SELECT_SPLIT:-val}"
# Subset of sweep stages, e.g. STAGES="base_lr minsnr_gamma". Empty = all of them.
STAGES="${STAGES:-}"
# Extra tune.py flags, e.g. EXTRA_ARGS="--dry-run" for a pre-flight, or
# EXTRA_ARGS="--select-split val" / "--phase final --final-seeds 0 1 2 3 4".
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR=ldt/results/tune_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/noaa_uk_h168_${RUN_TAG}_$(date +%Y%m%dT%H%M%S).log"

# --------------------------------------------------------------------------
# COST (re-measured on an RTX A6000, 2026-07-28, chirp core, this cache).
#
# CORRECTION to the previous note in this file: it charged ONE forward per DDIM
# step, but GUIDANCE_STRENGTH=(1.0, 2.0) makes CFG active, so every step runs
# TWO forwards (conditional + unconditional, llapdiff.py `cfg_active`). The old
# "52 min / 30 h per trial / 29 days" figures were therefore ~2x optimistic.
#
#   denoiser forward, B=15 rows   14.11 ms fp32 / 10.76 ms TF32 (measured)
#   train step (B=15)             70 ms -> 77 s per epoch (1086 batches)
#   ONE full val CRPS eval        146 batches x 25 samples x 64 steps x 2 fwd
#                                 = 110 min, and it runs every 5 epochs
#   => ~94% of wall-clock is the val eval, not training
#
#   earliest possible early stop (157 epochs)   60 h/trial -> 23 trials = 60 days
#
# Peak GPU memory is only ~1.3 GiB, so this is pure time, not capacity.
#
# The fix is that the trainer's val CRPS is used ONLY to pick the best epoch and
# to early-stop -- it does not need the reported protocol. `_sampling_kwargs`
# resolves EVAL_* for the trainer's val evals and TEST_* for everything that
# produces a reported number, so the two are separable. These are base-config
# (not preset-stamped), so an edit in llapdiffusion/configs/config.py holds:
#
#   EVAL_STEPS       = 16   # vs GEN_STEPS=64            -> 4x
#   EVAL_NUM_SAMPLES = 5    # vs NUM_EVAL_SAMPLES=25     -> 5x
#   EVAL_MAX_BATCHES = 48   # strided over the 146       -> 3x
#   EVAL_SEED        = 4242 # pairs the estimate across epochs (was global RNG)
#
# = 61x off the val eval, taking a trial from ~60 h to ~2.8 h. NONE of these
# reach a reported number: the harness scores checkpoints via eval_sampling.py
# and --phase final re-reads test, both at the full 25 x 64 protocol.
#
# Do NOT raise DOWNSTREAM_EVAL_EVERY without lowering EARLY_STOP: EARLY_STOP
# counts EVALS, so at every=10 the patience becomes 200 epochs and trials get
# LONGER, not shorter.
# --------------------------------------------------------------------------
echo "[run] noaa_uk h=168 | arms=$ARMS | run-tag=$RUN_TAG | GPU=$CUDA_VISIBLE_DEVICES"
echo "[run] stages=${STAGES:-<all>} | select-split=$SELECT_SPLIT | log=$LOG"
python - <<'COST'
from llapdiffusion.configs import config
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset
from llapdiffusion.trainers import train_val_llapdiff as tv

apply_dataset_preset(config, "noaa_uk", pred=168)
ev = tv._sampling_kwargs(config, prefix="EVAL")
te = tv._sampling_kwargs(config, prefix="TEST")
default_n = int(getattr(config, "NUM_EVAL_SAMPLES", 25))
ev_n, te_n = int(ev.get("num_samples", default_n)), int(te.get("num_samples", default_n))
cap = int(getattr(config, "EVAL_MAX_BATCHES", 0) or 0)
nb = min(cap, 146) if cap > 0 else 146
# Measured per-forward at B=15 rows; x2 because CFG runs cond + uncond per step.
fwd_ms = 10.76 if getattr(config, "ALLOW_TF32", False) else 14.11
every = int(config.DOWNSTREAM_EVAL_EVERY)
# Measured 2026-07-28 on this cache/GPU: 82.4 s per 1086-batch epoch (76 ms/step).
# Training did NOT get faster with the 2026-07-28 changes -- the step is dominated by real
# GPU compute in two small cond/uncond passes, where TF32 buys little at ~15 rows. It is now
# the majority of a trial, so the next lever is LAPLACE_K (see CMD_RUNBOOK §3), not the eval.
epoch_s = 82.4
per_eval = nb * ev_n * int(ev["steps"]) * 2 * fwd_ms / 1000 / 60            # minutes
per_trial = (157 / every) * per_eval + 157 * epoch_s / 60                  # minutes
print(f"[cost] val  protocol: steps={ev['steps']} samples={ev_n} batches={nb}/146 "
      f"every={every} epochs  seed={getattr(config, 'EVAL_SEED', None)}")
print(f"[cost] test protocol: steps={te['steps']} samples={te_n}  <- the reported numbers")
print(f"[cost] ~{per_eval:.1f} min per val eval -> ~{per_trial/60:.1f} h per trial "
      f"(earliest early stop) -> ~{per_trial*23/1440:.1f} days for 23 trials")
COST

python finetuning/tune.py \
  --dataset-key noaa_uk \
  --preds 168 \
  --arms $ARMS \
  --run-tag "$RUN_TAG" \
  --select-split "$SELECT_SPLIT" \
  ${STAGES:+--stages $STAGES} \
  ${EXTRA_ARGS} \
  2>&1 | tee "$LOG"
