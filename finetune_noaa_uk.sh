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
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
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

export CUDA_VISIBLE_DEVICES="2"
RUN_TAG="${RUN_TAG:-fixed_bugs}"
ARMS="${ARMS:-d}"
# Subset of sweep stages, e.g. STAGES="base_lr minsnr_gamma". Empty = all of them.
STAGES="${STAGES:-}"
# Extra tune.py flags, e.g. EXTRA_ARGS="--dry-run" for a pre-flight, or
# EXTRA_ARGS="--select-split val" / "--phase final --final-seeds 0 1 2 3 4".
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR=ldt/results/tune_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/noaa_uk_h168_${RUN_TAG}_$(date +%Y%m%dT%H%M%S).log"

# --------------------------------------------------------------------------
# COST (measured on an RTX A6000, 2026-07-27, chirp core, this cache):
#
#   train step (B=15)            70 ms  -> 77 s per epoch (1086 batches)
#   ONE val CRPS eval            52 min (146 val batches x 25 samples x 64 DDIM
#                                        steps) and it runs every 5 epochs
#   => 89% of wall-clock is the val eval, not training
#
#   earliest possible early stop (157 epochs)   30 h/trial -> 23 trials = 29 days
#   full 600-epoch schedule                    116 h/trial -> 23 trials = 111 days
#
# Peak GPU memory is only ~1.3 GiB, so this is pure time, not capacity.
#
# The eval protocol is what costs, and it is base-config (not preset-stamped),
# so editing llapdiffusion/configs/config.py holds for every subsequent run:
#
#   EVAL_STEPS = 16          # NEW key. _sampling_kwargs(prefix="EVAL") reads it
#                            # before falling back to GEN_STEPS=64, and ONLY the
#                            # trainer's val evals use prefix="EVAL" -- the final
#                            # test read uses prefix="TEST". 4x, no effect on
#                            # reported numbers.
#   NUM_EVAL_SAMPLES = 5     # during tuning this only affects the in-trainer val
#                            # eval: run_trial.py sets FINAL_TEST_EVAL="skip" and
#                            # the harness scores separately via
#                            # eval_sampling.py --num-samples. 5x.
#   DOWNSTREAM_EVAL_EVERY = 10   # halves the eval count; EARLY_STOP counts EVALS,
#   EARLY_STOP = 10              # so halve it too to keep the same patience in
#                                # epochs.
#
# Together those take a trial from ~30 h to ~1 h. Restore 25 / 64 before the
# --phase final read so the reported test numbers stay on protocol.
# --------------------------------------------------------------------------
echo "[run] noaa_uk h=168 | arms=$ARMS | run-tag=$RUN_TAG | GPU=$CUDA_VISIBLE_DEVICES"
echo "[run] stages=${STAGES:-<all>} | log=$LOG"
python - <<'COST'
from llapdiffusion.configs import config
n, s, every = config.NUM_EVAL_SAMPLES, getattr(config, "EVAL_STEPS", config.GEN_STEPS), config.DOWNSTREAM_EVAL_EVERY
per_eval = 146 * n * (s / 64) * 0.85 / 60          # min, from the measured 0.85 s/draw at 64 steps
per_trial = (157 / every) * per_eval + 157 * 77 / 60
print(f"[cost] NUM_EVAL_SAMPLES={n} EVAL_STEPS={s} DOWNSTREAM_EVAL_EVERY={every}")
print(f"[cost] ~{per_eval:.0f} min per val eval -> ~{per_trial/60:.1f} h per trial (earliest early stop)"
      f" -> ~{per_trial*23/1440:.1f} days for 23 trials")
COST

python finetuning/tune.py \
  --dataset-key noaa_uk \
  --preds 168 \
  --arms $ARMS \
  --run-tag "$RUN_TAG" \
  ${STAGES:+--stages $STAGES} \
  ${EXTRA_ARGS} \
  2>&1 | tee "$LOG"
