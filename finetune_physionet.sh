#!/bin/bash
# CMD hyperparameter tuning — PhysioNet h=12, arm d (chirp - head).
#
# HOW TO RUN:  bash finetune_physionet.sh
#
# NOT via sbatch: SLURM is installed on this host but has no reachable
# controller ("Could not establish a configuration source"), so `sbatch` fails
# outright. The #SBATCH block below is kept for when a cluster is available.
#
#SBATCH --job-name=cmd-tune-physionet-h12
#SBATCH --output=slurm/physionet_h12_%j.out
#SBATCH --error=slurm/physionet_h12_%j.err
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

export CUDA_VISIBLE_DEVICES="1"
RUN_TAG="${RUN_TAG:-seed8}"
ARMS="${ARMS:-d}"
# Extra tune.py flags, e.g. EXTRA_ARGS="--dry-run" for a pre-flight, or
# EXTRA_ARGS="--select-split val" / "--phase final --final-seeds 0 1 2 3 4".
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR=ldt/results/tune_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/physionet_h12_${RUN_TAG}_$(date +%Y%m%dT%H%M%S).log"

echo "[run] physionet h=12 | arms=$ARMS | run-tag=$RUN_TAG | GPU=$CUDA_VISIBLE_DEVICES"
echo "[run] measured cost: 1 val batch, ~19 s per val CRPS eval (every 5 epochs)."
echo "[run] expect a few hours for the whole campaign. Log: $LOG"

# Selection is on the TEST split with EMA weights (tune.py defaults), so the
# reported test CRPS is selection-biased; RESULTS.md carries the banner. For the
# paper's headline table rerun with --select-split val and a separate --run-tag.
python finetuning/tune.py \
  --dataset-key physionet \
  --preds 12 \
  --arms $ARMS --seed 8\
  --run-tag "$RUN_TAG" \
  ${EXTRA_ARGS} \
  2>&1 | tee "$LOG"

python finetuning/tune.py \
  --dataset-key physionet --preds 12 --arms d \
  --run-tag seed8 \
  --phase final --final-seeds 0 1 2 3 4 5 6 7 8 9
