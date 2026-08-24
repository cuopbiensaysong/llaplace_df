#!/usr/bin/env bash
# Nugget ablation on the WIDE LR grid, one (cell, arm, seed) per GPU.
#
#   bash finetuning/run_abl_widegrid.sh <cell> <arm> <seed> <gpu> [--dry-run]
#     cell: bms_air | noaa_uk      arm: base | nonugget
#
# Why this exists rather than `run_campaign_job.sh abl_*`:
#   1. that branch HARDCODES `--lr-grid 3e-4 1e-4 3e-5 1e-5` and ignores $LR_GRID, and both
#      arms selected its lower bound on 10/10 bms_air seeds, so the arms may be under-trained;
#   2. its lock is overridden by the bare-lock migration shim on noaa_uk, which silently
#      skips any re-run of a seed that already has a pre-namespace lock.
# Writes NEW filenames (`*_grid7_s<seed>.json`); the grid5 outputs are left untouched.
set -u
CELL="${1:?cell}"; ARM="${2:?arm}"; SEED="${3:?seed}"; GPU="${4:?gpu}"; DRY="${5:-}"
case "$CELL" in bms_air) CHUNK=32 ;; noaa_uk) CHUNK=128 ;; *) echo "bad cell"; exit 2 ;; esac
case "$ARM"  in base) EXTRA=() ;; nonugget) EXTRA=(--no-nugget) ;; *) echo "bad arm"; exit 2 ;; esac

source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
export PYTHONUNBUFFERED=1
cd "$LLAPDIFF_SRC" || exit 1
RESOLVED=$(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')
case "$RESOLVED" in "$LLAPDIFF_SRC"/*) ;; *) echo "FATAL: resolved $RESOLVED"; exit 2 ;; esac

# Parity with run_campaign_job.sh's environment.
export LLAPDIFF_EVAL_STEPS="${LLAPDIFF_EVAL_STEPS:-16}"
export LLAPDIFF_EVAL_NUM_SAMPLES="${LLAPDIFF_EVAL_NUM_SAMPLES:-5}"
export LLAPDIFF_EVAL_MAX_BATCHES="${LLAPDIFF_EVAL_MAX_BATCHES:-48}"
export LLAPDIFF_EVAL_SUBSET_MODE="${LLAPDIFF_EVAL_SUBSET_MODE:-stride}"
export LLAPDIFF_EVAL_SEED="${LLAPDIFF_EVAL_SEED:-4242}"
export LLAPDIFF_VAL_DIAG_EVERY="${LLAPDIFF_VAL_DIAG_EVERY:-5}"

OUT="finetuning/results/cmd_gp/${CELL}_h168/abl_${ARM}_grid7_s${SEED}.json"
if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; exit 0; fi
CMD=(python -m llapdiffusion.tools.run_option_a_probe
  --dataset-key "$CELL" --pred 168 --eval-chunk "$CHUNK"
  --heads chirp_gp --summary-tokens 16
  --lr-grid 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7 --weight-decay 1e-3
  --epochs-mse 150 --epochs-nll 80 --patience 20 --num-samples 25 --seed "$SEED"
  --tag "abl_${ARM}_grid7" "${EXTRA[@]}" --out-json "$OUT")
if [ "$DRY" = "--dry-run" ]; then echo "CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"; exit 0; fi
echo "[start] $CELL abl_$ARM seed=$SEED gpu=$GPU -> $OUT"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "${OUT%.json}.log" 2>&1
echo "[done] rc=$?"
