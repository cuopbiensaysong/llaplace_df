#!/usr/bin/env bash
# GPU 0 queue: finish the cheap one-pass val seeds (Family A needs 10), then take
# diffusion seeds 3 and 5 (Family B needs 6 total, i.e. seeds 0-5).
#
# One job at a time per GPU, deliberately: running three probes across two GPUs is what
# OOM-killed seed 4. Sequential per GPU is slower per job and finishes sooner overall.
set -u
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
cd "$LLAPDIFF_SRC" || exit 1
export CUDA_VISIBLE_DEVICES=0

R=finetuning/results/cmd_gp/${CELL:-noaa_uk_h168}
LOG=$R/queue_gpu0.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "guard: $(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')"

for S in 5 6 7 8 9; do
  if [ -f "$R/R10_option_a_s${S}_grid5.json" ]; then say "one-pass seed $S already done"; continue; fi
  say "one-pass seed $S start"
  python -m llapdiffusion.tools.run_option_a_probe \
    --heads chirp_gp diag_gaussian lowrank_gaussian student_t \
    --summary-tokens 16 --lr-grid 3e-4 1e-4 3e-5 1e-5 --weight-decay 1e-3 \
    --epochs-mse 150 --epochs-nll 80 --patience 20 --num-samples 25 --seed "$S" \
    --out-json "$R/R10_option_a_s${S}_grid5.json" \
    > "$R/R10_option_a_s${S}_grid5.log" 2>&1
  say "one-pass seed $S rc=$?"
done

for S in 3 5; do
  say "u3 diffusion seed $S start (3 stages, expect many hours)"
  python finetuning/run_u3_uq.py train --dataset-key noaa_uk --pred 168 --seeds "$S" \
    > "$R/u3_train_seed${S}.log" 2>&1
  say "u3 diffusion seed $S rc=$?"
done
say "GPU0 queue done"
