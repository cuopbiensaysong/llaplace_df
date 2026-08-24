#!/usr/bin/env bash
# GPU 2 queue: wait out the in-flight one-pass seed 4, then take diffusion seeds 2 and 4.
# Family B (read T2) needs 6 seeds total, i.e. 0-5; 0 and 1 already exist.
set -u
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
cd "$LLAPDIFF_SRC" || exit 1
export CUDA_VISIBLE_DEVICES=2

R=finetuning/results/cmd_gp/${CELL:-noaa_uk_h168}
LOG=$R/queue_gpu2.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "guard: $(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')"

# Let the in-flight seed-4 probe finish before claiming the card.
while pgrep -f "run_option_a_probe.*--seed 4" > /dev/null; do sleep 60; done
say "one-pass seed 4 clear"

for S in 2 4; do
  say "u3 diffusion seed $S start (3 stages, expect many hours)"
  python finetuning/run_u3_uq.py train --dataset-key noaa_uk --pred 168 --seeds "$S" \
    > "$R/u3_train_seed${S}.log" 2>&1
  say "u3 diffusion seed $S rc=$?"
done
say "GPU2 queue done"
