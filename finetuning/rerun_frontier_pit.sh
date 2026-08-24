#!/usr/bin/env bash
# Re-run the frontier scoring for noaa_uk h=168 with PIT-ECE added.
#
# WHY. run_mixture_uq_eval never computed a PIT calibration error: it imported only
# coverage_and_width, crps, energy_score and variogram_score, so every mixture_M* and
# sampled_M* arm carries coverage at five levels and NO distribution-wide calibration number.
# That is the one calibration metric the paper reports for every other arm, and it is missing
# for exactly the arm the R9 gate selected as the model.
#
# `randomized_pit` already existed in path_scores.py and was simply never called here.
#
# OUTPUT GOES TO NEW FILES. The existing frontier_seed*_val.json are left untouched, so the
# frozen Family B read cannot move under our feet and the two runs can be diffed.
set -u
cd "$(dirname "$0")/.." || exit 1

source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
RESOLVED=$(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')
case "$RESOLVED" in "$LLAPDIFF_SRC"/*) ;; *) echo "FATAL: $RESOLVED"; exit 2 ;; esac

R=finetuning/results/cmd_gp/noaa_uk_h168
say() { echo "[pit] $*"; }

ckpt() {
  python - "$1" <<'PY' 2>/dev/null
import json, sys, pathlib
d = json.loads(pathlib.Path("finetuning/results/uq_u3/noaa_uk_h168/d/state.json").read_text())["trials"]
rec = d.get(f"s2_diffusion_uq::seed{sys.argv[1]}")
p = pathlib.Path(rec["checkpoint"]) if rec else None
if p is None or not p.exists():
    sys.exit(1)
print(p)
PY
}

one() {
  local seed="$1" gpu="$2"
  local out="$R/frontier_pit_seed${seed}_val.json"
  [ -f "$out" ] && { say "skip seed $seed (exists)"; return 0; }
  local ck
  ck=$(ckpt "$seed") || { say "seed $seed: no s2 checkpoint"; return 0; }
  say "seed $seed on gpu $gpu"
  # Same protocol as the original frontier run: 25 components, EMA weights, seed 42+n, so the
  # only difference between the two files is the added column.
  CUDA_VISIBLE_DEVICES="$gpu" python -m llapdiffusion.tools.run_mixture_uq_eval \
    --dataset-key noaa_uk --pred 168 --checkpoint "$ck" --split val \
    --components 25 --vae-entity-encode --weights ema --seed $((42 + seed)) \
    --score --score-samples 25 --out-json "$out" > "${out%.json}.log" 2>&1 \
    && say "seed $seed done" || say "seed $seed FAILED (see ${out%.json}.log)"
}

# GPU 0 is training the bms_air stage-1 VAE; leave it alone.
( for s in 0 2 5 7; do one "$s" 1; done ) &
( for s in 1 4 6;   do one "$s" 2; done ) &
wait

say "================ PIT-ECE, all arms, mean over seeds ================"
python - <<'PY'
import json, pathlib, statistics as st
R = pathlib.Path("finetuning/results/cmd_gp/noaa_uk_h168")
seeds = [0, 1, 2, 4, 5, 6, 7]
runs = {}
for s in seeds:
    p = R / f"frontier_pit_seed{s}_val.json"
    if p.exists():
        runs[s] = json.loads(p.read_text())["arm_scores"]
if not runs:
    print("no results"); raise SystemExit
arms = ["mixture_M1", "mixture_M2", "mixture_M4", "mixture_M8", "mixture_M16", "mixture_M25",
        "sampled_M2", "sampled_M4", "sampled_M8", "sampled_M16", "sampled_M25"]
print(f"seeds: {sorted(runs)}")
print(f"{'arm':14s} {'PIT-ECE':>16s} {'cov@80':>8s} {'width@80':>9s}")
for a in arms:
    v = [runs[s][a]["PIT_ECE"] for s in runs if a in runs[s] and "PIT_ECE" in runs[s][a]]
    if not v:
        continue
    c = [runs[s][a]["coverage_0.8"] for s in runs if a in runs[s]]
    w = [runs[s][a]["width_0.8"] for s in runs if a in runs[s]]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"{a:14s} {st.mean(v):9.4f} ±{sd:6.4f} {st.mean(c):8.3f} {st.mean(w):9.3f}")
PY
say "done"
