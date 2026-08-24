#!/usr/bin/env bash
# One campaign job, claimed with a filesystem lock so any number of machines can pull from
# the same queue on the shared volume without duplicating work.
#
#   ./run_campaign_job.sh u3      <SEED>   # 3-stage diffusion training (expensive, ~10h/stage)
#   ./run_campaign_job.sh onepass <SEED>   # one-pass head sweep (cheap, ~45 min)
#
# Set CUDA_VISIBLE_DEVICES before calling, or pass it as the 3rd argument.
# The lock is `mkdir`, which is atomic on POSIX and on NFS -- unlike `test -f && touch`,
# which races exactly when two nodes start together.
set -u

# NB: no braces in these defaults -- a '}' inside ${x:?...} closes the expansion early and
# silently becomes part of the value. That produced a job named "onepass SEED [GPU]}".
usage() {
  echo "usage: run_campaign_job.sh JOB SEED [GPU]" >&2
  echo "  JOB: u3 onepass gate frontier sensitivity sweep efficiency" >&2
  echo "       abl_base abl_nonugget abl_pmono abl_pgrid abl_noanchor" >&2
  exit 2
}
JOB="${1:-}"; SEED="${2:-}"
[ -z "$JOB" ] && usage
[ -z "$SEED" ] && usage
case "$JOB" in
  u3|onepass|sensitivity|sweep|efficiency|gate|frontier) ;;
  abl_base|abl_nonugget|abl_pmono|abl_pgrid|abl_noanchor) ;;
  *) usage ;;
esac
case "$SEED" in ''|*[!0-9]*) usage ;; esac
[ "${3:-}" != "" ] && export CUDA_VISIBLE_DEVICES="$3"

source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC
cd "$LLAPDIFF_SRC" || exit 1

# 🔴 The in-training validation profile. Without it a trial is ~21x slower and NOTHING else
# changes: one full val CRPS eval is 146 batches x 25 samples x 64 DDIM steps x 2 forwards
# (CFG runs cond+uncond) = ~110 min, every 5 epochs, i.e. >90% of the trial. `_sampling_kwargs`
# resolves EVAL_* for exactly the two in-training call sites and TEST_* for everything that
# produces a reported number, so this moves checkpoint *selection* only -- and the profile was
# validated on precisely this cell (noaa_uk h=168).
#
# It also matters for PARITY: seeds 0 and 1 of Family B were trained under this profile by
# `finetune_noaa_uk.sh`, so training later seeds without it would select checkpoints by a
# different rule within the same family.
export LLAPDIFF_EVAL_STEPS="${LLAPDIFF_EVAL_STEPS:-16}"
export LLAPDIFF_EVAL_NUM_SAMPLES="${LLAPDIFF_EVAL_NUM_SAMPLES:-5}"
export LLAPDIFF_EVAL_MAX_BATCHES="${LLAPDIFF_EVAL_MAX_BATCHES:-48}"
export LLAPDIFF_EVAL_SUBSET_MODE="${LLAPDIFF_EVAL_SUBSET_MODE:-stride}"
export LLAPDIFF_EVAL_SEED="${LLAPDIFF_EVAL_SEED:-4242}"
export LLAPDIFF_VAL_DIAG_EVERY="${LLAPDIFF_VAL_DIAG_EVERY:-5}"

# Results and locks are namespaced by CELL so a second dataset cannot overwrite the first's
# outputs or steal its lock. Locks were previously bare (`gate_seed0.lock`), which is fine for
# one dataset and silently wrong for two: a crypto worker would skip noaa_uk's gate as
# "already claimed" and vice versa.
#
#   DATASET_KEY=crypto PRED=100 ./run_campaign_job.sh gate 0
#
DATASET_KEY="${DATASET_KEY:-noaa_uk}"
PRED="${PRED:-168}"
CELL="${DATASET_KEY}_h${PRED}"

# The variogram score's [S,B,T,T,D] tensor scales with the latent width: 128 windows is
# 5.8 GB at d_z=16 and 8.7 GB at d_z=24, which OOMs a 24 GB card. 128 keeps noaa_uk's existing
# numbers bit-identical; wider latents need less.
EVAL_CHUNK="${EVAL_CHUNK:-128}"

# The learning-rate grid for the one-pass probe, and the tag that keeps its outputs and its
# lock separate from another grid's. On bms_air h=168 the shipped grid's LOWER bound was
# selected on 10/10 seeds for chirp_gp and 8/10 for the diagonal head, i.e. the grid did not
# bracket the optimum and the arms may be under-trained. A wider grid must be applied to
# EVERY head and reported for EVERY read -- widening it only where an arm lost would be
# selection on the outcome.
# Which U3 stages the `u3` job trains. A NEW cell must stop after s1_mean so its own
# CHIRP_UQ_INIT_VAR can be measured; without this the job trains s1 for tens of hours, hits
# the guard at s2, releases its lock, and the next worker repeats the failure instantly.
STAGES="${STAGES:-}"

LR_GRID="${LR_GRID:-3e-4 1e-4 3e-5 1e-5}"
GRID_TAG="${GRID_TAG:-grid5}"

R="finetuning/results/cmd_gp/${CELL}"
LOCKS=finetuning/locks
mkdir -p "$LOCKS" "$R"
LOCK="$LOCKS/${CELL}__${JOB}_seed${SEED}.lock"
# A different LR grid is a different experiment: it needs its own claim, or workers
# would see the previous grid's completed lock and skip.
[ "$GRID_TAG" != "grid5" ] && LOCK="$LOCKS/${CELL}__${JOB}_${GRID_TAG}_seed${SEED}.lock"

# Migration shim: noaa_uk's locks predate the namespace and are named without the cell
# prefix. Honour an existing bare lock for that cell so a job already claimed or completed
# under the old scheme is not re-run. New cells never take this path.
if [ "$CELL" = "noaa_uk_h168" ] && [ -d "$LOCKS/${JOB}_seed${SEED}.lock" ]; then
  LOCK="$LOCKS/${JOB}_seed${SEED}.lock"
fi

# The sibling-checkout hazard: this must resolve under $LLAPDIFF_SRC or the run silently
# trains the wrong tree. Fail loudly rather than produce a plausible wrong number.
RESOLVED=$(python -c 'import llapdiffusion;print(llapdiffusion.__file__)')
case "$RESOLVED" in
  "$LLAPDIFF_SRC"/*) ;;
  *) echo "FATAL: llapdiffusion resolved to $RESOLVED, not under $LLAPDIFF_SRC"; exit 2 ;;
esac

# Resolve a seed's s2_diffusion_uq checkpoint from the U3 campaign state, so the
# no-training jobs below do not hardcode trial-id hashes that change per run.
u3_ckpt() {
  CELL="$CELL" python - "$1" <<'PY' 2>/dev/null
import json, os, sys, pathlib
seed = sys.argv[1]
sp = pathlib.Path(f"finetuning/results/uq_u3/{os.environ.get('CELL','noaa_uk_h168')}/d/state.json")
if not sp.exists():
    sys.exit(1)
tr = json.loads(sp.read_text()).get("trials", {})
rec = tr.get(f"s2_diffusion_uq::seed{seed}")
p = pathlib.Path(rec["checkpoint"]) if rec else None
if p is None or not p.exists():
    sys.exit(1)
print(p)
PY
}

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[skip] ${JOB} seed ${SEED} already claimed by $(cat "$LOCK/owner" 2>/dev/null)"
  exit 0
fi
echo "$(hostname):${CUDA_VISIBLE_DEVICES:-?}:$(date -Iseconds)" > "$LOCK/owner"
# Release the claim if we die, so a crashed job can be retried; a COMPLETED job is
# protected by its output file instead (checked below).
trap '[ -f "$LOCK/done" ] || rm -rf "$LOCK"' EXIT

echo "[claim] ${JOB} seed ${SEED} on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"

case "$JOB" in
  onepass)
    OUT="$R/R10_option_a_s${SEED}_${GRID_TAG}.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; touch "$LOCK/done"; exit 0; fi
    # --dataset-key/--pred are NOT optional here: the probe defaults to noaa_uk h=168, so
    # without them a DATASET_KEY=bms_air worker silently re-runs the noaa_uk arm and writes it
    # into the bms_air results directory.
    python -m llapdiffusion.tools.run_option_a_probe \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --eval-chunk "$EVAL_CHUNK" \
      --heads chirp_gp diag_gaussian lowrank_gaussian student_t \
      --summary-tokens 16 --lr-grid $LR_GRID --weight-decay 1e-3 \
      --epochs-mse 150 --epochs-nll 80 --patience 20 --num-samples 25 --seed "$SEED" \
      --out-json "$OUT" > "${OUT%.json}.log" 2>&1
    RC=$?
    ;;
  u3)
    # run_u3_uq.py gives each seed its own DIFF_PRECOMPUTE_DIR, so concurrent seeds do not
    # race on the precompute cache (they did, once, and deleted each other's).
    python finetuning/run_u3_uq.py train --dataset-key "$DATASET_KEY" --pred "$PRED" --seeds "$SEED" \
      ${STAGES:+--stages $STAGES} \
      > "$R/u3_train_seed${SEED}.log" 2>&1
    RC=$?
    ;;

  # ---- jobs below need no training; they run on the s2 checkpoints that already exist ----
  gate|frontier)
    # Read R9 (the variance decomposition) and the M-convergence frontier, one per u3 seed.
    # Family B needs these on EVERY trained seed, not just 0 and 1. A seed whose training has
    # not finished exits 0 without claiming `done`, so the lock is released and the job is
    # simply retried by a later worker -- run the worker again whenever trainings land.
    CK=$(u3_ckpt "$SEED") || { echo "[wait] no s2 checkpoint for seed $SEED yet"; exit 0; }
    if [ "$JOB" = "frontier" ]; then
      OUT="$R/frontier_seed${SEED}_val.json"; SCORE="--score --score-samples 25"
    else
      OUT="$R/R9_gate_seed${SEED}_val.json";  SCORE=""
    fi
    if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; touch "$LOCK/done"; exit 0; fi
    python -m llapdiffusion.tools.run_mixture_uq_eval \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --checkpoint "$CK" --split val \
      --components 25 --vae-entity-encode --weights ema --seed $((42 + SEED)) $SCORE \
      --out-json "$OUT" > "${OUT%.json}.log" 2>&1
    RC=$?
    ;;
  sensitivity)
    # Does the R9 verdict survive the read protocol? Algorithm 2 takes the component law from
    # the FINAL reverse step; the earlier A2 protocol read it at t=T-1 on a pure-noise input,
    # which inflates the within-component term ~3x and so pushes R down. We currently only
    # INFER that R stays above 0.25 under that choice; this measures it.
    CK=$(u3_ckpt "$SEED") || { echo "no s2 checkpoint for seed $SEED yet"; exit 0; }
    python -m llapdiffusion.tools.run_mixture_uq_eval \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --checkpoint "$CK" --split val \
      --components 25 --vae-entity-encode --weights ema --within-at-timestep 999 \
      --out-json "$R/R9_sensitivity_tTminus1_seed${SEED}.json" \
      > "$R/R9_sensitivity_tTminus1_seed${SEED}.log" 2>&1
    RC=$?
    ;;
  sweep)
    # Appendix A2 / read R8: guidance x DDIM-step grid on val. A flat curve is pre-declared to
    # mean the reverse loop contributes nothing beyond parameter refinement.
    CK=$(u3_ckpt "$SEED") || { echo "no s2 checkpoint for seed $SEED yet"; exit 0; }
    python -m llapdiffusion.tools.run_u1_sweep \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --checkpoint "$CK" --split val --weights ema \
      --guidance 1.0 1.25 1.5 2.0 --steps 64 32 16 8 4 1 \
      --dynamic-thresh-p 0.995 --output-root "$R/A2_sweep_seed${SEED}" \
      > "$R/A2_sweep_seed${SEED}.log" 2>&1
    RC=$?
    ;;
  efficiency)
    # Appendix A5. One machine's numbers only -- wall-clock is not portable across the
    # cluster's mixed cards, so the report records the device and only ONE seed is run.
    CK=$(u3_ckpt "$SEED") || { echo "no s2 checkpoint for seed $SEED yet"; exit 0; }
    python -m llapdiffusion.tools.run_efficiency_bench \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --checkpoint "$CK" --weights ema \
      --horizons 24 48 96 168 --components 1 4 8 16 25 --repeats 9 \
      --out-json "$R/A5_efficiency_seed${SEED}.json" \
      > "$R/A5_efficiency_seed${SEED}.log" 2>&1
    RC=$?
    ;;
  abl_*)
    # One-pass ablations, each a paper table that is currently all-TBD. Same frozen encoder,
    # same LR grid, same schedule as the R10 arms -- only the named knob moves, so a
    # difference is attributable to it.
    case "$JOB" in
      abl_base)     EXTRA=""                           ;;   # same code as the others
      abl_nonugget) EXTRA="--no-nugget"                ;;   # read R15 / OPT{N}
      abl_pmono)    EXTRA="--parameterization p_mono"  ;;   # appendix A3
      abl_pgrid)    EXTRA="--parameterization p_grid"  ;;   # A3; a GRID law by construction
      abl_noanchor) EXTRA="--anchor-nodes 0"           ;;   # cost of Prop-C.1 correctness
    esac
    OUT="$R/${JOB}_s${SEED}.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; touch "$LOCK/done"; exit 0; fi
    python -m llapdiffusion.tools.run_option_a_probe \
      --dataset-key "$DATASET_KEY" --pred "$PRED" --eval-chunk "$EVAL_CHUNK" \
      --heads chirp_gp --summary-tokens 16 --lr-grid 3e-4 1e-4 3e-5 1e-5 \
      --weight-decay 1e-3 --epochs-mse 150 --epochs-nll 80 --patience 20 \
      --num-samples 25 --seed "$SEED" --tag "$JOB" $EXTRA \
      --out-json "$OUT" > "$R/${JOB}_s${SEED}.log" 2>&1
    RC=$?
    ;;
  *) echo "unknown job '$JOB'"; exit 2 ;;
esac

echo "[done] ${JOB} seed ${SEED} rc=$RC"
[ $RC -eq 0 ] && touch "$LOCK/done"
exit $RC
