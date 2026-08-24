#!/usr/bin/env bash
# A worker: pulls campaign jobs until none are left. Run one per GPU, on any machine that
# can see the shared volume. Locking is in run_campaign_job.sh, so workers never collide
# and it is safe to start more of them at any time.
#
#   CUDA_VISIBLE_DEVICES=0 ./run_campaign_worker.sh          # everything, expensive first
#   CUDA_VISIBLE_DEVICES=1 ./run_campaign_worker.sh u3       # diffusion seeds only
#   CUDA_VISIBLE_DEVICES=2 ./run_campaign_worker.sh onepass  # cheap seeds only
#
# Work remaining for the pre-registration (w_docs/noaa_uk_h168/PREREG_U3_noaa_uk_h168.md §11):
#   Family B  u3 seeds 2 3 4 5      -> 6 diffusion seeds total   <-- the bottleneck
#   Family A  onepass seeds 4..9    -> 10 one-pass seeds total
set -u
WHICH="${1:-all}"
cd "$(dirname "$0")/.." || exit 1
JOB=finetuning/run_campaign_job.sh

# Family B is registered at 6 seeds (0-5). 6-9 take it to 10, which matches the paper's
# standard for a headline row, gives insurance if a training dies, and would let the reads be
# analysed as one family rather than two if the split is ever questioned.
U3_SEEDS="${U3_SEEDS:-2 3 4 5 6 7 8 9}"
ONEPASS_SEEDS="${ONEPASS_SEEDS:-4 5 6 7 8 9}"
ABL_SEEDS="${ABL_SEEDS:-0 1 2}"
# These need only the s2 checkpoints that already exist (seeds 0 and 1), so they are
# available immediately and do not wait on the training queue.
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SENSITIVITY_SEEDS="0 1"     # does the R9 verdict survive the read protocol?
SWEEP_SEEDS="0 1"           # appendix A2 / read R8: guidance x DDIM steps
EFFICIENCY_SEEDS="0"        # appendix A5: one machine only, wall-clock is not portable

run_group() {
  local kind="$1"; shift
  for s in $@; do bash "$JOB" "$kind" "$s"; done
}

run_ablations() {
  # abl_base is the reference arm re-run from the SAME code as the variants: the R10
  # baselines predate the vectorised quadrature, and a code-version difference between
  # a baseline and its ablations is exactly the confound an ablation exists to avoid.
  for k in abl_base abl_nonugget abl_pmono abl_pgrid abl_noanchor; do
    run_group "$k" $ABL_SEEDS
  done
}

case "$WHICH" in
  u3)          run_group u3 $U3_SEEDS ;;
  evals)       run_group gate $EVAL_SEEDS; run_group frontier $EVAL_SEEDS ;;
  # Poll until every trained seed has been evaluated. Without this a worker started while the
  # trainings are still running finds no s2 checkpoints, prints [wait] for every seed and exits
  # in seconds -- so the eval pass would have to be launched by hand after each training lands.
  evals-watch)
    deadline=$(( $(date +%s) + ${EVALS_WATCH_HOURS:-16} * 3600 ))
    while :; do
      run_group gate $EVAL_SEEDS
      run_group frontier $EVAL_SEEDS
      pending=0
      for d in finetuning/locks/u3_seed*.lock; do
        [ -d "$d" ] || continue
        [ -f "$d/done" ] || pending=$((pending + 1))
      done
      if [ "$pending" -eq 0 ]; then
        echo "[worker] no trainings left in flight; final eval pass done"
        break
      fi
      [ "$(date +%s)" -ge "$deadline" ] && { echo "[worker] watch deadline reached"; break; }
      echo "[worker] $pending training(s) still in flight; re-checking in 10 min"
      sleep 600
    done ;;
  onepass)     run_group onepass $ONEPASS_SEEDS ;;
  sensitivity) run_group sensitivity $SENSITIVITY_SEEDS ;;
  sweep)       run_group sweep $SWEEP_SEEDS ;;
  efficiency)  run_group efficiency $EFFICIENCY_SEEDS ;;
  ablations)   run_ablations ;;
  # Everything that needs no training, for a spare GPU that should not start a 30-hour job.
  quick)       run_group gate $EVAL_SEEDS
               run_group frontier $EVAL_SEEDS
               run_group sensitivity $SENSITIVITY_SEEDS
               run_group sweep $SWEEP_SEEDS
               run_group efficiency $EFFICIENCY_SEEDS
               run_ablations ;;
  # Expensive first: a worker that dies early should have spent its time on the bottleneck,
  # and the cheap jobs can be mopped up by anything.
  all)         run_group u3 $U3_SEEDS
               run_group gate $EVAL_SEEDS
               run_group frontier $EVAL_SEEDS
               run_group sensitivity $SENSITIVITY_SEEDS
               run_group sweep $SWEEP_SEEDS
               run_group efficiency $EFFICIENCY_SEEDS
               run_ablations
               run_group onepass $ONEPASS_SEEDS ;;
  *) echo "usage: run_campaign_worker.sh [all|u3|evals|evals-watch|ablations|quick|onepass|sensitivity|sweep|efficiency]"
     exit 2 ;;
esac
echo "[worker] no unclaimed $WHICH jobs left on $(hostname)"
