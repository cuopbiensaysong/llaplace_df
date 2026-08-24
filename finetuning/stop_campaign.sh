#!/usr/bin/env bash
# Stop every campaign process on THIS machine, in the order that actually works.
#
# Why this exists: the process tree is
#     run_campaign_worker.sh -> run_campaign_job.sh -> run_u3_uq.py -> run_trial.py -> training
# and `pkill -f run_u3_uq` kills only the driver. `run_trial.py` is then **orphaned and keeps
# training on the GPU**, invisible to a `pgrep run_u3_uq` check. If the locks are cleared and
# workers restarted at that point, the orphan and the new job train the same seed into the same
# trial directory at the same time.
#
# Kill children first, parents last, so no parent respawns anything.
set -u
cd "$(dirname "$0")/.." || exit 1

say() { echo "[stop] $*"; }

for pat in "run_trial.py" "run_u3_uq.py" "run_option_a_probe" "run_mixture_uq_eval" \
           "run_efficiency_bench" "run_u1_sweep" "run_campaign_job.sh" "run_campaign_worker.sh"; do
  pids=$(pgrep -f "$pat" 2>/dev/null | tr '\n' ' ')
  if [ -n "${pids// /}" ]; then
    say "killing $pat -> $pids"
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
  fi
done
sleep 3

left=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
say "GPU compute apps still present (other users' jobs may legitimately appear):"
echo "${left:-  none}"

say "locks held by THIS host (released so the work can be reclaimed):"
host=$(hostname)
for d in finetuning/locks/*.lock; do
  [ -d "$d" ] || continue
  [ -f "$d/done" ] && continue
  case "$(cat "$d/owner" 2>/dev/null)" in
    "$host"*) echo "  $(basename "$d" .lock)"; rm -rf "$d" ;;
  esac
done
say "done. Locks owned by OTHER hosts were left alone -- stop those machines separately."
