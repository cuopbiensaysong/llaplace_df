#!/usr/bin/env bash
# Stop the five s3_oneshot_uq jobs (seeds 2,4,5,6,7) and KEEP seed 3 running.
#
# Run this on EVERY host that owns one of those jobs -- currently cuda10 and cuda12.
# It is safe to run on a host that owns none of them, and safe to run twice.
#
# Why not `pkill -f run_u3_uq`:
#   1. it kills every seed including seed 3, and
#   2. the tree is  worker -> run_campaign_job.sh -> run_u3_uq.py -> run_trial.py -> training,
#      so killing the driver ORPHANS run_trial.py, which keeps burning the GPU while
#      `pgrep run_u3_uq` reports nothing.
# This script kills children before parents, and only for the seeds named below.
#
# Why the locks are marked `done` BEFORE killing:
#   run_campaign_job.sh carries `trap '[ -f "$LOCK/done" ] || rm -rf "$LOCK"' EXIT`. If the
#   lock is released, any still-running worker re-claims that seed and restarts it. Marking
#   `done` first makes the trap leave the lock in place, so the seed stays retired.

set -u
cd "$(dirname "$0")/.." || exit 1

STOP_SEEDS="2 4 5 6 7"
KEEP_SEED=3
LOCKS=finetuning/locks

say() { echo "[stop-s3] $*"; }

# ---------------------------------------------------------------- 0. safety guard
# Seed 3 must survive. Record its pids now and verify at the end that they are untouched.
keep_pids=$(pgrep -f "run_u3_uq.py .*--seeds ${KEEP_SEED}( |$)" 2>/dev/null | tr '\n' ' ')
say "seed ${KEEP_SEED} driver pids on this host: ${keep_pids:-none}"

# ---------------------------------------------------------------- 1. park free seeds
# Seeds with no lock are claimable. With workers still alive, freeing a GPU here would let
# one start a fresh ~26 h training on seed 8 or 9. Park every unclaimed u3 seed first.
for n in 0 1 8 9; do
  d="$LOCKS/u3_seed${n}.lock"
  if mkdir "$d" 2>/dev/null; then
    echo "parked-by-request:$(hostname):$(date -Iseconds)" > "$d/owner"
    touch "$d/done"
    say "parked u3_seed${n} (was unclaimed)"
  fi
done

# ---------------------------------------------------------------- 2. retire the s3 seeds
for n in $STOP_SEEDS; do
  d="$LOCKS/u3_seed${n}.lock"
  [ -d "$d" ] && touch "$d/done"          # keep the lock so no worker re-claims this seed

  # the driver for exactly this seed; the trailing (space|end) stops "3" matching "34"
  drv=$(pgrep -f "run_u3_uq.py .*--seeds ${n}( |$)" 2>/dev/null | tr '\n' ' ')
  [ -z "${drv// /}" ] && { say "seed ${n}: no driver on this host"; continue; }

  # collect descendants (run_trial.py and below), deepest first
  kids=""
  frontier="$drv"
  while [ -n "${frontier// /}" ]; do
    next=""
    for p in $frontier; do
      c=$(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')
      next="$next $c"
    done
    kids="$next $kids"
    frontier="$next"
  done

  # parent job wrapper, so it cannot proceed to another stage
  par=""
  for p in $drv; do par="$par $(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"; done

  say "seed ${n}: killing children [${kids# }] then driver [${drv% }] then wrapper [${par# }]"
  for p in $kids $drv $par; do
    [ -z "$p" ] && continue
    case " $keep_pids " in *" $p "*) say "REFUSING to kill $p -- belongs to seed ${KEEP_SEED}"; continue ;; esac
    kill -9 "$p" 2>/dev/null
  done
done

sleep 3

# ---------------------------------------------------------------- 3. verify
say "--- seed ${KEEP_SEED} still alive? ---"
if pgrep -f "run_u3_uq.py .*--seeds ${KEEP_SEED}( |$)" >/dev/null 2>&1; then
  say "YES: seed ${KEEP_SEED} is still running (this is what we wanted)"
else
  say "NOTE: no seed ${KEEP_SEED} driver on this host (expected unless this host owns it)"
fi

say "--- remaining u3 drivers on this host ---"
pgrep -af "run_u3_uq.py" 2>/dev/null || echo "  none"

say "--- GPU compute apps (other users' jobs may legitimately appear) ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "  n/a"

say "done. Locks for seeds ${STOP_SEEDS} were marked done, so no worker will restart them."
