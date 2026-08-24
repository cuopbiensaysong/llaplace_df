# Running the CMD-GP campaign across machines

Any machine that can see the shared volume can join. Work is claimed with atomic `mkdir`
locks under `finetuning/locks/`, so you can start any number of workers on any number of
machines at any time without coordinating — a claimed job is skipped, not duplicated.

## Copy-paste

```bash
cd /vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df

# one worker PER GPU. Pick the mode by what the machine should help with:
CUDA_VISIBLE_DEVICES=0 nohup bash finetuning/run_campaign_worker.sh ablations > /tmp/w0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash finetuning/run_campaign_worker.sh ablations > /tmp/w1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash finetuning/run_campaign_worker.sh u3        > /tmp/w2.log 2>&1 &
```

Nothing to set up. Each script sources the venv, pins `PYTHONPATH`, and **hard-fails** if
`llapdiffusion` resolves outside this checkout — the sibling-checkout hazard otherwise trains
the wrong tree silently.

**One worker per GPU, not more.** Three concurrent jobs across two cards OOM-killed a run.

## Which mode

| mode | jobs | cost each | when to use |
|---|---|---|---|
| `ablations` | 15 (5 variants × 3 seeds) | ~45 min | **best use of a spare GPU now** — 3 paper tables, all `\TBD` |
| `u3` | seeds 8–9 left | ~30 h | the training bottleneck; only 2 slots remain |
| `evals` | gate + frontier, seeds 0–9 | ~25–40 min | **run after any `u3` training finishes** |
| `quick` | evals + sensitivity + sweep + efficiency + ablations | mixed | a GPU that must not start a 30-hour job |
| `all` | everything, expensive first | mixed | a dedicated machine |

Workers exit when nothing is unclaimed, so extra workers are harmless — they just stop.

## The one thing to remember

`evals` (gate + frontier) must be re-run **after** each `u3` training completes. A seed whose
training has not finished prints `[wait] no s2 checkpoint yet`, releases its lock and moves on,
so it is safe to run `evals` early and often — just re-run it as trainings land. Family B needs
these on **every** trained seed, not only seeds 0 and 1.

## Checking progress

```bash
ls finetuning/locks/                     # claimed jobs; each has an owner file
grep -c . finetuning/locks/*/done 2>/dev/null | wc -l   # completed
python -m llapdiffusion.tools.run_prereg_power --seeds 0 1 2 3 4 5 6 7 8 9
python -m pytest tests/ -q               # 550 expected
```

The power tool regenerates the seed statistics and the table the pre-registration's freeze
record needs (`w_docs/noaa_uk_h168/PREREG_U3_noaa_uk_h168.md` §6, §11).

## Stopping cleanly (do NOT just `pkill -f run_u3_uq`)

```bash
bash finetuning/stop_campaign.sh     # on EVERY machine that ran a worker
```

The process tree is `worker -> job -> run_u3_uq.py -> run_trial.py -> training`. Killing the
driver **orphans `run_trial.py`**, which keeps training on the GPU while `pgrep run_u3_uq`
reports nothing. Clearing the locks at that point and restarting makes the orphan and the new
job train the same seed into the same trial directory at the same time. The script kills
children before parents and releases only the locks this host owns.

## If a job dies

The lock releases on exit unless the job succeeded, so start a worker again and it retries.
To force a retry of a completed job, remove its lock directory. To park a job:

```bash
mkdir -p finetuning/locks/u3_seed9.lock && echo parked > finetuning/locks/u3_seed9.lock/owner
```

## 🔴 Do not touch the test split

Everything here runs on **val**. `FINAL_TEST_EVAL` stays `skip` and no tool in the queue reads
test. The test split is read once, after the freeze record in
`w_docs/noaa_uk_h168/PREREG_U3_noaa_uk_h168.md` §11 is complete.
