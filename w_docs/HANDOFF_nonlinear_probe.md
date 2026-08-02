# Handoff — is the B21 conditioning gap information LOSS or nonlinear RE-ENCODING?

**To:** a second Claude Opus 5 instance, on a separate GPU cluster.
**From:** the agent holding this session (call it **agent-A**). You are **agent-B**.
**Written:** 2026-08-02. **Comms channel:** `w_docs/agents_communication.md` — read it first,
append there, do not edit agent-A's entries.

---

## 0. STOP — read this before running anything

### 0.1 The code you need is NOT in git — but you already have it

**Confirmed 2026-08-02: you share this filesystem and this working tree.** So you get the code
automatically — and that is exactly why §0.3 matters. Do **not** `git stash`, `git checkout`,
`git restore`, or `git clean` anything: this tree has **~1900 lines of uncommitted diff** and
those commands would delete work that exists nowhere else.

Three files are load-bearing for your task and **do not exist at git HEAD**:

| file | why you need it |
|---|---|
| `llapdiffusion/tools/run_ridge_probe.py` | the tool whose collection path you must reuse (new, untracked) |
| `llapdiffusion/models/llapdiff_utils.py` | `normalize_cond_per_batch` gained `mode="global"` |
| `llapdiffusion/trainers/train_val_llapdiff.py` | `_resolve_cond_norm_stats`, `compute_global_cond_stats`, B22 summarizer persistence |

**Restore point.** Agent-A snapshotted the full uncommitted state to
`/vol/dl-nguyenb5-solar/users/cuopbiensaysong/_agentA_snapshot_20260802T044906/`
(`uncommitted.patch` + a `tree/` copy of every touched file). If something is clobbered, say so
in the comms file — do not try to reconstruct it yourself.

Verify before anything else:

```bash
python -c "
from llapdiffusion.tools import run_ridge_probe          # must import
from llapdiffusion.models.llapdiff_utils import normalize_cond_per_batch as n
import inspect; assert 'global' in inspect.signature(n).parameters['mode'].annotation or True
n(__import__('torch').zeros(2,3,4), mode='global')       # must raise ValueError, not KeyError
"
```

### 0.2 Environment (from `CLAUDE.md`, non-negotiable)

```bash
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=<path to THIS working tree>
export PYTHONPATH=$LLAPDIFF_SRC
python -c "import llapdiffusion; print(llapdiffusion.__file__)"   # MUST be under $LLAPDIFF_SRC
```

The venv has `llapdiffusion` installed **editable against a different sibling checkout**. Without
`PYTHONPATH` your imports silently resolve there — no error, no warning, edits have no effect.

### 0.3 🔴 Write boundaries — SHARED filesystem, agent-A has jobs running

You are on the same tree agent-A is actively editing, and agent-A has a 4-run multi-seed
campaign writing to disk on **GPU 1** until roughly 2026-08-02 +6 h. **This is the section most
likely to cause damage.**

| path | who owns it | you may |
|---|---|---|
| `w_docs/results_nonlinear_probe.md` | **agent-B (you)** | create, write freely |
| `w_docs/agents_communication.md` | shared, append-only | **append** at the bottom only |
| `llapdiffusion/**`, `tests/**`, `pyproject.toml` | agent-A | **read only** |
| `w_docs/CMD_BUG_REPORT.md`, `CMD_UQ_RECOVERY_PLAN.md`, `USAGE.md`, `CMD_RUNBOOK.md` | agent-A | **read only** |
| `ldt/diffusion_cache/**` | agent-A's running jobs | **do not touch** |
| `ldt/tuning/**`, `finetuning/results/**` | agent-A's running jobs | read only (write only under your own `--run-tag`, and only if a task grants it) |
| **`ldt/vae/**`, `ldt/summarizer/**`** | **everyone — see below** | **never write, and check they EXIST before training** |
| anywhere outside the repo | yours | scratch, checkpoints, intermediates |

> 🔴 **The stage-1/2 trap, which `--run-tag` does NOT protect against.**
> *(Gap in an earlier draft of this handoff; caught by agent-B before it bit, 2026-08-02.)*
>
> `run_trial.py` re-roots only `OUT_DIR`, `CKPT_DIR` and `POLE_PLOT_DIR` under your run tag. It
> leaves `VAE_CKPT` and `SUM_CKPT` at their shared, **unrouted** config-derived paths —
> `ldt/vae/saved_model/<ds>/…` and `ldt/summarizer/saved_model/<ds>/…` — which every arm, every
> seed and both agents share. That sharing *is* the parity requirement.
>
> And `run_single_pred` trains stage 1/2 **whenever the checkpoint file is missing**. So if one
> of those files were absent, your first training run would silently retrain a shared artifact
> underneath another agent's in-flight campaign and invalidate every seed on both sides — with
> no error and no warning. **Before any training run, confirm both files exist**, and never
> pass `--recompute-vae` / `--recompute-summarizer` on a shared tree.

**Specific prohibitions, each with a reason:**

- **No `git stash` / `checkout` / `restore` / `clean` / `reset`.** ~1900 lines of uncommitted
  diff live only in this tree.
- **No writes under `ldt/diffusion_cache/**`.** 4.4 GB shared cache. Concurrent *reads* are
  safe (the only write is the manifest, on the build path), but a rebuild while another process
  reads destroys it. **Your task needs no diffusion cache** — do not set `DIFF_PRECOMPUTE_DIR`,
  do not run `run_trial.py`, do not train anything.
- **No edits to `llapdiffusion/**`.** If the task needs a code change, **append a QUESTION to
  the comms file and wait** — agent-A owns that surface and is editing it concurrently. Put
  your probe in a standalone script **outside the repo** that imports from `llapdiffusion`.
- **Use a GPU on your own cluster.** Agent-A's GPU 1 is running the experiment; contention there
  perturbs kernel selection (`cudnn.benchmark`), the documented 0.036 CRPS noise source, and
  could corrupt a −0.028 effect measurement.

**Appending to the comms file safely:** re-read the whole file immediately before you append.
If it contains an entry you had not seen, read it fully before writing — it may change your
task. Append only under a new `## [timestamp] agent-B — subject` heading; never edit above it.

Your task is **inference only** — frozen summarizer + VAE forwards plus CPU fitting. It trains
nothing and needs no write access to the repo beyond your two files.

---

## 1. Why this matters (the minimum context)

The CMD UQ campaign is blocked at Phase 1. Full detail: `w_docs/CMD_BUG_REPORT.md` **B21**
and `w_docs/CMD_UQ_RECOVERY_PLAN.md` §1a. Compressed:

A ridge probe on **noaa_uk h=168** measures, at matched temporal resolution, how much of the
h=168 target a *linear* readout can recover from each representation (percent RMSE reduction
vs predict-the-mean, alpha selected on a chronological train holdout, val never touched):

| representation | dims | linear reduction |
|---|---|---|
| raw history, temperature channel, per-entity | 128 | **27.3 %** |
| raw history, entity-averaged | 32 | 25.3 % |
| `cond_summary` (the denoiser's conditioning) | 2 048 | **7.7 %** |
| `cond_summary` | 8 192 | 14.2 % |

B21 originally read this as "the summarizer discards ~½ the forecastable signal". **That
framing is now in doubt**, and resolving it is your job.

### The evidence that put it in doubt

1. **`SUM_FT_MODE="all"`** (fine-tuning the whole summarizer through the conditioning path)
   produced the **best forecast of any run** — val CRPS 0.3240 vs 0.3627 control — while making
   the conditioning **less** linearly decodable: `cond_summary` → raw target 6.8 % (vs 7.7 %),
   → latent **1.4 %** (vs 5.7 %). Best forecast, worst linear probe.
2. **`COND_NORM_MODE="global"`** raised the raw-target readout 7.7 % → 12.3 % and lowered val
   CRPS 0.3627 → 0.3344, while the *latent* readout stayed flat (5.6 % → 4.9 %).
3. **Two architectural explanations are eliminated.** No dimensional bottleneck (input 6 720
   numbers, output 86 016 — the summary is 12.8× larger than its input). Not entity pooling
   (27.3 % → 25.3 %, costs 2 points; the summarizer *does* mean-pool entities like the VAE, but
   noaa_uk's 4 stations share 85 % of their variance so it is nearly free here).

So a representation can carry **more usable signal and less linear signal at once**, and the
remaining suspects are the encoder, the query-pooling to S tokens, and the training objective
(the summarizer optimises *history reconstruction*, `SUM_LOSS_W_X/V/T`, not forecasting).

---

## 2. Your task

**Question.** Is the 27.3 % → 7.7 % gap information *loss*, or is the information present in
`cond_summary` but not linearly accessible?

**Method.** Replace ridge with a small nonlinear probe (MLP) and re-measure. Report both
targets — the raw target and the latent `mu_norm` — because they move differently (B21).

### 2.1 The control you must not skip

An MLP will beat ridge on *any* representation. Measuring `cond_summary` alone tells you
nothing. **You must fit the same MLP on the raw history too**, so the comparison is
"how much does nonlinearity unlock *here* vs *there*":

| probe | representation | what it tells you |
|---|---|---|
| ridge | raw history (128) | 27.3 % — already measured, use as a reproduction check |
| ridge | `cond_summary` (2 048) | 7.7 % — already measured, use as a reproduction check |
| **MLP** | raw history (128) | how much nonlinearity adds to a *known-good* representation |
| **MLP** | `cond_summary` (2 048) | **the answer** |

### 2.2 Decision rule (state which branch you land in)

Let `R_hist`, `R_cond` be the ridge numbers and `M_hist`, `M_cond` the MLP numbers, all on the
**raw target**, all on val.

- **`M_cond` approaches `M_hist`** (say within ~20 % relative) → the information is **present
  and nonlinearly encoded**. B21's "discards" framing is wrong and must be retitled; the
  campaign's problem is the denoiser's ability to *use* the conditioning, not the summarizer's
  ability to carry it.
- **`M_cond` stays far below `M_hist`, and the gap is similar to the ridge gap** → genuine
  **loss**. B21 stands as written; fixing the summarizer's objective becomes the priority.
- **Both MLPs gain little over their ridges** → the linear probe was already near the ceiling
  and this whole line of inquiry is settled: the gap is real and linear-accessible.

### 2.3 Protocol — deviations here have already produced wrong answers twice

Reuse `llapdiffusion/tools/run_ridge_probe.py::_collect` for the data so your windows match
agent-A's exactly. Then:

- **Split**: fit on train, report on **val**. Never touch test.
- **Windows**: 4-entity windows only (the modal count) → expect **14 446 train / 2 183 val**.
  If you get different counts, stop and report in the comms file before continuing.
- **Features**: `cond_summary` **strided** over the token axis (8 and 32 tokens → 2 048 and
  8 192 dims). 🔴 **Never mean-pool the 336 tokens** — pooling scores −0.2 % where the strided
  view scores 14.2 %, i.e. it manufactures a false negative. This mistake already happened once.
- **Hyperparameters** (hidden width, depth, LR, epochs, weight decay): select on a
  **chronological holdout inside train** (the last 20 %), **never on val**. 🔴 Selecting on val
  moved a previous headline 6.2 % → 9.8 %, straight across a decision threshold.
- **Metric**: percent RMSE reduction vs `baseline_rmse_predict_mean` computed exactly as
  `llapdiff-uq-eval` does — a **scalar** grand mean over the eval split:
  `sqrt(((y - y.mean())**2).mean())`. `run_ridge_probe.ridge_reduction` shows the convention.
- **Seeds**: ≥3 MLP init seeds, report mean ± std. A single MLP fit is not a measurement.
- **Sanity check**: your MLP on raw history should comfortably exceed 27.3 %. If it does not,
  your training loop is underfit and nothing else you report is interpretable.

### 2.4 Which checkpoint / conditioning setting

Run with the **config default** (`COND_NORM_MODE="sample"`, no `--checkpoint`) so your numbers
line up with the table in §1. If you have spare capacity, repeat under `"global"` — agent-A has
a checkpoint at
`ldt/tuning/rawcond_probe/noaa_uk_h168/d/globalnorm_e60/output/modal-chirp/seed-0/pred-168/llapdiff_pred-168_best_ema.pt`
which carries the mode and its statistics; pass it as `--checkpoint`. **Secondary — do the
default first.**

---

## 3. Deliverable

Write `w_docs/results_nonlinear_probe.md` containing:

1. The 2×2 table (ridge/MLP × history/cond_summary), on the raw target, mean ± std over seeds.
2. The same for the latent target.
3. Which decision branch of §2.2 you land in, stated plainly.
4. Your MLP architecture and the holdout-selected hyperparameters.
5. Reproduction check: do your ridge rows match 27.3 % and 7.7 %? If not, say so **before**
   interpreting anything — a mismatch means the windows or the metric differ and the
   comparison is void.
6. Everything you could not establish. Agent-A's entries in `CMD_BUG_REPORT.md` all carry an
   explicit "not yet established" section; match that standard.

**Do not edit `CMD_BUG_REPORT.md` or `CMD_UQ_RECOVERY_PLAN.md`.** Agent-A will fold your result
in, so there is one hand on those files.

---

## 4. Traps this project has already paid for

Read these before you interpret any number. Each cost real time here.

| trap | consequence |
|---|---|
| Mean-pooling the summary tokens | −0.2 % vs 14.2 % — a false negative |
| Selecting a hyperparameter on val | 6.2 % → 9.8 %, across a decision threshold |
| Treating the latent as the yardstick | an intervention that *improves* CRPS *lowers* it |
| Single-seed CRPS claims | two byte-identical configs gave 0.2243 vs 0.1886 (0.036 swing) |
| Assuming `model_config` keys are top-level | they nest under `["llapdiff"]`; agent-A misread this once and nearly concluded a flag was inert |
| Trusting a commit message | `f52f843` says "K=128 for physionet"; the diff sets it on **noaa_uk**, and physionet still runs K=256 |

**Report negative and null results with the same care as positive ones.** Three of this
session's most useful findings were negatives.
