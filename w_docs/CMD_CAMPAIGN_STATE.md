# CMD UQ campaign — state of play and handoff

**Written 2026-08-03, end of session.** Read this first, then `CMD_UQ_RECOVERY_PLAN.md`.
This file says where things stand and what to do next; the plan says how.

---

## 1. Status in one line

**The campaign is STOPPED at Phase 1 by its own stop rule.** The Phase-1b signal gate failed on
noaa_uk h=168 on both seeds and both mean sources, and the cause is pinned: the **conditioning**,
not the denoiser and not the data. **Phases 2–6 are on hold** — they score the calibration of a
predictive law around a mean the gate has just shown carries ~6.5 % against a 10 % bar.

**Next action is not a phase in the plan.** It is the B21 conditioning track (§5 below).

---

## 2. Where the work is — read this before touching git

| | |
|---|---|
| branch | **`cmd-uq-tooling-and-b21`** — **not** `main` |
| ahead of `main` by | **20 commits** |
| pushed | yes, `origin/cmd-uq-tooling-and-b21` |
| merged to `main` | **no** |

⚠️ `main` is at `f1788e3` and contains **none** of this campaign. A fresh agent that checks out
`main` will see none of the tooling fixes, none of B20–B22, and will re-derive results that are
already recorded. Start from `cmd-uq-tooling-and-b21`.

⚠️ **`.gitignore:46` ignores `finetuning/results/*`.** Two campaign records lived only there and
would not survive a clone; they now have tracked copies. **Edit the tracked copy:**

| tracked (canonical) | ignored working copy |
|---|---|
| `w_docs/CMD_PHASE0_WINDOW_AUDIT.md` | `finetuning/results/window_audit.md` |
| `w_docs/CMD_PHASE1B_GATE.md` | `finetuning/results/phase1b/gate_readout.md` |

Raw JSON (`gate_seed*.json`, `s1_gate_seed*.json`) and all checkpoints under `ldt/` remain
untracked by design — they are large and reproducible from the commands in the tracked copies.

🔑 **`.git/config` contains a GitHub personal access token in plaintext** in the `origin` URL. It
is not pushed, but it is readable by anyone with filesystem access — which on this machine
includes other agents. **Rotate it** and switch to SSH or a credential helper. Left as-is because
changing a remote is the owner's call.

---

## 3. Settled — do not re-litigate, do not re-run

| finding | evidence |
|---|---|
| PhysioNet h=12 is a smoke-test cell only | 13 train windows vs 11.75 M params (B19, Phase 0) |
| crypto / us_equity cannot serve the gate | 90 / 72 val windows; over-parameterized 6.0× / 8.9× |
| bms_air h=168 is **VOID** as a cross-check | the h=168 eligibility filter leaves mostly *single-entity* windows; 23 full-panel val windows |
| Gate-0 failures are **not** a capacity problem | the Gate-0 score equals the cell's own cross-entity mean share (B20) |
| noaa_uk h=168 is the right gate cell | Gate 0 = 0.922, forecastable at A = 28.6 %, 4.3× **under**-parameterized |
| the summarizer, not the VAE, is the bottleneck | 1a: A = 28.6 % raw history vs B = 7.7–16.4 % through the conditioning |
| `COND_NORM_MODE="global"` is **not adopted and not measured** | 5 seeds, mean −0.0086 CRPS, 4 of 5 inside the 0.036 band, and `EPOCHS=60` truncated 8 of 10 runs at their final eval |
| **the denoiser is not the bottleneck** | Phase 1b: rescaled 6.5 % mean vs a ridge ceiling of 5.6 % — **+0.9 pt** for 10.2 M params |
| seed noise band on CRPS is **0.036** | §4 retraction 1; every decision run needs ≥ 2 seeds |

Two corrections that cost real time and are easy to repeat — both are recorded in
`CMD_BUG_REPORT.md`, read them before designing a probe:

- **Never mean-pool the summary tokens** in a probe. Averaging 336 tokens scores −0.2 % where the
  strided view scores 14.2 % — pooling manufactures a FAIL out of working conditioning.
- **Select ridge alpha on a chronological train holdout, never on val.** Selecting on val moved a
  headline 6.2 % → 9.8 %, straight across the 10 % threshold a branch turns on.

---

## 4. What Phase 1b actually measured

Full detail in `w_docs/CMD_PHASE1B_GATE.md` and §1b of the plan. The three things that matter:

1. **The gate fails.** Both seeds (CRPS 0.32376 / 0.33229, spread 0.0085 ≪ 0.036 band), all four
   reads. Even granting a free optimal rescaling the model does not have, 6.5 % < 10 %.
2. **The denoiser sits at the ridge ceiling.** 10.2 M parameters, a nonlinear architecture and 355
   epochs buy **under one percentage point** over linear least squares on the same inputs. This
   converts B21 from an inference about probes into a measurement on the trained model.
3. **`--mean-source ddim` is the wrong statistic for a signal gate** and the plan no longer
   mandates it alone. `generate(eta=0.0)` returns **one sample** from a random `x_T`, with CFG on a
   `(1.0, 2.0)` schedule — 2.2–2.4× over-dispersion, enough to turn a +6 % model into a −3 %
   reading. Run **both** sources; the gap between them is diagnostic.

---

## 5. The next task: the B21 conditioning track

The gate will not move until **C** in Phase 1a (`cond_summary` → latent) clears 10 %. It currently
reads 4.9–6.8 %. **Re-entering 1b on any other cell tests nothing** — every cell carries the same
summarizer.

**Root mechanism (B21 §6, found with agent-B).** Every `LaplaceAE` pretraining head reads
`ctx_mean = context.mean(dim=1)` — `llapdiffusion/models/summarizer.py:637`, and lines 639–642 for
the four decoders. So of 336 token directions, **335 are unsupervised**; the objective only ever
constrains their mean. `COND_NORM_MODE="sample"` then removes exactly the one direction that *was*
supervised. That is the defect to fix, and **it has not been attempted** — everything tried so far
works downstream of it.

**Already tried, so do not repeat as if new:** `COND_POOL_USE_RAW=True`, `SUM_FT_MODE=all`,
`COND_NORM_MODE="global"` (5 seeds, inconclusive — see §3).

**Two named leads, both still open**, in the plan's debugging-track section:

1. **B15's residual gap.** Fixing per-batch conditioning normalization moved R²_cond from −0.001
   to **+0.210**, against a ridge ceiling of **+0.747** on the model's own `cond_summary`. The gap
   is unexplained and upstream of everything.
2. **`block_summary_adaln = False`.** Conditioning reaches the trunk only via cross-attention into
   K modal tokens whose residues come from `x_t`; at high noise those tokens are noise-dominated.
   **Phase 1b confirmed this independently**: `oneshot` at `t = T−1` scores corr 0.1903 / 0.1931
   against the trajectory's 0.3702 / 0.3395 — the model recovers half its correlation only *after*
   `x_t` becomes informative, and an ideal model would order these the other way. The two seeds
   land 1.5 % apart, so it is architectural, not a draw.

**How to know it worked:** re-run `llapdiff-ridge-probe --dataset-key noaa_uk --pred 168
--tokens 8`. Success is **C ≥ 10 %**, not a CRPS improvement — CRPS on this pipeline is dominated
by a per-entity prior and is not a health check (see the plan's Background).

---

## 6. Operational hazards specific to this repo

- **`PYTHONPATH=$LLAPDIFF_SRC` is not optional.** A sibling checkout is installed editable in the
  venv; without it, `import llapdiffusion` silently resolves elsewhere and edits appear to do
  nothing. See `CLAUDE.md`. Verify with
  `python -c "import llapdiffusion; print(llapdiffusion.__file__)"`.
- **`EARLY_STOP` counts evals, not epochs.** Patience is `EARLY_STOP × DOWNSTREAM_EVAL_EVERY`.
- **Do not launch a training run without setting `EPOCHS` explicitly.** The default is 600; a
  forgotten override once queued ~32 h of work. Validate override JSON with `json.load` *before*
  launching.
- **`*.json` globs over a results directory will count override files**, not just results.
- **Campaign eval profile** (~61× cheaper val, and what every recorded number above used):
  `LLAPDIFF_EVAL_STEPS=16 EVAL_NUM_SAMPLES=5 EVAL_MAX_BATCHES=48 EVAL_SUBSET_MODE=stride
  EVAL_SEED=4242 VAL_DIAG_EVERY=5`.
- **A second agent (agent-B) shared this filesystem.** Its channel is `w_docs/agents_communication.md`
  and its results are in `w_docs/results_nonlinear_probe.md`. Its shared-variance column supersedes
  an earlier one of mine; the leave-one-out figures (noaa_uk 76.5 %, noaa_us 46.8 %, crypto 42.4 %,
  us_equity 34.2 %) are the mediator of record. If another agent is run again, give it the same
  write restrictions recorded in `w_docs/HANDOFF_nonlinear_probe.md`.
- **415 tests pass** (`python -m pytest tests/ -q`). Keep them passing; several encode bugs that
  recurred (B16 EMA/raw, the `blocked_purged_split` collapse, the degenerate-window filter).

---

## 7. Open questions nobody has answered

1. Should `finetuning/results/window_audit.md` and the other results files be force-added to git,
   or is the tracked-copy arrangement in §2 the permanent answer? Asked twice, unanswered.
2. Should `cmd-uq-tooling-and-b21` be merged to `main`, or stay a long-lived branch?
3. The shared-variance reproduction discrepancy (my noaa_us 42 % vs agent-B's 49.0 %, drop rates
   11.4 % vs 1.7 %) was resolved by adopting agent-B's column wholesale, **not** by finding the
   bug. The two scripts filter differently and the cause was never identified.
