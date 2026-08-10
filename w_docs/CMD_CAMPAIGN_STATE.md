# CMD UQ campaign — state of play

**Start here.** Then `CMD_UQ_TABLE3_PLAN.md` for the live plan and Table 3.
Last updated 2026-08-10.

> ⚠️ **This file was rewritten 2026-08-10.** Its previous header carried a 🛑 banner reading
> *"No UQ table is obtainable on noaa_uk h=168 with this pipeline"*. **That verdict is
> withdrawn** — it rested on two measurement errors (single-draw scoring and guidance left on),
> both since corrected. Table 3 now exists and both pre-flight gates pass. The full previous
> text is at `archive/2026-08-uq-campaign/CMD_CAMPAIGN_STATE.md`.

---

## 1. Where things stand

| | |
|---|---|
| cell | **noaa_uk h=168**, arm `d` (chirp − head), `predict_type=x0` |
| stack | entity-encoded stage 1 (`VAE_ENTITY_ENCODE`) + shipped summarizer + `COND_POOL_NORM_MODE="global"` |
| Table 3 | **exists**, 2 of 5 seeds, val — see `CMD_UQ_TABLE3_PLAN.md` |
| gates | **G-b and G-c both PASS** (first time in the campaign) |
| branch | **`cmd-uq-tooling-and-b21`**, not `main` |
| next | seeds 2–4; then the two open questions in §4 |

`main` (`f1788e3`) contains **none** of this campaign. Start from the branch.

---

## 2. Settled — do not re-litigate, do not re-run

| finding | evidence |
|---|---|
| noaa_uk h=168 is the right cell | Gate 0 = 0.922 (0.995 entity-encoded), forecastable at A = 28.6 %, 4.3× **under**-parameterized |
| PhysioNet h=12 is a smoke-test cell only | 13 train windows vs 11.75 M params (B19) |
| crypto / us_equity cannot serve a gate | 90 / 72 val windows; over-parameterized 6.0× / 8.9× |
| bms_air is **VOID** at every horizon | A ≤ +1.8 % across h ∈ {24,48,96,168}; the target (PM2.5) is not forecastable, not a pipeline defect |
| **noaa_us h=168 is the Phase-2 cross-check cell** | A = 23.4 %, same target variable, independent sensor network |
| Gate 0 does **not** license a cell | bms_air has the highest Gate 0 (0.941) and zero forecastable signal. Gate 0 and 1a's **A** are independent |
| the B21 conditioning gap is genuine **loss**, not nonlinear re-encoding | oracle MLP reproduces the linear gap to 0.3 pt (`results_nonlinear_probe.md`) |
| **335 of 336 summarizer token directions are unsupervised** | every `LaplaceAE` pretraining head reads `context.mean(dim=1)` (`summarizer.py:637`); `COND_NORM_MODE="sample"` then removes the one direction that *was* supervised |
| the denoiser is not the bottleneck | Phase 1b: rescaled 6.5 % vs a ridge ceiling of 5.6 % — +0.9 pt for 10.2 M params |
| CRPS seed band is **0.036** | every decision needs ≥ 2 seeds |
| **guidance is monotone-harmful here; freeze `w = 1.0`** | w=1.0 → RMSE 0.4296 / CRPS 0.2565; shipped (1.0,2.0) ramp → 0.4513 / 0.2688; w=2.0 → 0.4754 / 0.2826 |

### The Phase-1 latent gate is retired

Four independent interventions moved the gate axis and val CRPS in **opposite** directions,
without exception. Mechanism: a faithful stage 1 (round-trip 0.922 → 0.995) moves genuinely
*unpredictable* per-entity content into the latent, so the fraction of latent variance explained
falls while the decoded forecast improves. A criterion anti-correlated with the objective cannot
gate the campaign. `CMD_UQ_TABLE3_PLAN.md` replaced it; the four-arm attribution table is there.

---

## 3. Metric traps — each produced a wrong verdict here before it was caught

Read these before designing any measurement.

| trap | cost when missed |
|---|---|
| **Never score a single draw.** `generate(eta=0)` returns ONE sample from a random `x_T`; the point forecast is `E[decode(z)]` over draws | 50 points of RMSE reduction; and 51 % of val `latent_mse` is draw noise (0.99 → 0.48 at a matched ensemble mean) |
| **Name the baseline.** `baseline_rmse_predict_mean` is computed on the **eval** split, so it leaks that split's level — 2.3× stronger than an honest train-mean baseline | swings a verdict by 100 pt |
| **Never mean-pool the summary tokens** in a probe | −0.2 % where the strided view scores 14.2 % — manufactures a FAIL |
| **Select ridge alpha on a chronological train holdout, never on val** | moved a headline 6.2 % → 9.8 %, across a decision threshold |
| **An "oracle-rescaled" model with c ≈ 0 collapses to the baseline itself** | scoring ≈ 0 % against it is a tautology, not a measurement |
| **Marginal calibration is not conditional calibration** | pooled PIT-ECE 0.05 while per-decile coverage runs 0.78 → 0.997 (§4) |
| **`val_diag_mse_raw` is an MSE under `mse` and an NLL under `gaussian_nll`** | produced a phantom "340× degradation" |

---

## 4. Open questions

1. **Seeds 2–4** to reach the pre-registered 5. The calibration columns are the deliverable and
   they are the ones that need the seeds; CRPS and the A3 efficiency claim are already stable.
2. 🔴 **Is the analytic law's heteroscedasticity informative?** Measured on seed 0:
   `corr(predicted variance, squared residual) = +0.016`, Spearman +0.057. Predicted variance
   spans 14.5× across deciles while the actual squared error spans 1.6× and does not track it.
   The law is *marginally* near-calibrated only because over- and under-confident deciles cancel
   when pooled. **Table 3's columns are all pooled statistics and cannot see this.** Confirm on
   seeds 2–4; consider a conditional-calibration column.
3. **The noaa_uk pre-registration was never written.** `PREREG_U3.md` is scoped to PhysioNet
   h=12. Recovery plan Phase 4 requires a new file before the test split is touched.
4. Should `cmd-uq-tooling-and-b21` merge to `main`? Asked repeatedly, unanswered.

---

## 5. Operational hazards specific to this repo

- **`PYTHONPATH=$LLAPDIFF_SRC` is not optional.** A sibling checkout is installed editable in the
  venv; without it `import llapdiffusion` silently resolves elsewhere and edits do nothing.
  Verify: `python -c "import llapdiffusion; print(llapdiffusion.__file__)"`.
- **`EARLY_STOP` counts evals, not epochs.** Patience = `EARLY_STOP × DOWNSTREAM_EVAL_EVERY`.
- **Always set `EPOCHS` explicitly.** The default is 600; a forgotten override once queued ~32 h.
- **`--smoke` uses `setdefault("EPOCHS", 3)`**, so an `EPOCHS` in `--overrides-json` wins over it.
  Give the smoke its own override file.
- **Campaign eval profile** — all six variables carry the `LLAPDIFF_` prefix, and the run must
  log `[run_trial] validation-protocol profile from env: {...}` with **six** keys:
  ```bash
  export LLAPDIFF_EVAL_STEPS=16 LLAPDIFF_EVAL_NUM_SAMPLES=5 LLAPDIFF_EVAL_MAX_BATCHES=48 \
         LLAPDIFF_EVAL_SUBSET_MODE=stride LLAPDIFF_EVAL_SEED=4242 LLAPDIFF_VAL_DIAG_EVERY=5
  ```
  This is the *in-training* protocol only. Reported numbers use `prefix="TEST"` (steps 64,
  25 samples) and are unaffected by it.
- **Concurrent seeds need separate `DIFF_PRECOMPUTE_DIR`** — they otherwise race and delete each
  other's cache.
- **Concurrent drivers must not share `state.json` blindly.** `save_state` now merges per trial
  key; before that fix, seed 1's save silently erased seed 0's completed S1 record.
- **`.gitignore` ignores `finetuning/results/*`.** Tracked copies:
  `w_docs/CMD_PHASE0_WINDOW_AUDIT.md`, `w_docs/CMD_PHASE1B_GATE.md`. Edit the tracked copy.
- 🔑 **`.git/config` holds a GitHub PAT in plaintext** in the `origin` URL. Readable by anyone
  with filesystem access. **Rotate it.** Left as-is because changing a remote is the owner's call.
- **Keep the tests passing** — `python -m pytest tests/ -q`, currently **490**. Several encode
  bugs that already recurred once.

---

## 6. Document map

| file | what it is |
|---|---|
| **this file** | state, settled facts, traps, hazards |
| **`CMD_UQ_TABLE3_PLAN.md`** | the live plan of record + Table 3 |
| `CMD_BUG_REPORT.md` | the durable bug ledger (B1–B24) |
| `CMD_UQ_RECOVERY_PLAN.md` | standing rules + the U3 phase spec. **Its Phase 0–2 gate logic is superseded** |
| `CMD_RUNBOOK.md`, `USAGE.md`, `DEVELOPER_GUIDE.md` | how to run things |
| `results_nonlinear_probe.md` | the B21 loss-vs-re-encoding result |
| `PREREG_U3.md` | PhysioNet-scoped prereg — **superseded, replacement unwritten** |
| `claude_communication.md` | live agent channel; process log, not a source of record |
| `archive/2026-08-uq-campaign/` | full pre-cleanup originals of every file above |
