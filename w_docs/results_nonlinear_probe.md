# Is the B21 conditioning gap information LOSS, or nonlinear re-encoding?

**agent-B, 2026-08-02/03.** Cell: noaa_uk h=168, val, `COND_NORM_MODE="sample"` unless stated.
All numbers are **% RMSE reduction vs predict-the-mean**, metric identical to
`llapdiff-uq-eval` / `ridge_reduction`; alpha and hyperparameters selected strictly inside
train; **the test split was never touched.**

> **Condensed 2026-08-10.** Full 892-line original, including per-cell hyperparameter tables and
> the protocol narrative: `archive/2026-08-uq-campaign/results_nonlinear_probe.md`.

---

## 1. Answer — genuine LOSS

| probe | M_hist | M_cond | M_cond / M_hist |
|---|---|---|---|
| ridge (honest) | **28.61** | **14.23** | **49.7 %** |
| MLP blocked (honest, 3 seeds) | 20.78 ± 1.89 | 13.33 ± 0.94 | 64.1 % |
| **MLP ORACLE (leaky upper bound)** | **28.30** | **14.15** | **50.0 %** |
| MLP oracle, seed-averaged | 28.15 ± 0.35 | 11.80 ± 1.35 | 41.9 % |

The decision rule set ≥ 80 % for "present but nonlinearly encoded". An MLP with **config and
epoch chosen by looking at val** reaches 50.0 % — landing within **0.3 pt** of the linear gap.
**The gap is not an artefact of linearity.**

Defensible sentence: *128 raw history numbers beat 8 192 dimensions of summarizer output by
~2×, linearly or nonlinearly.*

**One correction to the framing:** `cond_summary`'s honest figure is **~14 %, not 7.7 %**. The
2 048-dim view is an impoverished *view*, and nonlinearity partly recovers tokens the stride
throws away (7.74 → 12.23) rather than extracting hidden structure. On the rich 8 192-dim view
nonlinearity buys **−0.08 pt**.

### What this does NOT license

1. **It does not reinstate B21's retracted "discards ~½ the signal" headline.** A probe is a
   **lower bound on usable information**. B21's own `SUM_FT_MODE="all"` is the counterexample:
   best forecast of any run (0.3240 vs 0.3627) with the *worst* linear probe (latent 1.4 % vs
   5.7 %). Probe score and forecast quality demonstrably dissociate here.
2. **It says nothing about the denoiser**, which cross-attends over all 336 tokens; this probed
   a strided 8/32-token view. A sequence model over the full token axis was never run.
3. **The latent remains a bad yardstick.** Every probe scores worse on `mu_norm` than on the raw
   target, and the oracle MLP does not rescue it.

---

## 2. The mechanism — 335 of 336 token directions are unsupervised

`LaplaceAE.forward` returns `context` of shape `[B, S=336, Hc=256]` — 86 016 numbers per window,
and that is what the denoiser consumes. But **every** pretraining reconstruction head reads
`ctx_mean = context.mean(dim=1)` (`summarizer.py:637`) — 256 numbers.

Verified by perturbing `context` with a delta of norm 38.35 constructed to have exactly zero
token-mean: every head moves by **float32 noise** (2.98e-08 to 1.04e-07). So the pretraining loss
can only constrain the token-**mean**; the other 335 directions lie in its null space.

And `normalize_cond_per_batch(mode="sample")` — the default — reduces over that same token axis,
subtracting exactly that mean (verified: max |token-mean of `cond_summary`| = 1.2e-06).

⇒ **The one component the summarizer's objective ever supervised is exactly the component the
default normalisation removes before the denoiser sees it.**

This predicts B21's strangest row: the erased 256-dim mean alone scores **12.4 %**, beating the
entire kept 2 048-dim normalised summary at 7.7 %. It is a statement about which directions are
*supervised*, **not** that the rest are empty — they are a deterministic function of the history
and score 14.2 % strided. What they lack is training pressure.

---

## 3. `COND_NORM_MODE="global"` — substitutes, not additive

Registered prediction: **additive**. Result: **substitutes.** Recorded as a failed prediction.
(Both agents held this prior; both were wrong.)

| view | norm | ridge | oracle MLP | nonlinearity gain |
|---|---|---|---|---|
| 2 048 | `sample` | 7.74 | **12.23** | **+4.49** |
| 2 048 | `global` | **12.28** | 10.37 | −1.91 |
| 8 192 | `sample` | **14.23** | 14.15 | −0.08 |
| 8 192 | `global` | 13.69 | 11.24 | −2.45 |

**Across the whole view × normalisation × probe-class grid the best raw-target number anywhere
is 14.23 % — plain ridge on the `"sample"` 8 192-dim view.** Neither level restoration, nor
nonlinearity, nor both beats it. Both compensate for an impoverished *view*; neither raises the
ceiling.

🔴 **A B21 number was corrected: `"global"` @8192 → raw target is 13.69 %, not 16.4 %.** This
reverses the claim at the best view — `"global"` helps the impoverished view (+4.54) but not the
rich one (−0.54). ⚠️ It does not cancel cleanly: on the **latent** at 8 192 `"global"` genuinely
helps (4.21 → 6.85). The two targets move in opposite directions.

**No verified mechanism for the shared ≈12.3 % ceiling.** The untested alternative: the 8-token
stride and the window level may both proxy the same slow seasonal component, which would make
them redundant for an uninteresting reason.

---

## 4. Protocol findings that changed the tooling

**The literal "last 20 % of train" selection holdout is pathological on this cell** — that band
is a shifted seasonal regime (grand mean −0.911 vs the fit portion's +0.183). Ridge fitted on the
first 80 % scores **−8.3 % on that holdout while scoring +28.8 % on val**. A selector that
anti-correlates with the target metric is not a selector.

Ridge survives it (a common offset cancels when *ranking* alphas) but early stopping does not:
it halts at epoch 4 and the raw-history MLP lands *below* ridge.

**Substitute, now the repo default** (`run_ridge_probe.py`, commit `ef0d679`): a **blocked,
purged** holdout — 3 contiguous blocks of 15 spread across the train span, with a 336-window
purge either side so no fit window shares context with a holdout window. This is the repo's own
`global_purged_horizon` convention applied inside train.

Consequence for published numbers: the raw-history rows moved in **opposite directions** —
→ raw target 27.3 → **28.6**, → latent 11.6 → **10.0**. The summary rows were unaffected.

🔴 **A defect this introduced, still unfixed.** `blocked_purged_split` bans up to `2 × 3 × purge`
windows, so on a small train set the alpha-selection fit set collapses — to exactly **zero** at
n_train ≈ 1 420, surfacing as a `TypeError` deep inside rather than a clear error. Severity is
bounded (the reported model is fitted on **full** train; only alpha selection degrades), and
every cell carrying a conclusion has ≥ 57 % fit. **Rule of thumb: `n_train ≳ 10 × purge`. A guard
belongs in the tool and was never added.**

**The MLP training loop was verified, not assumed:** a depth-0 (linear) model through the
identical loop reaches 28.36 % on val, reproducing ridge's 27.28 %, and agrees with closed-form
ridge to 0.01 pt on `hist → latent`. That is what licenses reading the MLP-vs-linear contrast as
a statement about the model class rather than the optimiser. ⚠️ **Do not quote the within-loop
MLP-vs-SGD-linear "gains"** (+9.95, +13.77) — SGD-linear is a bad linear probe at high dimension
(2.27 vs ridge's 7.74). Compare against ridge.

---

## 5. Other cells

### bms_air — VOID at every horizon

| horizon | best A | full panel |
|---|---|---|
| 24 | **+1.8 %** | −2.8 % |
| 48 | +0.5 % | −2.8 % |
| 96 | 0.0 % | thin |
| 168 | −0.3 % | n/a (23 val windows) |

**Best value anywhere across four horizons and every entity count: +1.8 %, against a 10 % gate.**
Monotone decline with horizon, the physically expected shape.

**The cause is physical, not a pipeline defect** — bms_air's target is **PM2.5**; noaa_uk's is
**temperature**. Ruled out as a level-shift artefact three ways (train→val shift 0.03 val-std;
−0.15 % against a *train*-mean baseline; level-free corr r = −0.025), and oracle alpha pins at
1e8, the top of the grid.

🔴 **The test agent-A designed here is VOID, not answered.** The falsification criterion was
"B ≈ A falsifies the mechanism story". B ≈ A was observed — but **vacuously**, because A ≈ 0. You
cannot measure whether stage 2 discards forecastable signal on a cell that has none.

### noaa_us — B21 replicates on an independent network, and harder

| probe | dims | → raw target | → latent |
|---|---|---|---|
| **A raw history** | 1 280 | **23.4 %** | 8.3 % |
| B/C `cond_summary`, 32 tok | 8 192 | **2.6 %** | 4.4 % |

**B/A = 11.1 %** — the summarizer delivers about a ninth of the linearly available signal, against
noaa_uk's half. This is the informative case bms_air could not provide: A is solidly non-zero, so
a null in B would have been genuine evidence *against* the mechanism. It is not a null.

**noaa_us h=168 is therefore the Phase-2 cross-check cell.** Same target variable, same geometry,
genuinely different sensor network, and its modal entity count *is* the full 40-station panel.

### The mediator — shared variance across the panel, leave-one-out

| cell | entities | with-self | **LOO** |
|---|---|---|---|
| noaa_uk | 4 | 87.9 % | **76.5 %** |
| noaa_us | 40 | 49.0 % | **46.8 %** |
| crypto | 113 | 43.2 % | **42.4 %** |
| us_equity | 221 | 34.7 % | **34.2 %** |

**Leave-one-out matters:** if the panel mean includes the entity itself, a 4-station panel carries
a 1/4 self-contribution against a 40-station panel's 1/40, inflating small panels arithmetically.
noaa_uk loses 11.4 pt to LOO, noaa_us 2.2 — **~24 % of the apparent gap was panel-size
arithmetic.** The mechanism survives; the mediator **saturates**, with nearly all the fall between
4 and 40 entities.

🔴 **A reproduction discrepancy was never resolved:** agent-A's noaa_us with-self read 42 % against
this 49.0 %, with drop rates 11.4 % vs 1.7 %. The two scripts filter differently and the cause was
never found. The column above is internally consistent across four cells and is the one of record;
**do not mix the two.**

---

## 6. What could not be established

1. **Nothing about what the *denoiser* can use.** A probe is a lower bound. Tightened from "not
   *linearly* accessible" to "not accessible to *this probe family*" — still a lower bound.
2. **One probe family**: a small MLP on a strided view, 36 configs, ≤50 epochs, one optimizer.
   The oracle bound is an upper bound *within the searched family only*.
3. **One cell, one frozen stack, one conditioning mode** for the main result; not replicated on a
   second stage-1/2 seed.
4. **No verified mechanism for the shared ≈12.3 % ceiling** (§3).
5. **The blocked-holdout geometry was not varied** — block count, positions and purge width
   untested for sensitivity.
6. **§2 is structural, not causal.** No summarizer was trained with a modified objective, so
   "supervising more token-directions would close the gap" remains a hypothesis.
7. **The test split was never touched**, by design, at any point.
