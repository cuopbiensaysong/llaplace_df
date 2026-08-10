# CMD UQ — plan of record for a defensible Table 3

**Written 2026-08-07.** Supersedes the *gate logic* of `CMD_UQ_RECOVERY_PLAN.md` (Phases 0–2);
that document's Phases 3–6 remain the specification for the U3 ablation itself and are carried
forward here by reference. Read `CMD_CAMPAIGN_STATE.md` §1b/§1c first for how we got here.

## Why this file exists

The recovery plan gates the campaign on a latent-space criterion. **Four independent
interventions have now shown that criterion is anti-correlated with the deliverable:**

| arm | stage 1 | pool norm | `ddim` resc | Δ gate | val CRPS | Δ CRPS |
|---|---|---|---|---|---|---|
| control (2 seeds) | shipped | inherit | 6.41 % | — | 0.3280 | — |
| `b23only` | shipped | global | 4.82 % | −1.59 | 0.31513 | −0.013 ✅ |
| `s1only` | entenc | inherit | 3.01 % | −3.40 | **0.30179** | −0.026 ✅ |
| both (2 seeds) | entenc | global | 2.09 % | −4.32 | 0.3197 | −0.008 ✅ |

Every arm improves the forecast and degrades the gate. Mechanism (B20/B24): a faithful stage 1
(round-trip 0.922 → 0.995) moves genuinely **unpredictable** per-entity content into the latent,
so the *fraction* of latent variance explained falls while the decoded forecast improves.

**A good UQ result does not need a strong mean. It needs an honest mean and a law that can
express heteroscedasticity.** The campaign conflated those two, which is why effort went into
"does the mean carry signal" while every UQ-specific defect sat untouched.

---

## Phase 0 — data-space entry condition ← THE DECISION POINT

**Threshold, frozen before measuring: ≥ 10 % RMSE reduction vs predict-the-mean, in DATA space,
on val.** This is not a new number — it is the recovery plan's own "usable headroom" bar,
applied in the space it was written for instead of the whitened latent it was misapplied to
(B24). Report it also as a fraction of the 28.6 % that a linear probe on raw history achieves.

Measured by `finetuning/results/table3/data_space_entry.py`: take the model's predicted latent
(both mean sources), decode through the real VAE decoder, score against the targets and against
a predict-the-mean baseline in data space. No UQ head required, so it runs on the S1 arms.

**Stop conditions.**
- **≥ 10 %** → proceed to Phase 1.
- **< 10 %** → **STOP.** noaa_uk h=168 does not support a UQ table; the answer is a different
  cell, not more fixes. Report and wait.
- Report the round-trip ceiling (91.3 %) alongside, so the reader can see how much of the gap is
  forecasting and how much is reconstruction.

### 🔴🔴 RETRACTED — the first execution below was WRONG, on two of my own errors

**Do not act on the "STOP" section that follows.** Re-measured with the correct protocol, the
model **beats persistence by 17.8 %**. Two mistakes, both mine, both in the measurement:

1. **I scored a single sample, not the ensemble mean.** `generate(eta=0)` returns ONE draw from
   a random `x_T`. The point forecast is `E[decode(z)]` over draws — the thing the trainer's
   CRPS is built on. Averaging 12 draws moves the model from −72 % to −21 % on its own.
2. **Classifier-free guidance was on**, at the shipped `(1.0, 2.0)` ramp, and it is
   **monotonically harmful here**.

#### ✅ Phase 5 (the U1 guidance audit) — executed early, because it turned out to be the bug

300 val windows, 12 draws, `s1only` seed 0, decoded then averaged. The recovery plan mandates
this audit "before any calibration conclusion" and records that the PhysioNet U3 run violated it;
it had **never been run on this cell**.

| guidance | RMSE | vs persistence | vs val-mean | CRPS |
|---|---|---|---|---|
| **w = 1.0 (off)** | **0.4296** | **+17.83 %** | −7.73 % | **0.2565** |
| w = 1.25 | 0.4373 | +16.36 % | −9.67 % | 0.2604 |
| w = 1.5 | 0.4486 | +14.18 % | −12.51 % | 0.2667 |
| w = 2.0 | 0.4754 | +9.07 % | −19.22 % | 0.2826 |
| ramp (1.0, 2.0) p=0.3 — **the shipped default** | 0.4513 | +13.67 % | −13.19 % | 0.2688 |
| *persistence* | 0.5228 | — | −31.1 % | — |

**Strictly monotone in w, on RMSE and CRPS alike. w = 1.0 wins both.** Mechanism: CFG extrapolates
along (conditional − unconditional); when the conditioning carries little signal that difference
is mostly noise, so guidance amplifies noise. This is also the measured source of the 2.2–2.4×
over-dispersion recorded in §1b.

⇒ **FREEZE `GUIDANCE_STRENGTH = 1.0` for noaa_uk h=168.** Do not inherit the global `(1.0, 2.0)`
default. Beyond the accuracy gain this is a *validity* requirement: w > 1 sharpens the predictive
distribution and is indistinguishable from miscalibration, so every PIT/coverage number in
Table 3 would otherwise be confounded.

#### Where the entry condition actually stands

| baseline | model (w=1.0, ensemble mean) | verdict |
|---|---|---|
| **persistence** — repeat the last observed value | **+17.83 %** | a real forecast |
| predict the **val** mean — what I froze the threshold against | **−7.73 %** | fails my 10 % bar |

**By the letter of the threshold I froze, Phase 0 still fails**, and I am not going to switch
baselines after the fact to make it pass. But the baseline deserves scrutiny that it did not get
when I froze it: the *eval-split* mean is effectively **climatology**, it beats persistence by
31 %, and beating climatology by 10 % at a 7-day horizon is a very high bar in weather
forecasting. **Whether the entry condition should be persistence-relative is a decision for the
owner**, and it should be made on its merits, not on this measurement.

What is no longer in doubt: there IS a forecast here — it beats the standard trivial baseline by
17.8 % — so a UQ table is not obviously vacuous, which is what the section below claimed.

---

### 🛑 SUPERSEDED — first execution, retained for the record (its verdict is retracted above)

600 val windows, `--weights ema`, both mean sources, plus an **oracle** scale correction (the
optimal scalar fitted *on val* — a courtesy the model does not have, included so "no signal" is
distinguished from "signal at the wrong scale"):

| | control — shipped stack, the campaign's own Phase-1b seed 0 | `s1only` — the best repaired stack |
|---|---|---|
| `oneshot` | −116.13 % | −127.74 % |
| `ddim` | −73.75 % | −75.63 % |
| **`oneshot` + oracle scale** | **−1.12 %** (c = **−0.018**) | **−0.17 %** (c = 0.018) |
| **`ddim` + oracle scale** | **−1.11 %** (c = **−0.027**) | **−0.21 %** (c = 0.003) |
| round-trip ceiling | 65.47 % | **89.55 %** |

**The forecast is far worse than trivial baselines, on both stacks.** This is a property of the
**pipeline**, not of this session's changes — the control is the campaign's own gate checkpoint.

#### 🔴 Correction to the first reading of this table, and the baseline trap behind it

The oracle-rescaled rows were over-interpreted on first pass. With `c ≈ 0` the "rescaled model"
**collapses to the val-mean oracle itself**, so scoring ≈ 0 % against a val-mean baseline is a
tautology, not a measurement. The rows that mean something are the raw ones — and they are worse.

The baseline definition also had to be checked, because it swings the verdict by 100 pt:

| predictor (600 val windows, s1only seed 0) | RMSE |
|---|---|
| **model** (`ddim`, decoded) | **0.6861** |
| predict **val** mean — the gate's convention | **0.3989** |
| predict **train** mean — honest, no leak | 0.9294 |
| **persistence** — repeat the last observed value | **0.5463** |

| model's RMSE reduction vs | % |
|---|---|
| predict val mean | **−71.98** |
| predict train mean | +26.18 |
| **persistence** | **−25.58** |

⇒ **The model loses to persistence by 25.6 %.** That is the baseline-independent statement, and
it is the one that matters: a forecast that cannot beat "repeat the last value" cannot anchor a
UQ table. It beats the *train* mean (+26 %) only because it knows the current level, which
persistence also knows and exploits better.

⚠️ **The gate's `baseline_rmse_predict_mean` is computed on the EVAL split**, so it leaks that
split's level — here it is **2.3× stronger** than the honest train-mean baseline (0.399 vs
0.929), because val is a low-variance regime. Even *persistence* loses to it by 37 %. Any future
threshold must name its baseline; "predict-the-mean" is not one number.

**It also explains everything else measured.** The recovery plan's background already warned that
disabling conditioning entirely costs only CRPS 0.2606 → 0.2990, i.e. **CRPS ≈ 0.30 is the
no-forecast level**, and that "CRPS on this pipeline is dominated by a per-entity prior, not by
the forecast". This is that warning, quantified in the right space with a baseline attached:

- why every conditioning intervention failed to open the gate — there was nothing to deliver;
- why the gate and CRPS dissociate across four arms — both measure noise around zero signal;
- why "the denoiser sits at the ridge ceiling" — in data space that ceiling is ≈ 0;
- why CRPS 0.3280 → 0.30179 looked like progress: it is movement inside an unconditional model.

**⇒ No UQ table is obtainable here.** Calibrating intervals around a forecast equal to the mean
is true but vacuous. Phases 1–3 below are NOT started.

### The number that says where to look next

A ridge regression on **128 raw history numbers achieves 28.6 %** on this exact cell. The
pipeline delivers **0 %**. So the loss is total, and the decomposition localises it:

| stage | delivers |
|---|---|
| raw history → target (linear) | **28.6 %** |
| → after the summarizer (`B`, best view) | 14.2 % |
| → conditioning → latent (`C`, linear probe) | 7.3 % |
| → **the trained denoiser, in data space** | **≈ 0 %** |

The largest single drop is the **last** one: a ridge regression from `cond_summary` to the latent
beats a trained 10.2 M-parameter denoiser that sees the same input. **The defect is in stage 3**,
and everything this campaign has spent months on is upstream of it.

**Cheapest decisive follow-up (~30 min, CPU, no training):** fit ridge `cond_summary → latent` on
train, decode its val predictions through the VAE, and score with this same script. If the ridge
solution scores materially above 0 % in data space where the denoiser scores 0 %, the denoiser is
the defect and the conditioning work is a distraction. That is a *forecasting* investigation, not
a UQ one.

## Phase 1 — the four UQ-specific defects (no GPU, ~3 h)

None of these were ever tested by the gate, and each corrupts Table 3 independently.

| # | fix | why | status |
|---|---|---|---|
| 1a | `COND_POOL_NORM_MODE="global"` **for the U3 arms** | without it `chirp_field.uq_params` sees no history: `p0`/`q`/ρ̄ identical for every window, so A2/A3 score a structurally homoscedastic law (B23). A validity issue, not tuning | config, default is `"inherit"` — arms must opt in |
| 1b | **`CHIRP_UQ_INIT_VAR ≈ 1.0`** (not the 1e-2 library default) | Phase 3 rule: init the predicted variance near the residual scale. On this stack `latent_mse ≈ 0.96`, so the default is a **~100× mismatch** — the exact failure that made PhysioNet 3 % covered at nominal 90 %, with CRPS moving < 0.001 | set per run |
| 1c | **`ensemble_pit`** for A1 | `pit_calibration_error` / `reliability_curve` are law-agnostic; only `gaussian_pit` assumes a law. One helper gives A1 the same calibration metrics as A2/A3, scored by the *same* estimator | new function |
| 1d | **matched means across arms** | A1 uses a 25-draw ensemble mean, A2 a single DDIM pass, so the arms differ in mean *and* spread; CRPS gaps are partly location differences. Add an ensemble mean source to `AnalyticLawSampler` | code |

⚠️ **1d dissolves the efficiency headline for A1/A2** — with means matched, A2 costs what A1
costs. Report `A1 / A3` as the method-level speedup and label A1-vs-A2 "calibration at matched
cost", per the recovery plan's own note.

## Phase 2 — U3 training on the right stack

**Stack: `entenc` stage 1 + `COND_POOL_NORM_MODE="global"`.** Chosen deliberately: it is *not*
the best CRPS arm (`s1only` is, 0.30179 vs 0.3197), but `s1only` leaves the analytic law
homoscedastic and so cannot support a UQ ablation. Giving up ~0.018 CRPS buys arms that measure
what they claim to.

3 arms × 5 seeds, driver `finetuning/run_u3_uq.py`. ~150 GPU-h; ~75 h wall on 2 GPUs.

## Phase 3 — freeze guidance, then score

`llapdiff-u1-sweep` on val per arm, `w ∈ {1.0, 1.25, 1.5, 2.0}` × `GEN_STEPS ∈ {16, 32, 64}`,
`--weights ema`. **Guidance w > 1 sharpens the predictive distribution and is indistinguishable
from miscalibration**, so it must be frozen before any arm is scored. Also addresses the
measured **2.2–2.4× over-dispersion** of `generate(eta=0)` (random `x_T` + CFG), which inflates
A1's intervals directly.

Then Table 3: `CRPS ± std | MSE ± std | PIT-ECE | coverage @ 0.8 | mean interval width | wall (s)
| denoiser passes`.

## Standing rules carried forward

- `--weights ema` on every eval (B16). Campaign eval profile = all six `LLAPDIFF_*` variables;
  confirm the six-key log line.
- Score CPU or GPU consistently — they differ 0.4–0.8 % relative; never mix within a table.
- ≥ 2 seeds for any decision; CRPS seed band is **0.036**.
- Concurrent stage-3 runs with **different** stage-1/2 artifacts need separate
  `DIFF_PRECOMPUTE_DIR` — they otherwise race and delete each other's cache.
- Never overwrite a frozen artifact; new architecture ⇒ new filename.
- Register predictions before launching. *(Scorecard so far: 1 of 5 correct — which is the
  argument for measuring rather than reasoning.)*

---

## 🟢 Table 3 EXISTS — 2026-08-10, noaa_uk h=168, val, 2 seeds

First numbers with every fix in force. Provenance is recorded per row in each eval JSON's
`resolved` block: `vae_entity_encode=True`, `mean_source=ensemble25`, `variance_timestep=999`,
`guidance_strength=1.0`.

| Arm | CRPS | MSE | PIT-ECE | cov @ 0.8 | width | **passes/batch** | wall (s) |
|---|---|---|---|---|---|---|---|
| A1 sampled (25 draws) | **0.3119 ± 0.0020** | 0.3165 | 0.0250 ± 0.0278 | 0.7602 ± 0.0326 | 1.178 | 1600 | 1681 |
| A2 analytic, matched mean | 0.3184 ± 0.0004 | 0.3271 | 0.0446 ± 0.0020 | 0.7098 ± 0.0680 | 1.088 | 1604 | 1725 |
| A3 one-shot analytic | 0.3555 ± 0.0236 | 0.4004 | 0.1000 ± 0.0205 | 0.7225 ± 0.0702 | **1** | **21** |

Both pre-flight gates PASS (G-b NLL health, G-c one-shot conditioning), the first time in this
campaign.

### What is resolved at n=2, and what is NOT

**Resolved — A3's efficiency.** 1 denoiser pass against 1600 (**1600×**, 82× wall) for CRPS
+14 % and PIT-ECE 0.100 vs 0.025. The ordering holds on both seeds. This is the method-level
speedup the plan says to report; the A1/A2 ratio is 1.00 by construction after Fix 1d.

**Resolved — CRPS ordering.** A1 < A2 < A3 on **both** seeds (0.3105/0.3133 < 0.3181/0.3186 <
0.3388/0.3722). Sampled beats analytic by ~2 %, consistently.

🔴 **NOT resolved — the A1-vs-A2 calibration comparison, which is what this table exists for.**
The arms **swap ranking between seeds**:

| | seed 0 | seed 1 |
|---|---|---|
| A1 coverage @ 0.8 | 0.7371 | **0.7832** |
| A2 coverage @ 0.8 | **0.7578** | 0.6617 |
| A1 PIT-ECE | 0.0446 | **0.0053** |
| A2 PIT-ECE | **0.0432** | 0.0460 |

A1's PIT-ECE is 0.0250 ± **0.0278** — the seed std exceeds the mean, and an 8× swing between two
seeds. **Two seeds cannot separate these arms.** Any claim that the analytic law matches (or
misses) the sampled ensemble's calibration needs the pre-registered 5 seeds; A2's own spread is
tight (0.0446 ± 0.0020), so the noise is on the A1 side.

⚠️ **Wall-clock is not a clean measurement here** (1681 ± 1044 s): the two seeds ran concurrently
on different GPUs with contention. Quote `denoiser_passes_per_batch`, which is exact and
architecture-independent — this is why it was added.

### Next

Seeds 2–4 (~3.3 days at 2 GPUs) to reach the pre-registered 5. The calibration columns are the
deliverable and they are the ones that need the seeds; CRPS and the A3 efficiency claim are
already stable.
