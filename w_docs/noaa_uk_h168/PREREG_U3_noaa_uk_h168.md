# Pre-registration — NOAA-UK h=168 (CMD-GP / CMD-D campaign)

**Status: DRAFT, not yet frozen.** Freezing requires §11 to be completed and committed.
**The test split may not be touched until this document is frozen.**


|                       |                                                                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Cell                  | `noaa_uk`, `PRED=168`, arm `d` (chirp core, no output head)                                                                         |
| Supersedes            | `PREREG_U3.md` (PhysioNet-scoped, retired by B19)                                                                                   |
| Written               | 2026-08-15                                                                                                                          |
| Repo state at writing | branch `review_2`, commit `dd37486`                                                                                                 |
| Author discipline     | every number in §2 was measured on **val** before this document existed and is declared as such; nothing here is presented as blind |


---



## 0. Why this document exists, and what it can and cannot claim

`CMD_UQ_RECOVERY_PLAN.md` §4 blocks the test split until a NOAA-UK-scoped pre-registration
exists. `PREREG_U3.md` §5 lists what the replacement must fix: thresholds stated against
*measured* baselines, outcome branches that cover the orderings actually observed, the
measured `CHIRP_UQ_INIT_VAR`, and a **conditional**-calibration criterion. All four are
addressed below.

**This is a confirmatory pre-registration, not a blind one.** The hypotheses in §4 were formed
from val measurements listed in §2. That is the intended use of a train/val/test split: val
generates and selects, test confirms once. What this document buys is that the reads, their
directions, their decision rules, their multiplicity control and their seed counts are fixed
*before* the test split is seen, so the test read cannot be reshaped by its own outcome.

Anything not listed in §4 is exploratory and will be labelled exploratory in the paper,
however interesting it turns out to be.

---



## 1. Frozen stack and cell

Identical artifacts for every arm — the parity requirement. Verified by SHA-256 (first 16 hex
digits) at the time of writing:


| artifact                     | path                                                                           | sha256[:16]        |
| ---------------------------- | ------------------------------------------------------------------------------ | ------------------ |
| stage-1 VAE (entity-encoded) | `ldt/vae/saved_model/noaa_uk/pred-168_ch-16_entity_entenc_elbo.pt`             | `11f691a6454363d7` |
| stage-2 summarizer           | `ldt/summarizer/saved_model/noaa_uk/168-16-summarizer.pt`                      | `1ec31480fd86ca3d` |
| s2 denoiser, seed 0          | `ldt/tuning/uq_u3/noaa_uk_h168/d/95d4fd4fb7/.../llapdiff_pred-168_best_ema.pt` | `928528c23314d71c` |
| s2 denoiser, seed 1          | `ldt/tuning/uq_u3/noaa_uk_h168/d/ba0ea55d79/.../llapdiff_pred-168_best_ema.pt` | `fdbcb18fae96d8f4` |
| s2 denoiser, seed 2          | `ldt/tuning/uq_u3/noaa_uk_h168/d/970d110e30/.../llapdiff_pred-168_best_ema.pt` | `4820f7c1853a0d44` |
| s2 denoiser, seed 5          | `ldt/tuning/uq_u3/noaa_uk_h168/d/fdba8c4e22/.../llapdiff_pred-168_best_ema.pt` | `90b94ffd705c08bd` |
| s2 denoiser, seed 6          | `ldt/tuning/uq_u3/noaa_uk_h168/d/85384e2afc/.../llapdiff_pred-168_best_ema.pt` | `e6ffe694dd31d28b` |
| s2 denoiser, seed 7          | `ldt/tuning/uq_u3/noaa_uk_h168/d/42dc3ccf3b/.../llapdiff_pred-168_best_ema.pt` | `feddb7c7aee213ec` |


Seeds 2–7 appended 2026-08-17 when Family B reached its registered size of 6. Seeds 0 and 1
were **re-hashed at the same time and reproduce the digests recorded above**, confirming the
two pre-existing checkpoints were not touched by the later campaign. The four new rows are the
`s2_diffusion_uq` checkpoints resolved from
`finetuning/results/uq_u3/noaa_uk_h168/d/state.json`, which is the same resolution path the
eval jobs use, so the hashed file and the scored file are the same file by construction.

Cell geometry (from `CMD_PHASE0_WINDOW_AUDIT.md`): 16,286 train / 2,183 val / 4,702 test
window starts, 4 entities per window, `d_z = 16`, `K = 128`, `CHIRP_NUM_BASIS = 4`. This is the
only cell in the project that is **under**-parameterised (4.3×) and it is why it is the gate
cell; PhysioNet h=12 (13/5/5 windows, B19) and crypto/us_equity (90/72 val windows) cannot
support these reads and are not used.

**Measured** `CHIRP_UQ_INIT_VAR` **for this cell: 1.0** (not the 25.0 calibrated for PhysioNet
h=12). Derived per `CMD_UQ_RECOVERY_PLAN.md` §3 from the matched-ensemble latent MSE ≈ 0.96;
the shipped checkpoints were trained at this value and it is fixed for all further arms.

---



## 2. What is already known from val (declared, not blind)

Every figure below is **val**, NOAA-UK h=168, 146 batches / 5,867,904 latent elements, seeds
0 and 1. Sources: `finetuning/results/cmd_gp/`.

**Baselines on this cell.** Honest train-mean MSE on val = **0.9185** (a constant fitted on
train, applied to val). The in-split mean baseline is 0.9184 — on this cell the two coincide
to 4 decimal places, so the 2.3× leak `PREREG_U3.md` §5 warns about does **not** bite here.
Both are reported anyway.

**Gate (variance decomposition).** `R = 0.6389` (seed 0), `0.6801` (seed 1); within-component
0.2146 / 0.2267, between-component 0.3798 / 0.4820. Stable from `M=2` to `M=25`.

**Frontier at** `M=25`**, 25-member ensembles.**


| arm                | ES (unb.) s0/s1 | VS s0/s1        | CRPS s0/s1      | cov@80 s0/s1  |
| ------------------ | --------------- | --------------- | --------------- | ------------- |
| CMD-D(25) mixture  | 32.804 / 32.667 | 4063.9 / 4075.6 | 0.4871 / 0.4885 | 0.705 / 0.736 |
| CMD-D + 25 samples | 33.534 / 33.056 | 4877.8 / 4594.1 | 0.4974 / 0.4949 | 0.605 / 0.655 |
| CMD-D(4) mixture   | 35.300 / 35.746 | 4159.6 / 4199.9 | 0.5264 / 0.5361 | 0.634 / 0.654 |


**One-pass arms (R10), matched LR grid** `{3e-4,1e-4,3e-5,1e-5}` **selected on val.**


| arm               | VE vs train-mean s0/s1 | ES s0/s1        | VS s0/s1        | PIT-ECE s0/s1   |
| ----------------- | ---------------------- | --------------- | --------------- | --------------- |
| CMD-GP (chirp)    | 8.69 % / 7.67 %        | 33.391 / 33.627 | 4232.1 / 4631.5 | 0.0073 / 0.0045 |
| diagonal Gaussian | 9.93 % / 10.14 %       | 33.098 / 33.057 | 5387.7 / 5461.4 | 0.0133 / 0.0157 |
| low-rank Gaussian | 9.77 % / 10.13 %       | 33.134 / 33.058 | 5385.7 / 5445.7 | 0.0143 / 0.0166 |
| Student-t         | 11.65 % / 11.85 %      | 33.022 / 32.991 | 6802.5 / 6827.3 | 0.0376 / 0.0381 |


**Seed-to-seed spreads (n=2, indicative only, not an estimate of σ).** `R` 0.041; ES: mixture
0.136, sampled 0.477, CMD-GP 0.236, diagonal 0.041; VS: mixture 11.7, sampled 283.7, CMD-GP
399.4, diagonal 73.7; CRPS: mixture 0.0015, sampled 0.0025.

**The paired contrasts the test read will confirm**, per seed (val):
`VS(diagonal) − VS(CMD-GP)` = 1155.6 and 829.9;
`VS(sampled 25) − VS(mixture 25)` = 813.9 and 518.5;
`VE(diagonal) − VE(CMD-GP)` = +1.24 and +2.47 points.

---



## 3. Arms carried to test

Only these. Adding an arm after freezing requires a dated amendment (§10).


| id  | arm                                                      | passes | depth |
| --- | -------------------------------------------------------- | ------ | ----- |
| A1  | CMD-D + 25 samples (conventional empirical ensemble)     | 1600   | 64    |
| A2  | CMD-D(M) mixture, `M ∈ {1,2,4,8,16,25}`, 25-member draws | 64M    | 64    |
| A3  | CMD-GP (Option A, one pass, chirp-modal law)             | 1      | 1     |
| A4  | same-encoder diagonal Gaussian head                      | 1      | 1     |
| A5  | same-encoder low-rank-plus-diagonal Gaussian head        | 1      | 1     |
| A6  | same-encoder Student-t head                              | 1      | 1     |


A3–A6 share the frozen encoder of §1 and an identical LR grid selected on **val**. A1 and A2
share the same reverse trajectories, so no A1-vs-A2 gap can be a generation artefact.

---



## 4. Pre-registered reads

**Two confirmatory families, plus threshold and descriptive reads that are not tested.**


|              | reads      | arms needed   | cost/seed        | seeds        | Holm threshold | Wilcoxon floor |
| ------------ | ---------- | ------------- | ---------------- | ------------ | -------------- | -------------- |
| **Family A** | T3, T5     | one-pass only | ≈45 min          | **10**       | 0.025          | n≥6 ✓          |
| **Family B** | T2         | diffusion     | ≈10 h × 3 stages | **6**        | 0.050          | n≥5 ✓          |
| threshold    | T1         | diffusion     | —                | 6            | n/a            | n/a            |
| descriptive  | T4, T6, T7 | both          | —                | as available | n/a            | n/a            |


The split is **by scientific question, and it is a choice that must be justified rather than
assumed**. Family A asks *is the chirp-modal structure carrying the result* (the novelty
control); Family B asks *does the structured law beat the conventional sampled ensemble*.
They support different claims and neither is used to shore up the other. Splitting families
does buy power, so we state plainly that the alternative — a single family of three at
`0.0167`, needing n≥6 for both halves — is also defensible and was rejected only because
Family B's seeds cost 13× Family A's. **If a reviewer prefers the single family, every read
here still clears it at the registered seed counts**, since 10 and 6 both exceed 6.

T1 is a **threshold** read (`R` against 0.10/0.25), not a hypothesis test, so it carries no
multiplicity burden and is excluded from both families. T4 is registered as a *direction*
(adverse to the method) and reported without a test — it is evidence about where the method
does not help, and inflating the family to test it would cost the reads that matter.

Primary metric family is **path-level**: energy score and variogram score (`p = 0.5`,
`w_rr' = |t̃_r − t̃_r'|^{-1}` truncated at the median gap, uniform channel weights). Secondary:
CRPS, MSE. Calibration: PIT-ECE, coverage at {50,80,90,95,99} with width, and the conditional
criterion of §4.7. All scores use the identical estimator with **both** the naive and unbiased
variants reported (§6).

### T1 — the gate (confirmatory)

`R = tr Cov_{z_T}(μ) / tr Cov(z₀|H)` on test, `M = 25`.
**Hypothesis:** `R > 0.25`. **Rule:** `R < 0.10` → Option A; `R > 0.25` → Option B; between →
inconclusive and both arms carried. Reported per seed and pooled.

### T2 — structured law vs sampled ensemble, at matched cost and matched ensemble

A2(`M=25`) against A1 on **VS** (primary) and **ES** (secondary).
**Hypothesis (one-sided):** A2 < A1 on both. **Test:** one-sided Wilcoxon signed-rank paired
by seed, α = 0.05, Holm-corrected within the family of §4.

### T3 — R10, the novelty control, on the axis the paper's claim lives on

A3 against A4 on **VS**.
**Hypothesis (one-sided):** A3 < A4. **Test:** as T2.

### T4 — R10, the novelty control, on the marginal axis

A3 against A4 on **variance explained vs the train-mean baseline** and on **ES**.
**Hypothesis (one-sided, pre-declared as ADVERSE to the method):** A3 is **worse** than A4 on
both — i.e. `VE(A3) < VE(A4)` and `ES(A3) > ES(A4)`. This direction is registered because val
shows it on both seeds; registering the direction we expect to lose on is the point of
pre-registering at all. Reported regardless of outcome.

### T5 — calibration

A3 against A4 on **PIT-ECE**. **Hypothesis (one-sided):** A3 < A4.

### T6 — `M`-convergence of the mixture

A2 at `M ∈ {1,2,4,8,16,25}` on ES, VS, CRPS, PIT-ECE, coverage@80.
**Descriptive, no test.** Pre-declared reading: if VS at `M=4` is below A1's VS, the paper may
state that path coherence is obtained at 6.25× fewer trajectories than the conventional
ensemble; if ES at `M=4` is above A1's ES, the paper must state that marginal sharpness is
**not**. Both halves are reported together or neither.

### T7 — `M=1` is structurally miscalibrated

A2 at `M=1` coverage@80 against nominal 0.8.
**Hypothesis:** materially below nominal. **Pre-declared reading:** with `R > 0.25` confirmed
in T1, a single component is missing a quantified fraction of the predictive variance by
construction, and its under-coverage is reported as structural, **not** as a tie and not as
something recalibration of `q_k, p⁰_k` could repair.

### 4.7 Conditional calibration (the criterion `PREREG_U3.md` §5 requires)

Pooled PIT-ECE hides the failure it is supposed to detect: on this project's own measurement,
predicted variance spanned 14.5× across deciles while squared error spanned 1.6×, with pooled
PIT-ECE looking healthy at 0.052 because over- and under-confident deciles cancel.

Therefore, for **every** arm reporting a law, we additionally report:

1. `corr(predicted variance, squared residual)` — Pearson and Spearman;
2. coverage@80 **stratified by decile of predicted variance**;
3. the **spread ratio**: (range of E[predicted var] across deciles) / (range of E[residual²]
  across deciles).

**Pre-declared reading:** an arm whose decile-stratified coverage varies by more than 0.15
across deciles is reported as **conditionally miscalibrated** even if its pooled PIT-ECE is
small, and its pooled calibration number is not quoted without this qualifier.

---



## 5. Equivalence margins, and where we decline to set one

The paper's rule is `δ = 0.25 g`, with `g` the gap between the **published** fixed-pole CRPS
and the **published** strongest competing method on the dataset — both from the literature,
never from our own reproduction, so that a weak reproduction cannot widen our own margin.

**CRPS — no equivalence read is made. Amended 2026-08-17 (decision A2, §10).** The rule above
was applied and does not yield a usable margin on this dataset. The published LLapDiff NOAA-UK
CRPS is **1.922**, against a recorded competitor value of 0.6392, so

```
g = 0.6392 − 1.922 = −1.283      →      δ = 0.25 g = −0.321
```

A negative margin is not a margin, and taking `|g|` gives `δ = 0.32`, wider than the entire
CRPS range any arm here attains (0.31–0.63) — every result would pass, so the read would be
vacuous rather than defensive. **We therefore pre-register no equivalence read on CRPS**, which
is the alternative the rule itself permits. CRPS is reported **descriptively** alongside the
other arms in the table, exactly as the fixed-pole paper reports it, with seed counts and
standard deviations shown. No CRPS claim of the form "equivalent to" or "not worse than" is
made anywhere in the paper.

This does **not** affect T2's CRPS *superiority* read (§4), which is paired within this campaign
between two arms of our own and needs no external anchor.

⚠️ **Scale check required before the CRPS column is published** (this replaces the margin
computation, and it is the only requirement that remains): our reported CRPS must land in the
same numeric range as the other entries of the table it joins. Published LLapDiff is 1.922
while our in-harness reproduction of the same method is 0.5491 — a 3.5× gap for one method,
which is a units difference, not a reproduction-quality difference. Whatever number goes into
the table must be produced on the same scale as the rest of that table's column, or the
comparison is meaningless regardless of how many seeds it averages.

⚠️ **Transcription hazard.** `CMD_PHASE1B_GATE.md` and `finetuning/results/phase1b/` contain
the value **0.63927** in the one-shot row. It is a *latent MSE*, not a CRPS, and its first four
digits coincide with TimeGrad's published CRPS 0.6392. Do not let it stand in for the missing
anchor.

**ES and VS.** No published anchor exists for these metrics on this dataset for any method, so
the rule has nothing to anchor to, and any margin chosen now would be chosen after seeing the
val effect sizes in §2. **We therefore pre-register no equivalence read on ES or VS.** All
ES/VS reads are directional (T2–T4). A tie on a directional test is reported as "no detected
difference", explicitly **not** as equivalence.

We also record, as `PREREG_U3.md` did, that a seed-std margin was considered and rejected: it
becomes more permissive exactly when an arm is noisy.

---



## 6. Seeds, power, and the estimator

**Seeds: 10 for Family A, 6 for Family B.** Not the ≥5 the original registration and the draft
both state. The reason is a hard property of the chosen test, and it is worth spelling out
because a single family of five at ≥5 seeds cannot reject anything:

- the smallest one-sided p a Wilcoxon signed-rank test can attain at `n` pairs is `2^{-n}`;
- Holm across a family of `k` requires the leading read to clear `0.05/k`;
- so the seed floor is set by discreteness, not by variance:


| family size | Holm threshold | smallest n with `2^{-n} ≤ threshold` |
| ----------- | -------------- | ------------------------------------ |
| 1           | 0.050          | 5                                    |
| 2           | 0.025          | 6                                    |
| 3           | 0.0167         | 6                                    |
| 4           | 0.0125         | 7                                    |
| 5           | 0.010          | 7                                    |


The registered counts (10 and 6) clear their thresholds with margin, and 10 also leaves room
if a seed is lost. Costs: A3–A6 are ≈45 min/seed including the LR sweep; A1/A2 need the full
three-stage U3 pipeline at ≈10 h/stage, so Family B's four additional seeds are the campaign's
dominant compute item.

**Permitted fallback if Family B's compute is unavailable:** shrink the *family*, never the
seed count — a smaller family lowers the Holm threshold. Family B is already size 1, so its
floor is 5; below that no valid read exists and T2 must be reported as descriptive. Any such
change is an amendment under §10 and must be made **before** the test read.

**Power.** Computed by `llapdiff-prereg-power` from the val seed JSONs, so the number in §11 is
derived rather than asserted.

### 6.1 Final val power table — FROZEN at the registered seed counts

Re-run 2026-08-17 at the registered counts (Family A = 10, Family B = 6), from
`finetuning/results/cmd_gp/prereg_power_familyB6.json`. **This is the table §11 requires.**
Family B's seeds are U3 `{0, 1, 2, 5, 6, 7}` — seeds 3 and 4 were still training when the
family reached its registered size of 6 and are **not** included; see the note below.

The tests are **paired by seed**, which removes the shared seed effect and is far tighter than
the per-arm spreads suggest. `p` is the one-sided Wilcoxon signed-rank p-value on the paired
differences, oriented so that positive means the registered direction:


| read | metric         | family | n   | paired mean Δ | paired sd | `d_z`    | signs | `p`         | Holm α | verdict                     |
| ---- | -------------- | ------ | --- | ------------- | --------- | -------- | ----- | ----------- | ------ | --------------------------- |
| T2   | VS(p=0.5)      | B      | 6   | 773.33        | 145.36    | **5.32** | 6/6   | **0.0156**  | 0.050  | **rejects**                 |
| T2   | ES unbiased    | B      | 6   | 0.6109        | 0.1455    | **4.20** | 6/6   | **0.0156**  | 0.050  | **rejects**                 |
| T2   | CRPS unbiased  | B      | 6   | 0.00887       | 0.00204   | **4.36** | 6/6   | **0.0156**  | 0.050  | **rejects**                 |
| T3   | VS(p=0.5)      | A      | 10  | 992.10        | 165.84    | **5.98** | 10/10 | **0.00098** | 0.025  | **rejects**                 |
| T5   | PIT-ECE        | A      | 10  | 0.00860       | 0.00302   | **2.85** | 10/10 | **0.00098** | 0.025  | **rejects**                 |
| T4   | VE *(adverse)* | —      | 10  | 0.01546       | 0.01421   | 1.09     | 9/10  | 0.0049      | —      | adverse direction confirmed |
| T4   | ES *(adverse)* | —      | 10  | −0.40489      | 0.31318   | 1.29     | 9/10  | 0.0049      | —      | adverse direction confirmed |


Per-arm means (val): T2 mixture `M=25` vs sampled `M=25` — VS 4061.2 ± 111.6 vs 4834.6 ± 217.5;
ES 32.507 ± 0.368 vs 33.118 ± 0.442; CRPS 0.4852 ± 0.0056 vs 0.4941 ± 0.0067.

**T1 (the gate) at 6 seeds:** `R = 0.6256 ± 0.0325`, per-seed
`{0.639, 0.680, 0.636, 0.596, 0.604, 0.600}` — every seed above the pre-registered 0.25, so
**Option B stands**, now on 6 seeds rather than 2.

Three readings this table forces, none of which loosen anything:

1. **Family B's three p-values sit exactly on the discreteness floor** `2^{-6} = 0.0156`**.** With
  6 seeds and unanimous signs that is the smallest one-sided p the signed-rank test can
   produce; the test is saturated. The evidence for T2 is therefore carried by the effect sizes
   (`d_z` 4.2–5.3) and the unanimity, not by any margin below the threshold. Reported that way.
2. **T3 strengthened again as seeds were added** — `d_z` 4.31 (n=2) → 5.27 (n=8) → **5.98
  (n=10)**, with 10/10 signs. That is what a real effect does under added seeds and what a
   selection artefact does not.
3. **T4 is no longer unanimous, and it is the same seed both times.** Seed 8 flips *both* T4
  metrics (VE −0.0115, ES +0.1776) while the other nine confirm. T4 was registered as the read
   we expect to lose, and the adverse direction still holds at `p = 0.0049`; the honest
   statement is "CMD-GP is worse on the marginal axis on 9 of 10 seeds", not "on every seed".
   Seed 8 is not excluded and no post-hoc rule is applied to it.

**Deviation from the registered seed *identities* (not the count).** §6 registered Family B as
"U3 seeds 2–5". Seeds 3 and 4 trained ~1.5× slower than their peers under GPU contention, so
seeds 6 and 7 completed first and the family reached its registered size of **6** with
`{0,1,2,5,6,7}`. The count, the arms, the split and the test are all unchanged; only which
seeds filled the six slots differs, and the slots were filled by completion order — a rule
fixed before any of these seeds were scored, not chosen after seeing them. Seeds 3 and 4 will
be reported in an appendix as additional val seeds once they land, and their arrival **cannot**
change the confirmatory read, which is frozen here at n=6. Logged under §10.

**Variance is not the binding constraint — the test statistic is.** Every confirmatory effect
is large (`d_z` 2.8–6.0 at the final counts) and the variance-based `n` is 2 throughout. But a
one-sided Wilcoxon
signed-rank test **cannot reach** `p < 0.05` **with fewer than 5 seeds at any effect size**,
because the smallest attainable one-sided p-value is `2^{-n}`: 0.25 at n=2, 0.125 at n=3,
0.0625 at n=4, **0.031 at n=5**. This is what fixes the floor at 5, and it is why 5 is stated
as a hard requirement rather than a convention. Holm correction across a family of five raises
the bar further, so headline rows use 10.

Achieved power is reported beside every read; any read short of 80 % is reported as
**inconclusive**, never as a tie. The recomputation-before-freeze requirement is **discharged**
by §6.1, which supersedes every earlier n=2 and n=8 figure in this document.

**Estimator parity.** All arms are scored with an identical estimator and an identical
25-member ensemble. Both the naive `1/(2m²)` and unbiased `1/(2m(m−1))` variants are reported
for ES and CRPS, with all pairs enumerated exactly at `m = 25`. **Any comparison whose sign
differs between the two estimators is reported as inconclusive.** This matters at small
ensembles: measured on val, a 2-member ensemble scored ES 33.46 unbiased against 44.49 naive.

⚠️ Recorded discrepancy: the shipped `evaluate_regression` sub-samples ≤200 of the 300 distinct
pairs at `m=25` from the **global** RNG, while `baselines/metrics.py` enumerates exactly — an
arm-dependent estimator difference. All reads in §4 use `llapdiffusion/models/path_scores.py`,
which enumerates exactly for every arm.

---



## 7. Protocol parity checklist (signed off per arm per table)


| item                       | required value                                                         |
| -------------------------- | ---------------------------------------------------------------------- |
| samples for sampled scores | 25                                                                     |
| DDIM steps / `η`           | 64 / 0.0                                                               |
| guidance `w`               | **1.0, frozen** (measured monotone-harmful on this cell)               |
| dynamic thresholding       | `DYNAMIC_THRESH_P = 0.0`                                               |
| weights                    | **EMA**, via the alias-proof transplant (`--weights ema`; B16 is open) |
| `COND_POOL_NORM_MODE`      | `global`, with populated stats                                         |
| `VAE_ENTITY_ENCODE`        | `True`                                                                 |
| split                      | chronological 0.7/0.1/0.2, split before windowing                      |
| upstream artifacts         | the §1 hashes, identical across arms                                   |
| `CHIRP_UQ_INIT_VAR`        | 1.0                                                                    |
| time origin                | `last_history` (`t̃_r = t^q_r − t_i`)                                  |


`COND_POOL_NORM_MODE` is on the checklist because under the shipped default the pole and UQ
conditioning is identically zero (B23) and the law is structurally homoscedastic — a row
produced that way is history-blind and means nothing. The mode is read off the **rebuilt
model**, not the config, and recorded in every results JSON.

---



## 8. Correctness gates — all must pass before any test read


| gate                                                | criterion                          | current                                                                                          |
| --------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| (a) test suite                                      | all green                          | **550 passing** *(re-run 2026-08-17)*                                                            |
| (b) variance recurrence vs dense quadrature         | exact for piecewise-constant decay | 3.7e-16                                                                                          |
| (c) constant-pole closed form                       | float precision                    | 4.4e-07 (fp32)                                                                                   |
| (d) Markov joint sampling vs Cholesky               | equal in distribution              | pass                                                                                             |
| (e) analytic intervals from an MSE checkpoint       | assertion refuses                  | pass                                                                                             |
| (f) marginalization consistency, (P-exact)/(P-mono) | ≤1e-6 relative                     | **requires the anchored variance** (5.6e-07 / 0.0); the query-grid quadrature fails at 2.975e-02 |
| (g) component-mean reconstruction                   | 0 against the returned tensor      | 0.00e+00                                                                                         |


🔴 **Gate (f) constrains what may be claimed, and it applies unequally to the arms.**

- **A3 (CMD-GP)** is the only arm that can satisfy it, and already does: `ChirpGPHead` computes
the variance against a fixed 128-node anchor grid by default, so its law is projective in
mean *and* kernel. A3 may therefore make a Gaussian-**process** claim.
- **A1/A2 (CMD-D)** cannot satisfy it at all, and not because of the quadrature: under
Option B the parameters are read off a reverse trajectory indexed by the requested query
grid, which is Proposition C.1's failure mode (i). Anchoring the variance would not rescue
them. A1/A2 are reported as **grid laws**, and the mixture is described as a law on the
requested grid, never as a process.
- **A4–A6** are per-time heads with no cross-time kernel; the property is vacuous for them and
is not reported.

This is a claim restriction, not a blocker on running.

---


## 9. Multiplicity, and the single-touch rule

Holm correction **within** each family of §4: {T3, T5} and {T2}. T1 is a threshold read and
T4/T6/T7 are descriptive; none carry a multiplicity burden, and none may be converted to a
test after the fact.

The test split is evaluated **once per final configuration per seed**. `FINAL_TEST_EVAL` stays
`skip` for every tuning run. Any re-run of a test cell after seeing its result is an
amendment (§10) and the original result is reported alongside.

### No test-split pilot

It is explicitly **not permitted** to read 1–2 test seeds, inspect them, and then decide the
family size, the seed count, or whether to keep debugging. Two reasons, and the second is the
worse one:

1. choosing the family after seeing the outcome is the multiplicity problem, not a control
   for it — the Holm threshold has to be fixed before the data is seen;
2. "look, then go back and debug" converts test into a second validation set, permanently and
   silently, and no later analysis can undo it.

The design questions a pilot would answer are answered on **val**, which is what §2 and
`llapdiff-prereg-power` are for: at four val seeds every registered contrast is sign-consistent
with `d_z` between 1.8 and 3.7, so the seed counts above are set by the discreteness table,
not by uncertainty about the effects.

**If a test read disagrees with val, that is the result** — it means the val effect did not
generalise — and it is reported as such. Debugging happens on val.

---

## 10. Amendment policy

Amendments are dated, appended, and never rewrite prior text. Each states: trigger, what it
changes, and whether any test-split result informed it. An amendment that is informed by a
test result converts the affected read to exploratory.

### A1 — 2026-08-17: Family B's six seed slots filled by completion order

**Trigger.** §6 registered Family B as "U3 seeds 2–5". Six U3 trainings (seeds 2–7) were run
concurrently across two hosts; seeds 3 and 4 ran ~1.5× slower than their peers under GPU
contention (s1 wall-clock 8.9 h and 9.1 h against 6.1 h for seed 5). Seeds 6 and 7 reached a
scorable `s2_diffusion_uq` checkpoint first, so the family reached its registered size of 6
with `{0, 1, 2, 5, 6, 7}`.

**What it changes.** The *identity* of four seed slots. It does **not** change the seed count
(6), the arms, the split (val), the test (one-sided Wilcoxon signed-rank), the Holm threshold
(0.050, family of size 1), or any metric definition.

**Was any test-split result involved?** No. The test split has not been read. The substitution
rule — fill the six slots in completion order — is a property of the job queue and was fixed
before any of seeds 2–7 was scored, not selected after inspecting results.

**Residual risk, stated plainly.** Completion order is not independent of the seed: a seed that
trains faster may early-stop sooner, which could correlate with fit quality. The observed
spread is attributable to host contention rather than to convergence (seeds 3 and 4 shared a
host with a third job), but this cannot be proven from wall-clock alone. The mitigation is that
seeds 3 and 4 **will** be reported as additional val seeds when they land, so the substitution
is auditable: if their T2 differences are consistent with the six frozen seeds, the concern is
answered empirically. Their arrival cannot change the confirmatory read, which is frozen at
n = 6 in §6.1.

### A2 — 2026-08-17: no CRPS equivalence read is pre-registered

**Trigger.** The published LLapDiff NOAA-UK CRPS was obtained: **1.922**. Against the recorded
competitor value of 0.6392 this gives `g = −1.283` and `δ = −0.321` — a negative margin, and
`|δ| = 0.32` would exceed the entire CRPS range any arm here attains (0.31–0.63), so every
result would pass and the read would be vacuous.

**What it changes.** §5's CRPS paragraph. **No equivalence or non-inferiority claim is made on
CRPS anywhere in the paper.** CRPS is reported descriptively alongside the other arms, with
seed counts and standard deviations, as the fixed-pole paper reports it. This is the second
branch the freeze record always permitted, not a new allowance.

**What it does not change.** T2's CRPS *superiority* read (§4) is paired between two arms of
our own campaign and needs no external anchor; it is unaffected and remains confirmatory.

**Was any test-split result involved?** No.

**Carried forward in place of the margin.** The scale check in §5: our reported CRPS must land
in the same numeric range as the column it joins. Published LLapDiff (1.922) and our in-harness
reproduction of the same method (0.5491) differ by 3.5×, which is a units difference rather
than a reproduction-quality one, so this check is not a formality.

**Open transcription question, flagged not resolved.** §5 already warns that `0.63927` appears
in this project's phase-1b results as a *latent MSE*, coinciding in its first four digits with
the "TimeGrad published CRPS 0.6392". With the published LLapDiff at 1.922, the value that now
looks out of range is **0.6392**, not 1.922. Whether `tab:main`'s competitor entry is a
transcription of our own latent MSE should be checked against the source before that column is
published. This does not affect A2's decision, which holds under either reading.

---

## 11. Freeze record — TO COMPLETE BEFORE TOUCHING TEST

- [x] Published LLapDiff NOAA-UK h=168 CRPS pinned in §5, `δ_CRPS` computed
      *(or: no CRPS equivalence read is made, and §4 is amended to say so)*
      — **resolved via the second branch, 2026-08-17.** Published value **1.922** yields
      `g = −1.283` and no usable margin, so no CRPS equivalence read is pre-registered; §5 is
      amended and the decision recorded as A2 above. The §5 scale check carries forward.
- [x] **Family A:** 10 val seeds of A3–A6 — seeds 0–9, all present *(verified 2026-08-17)*
- [x] **Family B:** 6 val seeds of A1–A2 — U3 seeds `{0,1,2,5,6,7}`, gate + frontier present for
      each; checkpoint hashes appended to §1. *Seed identities differ from the registered
      "2–5"; the count does not. See amendment A1.*
- [x] `llapdiff-prereg-power` re-run at the final seed counts, its table pasted into **§6.1**
      *(2026-08-17)*
- [x] Gates (a)–(g) of §8 re-run green — **550 passed** *(2026-08-17)*; (b)–(g) are encoded as
      regression tests and are green within that run. ⚠️ Re-run on the **frozen commit** once
      the commit below exists — the working tree currently has uncommitted changes, so this is
      green on the tree, not on a commit.
- [x] Anchored-variance path enabled for every arm making a process claim (gate (f)) —
      `ChirpGPHead` defaults to a 128-node anchor grid, and per §8 A3 is the only arm making a
      process claim; A1/A2 are reported as grid laws.
- [ ] This file committed; commit hash recorded here: `__________`
      *(HEAD is `dd374863b3` "fixed B20 21 23 24", 2026-08-10 — predates this campaign)*
- [ ] Date frozen: `__________`

Until every box is ticked, the test split stays untouched.

**Status 2026-08-17:** 6 of 8 boxes ticked. Both remaining boxes are procedural — commit hash
and freeze date — and are the author's to take. No substantive experimental or statistical item
is outstanding. **The test split remains untouched.**
