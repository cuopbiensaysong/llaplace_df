# CMD-GP / CMD-D — experimental results for the paper revision

**Self-contained.** Every number needed to fill the experimental tables is in this document.
No other file needs to be opened.

Generated 2026-08-17.

---

## 0. Read this first — three facts that govern how the numbers may be used

**(1) There are two score spaces, and mixing them produces a wrong claim.**


| space                                      | which results                                | typical CRPS magnitude |
| ------------------------------------------ | -------------------------------------------- | ---------------------- |
| **latent** (16-dim VAE code, standardised) | §3 frontier, §4 one-pass heads, §7 ablations | 0.48 – 0.67            |
| **data** (decoded observations)            | §9 three-arm comparison                      | 0.31 – 0.36            |


The path-level scores (energy score, variogram score, and the CRPS reported beside them) are  
computed **against latent targets**. The three-arm CRPS/MSE/coverage numbers in §9 are computed  
**after decoding**. A latent CRPS of 0.485 and a data-space CRPS of 0.312 describe the same  
model. **Only §9 is comparable to a published data-space CRPS.**

**(3) The evaluation cell is regularly sampled and fully observed.** This bounds what the
real-data evidence can support; see §10, which is the single most important item for framing.

---



## 1. Cell, stack and protocol

Dataset `noaa_uk`, forecast horizon `h = 168`, chirp core, no output head.


| property               | value                                 |
| ---------------------- | ------------------------------------- |
| window starts          | 16,286 train / 2,183 val / 4,702 test |
| entities per window    | 4                                     |
| latent dimension `d_z` | 16                                    |
| modes `K`              | 128                                   |
| chirp basis functions  | 4                                     |
| val elements scored    | 5,867,904 (146 batches)               |
| parameter ratio        | under-parameterised 4.3×              |


Frozen across every arm: entity-encoded stage-1 VAE, shipped stage-2 summarizer, `predict_type = x0`, EMA weights, DDIM 64 steps with `η = 0`, **guidance frozen at** `w = 1.0`, pooled
conditioning in `global` normalisation mode, chronological 0.7/0.1/0.2 split, 25-member
ensembles, initial predicted variance 1.0.

Two protocol choices that are load-bearing:

- **Guidance is frozen at 1.0 because it was measured to be monotone-harmful on this cell**
(`w = 1.0` → RMSE 0.4296 / CRPS 0.2565; ramp (1.0, 2.0) → 0.4513 / 0.2688; `w = 2.0` →
0.4754 / 0.2826). With weak conditioning, classifier-free guidance extrapolates along a
difference that is mostly noise. This is a *validity* requirement, not an accuracy tweak:
`w > 1` sharpens the predictive distribution and is indistinguishable from miscalibration, so
every coverage and PIT number would otherwise be confounded.
- **Pooled conditioning must be in** `global` **mode.** Under the library's shipped defaults the
pole/UQ conditioning vector is identically zero (absolute max 7.9e-08), so the poles, the
frequencies and the entire closed-form law see *no history* and the predictive law is
structurally homoscedastic. Any result produced without this setting is history-blind and
invalid for a heteroscedasticity claim. Every number in this document was produced with it on.

---



## 2. The gate: is the predictive law one Gaussian process, or a mixture? (read R9 / T1)

The law of total covariance is applied to the terminal draw `z_T`:
`Cov(z₀ | H) = E_{z_T}[K] + Cov_{z_T}(μ)`, and `R = tr Cov_{z_T}(μ) / tr Cov(z₀ | H)`.

Pre-registered decision rule, fixed before the measurement: `R < 0.10` → Option A (a single
history-conditioned GP is the law); `R > 0.25` → Option B (the law is a mixture); between →
carry both.

**6 seeds, val,** `M = 25` **reverse trajectories, component law read at the final reverse step:**


| `M` | within `E[K]` | between `Cov(μ)` | total  | `R` (mean ± sd, n=6) |
| --- | ------------- | ---------------- | ------ | -------------------- |
| 1   | 0.2572        | 0.0000           | 0.2572 | 0.0000               |
| 2   | 0.2570        | 0.4286           | 0.6856 | 0.6256 ± 0.0312      |
| 4   | 0.2566        | 0.4286           | 0.6852 | 0.6259 ± 0.0317      |
| 8   | 0.2567        | 0.4280           | 0.6847 | 0.6255 ± 0.0323      |
| 16  | 0.2566        | 0.4284           | 0.6850 | 0.6258 ± 0.0325      |
| 25  | 0.2566        | 0.4281           | 0.6847 | **0.6256 ± 0.0325**  |


Per-seed `R` at `M = 25`: 0.6389, 0.6801, 0.6356, 0.5956, 0.6037, 0.5998 — **every seed above
the 0.25 threshold, none near 0.10.**

### Verdict: Option B. The predictive law is a mixture of chirp-modal GPs, not a single GP.

Two consequences that must reach the paper:

- `M = 1` **is not merely worse, it is structurally miscalibrated.** At `M = 1` the between
term is zero by construction, so a single component carries 0.257 of a total 0.685 — it is
**missing 63 % of the predictive variance by construction**, not by estimation error.
- `R` **is flat from** `M = 2` **to** `M = 25` (0.6256 → 0.6256). The decomposition is fully
resolved by two components; additional components refine the *score*, not the ratio.



### Sensitivity to the read protocol — report this, do not assert protocol-independence

The gate reads the component covariance at the **final** reverse step. An earlier protocol read
it at `t = T−1` on a re-noised component mean. Measured on 2 seeds:


| read protocol                          | seed 0 `R` | seed 1 `R` | within term   |
| -------------------------------------- | ---------- | ---------- | ------------- |
| final reverse step (the specified one) | 0.6389     | 0.6801     | 0.215 / 0.227 |
| `t = T−1` (adverse)                    | **0.2704** | **0.4504** | 1.027 / 0.589 |


The adverse read inflates the within-component term 2.6–4.8× and drops `R` to 0.270 on seed 0 —
**still Option B, but clearing the threshold by only 0.020.** The verdict survives, and the
paper should say so with this table attached rather than claiming the result is
protocol-independent.

The 2.6–4.8× swing is itself diagnostic: a within-component variance that collapses as the
diffusion time goes to zero is the signature of a head estimating *denoising* uncertainty
rather than *forecast* uncertainty.

---



## 3. Structured mixture law vs sampled ensemble, at matched cost (read T2)

Both arms use the same checkpoint and the same 25 reverse trajectories. `mixture_M(m)` draws 25
ensemble members from `m` fitted Gaussian components (sampling a Gaussian is free — this is the
efficiency claim); `sampled_M(m)` has exactly `m` members. At `m = 25` the ensembles are
matched at 25 members, so the comparison is calibration at equal cost, not a speedup.

**6 seeds, val, latent space. Lower is better for ES, VS, CRPS.**


| arm                | ES (unbiased) | ES (naive) | VS (`p=0.5`) | CRPS (unbiased) | CRPS (naive) |
| ------------------ | ------------- | ---------- | ------------ | --------------- | ------------ |
| mixture `M=1`      | 44.219        | 44.951     | 4505.8       | 0.6691          | 0.6799       |
| mixture `M=2`      | 38.124        | 39.098     | 4274.9       | 0.5735          | 0.5880       |
| mixture `M=4`      | 35.192        | 36.283     | 4163.2       | 0.5274          | 0.5438       |
| mixture `M=8`      | 33.911        | 35.054     | 4115.4       | 0.5073          | 0.5245       |
| mixture `M=16`     | 32.989        | 34.169     | 4078.8       | 0.4928          | 0.5106       |
| **mixture** `M=25` | **32.507**    | 33.705     | **4061.2**   | **0.4852**      | 0.5033       |
| sampled `M=2`      | 33.141        | 44.879     | 5496.8       | 0.4944          | 0.6713       |
| sampled `M=4`      | 33.111        | 38.986     | 5136.7       | 0.4940          | 0.5825       |
| sampled `M=8`      | 33.109        | 36.045     | 4955.6       | 0.4940          | 0.5382       |
| sampled `M=16`     | 33.127        | 34.594     | 4866.5       | 0.4942          | 0.5163       |
| sampled `M=25`     | 33.118        | 34.057     | 4834.6       | 0.4941          | 0.5083       |


**Coverage and interval width at** `M = 25` (nominal / empirical, 2,183 val windows):


| arm            | cov@50 | cov@80    | cov@90    | cov@95    | cov@99 | w@80  | w@95  |
| -------------- | ------ | --------- | --------- | --------- | ------ | ----- | ----- |
| mixture `M=25` | 0.465  | **0.731** | **0.815** | **0.865** | 0.895  | 1.894 | 2.730 |
| sampled `M=25` | 0.383  | 0.630     | 0.720     | 0.778     | 0.816  | 1.491 | 2.120 |
| mixture `M=4`  | 0.421  | 0.656     | 0.734     | 0.782     | 0.815  | 1.755 | 2.385 |
| sampled `M=4`  | 0.271  | 0.432     | 0.477     | 0.497     | 0.512  | 1.017 | 1.223 |
| mixture `M=1`  | 0.228  | 0.404     | 0.484     | 0.542     | 0.588  | 1.128 | 1.620 |


Both arms are **under**-covered; the mixture is substantially less so at every level.

### Statistical read (pre-registered, one-sided Wilcoxon signed-rank, paired by seed, n = 6)


| metric          | mixture `M=25`  | sampled `M=25`  | paired Δ          | `d_z`    | seeds confirming | `p`        |
| --------------- | --------------- | --------------- | ----------------- | -------- | ---------------- | ---------- |
| VS (`p=0.5`)    | 4061.2 ± 111.6  | 4834.6 ± 217.5  | 773.33 ± 145.36   | **5.32** | 6/6              | **0.0156** |
| ES (unbiased)   | 32.507 ± 0.368  | 33.118 ± 0.442  | 0.6109 ± 0.1455   | **4.20** | 6/6              | **0.0156** |
| CRPS (unbiased) | 0.4852 ± 0.0056 | 0.4941 ± 0.0067 | 0.00887 ± 0.00204 | **4.36** | 6/6              | **0.0156** |


All three reject at the pre-registered threshold of 0.050.

⚠️ `p = 0.0156` **is exactly** `2⁻⁶`**, the discreteness floor of the test.** With 6 seeds and
unanimous signs this is the smallest one-sided p-value the signed-rank test can produce; the
test is saturated. The paper must present the evidence as the effect sizes (`d_z` 4.2–5.3) plus
unanimity, **not** as a comfortable margin below threshold.

### The two findings in this table that are more interesting than the p-values

1. **The mixture's advantage is overwhelmingly on the *path* score, not the marginal one.**
  VS improves by 16.0 %; CRPS by only 1.8 %. CRPS is computed per query time and averaged, so
   it is blind to temporal dependence by construction (see §6). The variogram score is the
   instrument that can see what the structured law provides.
2. **The sampled arm does not improve with more members; the mixture does.** From `M=2` to
  `M=25` the sampled arm's ES moves 33.141 → 33.118 (0.07 %) and its CRPS 0.4944 → 0.4941,
   while the mixture moves 38.124 → 32.507 (14.7 %) and 0.5735 → 0.4852 (15.4 %). The mixture
   *starts worse* — at `M ≤ 8` it is worse than the sampled arm on ES and CRPS — and crosses
   over between `M = 8` and `M = 16`. **The crossover is a real, reportable limitation: the
   structured law needs enough components to pay off.** On VS the mixture wins at every `M`,
   including `M = 1` (4505.8 vs 5496.8).



### Estimator parity — a live discrepancy

Both the naive `1/(2m²)` and unbiased `1/(2m(m−1))` estimators are reported above, from the
same draws, with all pairs enumerated exactly at `m = 25`. `naive − unbiased = +spread/(2S)`
exactly (verified to 1e-5). **The naive estimator is biased upward, i.e. in the direction that
flatters a closed-form arm scored against a sampled one.** The effect is large at small
ensembles: `sampled_M=2` scores ES 33.14 unbiased against 44.88 naive.

⚠️ The shipped regression-evaluation path estimates the spread term by Monte-Carlo over ≤200 of
the 300 distinct pairs at `m = 25`, drawn from the global RNG, while the baselines path
enumerates exactly — an **arm-dependent estimator difference**. All results in this document
use a single implementation that enumerates exactly for every arm. Any prose in the paper
claiming the naive estimator is used throughout, or quoting a "≈4 % of the spread term" figure,
does not match what was computed and needs rewriting.

---



## 4. Same-encoder distributional heads — is it the chirp structure, or just an NLL head? (reads T3, T4, T5)

All four heads are trained on the **identical frozen history summary** with an identical
learning-rate grid `{3e-4, 1e-4, 3e-5, 1e-5}` selected on validation per arm, identical
schedule (150 mean epochs, 80 NLL epochs, patience 20), one forward pass, no diffusion.
This is the novelty control: same encoder, different output head.

**10 seeds, val, latent space. Mean ± sd across seeds.**


| head                | variance expl. vs train-mean | latent MSE          | ES (unb.)          | VS (`p=0.5`)       | CRPS (unb.)     | PIT-ECE             | cov@80 | width@80 | params    | train s |
| ------------------- | ---------------------------- | ------------------- | ------------------ | ------------------ | --------------- | ------------------- | ------ | -------- | --------- | ------- |
| **CMD-GP (chirp)**  | 8.56 % ± 1.44                | 0.8399 ± 0.0133     | 33.470 ± 0.310     | **4486.1 ± 177.9** | 0.5022 ± 0.0052 | **0.0064 ± 0.0024** | 0.742  | 2.044    | 1,464,272 | 431     |
| diagonal Gaussian   | 10.11 % ± 0.22               | 0.8257 ± 0.0021     | 33.065 ± 0.043     | 5478.2 ± 41.5      | 0.4964 ± 0.0008 | 0.0150 ± 0.0009     | 0.748  | 2.001    | 1,196,832 | 15      |
| low-rank + diagonal | 10.05 % ± 0.19               | 0.8262 ± 0.0018     | 33.074 ± 0.037     | 5481.3 ± 44.5      | 0.4966 ± 0.0007 | 0.0153 ± 0.0006     | 0.748  | 2.002    | 1,213,280 | 16      |
| Student-`t`         | **11.60 % ± 0.15**           | **0.8120 ± 0.0014** | **33.006 ± 0.038** | 6723.3 ± 152.3     | 0.4965 ± 0.0009 | 0.0363 ± 0.0023     | 0.829  | 2.489    | 1,196,833 | 20      |


Baseline: train-mean latent MSE on val = **0.9185** (a constant fitted on train, applied to
val). The in-split mean baseline is 0.9184 — the two coincide to four decimals on this cell, so
the in-split leak that would otherwise inflate the baseline does **not** bite here. Both are
reported.

### Statistical read (one-sided Wilcoxon, paired by seed, n = 10)


| read                      | contrast                           | paired Δ          | `d_z`    | seeds confirming | `p`         | threshold |
| ------------------------- | ---------------------------------- | ----------------- | -------- | ---------------- | ----------- | --------- |
| **T3**                    | VS(diagonal) − VS(chirp)           | 992.10 ± 165.84   | **5.98** | 10/10            | **0.00098** | 0.025     |
| **T5**                    | PIT-ECE(diagonal) − PIT-ECE(chirp) | 0.00860 ± 0.00302 | **2.85** | 10/10            | **0.00098** | 0.025     |
| T4 *(registered adverse)* | VE(diagonal) − VE(chirp)           | 0.01546 ± 0.01421 | 1.09     | 9/10             | 0.0049      | —         |
| T4 *(registered adverse)* | ES(chirp) − ES(diagonal)           | 0.40489 ± 0.31318 | 1.29     | 9/10             | 0.0049      | —         |




### Verdict: a split decision, and it is stable

**The chirp-modal structure wins decisively on the two axes the paper's claim lives on:**

- **path coherence** — VS 4486 vs 5478, an 18.1 % improvement, `d_z = 5.98`, 10/10 seeds;
- **calibration** — PIT-ECE 0.0064 vs 0.0150, **2.3× better**, `d_z = 2.85`, 10/10 seeds.

**And it loses on the marginal axis:** variance explained 8.56 % vs 10.11 % (and 11.60 % for
Student-`t`), latent MSE 0.8399 vs 0.8257, ES 33.470 vs 33.065. This was **pre-registered as
the outcome expected to lose**, before the 10-seed measurement.

Three qualifications the paper must carry:

1. **T4 lost its unanimity at 10 seeds.** It was 8/8 at eight seeds; at ten it is 9/10. Seed 8
  flips *both* T4 metrics (VE −0.0115, ES +0.1776) while the other nine confirm. The honest
   statement is "worse on the marginal axis on 9 of 10 seeds", not "on every seed". Seed 8 is
   not excluded and no post-hoc rule is applied to it.
2. **The chirp head is markedly less stable across seeds.** Its VS standard deviation is 177.9
  against the diagonal head's 41.5 (4.3×), and its VE sd is 1.44 points against 0.22 (6.5×).
   It also selects different learning rates per seed (1e-5 on five seeds, 1e-4 on four, 3e-4 on
   one) where the diagonal head concentrates on 3e-5. This variance is a genuine cost and
   should not be omitted from the table.
3. **Student-**`t` **is the best marginal forecaster and the worst path forecaster.** Highest VE
  (11.60 %), lowest ES, and VS 6723 — 50 % worse than the chirp head — with PIT-ECE 0.0363,
   5.7× worse. It buys marginal sharpness by inflating tails (cov@80 = 0.829 against a nominal
   0.80, width 2.489 against 2.044). This is the cleanest illustration in the whole campaign
   that marginal and path quality are different objectives.



### Cost

CMD-GP trains in 431 s against the diagonal head's 15 s (29×) and uses 1.46 M parameters
against 1.20 M (1.22×). Its advantage is not free.

---



## 5. Marginalization consistency — is CMD-GP a Gaussian *process*? (read R11)

Test: predict on a full grid of 24 irregular query times, then re-predict on subsets
(25 %, 50 %, 75 %) and with points inserted before / between / after the existing queries.
A genuine process must return the exact marginal. Conditioning vector held fixed, so any
discrepancy is attributable to the time map and the pole/variance integration alone.
Discrepancies are relative; worst case over 6 cases, 3 seeds.


| variance construction                             | parameterization | Δ mean (rel) | Δ kernel (rel) | gate @ 1e-6 |
| ------------------------------------------------- | ---------------- | ------------ | -------------- | ----------- |
| **query-grid quadrature (as originally shipped)** | P-exact          | 3.611e-07    | **2.975e-02**  | **FAIL**    |
| query-grid quadrature                             | P-mono           | 9.627e-08    | **1.369e-04**  | **FAIL**    |
| **fixed anchor grid, 64 nodes (the repair)**      | P-exact          | —            | **5.610e-07**  | **PASS**    |
| fixed anchor grid, 64 nodes                       | P-mono           | —            | **0.000e+00**  | **PASS**    |
| fixed anchor grid, 256 nodes                      | P-exact          | —            | 5.572e-07      | PASS        |
| fixed anchor grid, 1024 nodes                     | P-exact          | —            | 5.608e-07      | PASS        |
| *control:* re-anchored origin                     | P-exact          | 2.347e+00    | 3.685e-01      | control     |
| *control:* re-anchored origin                     | P-mono           | 1.764e+00    | 4.473e-01      | control     |
| *control:* P-grid                                 | P-grid           | 1.027e+00    | 2.490e-01      | control     |




### Finding: the original construction fails Proposition C.1, and the mean is not the problem

The mean is projective to float32 noise (3.6e-07) — the closed-form integrated poles really are
functions on `[0, H]`. **The kernel is not.** The process-variance term is evaluated by a
recurrence over the *requested query nodes*, so it is a function of the query **set**, not of
time. The per-case breakdown pins the mechanism exactly:


| case                          | queries | Δ mean    | Δ kernel      |
| ----------------------------- | ------- | --------- | ------------- |
| drop to 25 %                  | 6       | 1.03e-07  | 2.910e-02     |
| drop to 50 %                  | 12      | 7.03e-08  | 1.265e-02     |
| drop to 75 %                  | 18      | 0.000e+00 | 1.825e-03     |
| insert **before** all queries | 24      | 0.000e+00 | 7.392e-05     |
| insert **between** queries    | 24      | 0.000e+00 | 1.340e-04     |
| insert **after** all queries  | 24      | 0.000e+00 | **0.000e+00** |


The discrepancy scales monotonically with coarsening, and inserting a point *after* every
existing query changes nothing **exactly**. That is the signature of a grid-dependent
quadrature and of nothing else: adding a node before or between refines the quadrature for
every later point; adding one after refines nothing.

**As originally implemented, the model is a Gaussian process in its mean and a grid law in its
kernel**, so the proposition does not hold for the object being evaluated.

### The repair works, is exact, and is free

Running the same recurrence on a fixed anchor grid that does not depend on the query set, then
propagating exactly from the enclosing anchor node to each query time, makes both terms
functions of `t` alone. The residual 5.6e-07 under P-exact is float32 noise, not
discretisation — it is the same order as the mean's 3.6e-07 and **does not decrease** from 64
to 1024 nodes. P-mono is bit-exact.

64 anchor nodes suffice, so the cost is `O(K·N)` with small `N`, independent of the horizon.
§7 shows the repair costs nothing measurable in accuracy, and §8 shows the anchored path is
also *faster*. **There is no accuracy/correctness trade-off to argue about.**

Two constraints the construction must state: the anchor horizon must be a property of the
window, never the maximum query time (deriving it from the queries smuggles the query set back
in and undoes the repair for exactly the insert-after case); and the anchor grid must start at
`t = 0`, where the initial condition sits.

---



## 6. Machinery validation, and calibration of the scoring instruments



### The variance recurrence computes what the theory says


| check                                                             | criterion                      | result                               |
| ----------------------------------------------------------------- | ------------------------------ | ------------------------------------ |
| exactness for piecewise-constant instantaneous decay              | rel ≤ 1e-6                     | **3.674e-16** (float64)              |
| constant-pole closed form `q(1−e^{−2ρt})/(2ρ)`                    | float precision                | **4.419e-07** (float32 eps ≈ 1.2e-7) |
| `t → ∞` limit equals the algebraic-Lyapunov steady state `q/(2ρ)` | rel ≤ 1e-5                     | pass                                 |
| small-`Δρ̄` series branch                                         | rel ≤ 1e-8                     | pass                                 |
| cross-time kernel diagonal equals the marginal variance           | exact                          | pass                                 |
| cross-time kernel vs constant-pole closed form                    | rel ≤ 1e-5                     | pass                                 |
| cross-time kernel positive semi-definite                          | min eigenvalue ≥ 0             | **+3.008e-02**                       |
| Markov joint sampling vs the analytic kernel                      | max rel err < 3 % at `S = 2e5` | pass                                 |
| Markov joint sampling vs Cholesky of the materialised kernel      | equal in distribution          | pass (max cov diff < 4 %)            |
| nugget lands on the diagonal only                                 | exact                          | pass                                 |
| amplitude `α` scales mean by `α`, covariance by `α²`              | rel ≤ 1e-6                     | pass                                 |
| sampling is differentiable                                        | finite nonzero gradients       | pass                                 |


**Quadrature order — a stronger claim than the paper currently makes.** Drifting pole
`ρ(s) = 0.4 + 0.3 cos(πs)` on `[0,4]`, `q = 0.5`, against a dense float64 reference:


| grid spacing | `T` | max rel err | observed order |
| ------------ | --- | ----------- | -------------- |
| 0.2500       | 16  | 7.600e-03   | —              |
| 0.1250       | 32  | 1.900e-03   | 2.00           |
| 0.0625       | 64  | 4.750e-04   | 2.00           |
| 0.0312       | 128 | 1.190e-04   | 2.00           |
| 0.0156       | 256 | 2.976e-05   | 2.00           |


**Reportable:** the exponential-integrator recurrence is *exact* for piecewise-constant
instantaneous decay (machine precision) and **second-order accurate** in the query-grid spacing
otherwise, with the order measured at 2.00 across four refinements.

### The scoring instruments, calibrated on a known ground truth

Two forecasts with **identical marginals** and opposite temporal dependence — a true stationary
AR(1) with `φ = 0.9` (so every marginal is standard normal) against i.i.d. noise. Averaged over
400 truths at `S = 2000`. The ratio is (i.i.d. forecast) / (correct forecast); 1.0 means blind.


| score                       | correct | i.i.d. | ratio      |
| --------------------------- | ------- | ------ | ---------- |
| CRPS                        | 0.5519  | 0.5519 | **1.0000** |
| energy score                | 1.9633  | 2.0541 | **1.046**  |
| variogram score (`p = 0.5`) | 2.8199  | 7.8539 | **2.785**  |


- **CRPS is exactly blind to temporal dependence** — identical to four decimal places, not
merely "less sensitive". It is computed per query time and averaged, so it cannot be
otherwise.
- **The energy score penalises a forecast with zero temporal dependence by only 4.6 %.**
Certifying path coherence with the energy score alone would be choosing the instrument least
able to detect the problem.
- **The variogram score separates them by 2.8×.**

This is the quantitative justification for reporting path-level scores at all, and it explains
§3 and §4 directly: the mixture and the chirp head both win far more on VS than on ES or CRPS
because VS is the only one of the three that can see what they provide.

**Known blind spot, worth stating.** The variogram score cannot detect a permutation of query
times for a process whose pairwise increments are exchangeable — e.g. a common random level
plus i.i.d. noise, where the pairwise absolute difference has the same law for every pair.
Measured: shuffling such a process changes VS by 4.6 %, against 2.8× for the AR(1). Diagnostics
that compare *predicted against empirical lag covariance* directly are not subject to this.

---



## 7. Ablations: nugget, parameterization, and the cost of the consistency repair

One-pass chirp head, 3 seeds each, identical LR grid selected on val per arm. All five variants
come from the **same code** — the baseline is re-run alongside the variants, because a
code-version difference between a baseline and its ablations is exactly the confound an
ablation exists to remove.


| variant                             | VE %      | ES ↓       | CRPS ↓    | VS ↓     | PIT-ECE ↓ | cov@80 | width@80 | NLL obj ↓ |
| ----------------------------------- | --------- | ---------- | --------- | -------- | --------- | ------ | -------- | --------- |
| baseline: P-exact + nugget + anchor | 8.26      | 33.573     | 0.504     | 4605     | 0.007     | 0.746  | 2.076    | 1.347     |
| no nugget                           | 8.83      | 33.468     | 0.501     | 4308     | 0.007     | 0.738  | 2.011    | 1.351     |
| query-grid variance (no anchor)     | 8.44      | 33.535     | 0.503     | 4563     | 0.008     | 0.741  | 2.048    | 1.346     |
| P-mono                              | 8.70      | 33.454     | 0.502     | 4332     | 0.006     | 0.743  | 2.056    | 1.343     |
| **P-grid**                          | **11.72** | **32.860** | **0.491** | **3929** | 0.007     | 0.749  | 2.016    | **1.312** |


Per-seed variogram scores, showing the stability difference:


| variant  | seed 0 | seed 1 | seed 2 | spread |
| -------- | ------ | ------ | ------ | ------ |
| baseline | 4338   | 4632   | 4846   | 508    |
| P-grid   | 3932   | 3937   | 3917   | **20** |




### Three readings

**1. The nugget has no detectable effect on this cell.** ES 1.00×, VS 0.94×, PIT-ECE 0.91×,
width 0.97× — the no-nugget arm is marginally *better* on several. This follows from rank
arithmetic: here `d_z = 16` and `2K = 256`, so `d_z < 2K` and the covariance is generically
**full rank without a nugget**. The nugget is motivated by a `d_z > 2K` singularity, a regime
this cell never enters. **Report as "no detectable effect where the rank bound does not bite",
and scope the necessity claim to configurations where it does.**

**2. The consistency repair is free.** Anchoring the variance to a query-independent grid costs
nothing measurable: ES 1.00×, VS 0.99×, width 0.99×. Combined with §8's finding that the
anchored path is also faster, the correct construction is strictly better than the one it
replaces.

**3. 🔴 P-grid — the parameterization excluded from process claims — is the best forecaster
here, by a wide and unusually stable margin.** VE 11.72 % against 8.26 % (+42 % relative), VS
3929 against 4605 (−15 %), best CRPS, best NLL, and a per-seed VS spread of **20** against the
baseline's **508**.

The reason is §10: the query grid on this cell is perfectly regular and identical in every
window. P-grid's only defect is that its cumulative-trapezoid integration depends on the
requested grid — **on this cell that defect is never exercised**, so P-grid keeps its extra
pointwise flexibility for free. Reporting "P-grid is best" without the regularity caveat would
invite a reviewer to ask why the paper excludes it.

---



## 8. Efficiency

Measured on one NVIDIA RTX PRO 4000 Blackwell; wall-clock is not portable across mixed cards,
so the device is recorded and one machine produces the table.

Both first-order linear recurrences in the closed form were Python loops over query times,
paying `T` kernel launches each. Both have exact closed forms and are now evaluated without a
loop: the **variance** recurrence unrolls to a `logcumsumexp` (accumulated in float64, because
the exponent reaches ~720 at a long horizon with a fast mode); the **sampler** exploits the
rotation–scaling normal form, where the 2-D state is one complex number so the recursion
telescopes into a single cumulative sum.


| component              | `h=24` before → after | `h=168` before → after | speedup @168 |
| ---------------------- | --------------------- | ---------------------- | ------------ |
| variance recurrence    | 5.76 → **0.54** ms    | 39.38 → **0.77** ms    | **51×**      |
| joint sampling, Markov | 8.40 → **0.79** ms    | 57.09 → **3.55** ms    | **16×**      |
| **CMD-GP, one pass**   | 32.76 → **2.85** ms   | 33.18 → **2.84** ms    | **11.7×**    |


Unchanged, as expected: denoiser forward 9.48 ms, kernel materialisation 11.41 ms, summarizer
9.11 ms, decoder 1.31 ms/path, single-component diffusion 317.9 ms.

**CMD-GP is flat in the horizon** — 2.850 ms at `h=24` against 2.842 ms at `h=168` — because
the anchored variance costs `O(K·N_anchor)`, independent of `h`.

The vectorisation was **not bought with error**: the variance scan is *more* accurate than the
loop it replaced (max rel error against a float64 reference 1.51e-07 vs 3.63e-07).

### The `O(Kh)` sampling claim now holds; before, it did not


| route                                                       | `h=168`       |
| ----------------------------------------------------------- | ------------- |
| Markov recursion                                            | **3.55 ms**   |
| Cholesky route (materialise 11.41 + factorise 0.25 + solve) | **≥ 11.7 ms** |


Markov is now **3.3× faster**. Previously the `O(h³)` route was ~3.8× *faster*, because the
`O(Kh)` route was a Python loop. The asymptotics were never wrong; the constants were.

**Cost of the fix:** sampler peak memory rises from 133 MB to 315 MB at `h=168`, because the
closed form materialises a larger intermediate. Still less than half what kernel materialisation
costs (698 MB).

### The honest speedup, and why it shrinks

CMD-GP against the 25-sample diffusion arm, same card:


| what is counted                   | CMD-GP   | CMD-D(25) | ratio     |
| --------------------------------- | -------- | --------- | --------- |
| denoiser evaluations              | 1        | 1600      | 1600×     |
| the arm's own compute             | 2.84 ms  | ≈7947 ms  | **2800×** |
| + summarizer (both pay it once)   | 11.95 ms | ≈7956 ms  | **666×**  |
| + decoder, 25 paths (both pay it) | 44.4 ms  | ≈7989 ms  | **180×**  |


**The defensible headline is the last row: 180×.** The ratio falls from 2800× to 180× as shared
costs are included. Note the arm's-own-compute ratio (2800×) *exceeds* the pass ratio (1600×) —
CMD-GP runs no denoiser at all, so it is not paying 1/1600th of a denoiser, it is paying for a
different and smaller network. That is a reason to quote the end-to-end number, not the
flattering one.

**Memory.** Kernel materialisation is the only component whose footprint moves with the
horizon: 131 MB at `h=24` to 698 MB at `h=168` — a 5.3× rise for a 7× horizon, consistent with
the `O(h²)` term.

---



### 8.1 Guidance × DDIM-step sweep — the reverse loop contributes almost nothing

Guidance `w ∈ {1.0, 1.25, 1.5, 2.0}` × DDIM steps `{64, 32, 16, 8, 4, 1}`, 2 seeds, val.
A flat curve was **pre-declared** to mean the reverse loop contributes nothing beyond
parameter refinement.

CRPS (mean of 2 seeds; ⚠️ this sweep ran with dynamic thresholding at 0.995 rather than the
frozen protocol's 0.0, so read the *variation*, not the absolute level):


| `w` \ steps | 64     | 32     | 16     | 8      | 4      | 1      |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 1.0         | 0.6343 | 0.6342 | 0.6341 | 0.6338 | 0.6336 | 0.6354 |
| 1.25        | 0.6344 | 0.6342 | 0.6341 | 0.6338 | 0.6337 | 0.6355 |
| 1.5         | 0.6344 | 0.6343 | 0.6341 | 0.6338 | 0.6338 | 0.6355 |
| 2.0         | 0.6347 | 0.6345 | 0.6342 | 0.6340 | 0.6340 | 0.6357 |


**The entire 24-cell grid spans 0.6336 to 0.6357 — a range of 0.33 %.** Going from 64 DDIM
steps to **1** changes CRPS by 0.17 %, and increasing guidance from 1.0 to 2.0 costs 0.06 %
(monotonically harmful, consistent with the frozen `w = 1.0`).

This is the pre-declared flat outcome, and it pairs with §3 to give a sharp statement:
**the number of reverse *steps* is nearly irrelevant, while the number of mixture *components*
is decisive** (`M = 1 → 25` moves CRPS 0.6691 → 0.4852, a 27 % improvement, in latent space).
The predictive value is in the ensemble over terminal draws, not in the refinement trajectory.

---



## 9. Three-arm comparison in **data space** — the only numbers comparable to a published CRPS

⚠️ **These are decoded, data-space numbers.** They are on a different scale from every score in
§3, §4 and §7, which are latent. 2 seeds, val.


| arm                                      | CRPS                | MSE    | MAE    | PIT-ECE | cov@80 | interval width | denoiser passes | wall (s) |
| ---------------------------------------- | ------------------- | ------ | ------ | ------- | ------ | -------------- | --------------- | -------- |
| **A1 diffusion, 25 samples**             | **0.3119 ± 0.0020** | 0.3165 | 0.4362 | 0.0250  | 0.760  | 1.178          | 1600            | 1681     |
| A2 diffusion, analytic law, matched mean | 0.3184 ± 0.0004     | 0.3271 | 0.4441 | 0.0446  | 0.710  | 1.087          | 1604            | 1725     |
| one-shot analytic (diffusion stage 3)    | 0.3555 ± 0.0236     | 0.4004 | —      | 0.1000  | 0.722  | 1.243          | **1**           | 21       |


A2's latent-space diagnostics: latent RMSE 0.9065, latent Gaussian NLL 1.3953, mean predicted
std 0.7960, latent PIT-ECE 0.0164.

The one-shot arm is the closest *diffusion-side* analogue to a one-pass structured law: 1 pass,
80× faster wall-clock, and **+14 % CRPS with the worst calibration of the three** (PIT-ECE 0.100
against 0.025). This is a real cost that the efficiency story must carry.

🔴 **Naming collision — do not merge these two arms.** The label `A3` is used for two different
models in this project's materials:


| label in context                               | what it is                                                  | CRPS                           | space      | passes |
| ---------------------------------------------- | ----------------------------------------------------------- | ------------------------------ | ---------- | ------ |
| `A3` in the pre-registration (reads T3/T4)     | **CMD-GP**, the one-pass chirp head — the Option-A flagship | **0.5022 ± 0.0052** (10 seeds) | **latent** | 1      |
| `A3_oneshot_analytic` in the training campaign | stage 3 of the **diffusion** pipeline                       | **0.3555 ± 0.0236** (2 seeds)  | **data**   | 1      |


Both are "one pass", which is why they collide, but they are different models trained by
different procedures and their CRPS values are on different scales. **The Option-A row of any
table must use 0.5022 (latent), never 0.3555.** The correct latent-space comparison is
CMD-GP 0.5022 against the mixture's 0.4852 and the diagonal head's 0.4964.

### On comparing to a published CRPS of ~0.53–0.54

Our best data-space val CRPS on this cell is **0.3119** (A1, 25 samples), i.e. **substantially
lower**. Before this comparison goes in the paper, three conditions must be checked, because
each could account for a difference of this size on its own:

1. **Split.** These are validation numbers. A published figure is normally test. This cell has
  2,183 val against 4,702 test windows.
2. **Normalisation.** Data-space CRPS depends on the scaler applied to observations. Unless the
  published number uses the same per-entity standardisation, the scales differ.
3. **Ensemble and estimator.** These use 25 members with exact pair enumeration and the
  unbiased estimator; the naive estimator would report a *higher* number.

**Do not claim a win over the published figure from this document alone.** The reconciliation is
a specific, small piece of work: confirm the published number's split, scaler and ensemble size,
then re-read the same quantity under the frozen protocol.

---



## 10. 🔴 The finding that most affects the paper's framing

**The evaluation cell is regularly sampled and fully observed.** Measured directly:

```
context inter-arrival gaps : min 1.0000  max 1.0000  unique values 1
query   inter-arrival gaps : min 1.0000  max 1.0000  unique values 1
max deviation of any window's grid from the first : 0.000e+00
observed fraction : 1.0000   (no missingness at all)
```

The paper is about irregularly sampled multivariate series with variable-wise, often
informative, missingness. **The cell carrying every headline result has neither irregular
spacing nor any missingness.**

Nothing measured is invalidated — R9, T2, T3–T5, R11 and the ablations all measure what they
claim on the data they use. But this bounds what those results can support:

- **The method's central capability is untested on real data here.** Closed-form evaluation at
arbitrary *irregular* query times without re-gridding is the axis on which this approach is
claimed to differ from grid recurrences. On a fixed regular grid a grid-based core needs no
re-gridding either, so the axis collapses.
- **Marginalization consistency is real but never *needed* on this cell.** R11's correctness
gate still matters for the process claim, but no forecast here ever varies its query set.
- **It explains why P-grid wins** (§7). P-grid's only defect is grid-dependence, and the grid
never varies.

The cell was selected on statistical power — window counts, forecastable signal, parameter
ratio — and it is the only cell that passes those. Irregularity was not a selection criterion,
and for this paper it should have been.

**Two honest options.** (a) Run the headline reads on a genuinely irregular cell as well — a
synthetic benchmark with Gamma-renewal gaps can carry this. (b) State plainly that the
real-data evaluation is on regular grids with masked missingness and that the irregular-query
capability is demonstrated on synthetics only. **(b) is honest and cheap; (a) is stronger.**

---



## 11. Statistical protocol and its current status

Two confirmatory families with Holm correction *within* each family:


| family | reads                                   | seeds | Holm threshold | result                                   |
| ------ | --------------------------------------- | ----- | -------------- | ---------------------------------------- |
| **A**  | T3 (path score), T5 (calibration)       | 10    | 0.025          | both reject at `p = 0.00098`             |
| **B**  | T2 (structured law vs sampled ensemble) | 6     | 0.050          | all three metrics reject at `p = 0.0156` |


The gate (§2) is a threshold read, not a hypothesis test. T4 is registered as a directional read
expected to fail.

**Why the seed counts are what they are.** The smallest one-sided p-value a Wilcoxon
signed-rank test can attain at `n` pairs is `2⁻ⁿ`, and Holm across a family of `k` requires the
leading read to clear `0.05/k`. The floor is therefore set by discreteness, not variance:


| family size | Holm threshold | smallest `n` with `2⁻ⁿ ≤ threshold` |
| ----------- | -------------- | ----------------------------------- |
| 1           | 0.050          | 5                                   |
| 2           | 0.025          | 6                                   |
| 3           | 0.0167         | 6                                   |
| 4           | 0.0125         | 7                                   |
| 5           | 0.010          | 7                                   |


A single family of five reads at 5 seeds **cannot reject anything at any effect size**. Splitting
into families of 2 and 1 is what makes the reads attainable. Every effect measured is large
(`d_z` 2.8–6.0) and the variance-based required `n` is 2 throughout — **variance was never the
binding constraint; the test statistic was.**

### What is still open before the test split may be read

1. **The published-CRPS anchor** (§9) — needed to set an equivalence margin. Not yet obtained.
  This blocks only the *equivalence* read; T2's CRPS *superiority* read is paired within this
   campaign and needs no external anchor.
2. **A frozen commit** — the results above were produced from a working tree with uncommitted
  changes.
3. **Seed-identity deviation, already logged.** Family B was registered as seeds 2–5; seeds 3
  and 4 trained ~1.5× slower under GPU contention, so seeds 6 and 7 completed first and the
   family reached its registered size of 6 with seeds {0, 1, 2, 5, 6, 7}. The count, arms,
   split, test and threshold are unchanged. Seeds 3 and 4 will be reported as additional
   validation seeds when they finish and cannot change the frozen `n = 6` read.

---



## 12. Summary of what the evidence supports

**Supported, with strong effect sizes and unanimous or near-unanimous seeds:**

1. The predictive law of the diffusion model **is a mixture**, not a single Gaussian process
  (`R = 0.6256 ± 0.0325`, 6/6 seeds above threshold). A single component is missing 63 % of the
   predictive variance by construction.
2. The structured mixture law **beats the sampled ensemble at matched cost and matched ensemble
  size** on all three path scores (`d_z` 4.2–5.3, 6/6 seeds), and is materially better
   calibrated (cov@80 0.731 vs 0.630).
3. The chirp-modal structure **beats a conventional NLL head on path coherence** (VS −18.1 %,
  `d_z` 5.98, 10/10) **and calibration** (PIT-ECE 2.3× better, `d_z` 2.85, 10/10), using the
   identical frozen encoder.
4. The closed-form machinery **computes what the theory says**, exactly for piecewise-constant
  decay and second-order otherwise (order 2.00 measured).
5. Exact joint sampling in `O(Kh)` by Markov recursion **is now genuinely faster** than the
  Cholesky route (3.3×), and the one-pass law is **flat in the horizon**.
6. End-to-end, the one-pass law is **180× cheaper** than the 25-sample diffusion arm.

**Contradicted or requiring the paper to be scoped back:**

1. **Proposition C.1 fails as originally implemented** — the mean is projective, the kernel is
  not, and this is true for *every* parameterization, not only P-grid. The repair is
   implemented, verified and free, but the construction must be stated as part of the model.
2. **The chirp structure loses on the marginal axis** (variance explained, MSE, energy score) to
  a plain diagonal Gaussian head, on 9 of 10 seeds.
3. **The nugget has no measurable effect** on this cell, because the rank bound motivating it
  never binds here.
4. **P-grid, the excluded parameterization, is the best forecaster on this cell** — a direct
  consequence of the cell being perfectly regular.
5. **The evaluation cell is regular and fully observed**, so the paper's central
  irregular-query capability is not exercised on real data.
6. **The structured mixture is worse than the sampled ensemble at** `M ≤ 8` on ES and CRPS,
  crossing over between `M = 8` and `M = 16`.
7. **Guidance is monotone-harmful on this cell** and is frozen at 1.0; any claim that guidance
  strength was tuned as a hyperparameter is not what happened.
8. **The estimator-parity description does not match the implementation** — the shipped
  regression path subsamples pairs from a global RNG while the baselines path enumerates
   exactly. Results here use exact enumeration throughout, but the paper's prose needs rewriting.

**Not yet established:** any test-split number; any result on an irregular or partially observed
real cell; any comparison against external baselines; any comparison against a published CRPS
(see §9).