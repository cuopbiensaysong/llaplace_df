# NOAA CMD-Mix readiness report — G0…G3 (ACTIONS §11)

Date 2026-08-25. Agent: NOAA. Governing plan
[`CMD_MIX_NOAA_UK_TEST_STRATEGY.md`](../CMD_MIX_NOAA_UK_TEST_STRATEGY.md); this work is
specified by [`CMD_MIX_NOAA_UK_V2_ACTIONS.md`](../CMD_MIX_NOAA_UK_V2_ACTIONS.md).

> ## ▲ CURRENT STATUS — this is the live verdict
>
> ```
> G0 = PASS   G1 = PASS   G2 = PASS   G3 = PASS      s4_authorized: true
> ```
>
> Authoritative machine-readable source: **`frozen/gate_index.json`**. If any statement in
> this document disagrees with that file, the file is right.
>
> **This document is an append-only log of three rounds.** Rounds 1 and 2 below are
> **SUPERSEDED historical records** and their verdict tables say `G1 BLOCKED`, which was
> true when written and is no longer true. Jump to
> [FINAL READINESS REPORT](#final-readiness-report--2026-08-25) for the current state.
>
> | round | G1 said | why it changed |
> |---|---|---|
> | 1 (superseded) | BLOCKED | B1: no four-way split existed |
> | 2 (superseded) | BLOCKED | B1 resolved; B3 raised (lockbox blocks); B2 unsigned |
> | **final (live)** | **PASS** | B1, B2, B3 all resolved — B3 by dated amendment A1 |
>
> **Still true and unchanged across all rounds:** every pre-existing result is
> DEVELOPMENT_ONLY, and **S4 smoke has not started** — it is blocked on converting the v2
> partitions to summary latents, see *Smoke stage — prerequisite*.

---

# Round 1 — initial gate report (SUPERSEDED 2026-08-25)

> **SUPERSEDED.** The verdict table in this section is a historical record. G1 was BLOCKED
> on B1 when this was written; B1 is now RESOLVED. See the CURRENT STATUS box above.

## Verdict (as of Round 1 — SUPERSEDED)

| gate | status |
|---|---|
| **G0** golden_cmdmix_v2 NOAA conformance | **PASS** |
| **G1** split / thresholds / hashes / exposure | **BLOCKED** *(superseded → now PASS)* |
| **G2** D=1 parity, projectivity, ordering | **PASS** |
| **G3** covariance validity and parameterization | **PASS — keep `p_exact`** |

**S4 MUST NOT LAUNCH** *(statement as of Round 1; superseded — S4 is now authorized,
though smoke has not started).* §11 requires all four lines to read PASS.

**Every prior result is DEVELOPMENT_ONLY.** No existing seed, checkpoint, complete-case
comparison, p-value or metric counts toward S4, model selection or the lockbox.

---

## G0 — NOAA-owned v2 conformance · PASS

`_campaign_scripts/cmd_mix/conformance_v2.py`. Imports the production NOAA mixture law
and **no Toy model implementation** — `cmdmix`, `cmd_toys` and the vendored core are
absent from its import graph by construction. The Toy-side
`check_fixture_v2.py --foreign-adapter` was NOT used to claim conformance; it still
instantiates Toy `CMDMix` and never invokes a NOAA adapter.

**Integrity**, pinned as literals in `fixture_pin.py` from ACTIONS §1, not read from the
adjacent JSON:

| item | expected | result |
|---|---|---|
| fixture NPZ sha256 | `803e787a…60155b` | OK |
| canonical metadata-body sha256 | `ad3b65b0…c345a8f` | OK |
| raw JSON-file sha256 | `bf2af25a…677d1a` | OK |
| configuration sha256 | `2502f0a9…18189a00` | OK |
| schema / eps_abs / eps_rel / jitter / dtype / Cholesky | 2 / 1e-8 / 1e-6 / 1e-10 / float64 / lower | OK |
| Toy canonical-core and implementation hashes | — | **declared divergence** (absent by design) |

NOAA adapter `3aa7c1e102cc5842ad7d87f16919b69886863c6910235c772564cd9f51e3fe6d`

NOAA checker `5073debbb55ee41cdacde3b4971268135b23ed519cdc7b1e1bd932bef34075cb`

**Numerics**, both cases × both Gaussian scorers (the batched `fast` path and the shipped
`path_scores` loop), 24 checks each, 96 in total, all pass:

| combination | worst absolute discrepancy | tolerance |
|---|---|---|
| fast / canonical | **2.842e-13** | 3.33e-04 |
| fast / semantic | 1.137e-13 | 3.72e-04 |
| shipped / canonical | 2.842e-13 | 3.33e-04 |
| shipped / semantic | 1.137e-13 | 3.72e-04 |

Everything other than `comp_logpdf_T` is at or below 1.7e-15. Covered: per-history gating,
component Gaussian scoring at jitter 1e-10 lower-Cholesky float64, the weighted
`logsumexp` through **one production helper shared with the trainer**, exact T→S
restriction on **both** covariance axes, direct-S vs restricted-T identity, mixture moments
including the between-component outer product, the pinned Cholesky factor, and a
deterministic sampler taking the fixture's labels and innovations.

The forbidden `Σ_d π_nd log p_d` is reproduced exactly and gated: mean gap 3.46 nats
(canonical) and 5.34 nats (semantic). A pooled gate — weights averaged over histories —
shifts the density by **0.636 nats on `semantic`** against 0.093 on `canonical`, so the
semantic case is what makes that defect detectable.

**Tamper self-test**: 44 golden-value tampers (one element each, both cases) and 7 adapter
tampers × 2 cases — pooled gate, uniform gate, the forbidden formula, one-axis restriction,
upper Cholesky, per-time relabelling, dropped between-component term — **all detected**.

The self-test found a defect in itself first: `cov_T_restricted` and `z_S` come out of the
npz non-C-contiguous, so `reshape(-1)[i] += eps` wrote into a copy and the tamper was
silently discarded. `.flat[i]` fixes it. Reported to the Toy agent.

---

## G1 — S3 freeze · BLOCKED *(Round 1 state; SUPERSEDED — G1 now PASSES)*

`_campaign_scripts/cmd_mix/freeze_s3.py` writes and validates
`frozen/s3_manifest.json`, `frozen/decision_thresholds.yaml` and
`frozen/prior_exposure.md`. Hashed: 1 data file, 2 upstream checkpoints, 8 llaplace_df core
files, 12 adapter/checker/driver files, 2 fixture files. No file missing. Environment, GPU
model, torch/CUDA versions and `llaplace_df` HEAD + `git status` are recorded.

### B1 — four-way split/exposure manifest · **BLOCKED**

The window cache holds only `E`, `mu`, `dt`, `mask`. **There is no site identifier and no
absolute timestamp per window.** The split that produced it is `global_purged_horizon` at
ratios 0.7 / 0.1 / 0.2, applied inside `run_experiment` and not recorded in the cache. So:

- an embargo **does** exist (the policy is purged by horizon), but its realised boundaries
  cannot be verified from the cache;
- "no station appears in two splits" cannot be checked at all;
- a **four-way** split cannot be derived from a two-way cache.

**Smallest unblock:** have `collect()` in
`llapdiffusion/tools/run_option_a_probe.py` carry two extra columns it already has — the
entity/site id and the absolute timestamp of the last history point — and re-collect the
cache once. That edits `llaplace_df`, so it needs explicit authorisation; I have not made
it.

One thing this does establish: **the test split has never been read.** The cache collects
`loaders[0]` and `loaders[1]` only; `loaders[2]` is never constructed. No test window has
entered any CMD-Mix artifact.

### B2 — thresholds · **PROPOSED, awaiting sign-off**

`frozen/decision_thresholds.yaml` is complete — practical superiority (1% relative, paired
CI above 0, Holm across D∈{2,4} at FWER 0.05), TOST equivalence at 0.5%, calibration (PIT
ECE ≤ 0.05, coverage error ≤ 0.03), component use (effective components ≥ 1.5 at D=2 and
≥ 2.5 at D=4, max occupancy ≤ 0.90, between-share R ≥ 0.05), cost (≤ 1.5× D=1 wall clock),
the expansion stability gate, and the per-epoch numerical-validity gate. **The numbers are
my proposal.** Pre-registered decision rules are a scientific commitment; they need your
sign-off before G1 can read PASS. None was chosen after seeing an S4 number — there are no
S4 numbers.

---

## G2 — production Gate A, repaired and expanded · PASS

`_campaign_scripts/cmd_mix/gate_a_v2.py`, on **128 real `noaa_uk` histories** at the real
`T = 168`, `d_z = 16`, real masks, in float64 **and** float32 with TF32 disabled, across
four grid families (real, irregular, long-gap, exact-ties) and `D ∈ {1, 2, 4}`.

**§5.1 query ordering.** `cross_time_kernel` resolves `min(t,t')` by tensor **index**, so an
out-of-order query set silently produces the wrong covariance and nothing raises. A
canonical sort now happens once at the public adapter boundary (`CMDMix.forward`), the
identical permutation is applied to targets and masks inside `mixture_nll`, chronological
order is **asserted** where the kernel relies on it, and the requested order is restored for
means, samples and both covariance axes. Measured: permuting the query set leaves weights,
component means, covariance, component density, mixture density, samples and decoder
outputs **bitwise** unchanged; inserting a query before, between and after the existing ones
leaves the law on the original times **bitwise** unchanged.

Incidental finding: the shipped `noaa_uk` cache is **already sorted**, so no past run had a
mis-ordered kernel on this cell. The boundary is insurance, not a repair of past numbers.

**§5.2 D=1 parity.** Mapped-checkpoint `CMD-Mix(1)` against the `ChirpGPHead` it was loaded
from, for both the `shared` and `full_experts` variants: `theta`, `rho_bar`, `omega_bar`,
`lam`, `p0`, `q`, `sigma2`, mean, var, the full cross-time covariance, the mixture NLL, the
decoder mean and variance — bitwise in float64; the log-weight is exactly 0; the parameter
count is identical and no gate module is constructed.

**§5.2 projectivity.** Direct-S laws against analytically restricted-T laws on weights,
component means, **both** covariance axes, mixture mean, mixture covariance and the mixture
density at one fixed target — exact under the production `anchor_nodes=128`.

**Positive control.** The requested-grid recurrence (`anchor_nodes=0`) is intentionally
inconsistent and the suite **rejects it**, on 15–16 of 16 history chunks per statistic.
Two corrections to how the control was originally posed, both of which were my errors:

- the predictive **mean** is exactly projective even at `anchor_nodes=0` (measured 0.000,
  matching `chirp_gp`'s own docstring) because only the *variance* quadrature is
  grid-dependent — requiring the mean to differ was asserting a defect the code does not
  have. The control is now gated on the covariance only;
- the control is gated on the **relative** discrepancy at 100× `eps_rel`. The combined
  absolute+relative criterion cannot serve, because the grid-dependent part is relatively
  large exactly where the covariance is absolutely small (max_rel 3.8e-04 at max_abs
  5.9e-09), so `eps_abs` swallows it.

**680 checks, 0 failures, 6/6 positive controls rejected.** Worst genuine discrepancy across the whole suite: **5.353e-01** on `f32/exact_ties/D=1 5.1 sample mean under permutation (ties: 256 draws, 6 sigma MC)`.

---

## G3 — covariance validity · PASS, `p_exact` retained

`_campaign_scripts/cmd_mix/audit_psd.py`. Runtime pinned:
`llaplace_df` HEAD `180a7466…5eabf0`, `laptrans.py` `357d7e90…b002f8`,
`chirp_gp.py` `42924c33…5b7befe`, `dist_heads.py` `5027b8bd…2b39ee80`.
**56 audited configurations** — real `noaa_uk` windows on a canonically sorted grid, at
init and under parameter perturbations to 32×, four targeted mechanism probes, the one
saved trained checkpoint (USHCN seed 0), in float64 / float32-TF32-off /
float32-TF32-on / float64-CPU. Every intervention acts on **parameters** and recomputes the
whole forward pass, so `Lambda`, `q_eff` and the covariance always come from one set of
dynamics.

| statistic | result |
|---|---|
| `min d(rho_bar)` over all 56 configurations | **+9.999e-05 > 0** — Assumption R never violated |
| float64: worst `eig_min` | **+8.07e-12** (positive) |
| float64: `min q_eff` | −2.8e-14 (cancellation round-off against `Lambda ~ 1e4`) |
| float64: fixed-jitter Cholesky failures, 28 configurations | **0** |
| float32: fixed-jitter Cholesky failures, 28 configurations | **136** |
| trained USHCN head | `min d(rho_bar)` +3.53e-04, `min q_eff` +5.11e-03, `eig_min` +6.48e-04, 0 failures, all precisions |

### §9 of the pilot plan is retracted

§9 concluded that constrained `p_exact` leaves Assumption R unenforced. **That is wrong.**
Two independent errors, both of which ACTIONS §6 anticipated:

1. **The method was invalid.** §9 perturbed `rho_bar` while keeping the trained head's
   `Lambda`. They are coupled through one quadrature, so that pair describes no law and its
   indefiniteness says nothing about the model.
2. **The code reading was wrong.** `_coeffs` returns `a_rho2 = c**2 ≥ 0`, rescaled so
   `Σ_m a_rho2 < rho_floor − rho_min`; with the centred basis `|φ_m − 1| ≤ 1`, so
   `d(rho_bar)/dt = rho_floor + Σ a_rho2(φ_m − 1) ≥ rho_min > 0`. `ChirpGPHead` **hard-codes**
   `rho_basis="centered"` and `growth_budget` is 0. `p0` and `q` are `softplus > 0`, so
   `q_eff = q·dt·(1−e^{−2x})/(2x) ≥ 0` for every real `x`. **The kernel is PSD by
   construction in exact arithmetic.**

### What the crashes actually were

**Rank collapse plus a relatively vanishing nugget, in float32.** As `q → 0`,
`Lambda → e^{−2 rho_bar} p0` and the kernel becomes a Gram matrix of `2K` cos/sin
features — rank at most `2K = 64` on a `T = 168` grid. Still PSD, but singular; only the
nugget lifts it, and a large modal part makes `SIGMA2_MIN = 1e-4` relatively negligible
(measured `nugget/diag = 1.3e-06`). Probe `q→0 + theta×1000`:

| precision | `eig_min` | numerical rank | Cholesky failures |
|---|---|---|---|
| float64 | **+8.16e-06** | 168/168 | **0** |
| float32, TF32 off | −1.73e-01 | 40/168 | 8 |
| float32, **TF32 on** | −1.29e+01 | 53/168 | **128/128** |

This also explains the cell-dependence that the `p_mono` story never did: USHCN has
`T = 12 < 2K = 32`, so the mechanism cannot bite — and USHCN never crashed. noaa_uk has
`T = 168 > 2K = 64`.

### Decision

**Keep `p_exact`.** Per ACTIONS §6, float64 passing where float32 fails is a precision
problem and is not evidence for `p_mono`. `p_mono` remains a separately named ablation and
is not adopted. Frozen instead: **float64** covariance construction and factorisation,
**TF32 disabled** (note that `llapdiffusion/configs/config.py` sets `ALLOW_TF32 = True` and
the quarantined campaign ran with it on), and a per-epoch validity panel.

Open, deliberately not acted on: the joint objective rewards driving `q → 0`, which is what
walks the kernel into the degenerate corner. A floor on `q` would remove the mechanism at
source but is a model change needing a dated pre-S4 amendment. Float64 + TF32-off removes
the observed failure without one.

---

## Objective, exactly

**Name:** `latent time-joint/channel-factorized composite NLL`. It is joint over time and
factorised over channels while the sampled law has cross-channel covariance, so it is
**not** an exact full latent joint likelihood and is never reported as one.

Per window, then mixed, then normalised — in that order:

```
l_nd     = component d's time-joint/channel-factorised NLL of window n
L_n      = -log sum_d exp( log pi_nd - l_nd )
L_scalar = ( sum_n L_n ) / ( sum_n n_scalar_n )
```

Component NLLs are **never** normalised before the `logsumexp`; weights are never uniform
and never averaged over histories. Saved per batch: raw per-window NLL, scalar count,
component NLLs, log-weights, responsibilities, jitter diagnostics, and the normalised
aggregate.

**Defect found and fixed:** the driver was returning `mixture_nll.mean()` — a per-*window*
mean. Mask density varies window to window, so that is a different objective from
`Σ L_n / Σ n_scalar_n`, not a rescaling of it. Now corrected.

Before S4 you must still choose ONE of ACTIONS §8's honest paths: a tractable likelihood
matching the full sampled latent law, or keep the composite NLL as a labelled
training/diagnostic score and use one common proper path score for selection. **Not yet
chosen — this is an open decision for you.**

## Numerical-failure policy, frozen

During S4, **any** adaptive-jitter escalation, Cholesky failure, NaN/Inf or skipped
non-finite step makes that arm/seed a **failed fit**. Implemented: `strict=True` raises at
the first failing block instead of rescuing it, and jitter diagnostics are returned per
component per call into the caller's dict — the module-global last-call counter is
deprecated and must not be used for eligibility. One predeclared retry rule applies
identically to every arm. **No complete-case subset created by differential failures may
ever be compared.**

## Frozen arms, seeds, budgets, output root

Arms: shared-encoder native capacity at `D ∈ {1,2,4}` (`D=8` diagnostic only);
`full_experts` capacity control; genuinely parameter-matched `D=1`; modal-budget-matched
`D=1`; the prespecified CMD-D reference. Seeds: smoke {0}, pilot {0,1,2}, expansion target
0–9. Budget: AdamW, wd 1e-3, batch 128, LR grid {1e-4, 3e-5, 1e-5}, patience 20, selection
by lowest inner-validation composite NLL. Checkpoints: initialization, post-warm-up,
best-inner-validation and selected, plus optimizer/scheduler/RNG state — **none of which the
quarantined campaign saved.** Output root: a fresh immutable directory per run, not yet
created because S4 is not authorised.

## What I need from you

1. **Authorise or decline the cache re-collection** that unblocks B1 (two extra columns in
   `collect()`, one re-collection run). Without it there is no verifiable four-way split and
   S4 cannot start.
2. **Sign off or amend `frozen/decision_thresholds.yaml`** (B2).
3. **Choose the §8 scientific path**: tractable full-latent likelihood, or composite NLL as
   a labelled score plus a common proper path score for selection.

---

# Round 2 — sign-off package (SUPERSEDED 2026-08-25)

> **SUPERSEDED.** G1 was BLOCKED on B3 (lockbox blocks) and B2 (unsigned thresholds) when
> this was written. B3 was resolved by dated amendment A1 and B2 by the principal
> investigator's direction to freeze. See the CURRENT STATUS box at the top.
>
> Note also that this section's thresholds are **v2**, since superseded by **v3**.

Actions taken under the authorization: versioned raw-data recollection with safeguards,
thresholds v2, §8 path 2, S4 still not launched.

## Verdict (as of Round 2 — SUPERSEDED)

| gate | status | change |
|---|---|---|
| G0 | PASS | unchanged; Toy delivered the real adapter contract, v2 bytes unchanged |
| G1 | **BLOCKED** *(superseded → now PASS)* | B1 **RESOLVED**; new **B3** raised; B2 still awaiting sign-off |
| G2 | PASS | unchanged |
| G3 | PASS | unchanged |

`frozen/gate_index.json` — 14 artifacts, each hashed, with **per-test maxima**, and one
machine-readable verdict: `s4_authorized: false`.

## 1. Exposure audit — the original test partition is burned

`frozen/exposure_audit.md`. **It was used for selection, not merely observed.**
`finetuning/results/fixed_bugs/noaa_uk_h168/d/state.json` carries
`selection: {"split": "test", "weights": "ema"}` with two CMD-D trials scored on test
(CRPS 0.5077 vs 0.5056) and files named `select_test_ema.json`. Seven noaa_uk records
declare that selection policy. `run_analytic_uq_eval.py` defaults to `--split test`.

Upstream is clean: VAE and summarizer took **no gradient** from test and selected on val;
each computed one post-hoc test scalar. CMD-GP is clean — 0 of 87 result files contain a
test key. **The contamination sits precisely on the CMD-D reference arm.**

Therefore **all three original partitions are unusable as a lockbox** — train by gradient,
val and test by selection — and re-partitioning 2022–2024 cannot produce a clean one.

## 2. The versioned cache — B1 RESOLVED

`_cmd_mix_data/noaa_uk_v2_20260825_r2/split_manifest.json`

Rebuilt from **raw per-station hourly arrays before windowing**, not by repartitioning
anonymous windows. 108,624 windows, each carrying `sample_id`, `station_id`, raw
observation ids (`obs_id_context_start/end`, `obs_id_forecast_start/end`) and **complete**
context/forecast time boundaries.

| partition | period | windows | 504-h blocks/station |
|---|---|---|---|
| train | 2022-04-06 → 2024-03-31 | 65,612 | 30 |
| dev_val | 2024-04-01 → 2024-08-15 | 11,140 | 6 |
| dev_test | 2024-08-16 → 2024-12-31 | 11,236 | 6 |
| **lockbox** | **2025-01-01 → 2025-08-24** | **20,636** | **11** |

- **Embargo**: a window joins a partition only if its **entire 504-hour span** lies inside
  it *and* is contiguous hourly. Stricter than the shipped `global_purged_horizon`, which
  purged on the target interval and let context reach across boundaries.
- **Zero overlap proven twice**: independently from raw observation ids and again from
  time boundaries. All six pairwise intersections are **0 shared observations**.
- **Normalization fitted on train only**: 0 non-train windows intersect the train
  observation-id range, per station.
- **Lockbox sealed**: `lockbox/SEALED.json`, hashes verified.
- The first build is retained as `noaa_uk_v2_20260825/` with `SUPERSEDED.txt`. Nothing
  deleted.

### Decoder validation gate — and two bugs it caught

The lockbox comes from the freshly fetched NOAA archive, so its decoder was validated
against the overlapping 2024 year before use. It caught:

1. **`pyisd` leaves precipitation in raw tenths of mm** while scaling every other channel.
   Unfixed, the lockbox would have carried a 10× smaller precipitation channel under an
   identical units label. Now exact on all four stations.
2. **Two of the four stations report every 10 minutes.** `resample('h').mean()` averaged
   six sub-hourly reports where the original took only the top-of-hour. Measured on
   03005099999: `mean` reproduced 10.6% of hours, `first` 99.5%, **`minute == 0` 100%**.

Final state: **exact on 3 of 4 stations across all five channels**; 98.63% on
03017099999. No alternative convention improves that residual and both plausible ones make
other stations worse, so it is attributed to **archive reprocessing**, not the decoder.
Consequently development reuses the **shipped arrays verbatim** (bit-identical to what
everything was fitted on) and only the lockbox is built from fresh data. Vintage caveat
recorded in the manifest.

## 3. B3 — NEW BLOCKER, needs your decision

**The lockbox yields 11 independent 504-hour blocks per station; the pre-registered
minimum is 12.**

The NOAA archive stops at **2025-08-24 for all four stations** (files last modified Oct
2025; 2026 is 404 on both the S3 and NCEI mirrors). 5,662 hours = 11 blocks. This is a
data ceiling, not a design choice — unlike `dev_test`, which I re-balanced from 5 to 6
blocks because development split sizes *are* a design choice and no S4 result exists.

`decision_thresholds.yaml` v2 says fewer than 12 makes the confirmatory claim
inadmissible. **I have not amended it.** Options:

- **A)** dated pre-S4 amendment to 11 blocks / 44 station-blocks. Legitimate in the sense
  that no S4 result exists, so it cannot be outcome-driven — but it is still a change to a
  pre-registered rule and is your call.
- **B)** descriptive fallback: development evidence only, no confirmatory claim.
- **C)** shorter blocks — **not recommended**: 504 h is context+horizon, so shorter blocks
  are not independent.

## 4. Thresholds v2 — every requested amendment applied

`frozen/decision_thresholds.yaml` (v2 supersedes the unapproved v1). Holm-adjusted bound
must clear the **full 1% margin**; superiority required against **both** D=1 and the
matched-capacity control; 0.5% equivalence **withdrawn**, replaced by 5% variogram
equivalence and 2% ES/CRPS non-inferiority resolved against a **hashed development
scale**; independent (station × time-block) units with a hierarchical bootstrap;
improved-unit fraction ≥ 0.60; Monte Carlo precision ≤ 20% of the margin with common
random numbers; smallest-D tie-breaker; entropy and R **demoted to diagnostics**, replaced
by three semantic-collapse gates (mean responsibility ≤ 0.90, second-largest weight ≥
0.05, normalized component-function distance ≥ 0.10); **major-rewrite criterion** —
matched-quality speedup with one-sided 95% lower bound ≥ 4×, with the 1.5× training ratio
demoted to a resource condition; `seed_sd/effect ≤ 0.25` **removed**, expansion now gated
on every arm finishing with zero validity failures; float64 + TF32-off + scale-aware PSD
tolerances + zero rescues frozen.

**12 `PENDING_SCALE_FREEZE` slots block S4** until the development scale is computed.

## 5. Power/precision — please read before approving

`frozen/power_analysis.json`. Development paired sd is **1.045% of scale**; the observed
development effect is **0.832%** — **below** the 1% margin.

| true effect | n pairs for 80% power |
|---|---|
| 1.5% | 37 |
| 2.0% | 11 |
| 3.0% | 5 |

At 10 pairs, minimum detectable effect is **1.75%**. If the true effect is what
development suggests, **no number of seeds can clear a 1% margin** — the point estimate
does not reach it. The design is powered for an effect ~2.4× larger than development has
shown. That is worth knowing before spending the compute.

## 6. §8 path 2 — scoring harness built and property-tested

`cmd_mix/scoring.py`, `reports/scoring_selftest.json`, all six checks pass:

- **Unbiased ES varies 0.0151 across M ∈ {4…256}; the naive estimator varies 0.4719** — a
  31× difference, and the naive one drifts monotonically, i.e. it rewards under-dispersion
  exactly as warned.
- CRPS matches the closed form to **0.0045**.
- **Variogram detects cross-channel dependence**: two ensembles with *identical marginals*
  and opposite channel correlation score 0.109 vs 0.608 (5.6×) on cross-channel pairs
  while the temporal component is unchanged. The cross-channel pairs earn their place.
- Calibration separates a calibrated ensemble (ECE 0.027) from a biased one (0.377).

Strategy §10.1 amended: latent NLL is **training/diagnostic only**; selection moves to the
observation-space variogram score; ES and calibration are ordered safeguards; the full
state-space likelihood is deferred behind the major-rewrite criterion.

## What is still open

1. **B3** — your decision (A, B or C).
2. **B2** — sign off or amend thresholds v2.
3. **Development-scale freeze** — needs the new partitions run through the frozen
   VAE/summarizer to produce latents, then the reference climatology scored through the
   full stochastic decoder. **Deliberately not started**: it is expensive and option B on
   B3 would change what it is for.

---

<a id="final-readiness-report--2026-08-25"></a>

# FINAL READINESS REPORT — 2026-08-25

> **THIS SECTION IS CURRENT.** It supersedes Rounds 1 and 2 above.

## Verdict

```
G0 = PASS   G1 = PASS   G2 = PASS   G3 = PASS      s4_authorized: true
```

`frozen/gate_index.json`, 18 artifacts, each hashed, with per-test maxima.
All four G1 blockers resolved: B1 (four-way split), B2 (thresholds), B3 (lockbox blocks).

## Amendment A1 — recorded as directed

`frozen/AMENDMENT_2026-08-25_A1.md`. Lockbox minimum 12 → 11 blocks. Cause: the verified
archive ceiling alone — all four stations end 2025-08-24, 2026 returns 404 on both
mirrors. Attested: no lockbox targets and no model results inspected; block length (504 h,
**not shortened**), boundaries, embargo, endpoint, models, analysis and practical margins
all unchanged. The 1% margin is **retained**, not moved toward 0.832%.

## Development scale — FROZEN, train + dev_val only

`frozen/development_scale.json`. `dev_test` and `lockbox` never opened; the partition
filter is asserted in code. No VAE involved — the scale is a property of the data and a
frozen climatological reference, both in observation space.

| field | value |
|---|---|
| VS_0.5 reference | 0.29007897178786074 |
| unbiased ES reference | 18.05023060729393 |
| CRPS reference | 0.44984389331861 |
| scale hash | `3e622a8f791efff152587224e65dc1dd628ebddcf048a8ae6ae8350b0e4da1a2` |
| variogram weights hash | `d494b9eaaa12787bcfe5fc9a222fe51f0230066025b71e5303b762eaf36d3b2d` |
| pairs | 11,460 — 5,050 temporal, 6,410 cross-channel |
| ensemble / CRN seed | 128 / 20260825 |

**All 12 pending fields resolved.** Derived margins: superiority 0.0029008, variogram
equivalence 0.0145039, ES non-inferiority 0.3610046, CRPS non-inferiority 0.0089969.

## Power — rebuilt on independent blocks

`cmd_mix/power.py`, `frozen/power_analysis.json`. The v2 figures are **withdrawn**: they
used sd across **optimizer seeds** on a training diagnostic, which is not the sampling
variance of a block-level contrast. The "1.75% MDE" is also withdrawn — it was
approximately the **observed** improvement needed for the bound to clear 1%, not a powered
MDE.

**Specification.** Endpoint: observation-space VS_0.5 over temporal + cross-channel pairs,
after the complete stochastic decoder, common random numbers. Contrast: paired
VS(comparator) − VS(CMD-Mix(D)). Scale: the hashed reference above. Unit: one independent
**504-hour block**. Allocation: 4 stations × 11 blocks = 44 station-blocks. Seeds are
**nested inside arms**, not units — averaging over S seeds divides the seed variance by S
and leaves the block variance untouched, so seeds cannot buy block-level precision. MCSE
enters additively, reduced by CRN. Multiplicity: Holm over 4 hypotheses, FWER 0.05, step 1
one-sided α = 0.0125. Formula: `Var = σ²_block/n_eff + σ²_seed/S + σ²_mc/S_mc`,
`n_eff = k·m/(1+(m−1)ρ)`, non-central t with `df = n_eff − 1`.

**Measured on dev_val:** σ_block = **6.293%** of scale, ρ = **0.2799**, **n_eff = 11.58**,
SE = **1.849%**.

| conclusion | observed contrast required |
|---|---|
| feasibility (bound > 0) | **3.33%** of scale |
| major rewrite (Holm bound > 1%) | **5.82%** of scale |

| true effect | power, feasibility | power, major rewrite |
|---|---|---|
| 2% | 0.264 | 0.039 |
| 3% | 0.450 | 0.100 |
| 5% | 0.811 | 0.370 |
| 8% | 0.991 | 0.860 |

**Stated in advance: a null result is a likely and legitimate outcome of this design.**

σ_block came from a persistence-vs-climatology contrast — two structurally dissimilar
predictors. A CMD-Mix(2)-vs-CMD-Mix(1) contrast pairs similar models under CRN with
strongly correlated errors, so 6.293% is a conservative **upper bound**. Re-estimation
from real paired arm scores is **required at smoke**, on train + inner validation only.

## Margin attainability — one open issue

| margin | half-width 3.33% | attainable | needs σ_block ≤ |
|---|---|---|---|
| variogram 5% | | **yes** | 9.44% |
| ES/CRPS 2% | | **NO** | 3.78% |
| the withdrawn 0.5% | | no | 0.94% |

Your concern was right in kind. The variogram equivalence margin is **already 5%**, not
0.5% — 0.5% was v1's margin, withdrawn in v2 — and 5% is attainable. But the **2% ES/CRPS
non-inferiority margin is not attainable** at the measured variability, even when the true
difference is exactly zero. I have **not** widened it, because widening a margin to fit
the variability of a *proxy* contrast is the same error in the other direction. It is
retained unchanged and revisited exactly once, at smoke, against the real paired σ.

## Upstream exposure beyond gradients — one leak found

| channel | status |
|---|---|
| **normalization** | **LEAK** |
| cached representations | clean |
| calibration | clean — no post-hoc layer exists |
| checkpoint selection | mixed — VAE/summarizer on val; **CMD-D on test**; CMD-GP never |
| architecture decisions | clean — `laplace_k=128` selected on val/ema |

The shipped `norm_stats.json` was computed over the **full series including test**, and
those statistics are applied at load time. Measured `|stored − train_only| / sd` up to
**0.0809** (mean 0.0346) across 4 stations × 5 channels; `mean_y` matches the full-series
mean to 3.9e-4 and the train-only mean only to 0.277. **Every model in the project trained
on inputs standardised with test-period information.** Small in magnitude, systematic in
kind, and it voids any claim the legacy cache had train-only preprocessing. The v2 cache
fixes it and the fix is verified.

## Station-stratified sensitivity — frozen

`03017099999` reproduces 98.63% of 2024 hours; the residual 120 of 8,784 are hours with no
minute-0 report in the current archive, typical magnitude 0.4 °C. Recorded as a **frozen
archive-version shift**, not a decoder defect. Every headline result is reported **pooled
and station-stratified**; leave-one-station-out is declared in advance; **`03017099999` is
never excluded, and never excluded after seeing results.**

## CMD-D comparator

Marked `MUST_BE_RETRAINED`. The existing checkpoint was selected on test and is
DEVELOPMENT_ONLY. The S4 comparator is fit on `train` and selected on `dev_val`, never
reading `dev_test` or `lockbox`.

## Smoke stage — prerequisite, stated honestly

S4 is authorized, but **smoke cannot start yet.** CMD-Mix consumes summary latents
`(E, mu, dt, mask)` produced by the frozen VAE + summarizer via `collect()`, which
operates on the project's own DataLoader. **The v2 partitions have not been converted to
that form.** Building it means constructing a loader over the v2 window index that matches
the project's batch contract exactly — `_sanitize_batch`, `_build_cond_summary_pair`,
`_latent_targets_for_batch`, `_flatten_dt` — and a subtle mismatch would silently produce
different conditioning from every previous run.

Planned with the same validation-gate discipline that caught the precipitation scaling and
the sub-hourly resampling: build the loader, run it over the **legacy** split boundaries,
and require the resulting `E/mu/dt/mask` to reproduce the shipped `splits.pt` to float
tolerance **before** it is used on the v2 partitions. I have not started smoke on an
unvalidated collection.

## A defect in my own tooling, found and fixed

`freeze_s3.py` re-emitted a v1 `decision_thresholds.yaml` literal on every `--force` run,
silently reverting the hand-maintained versioned file — and reporting success. It now
hashes the thresholds and never authors them; v1 is preserved as
`decision_thresholds_v1_historical.yaml`. Any earlier statement that v2 was in place at a
given moment should be read against that.

---

# RETRACTION — the normalization leak does not exist (2026-08-25)

In the FINAL READINESS REPORT above I reported, under *upstream exposure beyond
gradients*, that `norm_stats.json` was computed over the full series including test, that
those statistics are applied at load time, and therefore that **"every model in the
project trained on inputs standardised with test-period information."**

**That conclusion is false and is withdrawn.**

The file does contain full-series statistics — that part was right. But the loader
**discards them**. `run_experiment(norm="train_only")` calls
`_compute_train_only_norm_stats`, whose docstring reads *"Compute mean/std for X and Y
using ONLY rows that can appear in TRAIN contexts"*, and the call site then executes
`norm_stats = tr_norm`. I inferred the stored file's use from its presence and from the
`self.mean_x = norm_stats['mean_x']` assignment, without following the override.

Measured, by re-deriving the normalized context independently from raw bytes:

| normalisation used | max &#124;V − re-derived&#124; |
|---|---|
| stored full-series | **0.141** |
| **train-only** | **0 mismatches over 156 windows** |

Last train context row per station: {0: 16620, 1: 16596, 2: 14780, 3: 16476}.

The `upstream_exposure.normalization` entry in `decision_thresholds.yaml` is corrected
from `LEAK_FOUND_IN_LEGACY_CACHE` to `CLEAN`. **All five audited upstream channels are now
clean except checkpoint selection**, where the CMD-D test-split selection finding stands
unchanged and still requires retraining.

Found by the G4.1b neural-boundary audit — the check ordered specifically to attribute the
E/mu discrepancy. It attributed the discrepancy and corrected a wrong finding of mine in
the same pass.
