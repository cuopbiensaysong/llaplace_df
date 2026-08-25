# CMD-Mix(D) pilot — implementation plan (noaa_uk + USHCN)

Plan of record for `w_docs/CMD_MIX_NOAA_UK_TEST_STRATEGY.md`, extended to USHCN.
Written 2026-08-24. **Nothing here is a headline result**; per strategy §1 this is a viability
pilot whose output is evidence for a later decision.

---

## 0. What CMD-Mix(D) is, and what it is not

One vectorised network call per history produces weights $\pi_{nd}$ **and** $D$
history-conditioned CMD-GP parameter sets:

$$p(z_Q\mid H_n)=\sum_{d=1}^{D}\pi_{nd}\,\mathcal N\!\big(z_Q; m_{nd}(Q), K_{nd}(Q,Q)\big).$$

**$D$ is not $M$.** $M$ is the number of CMD-D *diffusion trajectories* — the mixture we already
measure in the frontier. $D$ is the number of *explicitly learned* components from a single
forward pass. Conflating them is the central hazard of this pilot; every artifact records both.

---

## 1. Dependency audit — what exists, what does not

| strategy requirement | status |
|---|---|
| production CMD-GP (= CMD-Mix(1) target) | ✅ `dist_heads.ChirpGPHead` |
| projective variance vs a fixed grid | ✅ `anchor_nodes>0` → `modal_lambda_projective` |
| deliberate inconsistent control (§7.2) | ✅ `anchor_nodes=0` → requested-grid `modal_increments` |
| exact mixture NLL by `logsumexp` | ✅ `path_scores.mixture_nll` (was dead code; now also used by `_campaign_scripts/joint_nll.py`) |
| per-component joint likelihood | ✅ `gaussian_joint_nll_per_channel` + `cross_time_kernel` |
| global-component joint sampling | ✅ `sample_joint_markov` (draw one label per path) |
| **CMD-Mix(D) head, D>1** | ❌ **does not exist — must be built** |
| **Toy B golden conformance fixture** | ⏳ **exists — separate machine, isolated codebase; not yet reachable from this filesystem** |
| **4-way nested split (train / inner / outer gate / lockbox)** | ❌ 3-way only (0.7/0.1/0.2) |
| `decision_thresholds.yaml` | ❌ must be authored before fitting |

### 1.1 ⏳ Pending dependency: the Toy B conformance fixture

Toy B is being run **on another machine against an isolated codebase**, so its golden fixture
is not reachable from this filesystem yet. It is *pending delivery*, not missing.

Strategy §2 requires the adapter to reproduce that fixture's expected weights, restricted
component laws, latent log densities and samples **before comparative fitting**, saving the
fixture hash and adapter/config hash in every run manifest. That step is therefore **deferred,
not skipped** — and it gates S4 (comparative fitting), not S2.

**What I need from the Toy B machine:** the versioned fixture file plus its hash. Anything that
pins the four semantics in §1.3 is enough to start; the full fixture is needed before S4.

The self-contained parts of Gate A (§7.1 D=1 regression, §7.2 projectivity, §7.3 mixture
algebra) do **not** depend on it and proceed now.

### 1.3 Semantic contract — record now, reconcile with Toy B on delivery

§4 requires both implementations to agree on the meaning of $D$, $M$, global component
sampling, `logsumexp` likelihood and query-set independence, and to *"record any semantic
deviation explicitly."* Writing this down **before** the fixture arrives is what makes a
deviation detectable rather than negotiable after the fact. This adapter commits to:

| semantic | this adapter's choice |
|---|---|
| $D$ | explicitly learned components from **one** forward pass; `nn.ModuleList` of $D$ `ChirpGPHead`s |
| $M$ | **never** used here; reserved for CMD-D diffusion trajectories |
| weights | `log_softmax(Linear(cond))`, shape `(N,D)`, retained per-history; **no weight head at all when $D=1$** |
| weight inputs | `cond` **only** — never `t_rel`, query summaries, or target-derived features |
| mixture likelihood | **weighted** $-\log\sum_d \exp(\log\pi_d - \mathrm{nll}_d)$ — deliberately *not* `path_scores.mixture_nll`, which assumes uniform $1/M$ and is correct only for the trajectory mixture |
| per-component likelihood | `cross_time_kernel` → `gaussian_joint_nll_per_channel`; **pseudo**-likelihood (joint over time, independent across channels) |
| global sampling | one component label per **whole path**, drawn from that history's $\pi_{nd}$ |
| query-set independence | `anchor_nodes>0` → `modal_lambda_projective` against a fixed grid |
| responsibilities | posterior $\propto \pi_d e^{-\mathrm{nll}_d}$, **diagnostic only** — uses the realised future, never available to a forecast-time gate (§10.4) |

If Toy B differs on any row, that is a recorded semantic deviation and must be reconciled
before the two are described as the same object.

### 1.2 The lockbox we already have — protect it

The campaign has **never read the test split** on any cell. That makes the existing test
partition a genuine untouched confirmation lockbox in the sense of §5.2 — which is unusual and
directly determines whether this pilot can ever be confirmatory rather than descriptive
(§5.2, Gate D). **No stage of this plan touches test.** The outer gate is carved from the
existing validation material with an embargo; the lockbox stays sealed until §8.3 stage 4.

⚠️ Exposure to disclose under §5.1: `R = 0.6256` (noaa_uk) and the whole Family A / frontier
campaign are **validation** numbers, and validation has been inspected repeatedly. They are
hypothesis-generating only and cannot gate model selection here.

---

## 2. Staging, with cost and what blocks what

| stage | content | compute | blocks |
|---|---|---|---|
| **S1** | CMD-Mix(D) experiment-local adapter (§4 interface) | none | S2–S5 |
| **S2** | **Gate A**: D=1 mapped regression, projectivity suite, mixture algebra, inconsistent control | **CPU, minutes** | S4 (hard stop on failure) |
| **S3** | 4-way split manifest w/ embargo, protocol manifest, `decision_thresholds.yaml` | none | S4 |
| **S4** | D∈{1,2,4} × seeds, both cells, + CMD-D anchors | **GPU, the whole cost** | S5 |
| **S5** | paired hierarchical analysis, gates B/C/D, decision report | CPU | — |

**S1 and S2 need no GPU at all.** That is the order to do them in: Gate A is a hard stop
(§13), so spending GPU on S4 before it passes would be wasted.

---

## 3. S1 — the adapter (strategy §4 interface)

Experiment-local, in `_campaign_scripts/cmd_mix/`, importing the shipped dynamical core.
`llaplace_df` is **not** edited (jobs run from it; §2 also forbids repo-wide refactors here).

Exposes, per §4: `logits`/`weights` `(N,D)`; query-set-independent component coefficients;
per-component means and covariance representation; per-component joint log-likelihoods and
exact mixture log-likelihood; global-component joint sampling; within/between moment shares;
network-evaluation count, sequential depth, wall time, memory.

**Design constraints that make D=1 exact.** The weight head is a separate module reading
`cond` only; for D=1 its softmax is exactly 1, so the mixture law *is* the component law and
`logsumexp` over one component is that component's own log-density. A CMD-Mix(1) built from D
independent `ChirpGPHead`s can therefore load an existing CMD-GP `state_dict` into its single
component and reproduce it **bitwise** (§7.1), rather than merely closely.

**Weights depend on history only** (§3): the weight head never sees `t_rel`, query summaries,
or target-derived features. Asserted in code, not just intended.

---

## 4. S2 — Gate A, the part that can run today

1. **D=1 mapped-checkpoint regression** (§7.1): load a trained `ChirpGPHead` into CMD-Mix(1);
   compare component parameters, means, covariance, joint/marginal likelihoods, seeded samples,
   parameter count. Exact equality expected for the mapped implementation regression.
2. **Projectivity** (§7.2): ≥100 histories spanning sites/channels/missingness/first-lead gaps.
   Random $S\subset T$ at 25/50/75 %; insertion before/between/after; unsorted permutations;
   dense clusters, long gaps, near-duplicate times; multiple channels and asynchronous
   channel-time queries. Compare weights, component means, covariance submatrices, mixture
   moments and mixture log density (restricted $T$-law vs direct $S$-law at fixed $z_S$).
   float64 reference, $\epsilon_{abs}=10^{-8}$, $\epsilon_{rel}=10^{-6}$; max norm for weights
   and scalar log density, scaled Euclidean for means, Frobenius for covariance blocks; report
   median / p95 / max absolute and relative error.
3. **Inconsistent positive control**: the same suite against `anchor_nodes=0` (requested-grid
   recurrence). **The suite must FAIL it.** A suite that passes the control is not a test.
4. **Mixture algebra** (§7.3): exact per-history `logsumexp`; analytic vs Monte-Carlo moments;
   global component label per path; sampling frequencies match history weights; covariance
   PSD/nugget; no future targets or query summaries reach the gate.

**A projectivity failure is a hard stop regardless of score** (§7.2).

---

## 5. S3 — splits. The part most likely to be got wrong

Current: 3-way `global_purged_horizon` 0.7/0.1/0.2. Required: four mutually exclusive
manifests (§5.2). Plan, per §5.2's "clean default":

* **inner validation** carved from the **end of the training period**;
* the existing **validation** material split **70/30** into **outer gate** / **lockbox-B**;
* **embargo** ≥ max context span + forecast horizon at every boundary, so overlapping windows
  cannot share raw observations. noaa_uk/bms_air: 336+168 native steps. USHCN: context span
  and horizon are in **days**, not steps — median context 104 d, H=12 *observed steps* ≈ 53 d
  median, so the embargo must be computed in physical days from `windows.parquet`, not steps.
* the untouched **test** split remains the primary confirmation lockbox (§1.2).

Partition at the highest defensible unit: **station** for USHCN (1 113 stations with windows)
where the protocol permits; **chronological blocks** for noaa_uk/bms_air.

⚠️ **USHCN-specific**: its own protocol is *temporal extrapolation*, and the dataset guide is
explicit that the same stations appear in all splits by design. A station-wise partition would
answer a **different question** than the frozen USHCN protocol. Default therefore: chronological
blocks with a day-based embargo, and station-wise only as a declared secondary stratum.

---

## 6. S4 — training, and the honest cost

Arms per §8.1, native-capacity and parameter-matched (§8.2 views 1–2 first; FLOP and
measured-latency matching at §8.2 views 4–5 for Gate C).

$D\in\{1,2,4\}$; $D=8$ diagnostic-only and predeclared, ineligible for the rewrite decision.
Staging per §8.3: smoke → 3 paired seeds → ≥5 (target 10) → lockbox once.

**Cost is unknown and must be measured, not estimated.** CMD-GP one-pass training on these
cells is the closest analogue: Family A ran ~14.7 min per (head, LR) fit on USHCN, ~32 fits per
seed. CMD-Mix(D) multiplies component work by $D$. Before committing seeds I will run one smoke
fit per cell and report a measured per-fit time.

---

## 7. Ordering decision

Do **S1 → S2 on both cells first**, on CPU, while the GPUs stay on the current campaign
(bms_air R8 grid, USHCN s1/s2). Gate A is a hard stop; it costs no GPU; and if the adapter is
wrong we learn it in minutes rather than after a seed sweep.

USHCN adds one dependency the strategy does not mention: its **s2/R9 result does not exist
yet**, so the CMD-D anchor for Gate C on USHCN cannot be built until the diffusion arm lands.
Gate A and Gate B on USHCN do **not** need it; Gate C does. Recorded, not worked around.

---

## Gate A — TRUSTED RUN (2026-08-24 20:54)

`gate_a_ushcn_TRAINED.json`. **GATE A: PASS**, and this is the first run whose verdict is
worth anything.

### Why the earlier three runs did not count

`gate_a_dz16` / `gate_a_dz24` / `gate_a_ushcn` all built a **randomly initialised** head with
**random Gaussian conditioning** at a **guessed horizon (120)**. Under that configuration the
predictive variance is dominated by the `e^{-2 rho_bar} p0` term, and the quadrature term `v_k`
— the only part that anchor-grid vs requested-grid integration can disagree about — contributes
at ~1e-7 relative. So the deliberate inconsistent control (`anchor_nodes=0`) deviated from
projectivity by less than the tolerance, i.e. **the suite could not see its own control**:

| A3 control (anchor_nodes=0), abs_max / rel_max | untrained, random cond | **trained, real cond** |
|---|---|---|
| `component_var`        | 3.19e-08 / 1.00e-07 ok | **1.79e-04 / 5.04e-04 FAIL** |
| `component_kernel`     | 3.95e-07 / 1.24e-07 ok | **1.36e-03 / 4.83e-04 FAIL** |
| `mixture_log_density`  | 6.35e-07 / 3.00e-05 ok | **7.19e-03 / 1.20e-03 FAIL** |
| control passes projectivity? | `True` (BAD) | `False` (correct) |
| **control DETECTED**   | **False** | **True** |

On USHCN's geometry the control passed outright (`detected=False`) — a suite that passes its
own negative control is not a test. `gate_a_dz16` "detected" it only by a hair, for the same
accidental reason. **The fix was realism, not tolerance tuning**: tolerances are unchanged at
`eps_abs=1e-8`, `eps_rel=1e-6`.

Note on reading the tables: the criterion is the standard combined form
`|a-b| <= eps_abs + eps_rel*|b|`, elementwise. A large `rel_max` on a near-zero denominator is
correctly tolerated, so `rel_max` alone is misleading — `abs_max` is what decides these rows.

### What the trusted run used

- Head: `_ushcn_experiments/ldt/heads/head_chirp_gp_s0.pt` — the **trained** Family A
  `chirp_gp` head, seed 0, selected lr 3e-06, val objective 1.4329.
- Architecture taken from the checkpoint's own kwargs, not guessed:
  `hidden=256, depth=2, k=16, num_basis=6, horizon=534.0, anchor_nodes=128, nugget=True,
  parameterization='p_exact'`, `cond_dim=4096`, `d_z=16`.
- Conditioning: `gate_a_fixture_ushcn.pt` — 128 **real** validation windows, `E (128,4096)`,
  real query offsets `dt` spanning **1.00 .. 98.00 days**, indices spread across the split
  because the loader does not shuffle.
- Correction to an earlier note: the `horizon 12` printed by `family_a_ushcn.py` is
  `mu.shape[1]`, the number of horizon **steps**. The head's `horizon` kwarg is **534.0**.
  My earlier guesses of both 120 and 12 were wrong; the checkpoint is now the source of truth.

### Separation achieved

| | A2 real model (`anchor_nodes=128`) | A3 control (`anchor_nodes=0`) |
|---|---|---|
| worst `abs_max` | **1.78e-15** | **7.19e-03** |
| verdict | PASS (must) | FAIL (must) |

~12 orders of magnitude. Projectivity of the real model holds at machine precision for
D in {1,2,4}; the control fails loudly. A1 (D=1 reproduces CMD-GP) is bitwise exact with
identical parameter counts; A4 mixture algebra passes at D in {1,2,4}.

**Gate A is cleared — the D sweep may now spend GPU.**

### Harness changes
`gate_a.py` gained `--trained-head` and `--fixture`. With `--trained-head` every component is
initialised from the trained CMD-GP and the architecture comes from the checkpoint; the control
overrides `anchor_nodes` only. A1 also runs against the trained head when one is supplied.

---

## §7 Encoder replication — design correction (2026-08-24)

Reporting parameters in four roles (encoder | component heads | gate | decoder) exposed a
confound that the totals hid: **`shared encoder` was 0 on every cell.** `ChirpGPHead` carries
its conditioning trunk internally, so D independent heads replicate the entire encoder.

| cell | d_z | trunk | head/comp | trunk % of D=1 | D=1 | D=2 full_experts | D=2 shared |
|---|---|---|---|---|---|---|---|
| noaa_uk | 16 | 1,114,624 | 349,648 | 76% | 1,464,272 | 2,936,738 (+100%) | **1,822,114 (+24.4%)** |
| bms_air | 24 | 1,114,624 | 483,288 | 70% | 1,597,912 | 3,204,018 (+100%) | **2,089,394 (+30.8%)** |
| ushcn   | 16 | 1,114,624 | 193,328 | 85% | 1,307,952 | 2,624,098 (+100%) | **1,509,474 (+15.4%)** |

76–85% of what D=2 added was duplicated encoder, not mixture capacity. A D=2 > D=1 win under
that accounting could not be attributed to the mixture.

### Decision of record
* **`shared` is the PRIMARY experiment** and the launcher default.
* **`full_experts` is an explicitly named capacity control** — kept, not deleted; a win there
  that does not replicate under `shared` is capacity, not mixture structure.
* **Every arm of the D sweep uses `mixture_joint`, D=1 included.** At D=1 the one-component
  logsumexp reduces to that component's joint Gaussian likelihood — the correct D→1 limit.
  D=1 `marginal` is retained only as an adapter plumbing regression against the published
  head; it is NOT the scientific control, and pairing it against D=2 `mixture_joint` would
  confound components with objective.

### Parameter sharing was not enough — the encoder had to be hoisted
The first implementation rebound each component's `trunk` to component 0's. That shares
*parameters* but not *compute*: `ChirpGPHead.forward` opens with `h = self.trunk(cond)`, so
every component still re-ran the encoder and the true cost stayed `D*(C_enc + C_head)`.
The encoder is now owned by `CMDMix`, components get `nn.Identity()`, and `h` is computed
once. Verified **bitwise identical** to the per-component path under matched weights
(max|Δ| = 0.000e+00 on mean and var), so this is a pure compute change.

### Cost model, measured not assumed
`--profile-cost` measures `C_encoder + D*C_component_heads + C_mixture_algebra` and reports
the residual against the measured forward. The residual is what distinguishes the variants:

| variant | predicted | measured | residual |
|---|---|---|---|
| shared | 8.380 ms | 8.308 ms | **−0.9%** (additive model holds) |
| full_experts | 6.642 ms | 8.165 ms | **+18.7%** (encoder runs D times) |

(CPU, batch 16, T=12 — a mechanism check. Reportable numbers need an uncontended GPU.)
Early signal to confirm on GPU: `C_mixture_algebra` dominated both (~10x the forward), so the
mixture algebra, not the encoder or heads, may be the binding cost.

### Gate A re-run after the change
Both variants PASS with identical margins — A1 bitwise (1,307,952 params both sides),
projectivity 6.421e-16 (D=1,2) / 1.284e-15 (D=4), control DETECTED. `gate_a.py --share-trunk`
now gates the primary variant; `load_cmd_gp_state` re-targets trunk keys when hoisted and
raises if two different encoders are loaded into one shared trunk.

---

## §8 The binding cost is the mixture algebra, and it was a kernel-launch pathology

GPU measurement (RTX A5000, noaa_uk, D=2 shared, batch 256, T=168):

| term | measured |
|---|---|
| `C_encoder` | 0.070 ms |
| `C_component_head` | 2.690 ms (x D) |
| **`C_mixture_algebra`** | **3856.725 ms** |
| predicted forward | 5.449 ms |
| measured forward | 5.464 ms (**residual +0.3%** — the additive model holds) |

The cost model `C_encoder + D*C_component_heads + C_mixture_algebra` is confirmed, and it
localises the whole problem: the mixture algebra is **706x the forward**. At 64 batches/epoch
that is ~4 min/epoch forward-only, so 80 NLL epochs ≈ 13-16 h *per arm* before backward.

### Root cause
`path_scores.gaussian_joint_nll_per_channel` loops `for b in range(B): for d in range(D)`,
factorising one [T_obs,T_obs] matrix per (window, channel). At noaa_uk sizes that is
**256 x 16 = 4,096 sequential Cholesky calls per component** (8,192 at D=2), each its own GPU
kernel launch. The cost is launch overhead, not arithmetic — 32 batched 168x168 Choleskys
would be milliseconds. The loop is not a defect in the shipped code: its docstring is explicit
that per-window, per-channel selection is what stops an irregular mask silently dropping a
window. It is simply the wrong shape for this workload.

### Fix — `cmd_mix/fast_joint_nll.py`, experiment-local (llaplace_df untouched)
One batched factorisation that neutralises masked entries instead of indexing them out:
masked rows/cols zeroed with a unit diagonal (contributing log 1 = 0 to the determinant),
residual zeroed at masked times, and the surplus `0.5*log(2*pi)` constant subtracted exactly.
**Algebraically identical, not an approximation.** Verified against the shipped function in
float64:

| case | max abs | max rel |
|---|---|---|
| fully observed | 3.638e-12 | 2.396e-16 |
| random 70% observed | 2.558e-13 | 1.038e-15 |
| one channel fully missing | 1.819e-12 | 1.333e-16 |
| one window fully missing | 3.638e-12 | 2.396e-16 |

End-to-end through `CMDMix.mixture_nll`, real noaa_uk conditioning, D in {1,2}, real and
synthetic irregular masks: max rel 5.9e-16.

Selected by `CMDMix(fast_nll=True)` (the default); `--slow-nll` restores the shipped path.

### Gate A after the change
PASS on the shared variant, control still DETECTED. **Not byte-identical**: projectivity
margins moved 6.421e-16 -> 2.713e-14, which is the batched factorisation's different
floating-point reduction order, still ~8 orders below the 1e-6 tolerance, against a control
that fails at 7.2e-3.

### Open
The GPU speedup is **not yet measured** — CPU at B=6 shows only ~2.2x because the pathology is
launch overhead at B=256 on GPU. Re-profile before costing the sweep.

---

## §9 PSD diagnosis — the chirp kernel is CONDITIONALLY PSD, and `p_exact` does not enforce the condition

The noaa_uk joint-NLL arms kept failing Cholesky even after relative-jitter escalation to 1e6x.
Two earlier diagnoses of mine were wrong and are retracted: it is **not** a collapsed nugget
(`dist_heads` floors it at `SIGMA2_MIN = 1e-4`) and it is **not** curable by jitter.

### The construction
Per channel, `cross_time_kernel` computes
`K(t,t') = sum_k energy_k * Lambda_k(min(t,t')) * exp(-|d_rhobar_k|) * cos(d_omegabar_k)`.
`energy = c^2 + b^2 >= 0`; `cos(d_omega)` is PSD (Gram of `(cos w, sin w)`); `exp(-|d_rho|)`
is PSD **when rhobar is monotone** (OU under the time-change `u = rhobar`); and
`f(min(t,t'))` is PSD **only if f is non-negative and non-decreasing**. So the kernel is PSD
iff **Lambda is non-decreasing AND rhobar is non-decreasing (Assumption R)**.

### Measured, on the trained USHCN head (32 windows x 16 channels, float64)
| condition | min eigenvalue | indefinite blocks |
|---|---|---|
| baseline (assumptions hold) | **+8.64e-02** | 0 / 512 |
| Lambda non-monotone, 30% of steps | −3.96e-02 | 5 / 512 |
| rhobar excursion, amp 0.5 | −2.18e-01 | 90 / 512 |
| rhobar excursion, amp 2.0 | −3.38e+00 | 497 / 512 |

Breaking Assumption R is by far the more damaging. At amp 2.0, 97% of blocks are indefinite
with min eigenvalue −3.38 against an O(1) kernel scale — no jitter short of swamping the
kernel can rescue that, which is exactly what the 1e6x escalation failure was telling us.

### Why the condition is unenforced
`laptrans.integrated` under `parameterization="p_exact"` (what every run used, inherited from
the Family A config) computes `rho_bar = rho_floor * t_rel + rho_var`, where `rho_var` is a
**learned, unconstrained** combination of basis antiderivatives. Only the `rho_floor * t_rel`
term is monotone. (`growth_budget` defaults to 0, so Theorem B' excursions are OFF — that is
not the leak.) Nothing stops training from driving `rho_var` non-monotone.

`d_rho.clamp_min(_RHO_FLOOR)` with `_RHO_FLOOR = -20.0` does **not** enforce Assumption R
either — it permits `d_rho` down to −20, i.e. growth of `e^20 ~ 4.9e8`. It is an overflow
guard, not a validity guard.

### Why this appeared only now, and only on noaa_uk
Family A trained the **marginal** NLL, which never touches the off-diagonal kernel, so
rhobar's monotonicity was neither used nor punished. `mixture_joint` is the first objective
that trains through the cross-time covariance, and it can reach a lower loss by shaping
correlations in ways that leave the valid regime. USHCN (T=12) gives an excursion little room;
noaa_uk (T=168) gives it a great deal.

### Consequence for the results already collected
The surviving noaa_uk arms are **suspect, not merely incomplete**. An arm crashes only when a
block becomes indefinite *enough* to fail Cholesky; an arm with milder violations trains on a
covariance that is not a covariance, and its reported "NLL" is then not a log-likelihood. The
n=6 D=2-vs-D=1 comparison (mean +17.02, sd 21.38, sign test 4/6, p=0.688 — already not
significant) cannot be interpreted until validity is established per arm.

### Remedy
`parameterization="p_mono"` builds `rho_bar_var = sum_m u [softplus(v tau + b) - softplus(b)]`
whose t-derivative is non-negative by construction — i.e. Assumption R holds by design. It is
a **config change, not a code change**, and P-mono is a production variant named in strategy
§2 alongside P-exact. Required alongside it: a per-epoch diagnostic recording the fraction of
steps with `d_rhobar < 0` and the minimum kernel eigenvalue, so validity is asserted rather
than assumed.

---

## §10 — G0 and G3 under `CMD_MIX_NOAA_UK_V2_ACTIONS.md` (2026-08-25)

### §10.1 §9 is RETRACTED

**§9 concluded that constrained `p_exact` leaves Assumption R unenforced. That conclusion
is wrong and is withdrawn.** Two independent errors:

**The method was invalid.** §9's causal test perturbed `rho_bar` directly while keeping the
trained head's `Lambda`. `Lambda` and `rho_bar` are coupled through one quadrature, so a
hand-edited `rho_bar` with a stale `Lambda` is an inconsistent pair that need not describe
any law — indefiniteness there says nothing about the model. ACTIONS §6 called this out
before the audit was run, and it was right.

**The reading of the code was wrong.** §9 stated that `rho_var` is "a learned, unconstrained
combination of basis antiderivatives". It is not:

- `_coeffs` returns `a_rho2 = c**2`, i.e. **non-negative by construction**.
- Under `rho_basis="centered"` it then rescales by `rho_head/(rho_total + rho_head)` with
  `rho_head = rho_floor - rho_min`, so `sum_m a_rho2 < rho_floor - rho_min`.
- `Phi_rho = Phi - t_rel`, whose t-derivative is `phi_m - 1 = s_m cos(.)`, so
  `|phi_m - 1| <= 1`.
- Therefore `d(rho_bar)/dt = rho_floor + sum_m a_rho2 (phi_m - 1) >= rho_min > 0`.
- `ChirpGPHead` **hard-codes** `rho_basis="centered", omega_basis="centered"`, and
  `growth_budget` defaults to `0.0` and is never passed, so Theorem B' excursions are off.

`p0` and `q` are both `softplus(.) > 0`, so the implied innovation
`q_eff_r = Lambda_r - e^{-2 d_rhobar_r} Lambda_{r-1} = q dt_r (1 - e^{-2x})/(2x)` is
non-negative for every real `x`. **The shipped kernel is PSD by construction in exact
arithmetic.** `_RHO_FLOOR = -20.0` is an overflow guard on a quantity that is already
non-negative; it never had to act as a validity guard.

### §10.2 G3 — measured, `cmd_mix/audit_psd.py`

Runtime pinned: `llapdiffusion` resolves to this checkout, `llaplace_df` HEAD
`180a7466a735af494ff05eddbc148e92845eabf0`, `laptrans.py`
`357d7e90…b002f8`, `chirp_gp.py` `42924c33…5b7befe`, `dist_heads.py` `5027b8bd…2b39ee80`.
Real `noaa_uk` windows (`T=168`, `K=32`, `d_z=16`, `anchor_nodes=128`), canonically sorted
grid; the cache's `t_rel` was **already sorted**, so no past run had a mis-ordered kernel on
this cell. Every intervention is applied to PARAMETERS and the whole forward pass is
recomputed, per ACTIONS §6.

| state | precision | min d(rhobar) | min q_eff | eig_min | material neg | fixed-jitter Cholesky | numerical rank |
|---|---|---|---|---|---|---|---|
| init | f64 | +7.50e-04 | +9.89e-01 | +2.28e-03 | 0/128 | 0 | 168/168 |
| perturbed x2 | f64 | +1.01e-04 | +6.35e-05 | +5.37e+02 | 0/128 | 0 | 168/168 |
| perturbed x32 | f64 | +1.00e-04 | −2.8e-14 | +1.71e+06 | 0/128 | 0 | 168/168 |
| **USHCN TRAINED s0** | f64 | +3.53e-04 | +5.11e-03 | +6.48e-04 | 0/256 | 0 | 12/12 |
| probe `q→0` + `theta×1000` | **f64** | +7.50e-04 | +3.50e-09 | **+8.16e-06** | **0/128** | **0** | **168/168** |
| probe `q→0` + `theta×1000` | **f32, TF32 off** | +7.50e-04 | −1.79e-07 | **−1.73e-01** | 0/128 | **8** | **40**/168 |
| probe `q→0` + `theta×1000` | **f32, TF32 on** | +7.50e-04 | −1.79e-07 | **−1.29e+01** | **126/128** | **128** | **53**/168 |

`min d(rhobar) > 0` in **every one of the 40 audited configurations**, at every perturbation
scale up to 32x. Assumption R is never violated. The worst float64 `q_eff` is −2.8e−14
against `Lambda ~ 1e4` — cancellation round-off, not a negative innovation.

### §10.3 What actually caused the crashes

**Rank collapse plus a relatively vanishing nugget, in float32.** As `q -> 0` the quadrature
term vanishes and `Lambda_r -> e^{-2 rhobar_r} p0`, so

    K(t,t') -> sum_k E_k p0_k e^{-rhobar_k(t)} e^{-rhobar_k(t')} cos(omegabar_k(t) - omegabar_k(t'))

which is a Gram matrix of `2K` cos/sin features and therefore has rank **at most 2K = 64** on
a `T = 168` grid. It is still PSD — it is a Gram matrix — but singular, and only the nugget
lifts it off zero. Inflating the modal part makes the same `SIGMA2_MIN = 1e-4` nugget
relatively negligible (measured `nugget/diag = 1.3e-06`). In float64 the combination still
factorises cleanly; in float32 the numerical rank drops to 40/168 and the factorisation
fails. **TF32 turns 8 failures into 128/128.**

This is exactly ACTIONS §6's precision case, and its decision rule applies: *"If float64
passes and float32 fails, treat that as a precision problem and use the validated precision
for covariance construction/factorization. It is not evidence for `p_mono`."*

It also explains the cell-dependence that the `p_mono` story never did: USHCN has `T = 12`
against `2K = 32`, so `T < 2K` and the rank-collapse mechanism **cannot** bite. noaa_uk has
`T = 168` against `2K = 64`. USHCN never crashed; noaa_uk did.

### §10.4 G3 decision

**Keep `p_exact`.** It passes the recurrence and PSD gates under the frozen implementation.
`p_mono` remains a separately named ablation and is NOT adopted. Required instead, as
frozen S4 policy:

1. Build and factorise the covariance in **float64**; **TF32 disabled** for every correctness
   gate and for the objective.
2. Record the §6 panel per epoch: `min d(rhobar)`, `min q_eff`, `min Lambda`, unjittered
   `eig_min` with the scale-aware tolerance, numerical rank, `nugget/diag`, and Cholesky
   status — so validity is asserted per arm, never assumed.
3. Treat any adaptive-jitter escalation, Cholesky failure, NaN/Inf or skipped non-finite
   step as a **failed fit** (ACTIONS §7), not a rescued one.

An open model question, deliberately NOT acted on here: the joint objective rewards driving
`q -> 0`, which is what walks the kernel into the degenerate corner. A floor on `q` would
remove the mechanism at its source, but that is a model change and needs a dated pre-S4
amendment. Float64 + TF32-off is a runtime setting and removes the observed failure without
one.

### §10.5 G0 — NOAA-owned v2 conformance

`cmd_mix/conformance_v2.py` imports the production NOAA mixture law and **no Toy model**.
All four (scorer x case) combinations pass; worst discrepancy **2.842e-13** absolute on
`comp_logpdf_T` against a tolerance of 3.33e-04, everything else at or below 1.7e-15.
Fixture NPZ / canonical-metadata-body / raw-JSON / config hashes all match the literals
pinned in ACTIONS §1. NOAA adapter `7ed9e1b1…cb385898`, checker `5073debb…f34075cb`.
Self-test: 44 golden-value tampers and 7 adapter tampers x 2 cases, **all detected**.

The self-test found a defect in itself before it found anything else: `cov_T_restricted` and
`z_S` come out of the npz non-C-contiguous, so `reshape(-1)[i] += eps` wrote into a copy and
the tamper was silently discarded. `.flat[i]` fixes it. Any checker tampering the same way
has the same hole.
