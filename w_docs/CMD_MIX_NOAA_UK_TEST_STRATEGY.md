# CMD-Mix(D) Validation Strategy for `noaa_uk`

## 1. Purpose and decision to be made

This is a real-data viability pilot to run only after, or independently in
parallel with, the Toy B correctness/mechanism work. Its purpose is to decide
whether a small explicit CMD-GP mixture merits a major paper and architecture
rewrite.

The pilot must answer:

1. Does production `CMD-Mix(D)` remain projectively consistent on real
   irregular multivariate histories and through the observation decoder?
2. Does $D>1$ improve held-out joint/path quality over `CMD-Mix(1)`?
3. Can $D\le4$ approach CMD-D's quality and calibration at materially lower
   end-to-end cost?
4. Is any gain robust, history-dependent, and not merely increased parameter
   count or a few favorable windows?

This pilot is not a new headline result. Do not rewrite the manuscript or
replace Option A/Option B until the decision gates in Section 13 have passed.

## 2. Parallel-work boundary

This task is intended to run in parallel with the Toy B experiment.

- Work in an isolated branch/worktree and a `noaa_uk`-specific output
  directory.
- Do not edit Toy B code, Toy B artifacts, or the manuscript.
- Do not overwrite existing `noaa_uk` checkpoints, figures, or tables.
- If shared CMD-Mix code is unavailable, use an experiment-local adapter that
  follows Section 4. Record changes for later consolidation rather than doing
  a repository-wide refactor during the pilot.
- Treat all existing test-set numbers, including the reported
  $\mathcal R=0.6256$, as hypothesis-generating exploratory evidence.

Before implementation, resolve and freeze exact paths/identifiers for:

- repository root and training/evaluation entry points;
- `noaa_uk` dataset revision and split manifests;
- production P-exact/P-mono CMD-GP implementation;
- VAE, encoder, summarizer, decoder, and imputer checkpoints/configs;
- legacy CMD-GP and CMD-D configs/checkpoints;
- same-encoder capacity controls;
- versioned output root.

Do not silently substitute a different dataset, checkpoint, or baseline when a
resource is missing. Record the unresolved item and stop that dependent stage.

Consume the versioned golden conformance fixture produced by the Toy B task.
The `noaa_uk` adapter must reproduce its expected weights, restricted component
laws, latent log densities, and samples from supplied labels/innovations before
comparative fitting. Save the fixture hash and adapter/config hash in every run
manifest.

## 3. Probabilistic object and shared semantics

For each observed history $H_n$, one vectorized network call must produce
weights $\pi_{nd}$ and $D$ history-conditioned CMD-GP component parameter sets:

\[
p(z_Q\mid H_n)=\sum_{d=1}^{D}\pi_{nd}
\mathcal N(z_Q;m_{nd}(Q),K_{nd}(Q,Q)).
\]

For observations, require the pointwise emission

\[
p(y_Q\mid z_Q,H_n)=\prod_{q\in Q}p(y_q\mid z_q,H_n).
\]

For joint sampling, draw one component label for the entire forecast path.

Use $T_{\mathrm{forecast}}$ for physical forecast duration, $n_q=|Q|$ for the
number of requested time-channel locations, and $n_{\mathrm{scalar}}$ for the
number of scored target scalars after masking. The earlier shorthand $h$ in
$O(DKh)$ corresponds to $n_q$ only in a single-channel fixed-grid setting.

Non-negotiable definitions:

- $D$ is the number of explicit learned components.
- $M$ is the number of CMD-D diffusion trajectories.
- Weights retain shape `(N, D)` throughout training and evaluation. Never
  replace them with dataset-average weights.
- Weights depend on history only, not future targets, query-set summaries, or
  target-derived features.
- Pole, noise, and covariance functions may vary with physical time but must
  not be recomputed differently when other queries are inserted.
- `CMD-Mix(1)` must reproduce the existing one-pass CMD-GP under identical
  architecture and training/scoring semantics.
- Exact mixture NLL uses per-history `logsumexp` over all $D$ components.
- Observation-space consistency is claimed only for a pointwise/separable
  decoder that passes the full tests below.

Unless the observation decoder/likelihood is invertible or otherwise supplies
a tractable normalized density including every required Jacobian, `exact joint
NLL` means exact **latent-space** mixture NLL. Label it accordingly. Use common
sample-based observation-space scores for the main CMD-Mix versus CMD-D
comparison when an exact observation-space likelihood is unavailable. Every
emission factor and its parameters must remain query-set independent.

## 4. Minimum implementation and interoperability interface

Retain or expose:

- logits and weights, shape `(N, D)`;
- query-set-independent component-function coefficients;
- component means and covariance/state-space representation;
- per-component joint log likelihoods and exact mixture log likelihood;
- global-component joint sampling;
- within-component and between-component moment contributions;
- network-evaluation count, sequential depth, wall time, and memory.

Save per-window weights, responsibilities, component mean summaries, and
metrics. Aggregate-only files are insufficient for diagnosing whether gains
come from a small subset of windows.

The Toy B and `noaa_uk` implementations must agree on the meanings of $D$,
$M$, global component sampling, `logsumexp` likelihood, and query-set
independence. Record any semantic deviation explicitly.

## 5. Leakage and split policy

### 5.1 Audit prior exposure

Before training, document:

- the exact chronological/entity split used by every existing result;
- which validation and test metrics have already been inspected;
- whether test results influenced the proposed finite-mixture architecture,
  $D$ values, objectives, or thresholds;
- preprocessing, VAE, summarizer, and calibration fits and the data used for
  each;
- encoder, decoder, imputer, and any learned feature/checkpoint fits and the
  data used for each;
- whether overlapping windows cross split boundaries.

The existing $\mathcal R=0.6256$ must not serve as a test-set model-selection gate. It
suggests that one Gaussian may be insufficient; it proves neither that a
finite mixture works nor that diffusion is necessary.

### 5.2 Development, outer gate, and confirmation

Use chronological or entity-safe nested partitions:

1. **Training partition:** parameter fitting only.
2. **Inner validation:** early stopping and ordinary hyperparameter selection.
3. **Outer gate validation:** comparison and selection of $D$; do not tune on
   it repeatedly.
4. **Untouched confirmation lockbox:** opened only after the model, $D$,
   margins, seeds, and analysis are frozen.

Partition at the highest defensible independent unit: station/site where the
scientific protocol permits it, otherwise non-overlapping chronological blocks
within station. A 70%/30% outer-gate/lockbox division of the available
validation material is a reasonable default, but independence matters more
than the exact fraction. For chronological blocks, add an embargo at least as
long as the maximum context span plus forecast horizon so overlapping windows
cannot share raw observations across the boundary.

Make the four manifests mutually exclusive. A clean default is to carve inner
validation from the end of the original training period, then divide the
original validation material into outer gate and lockbox after applying the
embargoes. `Fresh outer evaluation` means evaluation on a genuinely untouched
replacement partition, not rerunning a modified protocol on the same outer
gate.

If the original test split has already influenced architecture decisions,
label it development evidence and designate a fresh chronological lockbox or
an honest repeated outer split for confirmatory claims. If no fresh lockbox is
possible, report this work as a descriptive pilot and do not present it as
confirmatory evidence.

Fit normalization and preprocessing on the training partition only. Freeze
windowing, history length, the existing production forecast horizon, channel set,
missingness handling, first-lead-gap semantics, VAE, and history summarizer
before the $D$ sweep. Every model must receive identical paired histories,
queries, and targets.

The current draft is expected to use $T_{\mathrm{forecast}}=168$ time units for
`noaa_uk`; verify the value and units against the frozen production manifest.
If it is not the accepted duration, do not change to 168 merely for this pilot.
Treat 168 as a separately preregistered additional stratum instead.

## 6. Freeze a protocol manifest

Before comparative fitting, save a machine-readable manifest containing:

- dataset revision and integrity hashes;
- entity/site list and exact timestamp boundaries for every partition;
- preprocessing and windowing code revision;
- history and forecast horizons and irregular query protocol;
- $D\in\{1,2,4\}$ as the selectable family. Predeclare $D=8$ as a
  diagnostic-only inner-validation escalation; it is not eligible for the
  major-rewrite lockbox decision;
- CMD-D $M$ settings used as quality and matched-compute anchors;
- $K$, pole basis, P-exact/P-mono choice, variance construction, nugget, and
  decoder;
- loss, optimizer, learning-rate grid, trial counts, early stopping, and
  initialization;
- calibration procedure and calibration partition;
- all seeds and estimator sample counts;
- final-model rule: one validation-selected checkpoint, a deterministic refit,
  or an ensemble; charge full ensemble inference cost if applicable;
- primary metrics and practical superiority/equivalence margins;
- hardware, precision, batch size, warm-up, repetitions, and timing method;
- selection and stopping rules from Section 13.

Also freeze a numeric `decision_thresholds.yaml` containing:

- finite-mixture primary-score superiority over $D=1$;
- CMD-D variogram-score equivalence and energy-score noninferiority;
- marginal calibration noninferiority;
- normalized component-distance and weight/responsibility occupancy cutoffs;
- minimum fraction of independent station/time outer units improved;
- target interval precision or power requirement and minimum independent outer
  units;
- major-rewrite latency/memory thresholds.

Every qualitative word in the final gates (`material`, `calibrated`,
`distinct`, `noncollapsed`, `acceptable`, and `beneficial`) must map to one of
these frozen numerical predicates. Require the relevant one-sided 95%
confidence bound to clear superiority/noninferiority and timing gates, and the
full 90% interval to lie within equivalence margins. The major-rewrite timing
criterion is a lower confidence bound of at least fourfold speedup over the
strong CMD-D reference; twofold is feasibility only.

Any change after outer-gate results requires a dated amendment and a fresh
outer evaluation.

## 7. Phase I: production correctness gate

### 7.1 D=1 mapped-checkpoint regression and retraining parity

Verify that `CMD-Mix(1)` reproduces the existing CMD-GP on fixed `noaa_uk`
histories and queries:

- component parameters;
- latent means and covariance;
- observation predictions;
- joint and marginal likelihoods;
- deterministic metrics and seeded samples;
- parameter count and inference path.

For sampling checks, explicitly share component labels and base Gaussian noise
where possible. Do not require naive bitwise equality from the same top-level
seed if one implementation consumes an additional categorical RNG draw.

Do not interpret $D>1$ until unexplained D=1 differences are removed.

Exact equality applies only to the mapped-checkpoint implementation
regression. The inferential $D=1$ baseline must then be retrained under the new
split and paired protocol. If the old checkpoint saw any lockbox information,
it is regression/context only and is ineligible for predictive comparison.

### 7.2 Projectivity tests

For trained but not test-selected models with `D = 1, 2, 4`, run at least 100
histories spanning sites/channels, missingness levels, and first-lead gaps.
If $D=8$ is escalated or reported, run the complete suite for $D=8$ as well.
For each history:

1. Draw random $S\subset T$ at 25%, 50%, and 75% of the full query set.
2. Compare prediction directly on $S$ with the $S$-marginal of prediction
   on $T$.
3. Insert new queries before, between, and after existing queries. Every
   inserted future query must remain after the final history observation.
4. Test unsorted permutations and restore both covariance axes correctly.
5. Use fresh irregular grids, dense clusters, long gaps, and repeated/nearly
   repeated times with declared duplicate-noise semantics.
6. Test multiple channels and asynchronous channel-time queries.
7. Compare weights, component means, component covariance submatrices, mixture
   moments, and mixture log densities. For density, compare the analytically
   restricted $T$-law and direct $S$-law at the same fixed $z_S$.
8. Repeat through the full observation decoder. For a stochastic/nonlinear
   pointwise emission, compare emission parameters or coupled pushforward
   distributions at shared queries, not only decoded first two moments.

Run the actual production time-varying $\rho/\omega/q$ and P-exact/P-mono path. A
constant-pole or latent-only test is insufficient. Use a history-anchored time
origin and no requested-query-grid integration whose result changes under
insertion.

Include P-grid/requested-grid recurrence as a deliberate inconsistent positive
control. The suite must detect it.

Use a float64 reference with

\[
\|a-b\|\le \epsilon_{\mathrm{abs}}
+\epsilon_{\mathrm{rel}}\|b\|.
\]

Use $\epsilon_{\mathrm{abs}}=10^{-8}$ and
$\epsilon_{\mathrm{rel}}=10^{-6}$ initially unless the manifest justifies
alternatives. Use max norm for weights and scalar log density, scaled
Euclidean norm for means, and Frobenius norm for covariance blocks. Report
absolute and relative median, 95th percentile, and maximum errors. A
projectivity failure is a hard stop regardless of score.

### 7.3 Mixture and sampling checks

- Confirm exact per-history `logsumexp` likelihood.
- Compare analytic mixture moments with Monte Carlo moments.
- Confirm global component selection over complete paths.
- Confirm sampling frequencies match history-specific weights.
- Check covariance PSD/nugget semantics and numerical stability.
- Confirm no future targets or query-set summaries enter the gate.

## 8. Phase II: staged model comparison

### 8.1 Required arms

- Frozen/reproduced current CMD-GP checkpoint and configuration.
- Retrained CMD-GP baseline using every paired inferential seed.
- `CMD-Mix(1)` / CMD-GP.
- `CMD-Mix(2)`.
- `CMD-Mix(4)`.
- `CMD-Mix(8)` only if $D=4$ clears a predeclared **inner-validation**
  escalation rule. Keep it diagnostic/secondary and do not introduce it after
  seeing the outer gate.
- Strongest practical CMD-D configuration selected under the same nested
  protocol and without the new lockbox. A setting previously selected using
  the inspected test split is context only.
- Retrained CMD-D inferential baseline using the paired seed protocol.
- Enough predeclared CMD-D $M$/step settings to bracket each finalist mixture's
  measured latency; identify the closest point within 10% if one exists.
- Same-encoder single Gaussian/low-rank head used in the current comparison.
- Parameter/FLOP-matched single-component capacity control.

A faithful MOSES run is desirable for final positioning but need not block the
first viability decision. If the existing comparator is not a faithful tuned
implementation, label it an `affine separable-flow/MOSES-style proxy`.

Frozen legacy checkpoints provide reproduction context only. Do not use one
frozen checkpoint as the inferential comparator for a multi-seed mixture;
retrain the main CMD-GP and CMD-D baselines under the paired protocol.

Within each trained CMD-D seed, vary $M$ and the predeclared inference-step
settings at inference rather than retraining a different diffusion model for
every frontier point. Count every denoising step times $M$ and keep $M$
distinct from the common score-estimation ensemble size $J$.

### 8.2 Two fairness views

Report these as distinct comparisons:

1. **Native capacity:** keep $K$ per component fixed, allowing parameters and
   computation to grow with $D$.
2. **Parameter match:** widen the $D=1$ head or modes until trainable parameter
   count matches the finalist mixture within the frozen tolerance.
3. **Modal-budget match:** hold the total number of component modes fixed
   across $D$.
4. **Estimated-FLOP match:** compare configurations within the frozen FLOP
   tolerance.
5. **Measured-latency match:** use profiled configurations, including CMD-D
   frontier points, within 10% latency where possible.

A native-capacity gain alone does not establish that mixture structure, rather
than extra parameters, caused the improvement.

### 8.3 Staged computation

1. **Smoke:** one training seed and a reduced training subset. Debug only.
2. **Exploratory validation pilot:** three paired full-data training seeds for
   $D\in\{1,2,4\}$ and the CMD-D anchors.
3. **Rewrite-decision validation:** at least five paired full-data optimizer
   seeds if the three-seed pilot passes. Target ten for optimizer-robustness in
   any main-paper claim; these seeds do not replace independent site/time
   units.
4. **Confirmation:** evaluate exactly once on the untouched lockbox the
   selected $D$, retrained $D=1$, prespecified retrained CMD-D reference, and
   prespecified matched-capacity control. Exclude unselected $D>1$ candidates.

The five/ten seeds are complete optimizer fits nested within the independent
station/time outer units. Use development to establish a stable initialization
or frozen convergence rule rather than silently choosing the best lockbox
start. A major rewrite additionally requires the precision/power and minimum
independent-outer-unit target frozen in `decision_thresholds.yaml`.

## 9. Training and model-selection parity

- Train every finite-mixture $D$ with one frozen objective. The default is
  exact latent joint mixture NLL via `logsumexp`. If the production training
  path cannot expose a valid exact likelihood, freeze one common proper
  sample-based joint/path objective before fitting and use it for $D=1$ and
  every $D>1$; keep exact latent NLL as a labeled diagnostic. Do not choose the
  objective after observing comparative results.
- Keep history encoder, VAE, decoder, query semantics, $K$, basis,
  parameterization, variance floor, and nugget identical across native-capacity
  mixture arms.
- Give each arm equal hyperparameter trial counts and comparable optimization
  budgets.
- Select early stopping and ordinary hyperparameters on inner validation.
- Select $D$ once on outer-gate validation using the frozen primary criterion
  and tie-breaking rule.
- Fit any recalibration on inner validation only, freeze it before outer-gate
  evaluation, and report raw and recalibrated results separately. Raw
  predictions remain primary.
- Record all failed fits, restarts, collapse remedies, and numerical repairs.
- Do not use component entropy or fitted between-component variance as an
  operational gate unless its threshold and selection regret are evaluated in
  a separate frozen study.

## 10. Measurements

### 10.1 Primary predictive measurements

> **AMENDED 2026-08-25 — objective path 2 adopted.** The paragraphs below replace the
> earlier hierarchy, which selected $D$ on latent NLL. Two things forced the change.
> First, the training score is **not** an exact latent joint likelihood: it is joint over
> time but factorised over channels, while the sampled law has nonzero cross-channel
> covariance. Calling it "exact latent joint NLL" overstated it. Second, a latent-space
> criterion cannot be compared across arms whose latent parameterisations differ, and it
> is not the quantity anyone cares about. Selection moves to observation space.

- **Training / diagnostic score:** *latent time-joint/channel-factorized composite
  mixture NLL*, normalized per predicted scalar. Named for what it is. It trains every
  arm and is reported as a diagnostic. **It never selects $D$ and never carries a
  comparative claim.**
- **Selection score:** observation-space **variogram score of order 0.5**, computed over
  **both** temporal pairs $(t,t')$ within a forecast window **and cross-channel pairs**
  $(c,c')$ at equal and unequal times, with predeclared weights frozen against the hashed
  development scale. A temporal-only variogram cannot see the cross-channel structure the
  sampled law actually has — which is precisely the structure the training score omits.
- **Ordered safeguards:** unbiased energy score (the $m(m-1)$ U-statistic; the naive
  estimator is biased toward under-dispersion, the exact failure a mixture can hide),
  then calibration.

Endpoint hierarchy:

1. **Select one $D$** on the observation-space variogram score, by the Holm-adjusted
   superiority rule in `decision_thresholds.yaml`. Ties resolve to the **smallest** $D$.
2. **Apply the safeguards in order**: unbiased energy score non-inferiority, then
   calibration. An arm failing either cannot be selected whatever its variogram score.
3. **Gate equivalence to CMD-D** on the same variogram score, then report the
   matched-quality speedup against the major-rewrite criterion.

Every score sample passes through the **complete stochastic decoder/emission** — VAE
decode plus emission noise — with common random numbers across arms. Latent-space samples
are not admissible for any reported score.

A full state-space likelihood matching the sampled latent law is **deferred**. It is
revisited only if CMD-Mix clears the major-rewrite criterion; implementing it before then
would be a large rewrite justified by nothing.

Do not compare a mixture NLL with a diffusion KDE surrogate as though they were the same
quantity — that hazard is unchanged, and is now moot for selection, since selection no
longer uses either.

Define all multivariate metrics in one frozen, train-fitted standardized space
or with preregistered channel weights. Freeze target-mask handling and the
normalization used for `per predicted scalar`. Use a fixed common evaluation
ensemble size $J$ for energy/variogram estimation and keep $J$ distinct from
CMD-D's internal component count $M$.

Use the energy-score U-statistic

\[
\widehat{\mathrm{ES}}
=\frac1J\sum_{j=1}^J\|x_j-y\|
-\frac{1}{2J(J-1)}\sum_{j\ne k}\|x_j-x_k\|,
\]

with the same $J\ge2$ and common random numbers for all laws. Freeze the exact
variogram estimator and all weights in the protocol rather than reporting
multiple estimators as interchangeable primary results.

### 10.2 Calibration and marginal measurements

- CRPS and marginal NLL.
- PIT/rank calibration error.
- Coverage at predeclared levels with width at matched coverage.
- Simultaneous path-band coverage.
- Calibration before and after any inner-validation-fitted recalibration.

Use common random numbers and the same unbiased/naive score estimators for
analytic and sampled arms. Report Monte Carlo standard error.

### 10.3 Dependence and path measurements

- Physical-lag covariance fit.
- Whitened/standardized residual autocorrelation.
- Cross-channel lag covariance.
- Running maxima, cumulative sums, threshold exceedance, and first-crossing
  distributions where meaningful.
- Score differences stratified by horizon position, gap CV, maximum gap,
  missingness, site/channel, and forecast difficulty.

The mixture claim requires a joint/path improvement, not only better CRPS or
coverage.

### 10.4 Real-data component-use diagnostics

There are no oracle branch labels on `noaa_uk`. Diagnose rather than narrate
components:

- per-history weight and responsibility occupancy;
- weight entropy and effective occupied count;
- fraction of histories dominated by each component;
- pairwise component-function distance over the complete
  `history x time x channel` tensor, normalized by predictive scale;
- between-component and within-component covariance shares;
- stability across seeds after diagnostic-only component matching;
- likelihood contribution of each component;
- correlation of component use with predeclared history covariates.

Add a history-weight ablation: replace each forecast's fitted weights with the
training/inner-validation mean weights while leaving all component functions
fixed. Evaluate it under the same outer protocol. If quality is unchanged
within the frozen margin or between-history weight variation is below its
frozen threshold, do not claim empirical value from history-conditioned
weights; describe only a finite process mixture whose component functions are
history-conditioned.

Posterior responsibilities use the realized future and are diagnostics only.
Never present them as information available to the forecast-time gate.

Before fitting, freeze numerical utilization/distinctness diagnostics. A
reasonable starting diagnostic is that at least two components each receive
at least 5% average validation responsibility and at least 10% of histories
assign at least 10% weight to a second measurably distinct component. Report
the full distributions and predeclare the component-distance tolerance. These
diagnostics define `systematic collapse`; they do not turn fitted
$\mathcal R$ into an operational model-selection gate.

Do not infer meteorological regimes from component indices without independent
evidence. Do not average component means over histories before measuring their
distance; cancellation can create a false collapse diagnosis.

### 10.5 Cost measurements

Report:

- parameter count and estimated FLOPs;
- training time and failed-run rate;
- synchronized inference latency and throughput;
- peak accelerator memory;
- batch size, hardware, precision, $T_{\mathrm{forecast}}$, $n_q$,
  $n_{\mathrm{scalar}}$, $K$, $D$, and $M$;
- network evaluations and sequential depth;
- separate cost of mean/marginals, joint likelihood, one path draw, and the
  common multi-sample scoring workload;
- decoder and mixture-aggregation overhead.

For CMD-D, count every denoiser step times $M$. For CMD-Mix likelihood, count
all $D$ component likelihoods. A sampled CMD-Mix path may select one
component after the weights are produced.

Use measured quality-at-matched-compute as the main efficiency result. Scope
$O(DK n_q)$ to vectorized component mean/marginal or Markov evaluation;
disclose full-kernel $n_q^2$ work and state-space likelihood costs separately. `One
network evaluation` means one batched invocation with work and memory growing
in $D$.

## 11. Statistical analysis

- Pair histories and seeds across methods.
- Use a paired hierarchical analysis or cluster bootstrap over independent
  station/time blocks, crossed with or nested inside training seed, as the
  primary 95% interval. Do not treat overlapping windows or query values as
  independent replicates.
- Report paired seed-level means and seed-only variability separately rather
  than treating five optimizer seeds as the entire generalization population.
- Freeze practical superiority and equivalence margins before opening the
  outer gate or lockbox.
- Use equivalence testing before saying the selected finite mixture matches
  CMD-D. Nonsignificance is not equivalence.
- Apply the predeclared multiple-comparison correction across the $D>1$
  versus $D=1$ family.
- Report heterogeneity and the fraction of histories improved, not only the
  average effect.
- Report Monte Carlo and optimization uncertainty separately where feasible.

The outer-gate $D$ sweep is model selection. On the lockbox, evaluate only the
single selected $D>1$ candidate and the prespecified comparison set:
retrained $D=1$, retrained CMD-D, and the matched-capacity control. Each is
evaluated exactly once; no unselected mixture candidate may be added.

## 12. Required artifacts

Produce a versioned package containing:

1. data/split exposure audit;
2. shared conformance-fixture hash and adapter pass report;
3. frozen protocol manifest and configurations;
4. code revision and environment/hardware record;
5. D=1 mapped regression and projectivity reports with raw maxima;
6. selected checkpoints, per-run learning curves, and complete validation,
   checkpoint, start, and final-model-selection records;
7. per-window, per-component, per-outer-unit, and per-seed machine-readable
   results;
8. failed-run and protocol-amendment log;
9. component-use/collapse diagnostics;
10. paired hierarchical statistical and equivalence analysis;
11. parameter/FLOP/runtime/memory/sequential-depth table;
12. quality-versus-$D$, quality-versus-measured-cost, and calibration plots;
13. stratified gain plots with uncertainty;
14. a decision report in the format below.

Do not overwrite or silently update current manuscript figures. The output of
this task is evidence for a later decision.

## 13. Decision gates

### Gate A: correctness

Pass only if:

- `CMD-Mix(1)` reproduces CMD-GP;
- production time-varying component functions and the full decoder pass the
  projectivity/permutation tests;
- history-specific weights and exact mixture algebra pass;
- the intentional inconsistent control is detected.

Failure is a hard stop.

### Gate B: real-data value beyond one component

Pass only if the validation-selected $D\le4$:

- materially improves over $D=1$ on the frozen primary joint/path metric;
- improves or remains within a frozen noninferiority margin on the other joint
  score and calibration safeguards;
- clears the frozen minimum fraction of independent station/time outer units
  improved and is not driven by a few histories, sites, or channels;
- uses multiple distinct components without systematic collapse;
- retains a material advantage against the matched-parameter/FLOP
  single-component control.

### Gate C: replacement value relative to CMD-D

Pass only if the selected $D\le4$:

- is equivalent to CMD-D on frozen quality and calibration margins, or has a
  predeclared acceptable quality tradeoff;
- has a matched-quality end-to-end latency speedup whose one-sided 95% lower
  confidence bound is at least fourfold, together with lower sequential depth;
- retains the cost advantage after decoder, component likelihood, memory, and
  scoring overhead are included.

An advantage between twofold and fourfold is feasibility only and should
normally be a secondary-model outcome unless predictive quality is better.

### Gate D: evidence integrity

Pass only if:

- $D$ and hyperparameters were selected without the confirmation lockbox;
- preprocessing, normalization, imputation, calibration, VAE, encoder,
  summarizer, decoder, and every other learned upstream module used no lockbox
  information;
- CMD-D and capacity-control configurations were selected under the same
  nested protocol rather than inherited from inspected test results;
- all prior test exposure and protocol changes are disclosed;
- only the selected mixture and the prespecified $D=1$, CMD-D, and
  matched-capacity comparators were evaluated once on the lockbox.

If no untouched lockbox exists, this gate cannot support a confirmatory claim;
report a descriptive pilot.

## 14. Rewrite decision and agent handoff

The final report must begin with exactly one of:

- **STOP — correctness failure.**
- **STOP — no real-data mixture benefit.**
- **KEEP AS APPENDIX/SECONDARY MODEL.** Benefit exists but requires large
  $D$, is unstable, is capacity-confounded, or lacks a material cost gain.
- **PROCEED TO MAJOR REWRITE.** A projective $D\le4$ mixture gives stable
  real-data joint/path benefit and matches CMD-D within frozen margins at
  materially lower cost.

Then report:

1. selected $D$ and its frozen selection rule;
2. split/exposure status and whether conclusions are exploratory or
   confirmatory;
3. primary paired effects with 95% intervals;
4. equivalence/noninferiority results;
5. maximum projectivity discrepancies;
6. component-use and collapse results;
7. matched-capacity comparison;
8. measured cost comparison;
9. all protocol deviations;
10. exact artifact paths.

The paper should be rewritten around CMD-Mix only if both this real-data gate
and the Toy B coherent-branch gate pass. A large between-component variance
ratio alone is not sufficient.

## 15. Permitted claim after success

The strongest defensible pre-rewrite conclusion is:

> On `noaa_uk`, a small explicit history-conditioned CMD-GP mixture improves
> the one-component joint/path forecast and approaches the selected CMD-D
> reference at materially lower measured inference cost, while preserving the
> tested projective-consistency conditions.

Do not generalize this to all datasets, universal multimodality, or complete
replacement of diffusion until the architecture is frozen and replicated on
additional real datasets.
