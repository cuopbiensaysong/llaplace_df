# Recommended actions for the `noaa_uk` CMD-Mix agent

## Status and decision

Pause new comparative fitting. The existing NOAA/USHCN runs remain useful as
development evidence for plumbing, memory, timing, and failure diagnosis, but
they do **not** count toward S4, model selection, or the confirmation lockbox.

Keep `p_exact` as the primary parameterization during the audit below. Do not
switch to `p_mono` on the basis of the previous rho-perturbation experiment.
`p_mono` remains a separately named ablation unless a frozen, internally
consistent `p_exact` run fails the production validity gate.

The governing experiment plan remains
[`CMD_MIX_NOAA_UK_TEST_STRATEGY.md`](./CMD_MIX_NOAA_UK_TEST_STRATEGY.md).
This document defines the immediate work required to unlock its S4 stage.

## 1. Pin and ingest golden fixture v2

Use fixture **v2**, not v1. Transfer the `.npz`, `.json`, and fixture README to
the NOAA machine without modifying them.

Source artifacts:

- `/vol/dl-nguyenb5-solar/users/cuopbiensaysong/cmd_toys/runs/cmdmix_toyB_v1/fixtures/golden_cmdmix_v2.npz`
- `/vol/dl-nguyenb5-solar/users/cuopbiensaysong/cmd_toys/runs/cmdmix_toyB_v1/fixtures/golden_cmdmix_v2.json`
- `/vol/dl-nguyenb5-solar/users/cuopbiensaysong/cmd_toys/runs/cmdmix_toyB_v1/fixtures/README.md`
- `/vol/dl-nguyenb5-solar/users/cuopbiensaysong/cmd_toys/runs/cmdmix_toyB_v1/check_fixture_v2.py` as a reference specification only

Expected identity:

| Item | Expected value |
|---|---|
| Schema | `2` |
| Fixture NPZ SHA-256 | `803e787a52fb02221b253acdecf8bf1db5c146e3510ee3b1ca9cdef75160155b` |
| Canonical metadata-body hash | `ad3b65b0dce21c3015d1de1fc3be1b42a2596016d149e23d169d6a0b5c345a8f` |
| Raw JSON-file SHA-256 | `bf2af25a74dd8a913756360975e7f002269b01b9d3df86f68a9f0eb40c677d1a` |
| Configuration hash | `2502f0a92f77b4b48de57624712d2a9104c15c53d4b1ecd84868ac0418189a00` |
| Canonical-core hash | `8b47bbe63cdf9decadc773f525faad9ac877b2b6067bda81b0369b6ef721505e` |
| Toy implementation hash | `c018f4e90d6687e17049bf0adc59faeedde2d3268fbb1913a5c490e24ad57c92` |

Pin the NPZ hash, canonical metadata hash, and raw JSON hash independently in
the NOAA frozen manifest. Do not rely only on expected hashes read from the
adjacent JSON.

### Important checker limitation

Do **not** claim NOAA conformance by merely running:

```text
python check_fixture_v2.py --foreign-adapter
```

The current Toy checker still instantiates Toy `CMDMix`; that flag only changes
how Toy/foreign code-hash differences are classified. It never invokes the
NOAA adapter.

## 2. Implement a real NOAA-owned v2 conformance bridge

Create a checker owned by the NOAA experiment that imports the production NOAA
mixture-law code and imports **no** Toy model implementation. Test the shared
law at the explicit component-law boundary:

1. Load fixture logits, component means, component covariances, targets,
   subset indices, global labels, innovations, and stored Cholesky factors.
2. Preserve the history dimension. Never average or broadcast weights across
   histories.
3. Convert fixture component means from `[N,D,T]` to `[N,D,T,1]` where the
   production interface requires a channel dimension.
4. Broadcast fixture covariance from `[D,T,T]` to the production component-law
   shape without changing its numerical values.
5. Run logits through the NOAA softmax/gating helper and reproduce weights.
6. Score each component using the same Gaussian component scorer used by the
   NOAA training path, in float64, using fixture jitter `1e-10` and a lower
   Cholesky convention.
7. Refactor weighted mixture `logsumexp` into one production helper so both
   training and the conformance checker call the identical implementation.
8. Restrict means and **both** covariance axes from `T` to `S`.
9. Compute full mixture moments, including

   \[
   \operatorname{Cov}(Z)=\sum_d\pi_d
   \left[K_d+(\mu_d-\bar\mu)(\mu_d-\bar\mu)^\top\right].
   \]

10. Add a deterministic conformance sampler accepting the supplied single
    global component label per path and supplied Gaussian innovations.
11. Check the forbidden control
    `sum_d pi_d * log p_d`; it must differ from the correct weighted
    `logsumexp` density by the pinned amount.
12. Run every check for both `canonical` and `semantic`. The semantic case has
    strongly history-varying weights and is required to detect pooled gates.

Use the fixture's combined criterion and named norm for every quantity:

\[
\|a-b\|\le 10^{-8}+10^{-6}\|b\|.
\]

Save a machine-readable report containing all absolute/relative median, p95,
and maximum discrepancies; fixture/config hashes; NOAA adapter/checker hashes;
and a single all-pass verdict. Add a tamper test proving the NOAA checker fails
when one golden value is changed.

This fixture validates mixture-law semantics. Its Toy-specific 23-element
`theta` must **not** be loaded into the production NOAA network. Production
CMD-GP functions, `p_exact`, projectivity, and the Markov sampler are covered
by the separate native gate below.

**Hard stop G0:** no S4 fit may start until both v2 cases pass with no numeric
failure and all fixture/config integrity hashes match.

## 3. Quarantine and document the current campaign

Before changing code:

- Snapshot current code, logs, JSON reports, environment, and hashes.
- Mark every existing NOAA and USHCN artifact `DEVELOPMENT_ONLY`.
- Preserve all crashes, skipped-step counts, adaptive-jitter behavior, and
  inspected comparisons in the exposure/amendment log.
- Do not include any current seed, checkpoint, complete-case comparison,
  p-value, or metric in S4 or lockbox counts.
- Retain old timing and memory measurements only as approximate resource
  planning data; remeasure final cost under frozen code.

This quarantine is necessary because the old runs predate v2, used multiple
joint-NLL code revisions, did not save checkpoints, enabled TF32, silently
rescued some covariance blocks, skipped nonfinite steps, and suffered
D-dependent attrition.

## 4. Freeze S3 before fitting

Create the missing four-way split/exposure manifest and
`decision_thresholds.yaml` required by the strategy. Freeze and hash:

- dataset/cache revision, sites, time boundaries, embargoes, and every split;
- normalization statistics and all preprocessing/windowing code;
- VAE, encoder, summarizer, decoder, imputer, and every upstream checkpoint;
- `D={1,2,4}`, with `D=8` diagnostic-only;
- `K`, component basis, `rho_basis`, `omega_basis`, `growth_budget`, anchor
  count, nugget, variance construction, and parameterization;
- training objective, score normalization, optimizer, LR grid, starts,
  patience, retry/failure rule, and checkpoint-selection rule;
- native-capacity, genuine parameter-matched, modal-budget-matched, and CMD-D
  comparison arms;
- hardware, precision, TF32 policy, batch size, timing method, and estimator
  sample counts;
- all practical superiority, equivalence, calibration, component-use, and
  cost thresholds.

Every run manifest must carry hashes for the fixture, metadata, data/splits,
all model/core/adapter/likelihood/driver files, complete config, upstream
checkpoints, and environment. Use unique immutable output directories. Any
code change during an S4 batch invalidates that batch.

**Hard stop G1:** do not fit until the split manifest, thresholds, hashes, and
prior-exposure log exist and validate.

## 5. Repair and expand production Gate A

### 5.1 Query ordering

`cross_time_kernel` determines chronological order by tensor index. Therefore:

- Canonically sort queries at the public adapter boundary.
- Apply the identical permutation to targets, masks, and channel indices.
- Assert chronological order inside the component/kernel implementation.
- Restore requested order for means, samples, and **both** covariance axes.

Test unsorted permutations on covariance, component density, mixture density,
sampling, and decoder outputs—not only means. Also test insertions before,
between, and after existing future queries.

### 5.2 D=1 and projectivity

- Run mapped-checkpoint `CMD-Mix(1)` versus CMD-GP regression for component
  parameters, latent means/covariances, decoder outputs, likelihoods, samples,
  parameter count, and inference path.
- Run `D={1,2,4}` on at least 100 representative real histories with NOAA's
  actual `T=168`, channels, masks, long gaps, irregular grids, near-duplicates,
  and asynchronous channel-time queries.
- Compare direct `S` laws with analytically restricted `T` laws for weights,
  component means, both covariance axes, mixture moments, and density at the
  same fixed target.
- Verify one global component label per whole path and history-specific label
  frequencies.
- Include the requested-grid recurrence as an intentionally inconsistent
  positive control and require the test suite to reject it.
- Exercise the full pointwise decoder/emission.

Run both a float64 reference and the proposed production precision. Report raw
maximum discrepancies under the combined tolerance.

**Hard stop G2:** any unexplained D=1, projectivity, permutation, decoder, or
positive-control failure stops fitting.

## 6. Audit covariance validity before choosing `p_exact` or `p_mono`

Begin with the uploaded constrained `p_exact`, centered bases, and
`growth_budget=0`. Verify the exact runtime import paths and hashes.

For every audited component and sorted grid, record:

- finite-status of all outputs;
- minimum instantaneous `rho`;
- minimum `diff(rho_bar)`;
- minimum `Lambda`;
- minimum implied innovation, including the origin transition,

  \[
  q^{\mathrm{eff}}_r=\Lambda_r-
  e^{-2(\bar\rho_r-\bar\rho_{r-1})}\Lambda_{r-1};
  \]

- covariance symmetry error and diagonal range;
- unjittered minimum/maximum eigenvalues and scale-aware PSD tolerance;
- condition estimate and fixed-jitter Cholesky status;
- agreement between the analytic/materialized covariance and the native
  Markov sampler covariance.

Recompute identical saved batches in:

1. float64;
2. float32 with TF32 disabled;
3. float32 with TF32 enabled, as a diagnostic only.

Disable TF32 for correctness gates. If float64 passes and float32 fails, treat
that as a precision problem and use the validated precision for covariance
construction/factorization. It is not evidence for `p_mono`.

Do not perturb `rho_bar` while retaining stale `Lambda`. Any causal
intervention must recompute `Lambda` from the same modified dynamics and then
re-evaluate `q_eff` and the covariance.

Decision rule:

- Keep `p_exact` if the frozen implementation passes recurrence and PSD gates.
- If it fails, first identify a runtime/config mismatch or implementation bug.
- Move to `p_mono` only by a dated pre-S4 amendment after a valid development
  comparison. If selected, rerun **all** D arms and comparators under
  `p_mono`; never pool parameterizations.

**Hard stop G3:** no S4 run may have an unexplained negative implied
innovation, material negative eigenvalue, nonfinite covariance, or failed
fixed-policy Cholesky.

## 7. Replace silent numerical rescue with an eligibility policy

- Save the selected checkpoint and the first failing batch/covariance tensors.
- Return and aggregate jitter/escalation diagnostics per component and batch;
  do not use a module-global last-call counter.
- During development, a rescue may be used only to capture diagnostics and
  must mark the fit ineligible.
- During S4, any adaptive jitter escalation, Cholesky failure, NaN/Inf, or
  skipped nonfinite step makes that arm/seed a failed fit.
- Never compare a successful complete-case subset created by differential
  failures. Apply only one predeclared, identical retry rule to every arm.
- Audit a fixed validation panel each epoch and audit every evaluation block
  at checkpoint selection.

## 8. Fix objective naming and normalization

The current score is time-joint but channel-factorized even though the sampled
law has cross-channel covariance. Call it
`latent time-joint/channel-factorized composite NLL`, not an exact full latent
joint likelihood.

For each window, first calculate raw component NLLs and then mix them:

\[
L_n=-\log\sum_d \exp(\log\pi_{nd}-\ell_{nd}).
\]

Only after `logsumexp`, normalize the aggregate:

\[
L_{\mathrm{scalar}}=
\frac{\sum_n L_n}{\sum_n n_{\mathrm{scalar},n}}.
\]

Do not normalize component NLLs before `logsumexp`, use uniform weights, or
average weights across histories. Save raw per-window NLL, scalar count,
component NLLs, weights, responsibilities, and normalized aggregate.

Before S4, choose and freeze one of these honest paths:

1. implement a tractable likelihood matching the full sampled latent law; or
2. retain the composite NLL as a labeled training/diagnostic score and use one
   common proper path score for scientific selection/comparison.

Freeze energy/variogram estimator definitions, standardized space, masks,
channel/lag weights, common ensemble size, common random numbers, and Monte
Carlo error reporting.

## 9. Save complete artifacts

For every trial and seed, save:

- initialization, post-warm-up, best-inner-validation, and selected checkpoints;
- optimizer, scheduler/scaler, and RNG states;
- complete learning curves and failure/retry records;
- per-window and per-independent-outer-unit metrics;
- weights, responsibilities, occupancy, entropy/effective count, component
  function distances, and within/between variance shares;
- history-mean-weight ablation outputs;
- raw numerical validity and projectivity reports;
- synchronized cost measurements and peak memory;
- a manifest linking every artifact to all frozen hashes.

The primary shared-encoder mixture is **not** parameter matched: it still adds
component heads as D grows. Report it as shared-encoder/native-capacity and add
a genuinely parameter-matched D=1 control, as required by the strategy.

Scope `O(D K n_q)` only to vectorized component mean/marginal or Markov
evaluation. Dense `mixture_joint` materializes time-time kernels and performs D
component factorizations; report that cost separately.

## 10. Fresh S4 staging

Begin only after G0-G3 and the protocol/objective freeze pass:

1. **Smoke:** one fresh paired reduced-data seed for `D={1,2,4}`, matched
   controls, and CMD-D anchors. Require zero skipped steps and zero adaptive
   rescues; save all checkpoints and replay the selected covariance in
   float64.
2. **Three-seed pilot:** three fresh paired full-data seeds for every frozen
   arm, with equal LR-trial counts, optimizer budgets, data, batch size,
   precision, masks, and scoring.
3. **Expansion:** proceed to at least five paired seeds, targeting ten, only if
   the predeclared stability and predictive gates pass.
4. **Outer selection:** select one `D<=4` under the frozen hierarchical
   analysis and multiplicity rule. Keep D=8 diagnostic-only.
5. **Lockbox once:** evaluate only the selected mixture, freshly retrained D=1,
   prespecified CMD-D reference, and matched-capacity control.

No current run contributes a seed or checkpoint to these counts.

## 11. Required handoff report before S4

Return one concise readiness report containing:

- `G0 PASS/FAIL`: v2 hashes, both-case numeric maxima, checker/adapter hashes;
- `G1 PASS/FAIL`: split, threshold, exposure, data/upstream/code manifests;
- `G2 PASS/FAIL`: D=1 parity and maximum production projectivity errors;
- `G3 PASS/FAIL`: recurrence/PSD/precision results and frozen parameterization;
- objective name and exact per-scalar formula;
- numerical failure policy;
- frozen arms, seeds, budgets, checkpoints, and output root;
- explicit statement that prior results are development-only.

Do not launch S4 unless all four lines say `PASS`.

## 12. Small remaining Toy-side task

You can create a communication channel with Toy agent through a file : /vol/dl-nguyenb5-solar/users/cuopbiensaysong/cmd_toys/communication.md 

No additional Toy fitting or scientific analysis is needed. Ask the Toy agent
for one handoff patch:

1. expose a real external-adapter callback/module contract in
   `check_fixture_v2.py`, or provide a small standalone contract that the NOAA
   checker can implement;
2. add a self-test that actually invokes a fake foreign adapter and detects a
   changed output;
3. directly check the stored arrays currently used only indirectly, including
   `comp_logpdf_S`, `mix_logpdf_T_interface`, and
   `mix_logpdf_S_from_T`;
4. publish checker/maker/self-test hashes and clarify the canonical metadata
   hash versus the raw JSON SHA;
5. restore released fixture permissions to read-only (`0444`).

These changes should not alter the frozen v2 NPZ values. If fixture bytes or
semantic values change, issue a new version and update the NOAA manifest before
conformance.
