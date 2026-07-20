# cmd_plan_v2 — Implementation Status & Compliance Audit

*Audited 2026-07-05 against `cmd_plan_v2.md` and `chirp_modal_method_update.md`, on the
combined branch (all phases). Test suite at audit time: **290 passing**. Operator
commands for every runnable item: `w_docs/CMD_RUNBOOK.md`.*

**Verdict: the core matches the method and the plan — every theorem-bearing code path
is implemented as specified and test-backed — with 4 deliberate (documented) deviations
and 5 genuine gaps listed below.**

---

## 1. Verified matches (plan/method item → evidence in code/tests)

| Plan / method item | Evidence |
|---|---|
| §1 symmetric architecture: no-MLP arms return `s·y_time` with `\|s\| ≤ 1` clamp; +MLP arms use LLapDiff's original head verbatim | `lapformer.py` forward; `test_no_head_path_clamps_skip_scale`, `test_output_head_matrix_builds_all_four_cells` |
| §1 parity defaults: 25 CRPS samples, DDIM 64 steps, η = 0, EMA 0.999, guidance/threshold loggable | `configs/config.py` (`NUM_EVAL_SAMPLES=25`, `GEN_STEPS=64`, `GEN_ETA=0.0`, `EMA_DECAY=0.999`); eval path `_sampling_kwargs` reads η default 0.0 |
| G1 Finding-1 corrected Option A + the three gate tests (model-level contraction of the returned tensor, envelope decay / no LayerNorm floor, loss-scale O(1)) | `test_full_model_contraction_bound`, `test_chirp_output_scale_o1_at_init`, `test_output_scale_preserved_for_chirp` |
| G1 Finding-2 window-scaled basis + wiggle-band + ω_max = π per native step | `test_chirp_is_nondegenerate_at_native_horizon`, `test_omega_stays_below_omega_max` |
| G3: all four 2×2 cells buildable (`--modal-type` × `--output-head`), seedable (`--seed`), collision-free routing (`modal-chirp/`, `head-<mode>/`, `seed-<n>/`) | routing + matrix tests; 4-cell one-epoch smoke passed |
| Eq. (3) chirp synthesis; Eq. (4) \|s\|-carrying bound; strict LTI generalization | `chirp_basis_matrix`, equivalence/contraction tests |
| Eq. (4′a)/(4′) Theorem B′: `h_k` nondecreasing, `γ_k = c_g[σ(g_k(t̃)) − σ(g_k(0))]` (exactly the doc's recipe, incl. harmless γ<0), bounds `ρ̄ ≥ ρ_min t̃ − c_g` and `e^{c_g}` synthesis bound, closed-form γ′ | `_growth_terms`; `tests/test_growth_budget.py` |
| Eq. (6)/(7) Theorem C (isotropic): `p⁰_k, q_k` heads, `s_k(t̃) = e^{−2ρ̄}p⁰ + v_k`, diagonal readout `Var(z_d)=Σ_k s_k (c²+b²)_kd` scaled by `s²`, Gaussian NLL, PIT/reliability/NLL metrics | `uq_params`, `modal_variance`, `return_variance`, `uq_metrics.py`; `tests/test_chirp_uq.py` (constant-ρ Lyapunov closed form reproduced to float precision) |
| Eq. (8) implied event-domain multipliers (Re/Im match the doc term-for-term) | `run_t1_pole_invariance._eq8_multiplier` |
| §4.4 anchoring remark: t̃ recentred to the earliest query; `generate` rejects non-monotone `dt` | `relative_time`, `generate` validation |
| §4.6: poles predicted once per denoising step; x0 loss (optionally NLL) supported | forward structure; `DIFF_LOSS_MODE` |
| H2 (i)(ii)(iii): 5 ground-truth pole tasks + freq-shift as regime switch; recovered-vs-truth pole overlay + best-mode RMSE; Prop-A.1 figure (companion err ≈ 4.6 vs normal form ≈ 2e-11) | `llapdiff-synthetic-chirp`, `plot_companion_vs_normal_form`; e2e smoke passed |
| U1 sweep (w × steps on **val**, clip-rate logging) / U3 three arms (sampled-UQ, analytic-UQ ddim-mean, one-shot `max_only`) | `llapdiff-u1-sweep`, `llapdiff-uq-eval --mean-source {oneshot,ddim}`, `TRAIN_T_SAMPLER="max_only"` |
| T1 tool (trajectory distance vs baseline + gap moments + Eq.-8 multipliers), T2 knob (`CHIRP_GROWTH_BUDGET`), T3 (existing imputation eval), T4 timing | `llapdiff-t1-poles`, `llapdiff-t4-timing`; smokes passed |
| Phase-4 parameterizations: P-exact / P-mono / P-grid behind `CHIRP_PARAMETERIZATION`, near-LTI init, ω-capped, compose with growth/UQ heads, checkpoint metadata | `tests/test_chirp_parameterizations.py` (14 tests) |

Beyond the plan (correctness additions made along the way): the ε-init fix for the
squared-coefficient **stationary trap** (all pre-2026-07-05 chirp checkpoints had frozen
constant poles — see the red box in `CMD_RUNBOOK.md` §0, audit one-liner included), the
NLL warm-start path (`_load_diff_init_state`), and the `--seed`/`--output-head` CLI +
routing infrastructure.

---

## 2. Deliberate deviations (documented; keep or revisit consciously)

1. **Finding-2 time constant `L`** — implemented as the *run's* horizon (`CHIRP_TIME_SCALE=None`
   → `config.PRED`), not the plan's "per-dataset max horizon in train split". Equivalent for
   single-horizon runs; differs when one command trains multiple horizons (each gets its own L).
   Pin a number in `config.py` if cross-horizon comparability of the function class matters.
2. **`q_k` is constant in time** — the doc allows `q_k(t̃)`; the constant case is Theorem C's
   sanity case and keeps the head minimal. Extension point exists (same basis machinery).
3. **`v_k` "closed-form under P-exact" (plan §U2 wording) is mathematically impossible** — the
   integrand `e^{2ρ̄(s)}q(s)` is exp-of-trig-polynomial (no elementary antiderivative).
   Implemented as an overflow-free exponential-integrator quadrature, exact for
   piecewise-constant poles; the method doc §4.5 has been corrected to say so.
4. **Variant details**: P-grid's ω uses `ω_max·sigmoid` (Nyquist by construction) instead of
   softplus; P-mono's instantaneous poles are weighted sigmoids (monotone-integral family)
   rather than literally "the integral of a softplus head". All stated properties hold
   (positivity, monotone integrals, no solver, near-LTI init, no stationary trap).

---

## 3. Genuine gaps (plan items NOT yet implemented), ranked by paper impact

1. ~~**H2 irregular sampling**~~ — **CLOSED 2026-07-05.** True Gamma renewal-gap
   sampling added to the generator (`_sample_gaps` + gap-aware discretization
   `phase = φ₀ + Σ2πf·Δ`, `envelope = e^{−Σρ·Δ}`; the regular grid is the unit-gap
   special case — bit-compatible caches, no RNG consumed). `llapdiff-synthetic-chirp`
   now defaults to `--gap-distribution gamma --gap-mean 1.0 --gap-shape 4.0`
   (`Var(Δ) = mean²/shape`, sweep `--gap-shape` for regimes); grid drawn once per
   cache and shared across entities (joint-collate requirement); realized gap moments
   in cache meta; gap regime tagged in cache path/rows/summary; `pole_truth/*.npz`
   stores sample `times`. Verified: 4 new unit tests (incl. exact closed-form check of
   the discretization) + GPU e2e smoke (realized Var 0.226 vs target 0.25; recovery
   figure on the irregular grid). Docs: runbook H2, USAGE §5.14, DEVELOPER_GUIDE map.
2. ~~**U2 data-space propagation**~~ — **CLOSED 2026-07-06.** `llapdiff-uq-eval` now runs
   the data-space comparison by default: `AnalyticLawSampler` duck-types LLapDiff inside
   the *unchanged* `evaluate_regression` — `generate()` returns draws from N(mean, Var)
   (law cached once per batch; one decoder pass per draw, no reverse diffusion) — so the
   analytic arm is scored with the identical masking/CRPS estimator/ensemble size/seed
   as the sampled baseline, and wall-clock is reported for both (`analytic_speedup_x`;
   8.2× at 5 samples in the GPU smoke). This also makes U3's three-way comparison
   apples-to-apples (`--mean-source oneshot|ddim` vs `data_space_sampled`). Decoder-MC
   is the propagation (strictly tighter than the delta method and matches the sampled
   arm's estimator). Verified: adapter unit test (caching/statistics/delegation/seeded
   draws) + GPU e2e. Remaining related item: data-space PIT *diagrams* stay with the
   Fig-4 plotter gap (#5). Docs: runbook U2/U3 rows, USAGE §5.15.
3. ~~**Pole-coefficient L2**~~ — **CLOSED 2026-07-06.** `CHIRP_COEFF_L2` (config, default
   0.0): `ChirpModalField.coefficient_penalty` (all three parameterizations; growth head
   excluded — `c_g` governs it) → `LapFormer`/`LLapDiff.pole_coefficient_penalty` →
   `diffusion_loss(coeff_l2=…)`, applied at the two TRAINING call sites only (never the
   val-diagnostic probe, so selection metrics stay clean); `coeff_penalty` in loss stats.
   Verified: unit tests (positivity/differentiability per variant; exact λ·penalty loss
   addition; loud lti guard) + e2e spy in real GPU training (value arrives at all calls;
   optimized loss inflated by λ·penalty). Note the correct soft-constraint dynamics: the
   penalty is quadratically small at ε-init and bites as coefficients grow — weight-absmax
   after 1 epoch is NOT a valid observable for it. Docs: runbook §3 row, USAGE §5.12.
4. ~~**U1's specific clip check at p = 0.995**~~ — **CLOSED 2026-07-06.** `llapdiff-u1-sweep`
   gained `--dynamic-thresh-p` / `--dynamic-thresh-max` overrides; the effective quantile is
   recorded in every output row alongside `clip_fraction_mean`. Verified e2e on the trained
   G3 cell-(d) checkpoint (physionet h=12, val): **clip fraction 0.23% at p=0.995** — the
   post-head-removal ẑ₀ barely brushes the threshold (the plan's question, first answer).
5. **T1 "gap-blind model" comparison arm + figure tooling for Figs 3–5** — T1 analyzes one
   checkpoint across regimes (no gap-blind baseline arm); no plotters yet for forecast slices
   with predictive intervals (Fig 3), reliability *diagrams* (Fig 4 — numbers exist, plots
   don't), or the stability-envelope figure (Fig 5 — the test exists, the plot doesn't).

**Addendum 2026-07-20 — H2(ii) recovery figure (Fig 2) rebuilt** after the first generated
figures failed review (`h2_pole_recovery_problems_fixes.md`, P1–P10 all implemented; status
block at its top). Selection is now output-contribution-ranked from a `modal_capture` of the
final denoising step (old `omega_rmse_best_*` metrics are meaningless), ω_eff/ρ_eff over all
modes is the primary recovered curve, both arms are scored (LTI overlay), windows are
stratified, a `--recovery-share-threshold` validity gate watermarks invalid selections, and
`--sweep-period` provides the within-window excursion the "LTI fails structurally" deliverable
requires (prereg clause amended in `cmd_plan_v2.md` §H2 before the freeze). Paper-grade H2
needs retraining on reswept caches with a horizon-sized `CHIRP_NUM_BASIS`.

**Plan-internal tension to decide (not a bug):** the method doc assumes x0-parameterization
throughout (§4.1, §4.6) while the G3 factorial defaults to v-prediction for LLapDiff parity;
the plan itself lists predict-type as a Tier-1 knob. The paper's CMD arm should eventually be
pinned to one story (x0 for method fidelity, or revise the doc).

---

## 4. Reference: where everything lives

- Operator commands + the three warnings (H3 geometry, NLL warm-start, frozen-coefficient
  bug + checkpoint audit one-liner): `w_docs/CMD_RUNBOOK.md`
- Flags & config semantics: `w_docs/USAGE.md` §5.12–5.15; internals: `w_docs/DEVELOPER_GUIDE.md` §7.5
- Method text (with the §4.5 v_k correction, the Theorem-B head-precision remark, and the
  "implementation notes / reproduction traps" bullet): `chirp_modal_method_update.md`
- Tests: `tests/test_chirp_modal.py`, `test_chirp_uq.py`, `test_growth_budget.py`,
  `test_chirp_parameterizations.py`, `test_synthetic_chirp_benchmark.py`
