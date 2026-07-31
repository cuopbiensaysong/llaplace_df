# CMD — Revised Submission & Experiment Plan (v2)

Supersedes the plan in `chat_summary_for_plan.md` (Part VII). Method reference:
`chirp_modal_method.md` (now Theorems A–D + B′).

---

## Changelog vs v1 — what changed and why

1. **In-harness LLapDiff reproduction moved from "action 2" to a Phase-0 gate.** It is not a
   baseline; it decides *which paper we are writing* (see the branch logic in §8).
2. **The ablation is now an explicit 2×2 factorial** (poles: chirp/LTI × head: MLP/none). v1 had
   only two of four cells.
3. **Claims language downgraded until seeded.** "Thesis validated" → "directionally encouraging"
   (Δ = 0.018 CRPS at 1 seed vs the baseline's reported ±0.01 std ≈ 1.5–2σ). "MSE beats the
   paper's" → "matched" (0.634 vs 0.638 is inside their ±0.02). The paper sentence is
   *"matched mean accuracy with the residual MLP deleted"* — stronger anyway.
4. **Theorem D and Theorem B′ now exist** → two new contributions (C4, C5), two new experiments
   (pole-invariance stress test; growth-budget ablation), and updated figure list.
5. **Four missing experiments added:** synthetic ground-truth chirp benchmark (the signature
   figure), Theorem-D pole invariance, imputation (LLapDiff's Fig.-3 counterpart), and the
   "why keep the diffusion" ablation for Phase 2.
6. **Protocol-parity checklist added** before any calibration conclusion (25-sample CRPS, DDIM
   steps, EMA, guidance w, dynamic thresholding).
7. **Finding-2 fix revised:** per-*dataset* time constant (train-split max horizon), not
   per-sample `L = max|t_rel|` (per-sample makes the function class sample-dependent and leaks
   horizon length).
8. **Symmetric-architecture rule:** every arm of every comparison uses the identical output path
   and identical tuning budget.
9. **Pre-registration block** (§1) so "select on val, touch test once" is auditable.
10. **Table hygiene:** debugging artifacts (0.469 naive Option A; pre-fix 0.367, which is not
    comparable post-Finding-2) are excluded from all paper tables.

---

## 0. Claims and contributions (final ordering)

Lead with the certificate story — Finding 1 earned it.

- **C1 (stability, the lead).** A stability certificate that holds *at the model output*, with the
  uncertified residual MLP deleted (Thm B) — noting the prior guarantee applied to a component
  that was not the returned tensor. Verified by a model-level contraction test shipped in the
  repo (cite in the reproducibility statement).
- **C2 (integrability).** Closed-form time-varying modal dynamics: the rotation–scaling normal
  form is exactly integrable (Thm A) and *necessarily* so — the companion form is not
  (Prop. A.1, WKB contrast). Strict generalization: constant poles recover LLapDiff verbatim.
- **C3 (analytic UQ).** Closed-form Lyapunov moment propagation (Thm C): exact Gaussian latent
  predictive law in one pass; calibration improvements measured by PIT/reliability.
- **C4 (renewal theory).** Theorem D: renewal averaging generalized to time-varying poles —
  *pathwise* contraction (strictly stronger than the baseline's mean-stability), the
  drift-corrected effective-pole map with the new ½·s′ₖ·E[Δ²] term, and a falsifiable
  invariance prediction on learned pole trajectories.
- **C5 (certified growth).** Theorem B′: a budgeted-excursion relaxation admitting genuine
  within-window amplitude growth with an explicit overshoot cap e^{c_g}; c_g = 0 recovers Thm B.
- **C6 (empirical).** Long-horizon accuracy, matched-or-better calibration, stability under
  extrapolation and gap stress, flat wall-clock — all against the in-harness reproduction.

**Precision rule for the writing:** time-varying poles buy frequency/decay drift, *not* per-mode
amplitude growth (that is C5's job). Never claim "transient growth" from chirp alone.

---

## 1. Pre-registration (freeze before Phase 1; keep in repo as `PREREG.md`)

- **Headline metric:** CRPS on NOAA-UK, h = 168 (secondary: MSE; tertiary: PIT calibration error).
- **Seeds:** 5 minimum for every reported cell (10 for headline table rows). Report mean ± std.
- **Selection:** all hyperparameters selected on the validation split; test evaluated once per
  final configuration per seed.
- **Comparison protocol (parity checklist — must match LLapDiff's):** 25 samples for CRPS;
  MSE on the sample mean; DDIM 64 steps, η = 0; EMA eval (decay 0.999); guidance w and dynamic
  thresholding logged per run; chronological 0.7/0.1/0.2 split; identical VAE + summarizer
  checkpoints across all arms.
- **Symmetric-architecture rule:** all arms share the identical output path (post Finding-1 fix:
  `output_skip_scale * y_time`, clamp |s| ≤ 1, no `head_proj(head_norm(·))` in any no-MLP arm;
  the +MLP arms use LLapDiff's original head). Confirm the LTI-no-MLP control was rebuilt after
  the head removal — if it still carries the head, its number is invalid.
- **Matched tuning budget:** every Tier-1 sweep (§7) runs with the same grid and trial count on
  chirp, LTI control, and the LLapDiff reproduction. Note: optimal guidance w may differ between
  arms (the modal-sum dynamic range changed) — that is allowed; unequal *budgets* are not.

---

## 2. Phase 0 — Correctness gates and the true target (blocks everything)

**G1. Land the two fixes, with tests.**
- Finding-1 corrected Option A (keep output scaling, delete only the LayerNorm head in no-MLP
  arms). New tests: (a) *model-level* contraction — the **returned tensor** obeys the
  |s|-carrying Thm-B bound over the full horizon at multiple diffusion steps; (b) envelope-decay
  assertion (no LayerNorm floor; catches regressions of the exact bug found); (c) loss-scale
  smoke test (initial loss O(1), not O(70)).
- Finding-2 with a **per-dataset** time constant: `L_dataset = max horizon in train split`
  (config-pinned), dividing `basis_freqs` in `_basis`/`integrated`. Assert wiggle/linear ratio in
  the intended O(0.01–0.1) band at the longest horizon; assert phase-unit consistency with
  ω_max = π per native step.

**G2. Reproduce LLapDiff (LTI + MLP) in our harness.** PhysioNet h=12 first (fast), then
NOAA-UK h=168. This defines the target; see §8 for the branch.

**G3. Fill the 2×2 factorial, ≥5 seeds each (PhysioNet h=12, cheap):**

| | + MLP head | − MLP head |
|---|---|---|
| **LTI poles** | (a) = in-harness LLapDiff | (b) control |
| **Chirp poles** | (c) redundancy probe | (d) = CMD |

Reads we are looking for: (d) > (b) → time variation carries accuracy; (c) ≈ (d) → the MLP is
redundant once poles vary (the cleanest kill-shot, stronger than cross-arm comparison);
(d) ≈ (a) → "delete the hack, keep the accuracy."

**G4. Protocol-parity audit** (checklist in §1) before interpreting any CRPS number. CRPS from m
samples is biased upward at O(1/m) — a sample-count mismatch masquerades as miscalibration.

**Exit criteria:** tests green; loss ~O(1); (a)–(d) filled with seeds; parity checklist signed off.

---

## 3. Phase 1 — Headline + the signature synthetic

**H1. NOAA-UK h=168:** full 2×2 + LLapDiff reproduction, 10 seeds, pre-registered metric. This is
the regime where fixed linearization drifts and chirp should separate. Add NOAA-US h=168 and
BMS h=168 as replications.

**H2. Synthetic ground-truth chirp benchmark (new; the signature figure).** Generate irregularly
sampled (renewal gaps, tunable Var(Δ)):
- linear and quadratic frequency chirps ω(t);
- ramp-damped oscillators ρ(t) increasing/decreasing;
- a regime-switch signal (piecewise poles);
- optionally one B′ case: growth-then-decay envelope (needs c_g > 0).

Deliverables: (i) LTI fails structurally while CMD tracks (forecast plots + CRPS/MSE);
(ii) **recovered ω̂ₖ(t), ρ̂ₖ(t) overlaid on ground truth** — the only place expressiveness is
*demonstrated* rather than argued, and the identifiability check that licenses the real-data
pole-trajectory figure; (iii) companion-form vs normal-form numerical-integration error (the
Prop.-A.1 figure).

*Amendment (2026-07-20, before PREREG freeze — see `h2_pole_recovery_problems_fixes.md`):*
recovery in (ii) is judged **only after the selection-validity gate passes**: recovered modes
are ranked by output contribution E_k = mean_t e^{−2ρ̄ₖ(t)}‖θₖ‖² captured from the final
denoising step of the evaluated forecast (never by coefficient variation, which anti-selects
zero-residue junk modes), the primary recovered curve is the E_k-weighted effective trajectory
over all modes, and the selected modes must explain ≥ 50% of output energy. A below-threshold
figure is stamped "selection invalid" and triggers a tool fix + rerun — a **third outcome**
distinct from "recovery works" and "identifiability fails" that counts against neither the
model nor the method. The first H2 figures (2026-07) failed exactly this gate; the model's
contributing modes were recovering fine. Additionally, "LTI fails structurally while CMD
tracks" in (i) requires the within-window pole excursion to exist: chirp tasks use the
triangle re-sweep (`--sweep-period ≈ window+horizon`); the legacy series-long ramp
(within-window Δω/ω ≈ 6%) is kept only for the cross-window stitched-recovery view.

**H3. Extrapolation / boundary-crossing stress** (mirror LLapDiff's Fig.-5 protocol) with
chirp-vs-LTI arms.

---

## 4. Phase 2 — Analytic UQ and calibration

**U1. Guidance-first calibration sweep.** Prime suspect for the CRPS-gap-with-matched-MSE
signature: w > 1 sharpens the predictive distribution. Sweep w ∈ {1.0, 1.25, 1.5, 2.0} × DDIM
steps {16, 32, 64} × samples {25} on val, per arm. Also check whether ẑ₀ brushes the dynamic
threshold (p = 0.995) after the head removal/rescale — log clip rates.

**U2. Implement Theorem C** (isotropic case): predict {qₖ(·), pₖ⁰}; closed-form vₖ(t̃) under
P-exact; latent Gaussian law (7); delta-method propagation through the decoder. Report
PIT/reliability diagrams and CRPS vs sampled-UQ at matched wall-clock.

**U3. "Why keep the diffusion" ablation (pre-registered, three arms):**
diffusion + sampled UQ vs diffusion + analytic UQ (7) vs **one-shot Gaussian NLL** with the same
synthesizer (no diffusion). Either outcome is a result: diffusion wins → justified; NLL ties →
an even faster model and an honest section. Scope the claim: Thm-C UQ is aleatoric, conditional
on predicted parameters — no epistemic component; say so.

---

## 5. Phase 3 — Theorem-driven experiments (new)

**T1. Theorem-D pole invariance.** Reuse LLapDiff's induced-missingness stress protocol
(coverage thresholds 0→80%), but evaluate the **learned pole trajectories**, not just CRPS:
(ρ̂ₖ(·), ω̂ₖ(·)) should stay ~invariant across gap regimes while implied event-domain
multipliers shift per Eq. (8). Report a trajectory-space distance across regimes for CMD vs the
effective-pole drift a gap-blind model exhibits. This converts Theorem D into a table.

**T2. Growth-budget (Thm B′) ablation.** c_g ∈ {0, log 2, log 5}: CRPS/MSE deltas; where the
learned excursions γₖ are spent (plot); whether ramping/trending sets (NOAA, finance) benefit.
Also preempts the "hard mean-reversion bias" objection in the rebuttal.

**T3. Imputation** (LLapDiff Fig.-3 counterpart, vs CSDI): 30% masked historical targets on
PhysioNet/Crypto/NOAA-UK, with t̃ anchored at the earliest query (per the §4.4 remark) so the
certificate covers imputation. One run, closes an obvious reviewer gap.

**T4. Efficiency table** (LLapDiff Tab.-5 format): wall-clock across horizons; target line is
"≈450 ms, flat in h, same as LLapDiff" — chirp adds per-query basis evaluation, so measure
before a reviewer asks. Include K-sweep parity (their Tab. 7).

---

## 6. Phase 4 — Full grid, remaining ablations, figures

- Full dataset × horizon grid (their Tab.-1 footprint) for the final table, 10 seeds on headline
  rows, 5 elsewhere.
- Chirp-specific ablations: num_basis M, ρ_min, parameterization (P-exact vs P-mono vs P-grid —
  expect P-grid to show integration error on wide gaps; that is a *feature* of the story), K,
  time-constant sensitivity, L2 on pole-function coefficients.
- **Figure list (final):**
  1. Pole *trajectories* ρₖ(t), ωₖ(t) — curves that move vs LLapDiff's static dots (cond vs
     uncond; across gap regimes → doubles as the T1 figure). Identifiability licensed by H2(ii).
  2. Synthetic ground truth vs recovered poles (H2).
  3. Long-horizon forecast slices with predictive intervals through missingness bands.
  4. Calibration: reliability/PIT, sampled vs analytic UQ.
  5. Stability envelope: model output vs the certified bound (the Finding-1 plot, now passing);
     extrapolation stress.
  6. Companion vs normal form integration error (Prop. A.1).
- **Table hygiene:** exclude 0.469 (naive Option A) and pre-fix 0.367 from every paper table —
  debugging artifacts; 0.367 is additionally incomparable post-Finding-2 (basis rescale changed
  the function class). Lab notes only.

---

## 7. Tuning tiers (revised)

- **Tier 0 (gates):** the two fixes + tests (§2). Not tuning — correctness.
- **Tier 1 (highest CRPS leverage, matched budget across arms):** guidance w, DDIM steps,
  EMA on/off, prediction type (x₀ vs v), min-SNR γ, sample count (fixed at 25 for reporting).
- **Tier 2 (chirp-specific):** num_basis, ρ_min, c_g, parameterization, K, time constant,
  pole-coefficient L2.
- **Tier 3 (capacity):** width/depth — only if Phases 1–2 leave a gap.

Rule: Tier-1 settings are tuned per-arm with identical grids/trials; Tier-2 applies to CMD only;
report the chosen settings for every arm in the appendix.

---

## 8. Branch logic (decided by G2)

- **If in-harness LLapDiff ≈ 0.32 (paper number reproduces):** the story is "close the gap":
  Tier-1 calibration work is the critical path; headline = (d) ≥ (a) with the MLP deleted +
  certificate + Thm C/D.
- **If in-harness LLapDiff ≈ 0.36 (reproduction gap):** the 0.320 anchor is retired from the
  narrative; the story is already "match + certify + strictly generalize": headline = (d) ≈ (c)
  ≈ (a)-in-harness with certification, and effort reallocates from CRPS-chasing to H2/T1/T2
  (theory-driven differentiation). State the reproduction transparently.

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chirp–LTI gap vanishes under seeds at h=12 | medium | expected (wrong regime); headline is h=168 — do not over-invest in PhysioNet |
| Calibration gap is protocol, not model | medium | §1 parity checklist + U1 before any model change |
| Poles unidentifiable → trajectory figure unconvincing | medium | H2(ii) synthetic recovery + cond/uncond + cross-regime consistency; selection-validity gate (contribution-ranked modes, share ≥ 50%) separates tool failure from identifiability failure |
| Chirp slower than LLapDiff | low | T4 early; basis evaluation is parallel — verify |
| "This is a time-varying diagonal SSM" review | high | related-work paragraph per method-file notes (S4D/S5/Mamba-style, CfC/LTC, LinOSS, Neural Flows, BOP-DMD, chirplets/nonstationary Prony); positioning: continuous-time closed form at arbitrary irregular queries, inside a diffusion denoiser, output-level certificate, analytic UQ |
| "Why diffusion at all" review | high | U3 pre-registered three-way ablation |
| Mean-reversion bias objection | medium | T2 (c_g ablation) + Thm B′ |
| Repro gap embarrassment | low | report transparently; §8 branch absorbs it |

---

## 10. Immediate next actions (unchanged order of urgency, tightened)

1. Land Finding-1-corrected + Finding-2 (per-dataset constant), add the three new tests
   (model-level contraction, envelope decay, loss scale); confirm loss ~O(1) and CRPS ≈ 0.365
   at 1 seed as a smoke check.
2. **G2 + G3:** in-harness LLapDiff (a) and chirp+MLP (c) on PhysioNet h=12; rerun (b), (d);
   5 seeds each; run the §1 parity audit. Decide the §8 branch.
3. Stand up NOAA-UK h=168 (full 2×2 + repro, 10 seeds) and, in parallel, the H2 synthetic chirp
   generator — the headline table and the signature figure.
