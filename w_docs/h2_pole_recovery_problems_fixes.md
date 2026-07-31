# H2 Pole-Recovery Figures — Problem Inventory & Fix Plan

> ## ✅ IMPLEMENTED 2026-07-20 (all of P1–P10; 316 tests green)
>
> **Step-1 diagnostic outcome (the expected case): the model was recovering fine —
> only the figure lied.** On the trained `synthetic_linear_chirp` seed-0 chirp
> checkpoint, the old criterion's top-4 "variation-energy" modes have output-energy
> share **exactly 0.0** (ρ≈41–111/step, ω≈0.9π — the absurd lines in the PDFs),
> while the top-4 modes by output contribution carry **98–99.5%** of the energy at
> ρ≈0.1–0.8/step, ω≈0.34–0.76 rad/step; ω_eff RMSE vs truth ≈ 0.05–0.09 rad/step.
> Genuine *model* findings now visible: ρ_eff overestimated (~0.1–0.2 vs truth
> 0.003) and ω_eff under-tracks the late test windows. NOTE these checkpoints were
> trained with the base-config `CHIRP_NUM_BASIS=256` (leaked class-2 knob; recovery
> JSON now records it).
>
> Where each fix landed:
> - **P3/P7** — `modal_capture` hook through `LapFormer.forward` → `LLapDiff.generate`
>   (final DDIM step, conditional branch) + `viz/plot_llapdiff_poles.py::modal_contributions`
>   (E_k, shares, residue norms, envelope mass, ω_eff/ρ_eff over ALL modes).
> - **P4/P9** — `_recover_pole_trajectories` rewritten: contribution ranking with
>   escalating top-N (`_select_top_modes`), `omega_eff_rmse`/`rho_eff_rmse` +
>   best-of-selected in CSV/JSON, per-mode diagnostics + metric definitions in the
>   recovery JSON, `--recovery-share-threshold` gate + "SELECTION INVALID" watermark.
> - **P2** — recovery runs for **both arms**; LTI overlays as flat gray dashed with
>   its own RMSE row. (Legacy-sweep observation: LTI's constant ω_eff matches truth
>   as well as chirp within windows — P5 was real.)
> - **P1/P8** — per-mode colors+linestyles, full legend with E-shares, joint
>   highlight mode, truth-scaled y-limits, ω_max=π reference, native-step unit guard.
> - **P6** — windows stratified across the test span (`_stratified_pick`), small
>   multiples with entity/start/t_norm annotations.
> - **P5** — `sweep_period` (triangle re-sweep) in `synthetic_regime_dataset.py` +
>   `--sweep-period` (cache-dir tagged, piecewise tasks reject/ignore), plus the
>   cross-window stitched figure `*_pole_recovery_series.pdf`.
> - **P10** — prereg amendment (third "selection invalid" category + sweep
>   requirement) in `cmd_plan_v2.md` §H2 + risk register, before the PREREG freeze.
>
> Regenerated (fixed-tool) figures/JSONs for the 5 trained (task, seed) pairs:
> `ldt/results/chirp_benchmark/recovery_v2_2026-07-20/` (originals left untouched).
> Paper-grade H2 needs retraining on `--sweep-period 144` caches (and a
> horizon-sized `CHIRP_NUM_BASIS`) per the runbook §2 H2 block.

*Diagnosed from `llaplace_df-update_method` source against the two generated PDFs
(`synthetic_linear_chirp seed=0`, `synthetic_quadratic_chirp seed=1`). Code references:
`llapdiffusion/viz/plot_llapdiff_poles.py::extract_chirp_pole_trajectories`,
`llapdiffusion/tools/run_synthetic_chirp_benchmark.py::{_recover_pole_trajectories,
_plot_recovery}` (+ the `if arm == "chirp"` caller block), and
`llapdiffusion/datasets/synthetic_regime_dataset.py::_pole_profiles`.*

---

## 0. Decoding the figures you have (what each line actually is)

- **Every colored line is the chirp model.** The LTI ("fixed poles") arm is **not in the
  figure at all** — recovery runs only under `if arm == "chirp"` in the benchmark tool.
- The 4 blue lines (ω panel) and 4 red lines (ρ panel) are the **top-4 modes ranked by
  coefficient "variation energy"** of the *same* chirp checkpoint, one line per mode.
- Alpha is the only differentiator: **α = 0.95 = the per-panel best-RMSE mode**, α = 0.25 =
  the other three. The dark blue and the dark red can be **different modes**
  (`best_omega_mode` and `best_rho_mode` are chosen independently).
- The dashed black line is the generator's ground truth for the one plotted window.
- Units are verified correct (`freq="1h"` → Δt in native steps; `CHIRP_TIME_SCALE → PRED`).
  The absurd values (ρ ≈ 87–118/step in fig 1; ω ≈ 2.75–2.87 ≈ 0.9π pinned near the cap)
  are therefore *real* pole values — **of the wrong modes** (see P3): a per-step decay of
  ~100 kills the envelope by t̃ ≈ 1, so those modes cannot be producing the forecast.

---

## 1. Problem inventory

### P1 — No legend / lines are visually undecodable  *(the reported symptom)*
- **Symptom:** several same-color lines per panel; only "ground truth" is labeled; the
  reader cannot tell chirp vs fixed poles, or mode from mode; alpha-only distinction dies
  in grayscale print.
- **Root cause:** `_plot_recovery` labels only the truth line; modes share one color;
  the highlighted mode differs per panel (P1b: independent `best_omega_mode` /
  `best_rho_mode` — misleading, since a reader assumes one component is highlighted).
- **Fix:** one color per mode from a colormap; legend entries
  `chirp mode #idx (E_k share = xx%)`; a single **jointly** selected highlight mode
  (top output-energy mode, see P3-fix) bolded in *both* panels; LTI overlay (P2) as flat
  gray dashed with its own legend entry; caption records entity id, `window_start`, and
  the window's t_norm span.
- **Acceptance:** every line in the PDF is identifiable from the legend alone; the same
  mode is highlighted in both panels.

### P2 — The fixed-pole (LTI) arm is missing from the figure
- **Symptom:** prereg Fig. 2 requires "LTI's flat recovered line (gray dashed)" for the
  structural-failure contrast; it is absent.
- **Root cause:** `extract_chirp_pole_trajectories` raises for LTI checkpoints; the
  caller only runs recovery for the chirp arm. No LTI pole-extraction path exists.
- **Fix:** add an LTI extraction (the LTI head's per-window conditioned constant poles via
  its `seed_poles`/`to_poles` path, same conditioning), overlay as flat gray dashed lines,
  and report its RMSE next to chirp's in the CSV/JSON so the "LTI fails structurally"
  claim has a number.
- **Acceptance:** figure shows chirp modes + LTI constants + truth; CSV contains
  `omega_rmse` for both arms.

### P3 — Mode selection anti-selects: the figure shows the model's junk drawer  *(core bug)*
- **Symptom:** selected modes have ρ ≈ 87–118/step (fig 1) or ≈ 2.5/step (fig 2) — decay
  that annihilates the envelope by t̃ ≈ 1–3, i.e., modes that contribute ~nothing to the
  forecast — and ω pinned near the ω_max = π cap (fig 1, 2.8 ≈ 0.9π).
- **Root cause:** ranking by `energy = a_rho2.sum(-1) + a_omega2.sum(-1)` (coefficient
  variation energy), **never consulting the residues θ**. Two compounding effects:
  (i) with K = 256 modes and a ~1-D sinusoidal target, the denoiser zeroes most residues;
  zero-residue modes get no gradient (and large-ρ̄ modes get vanishing envelope gradient),
  so their poles drift freely — the criterion selects exactly these unconstrained modes;
  (ii) the criterion is *positively correlated with ρ by construction*
  (instantaneous ρ = floor + Σa²φ, so "largest Σa²" ≡ "most inflated decay"), and large
  a_ω² pushes ω toward the cap via the headroom rescale — guaranteeing insane-looking
  panels whenever any junk mode's coefficients grow.
- **Fix:** rank by **output contribution**: capture (θ, ρ̄, ω̄) from the *actual
  generation* (forward hook on `LapFormer`'s synthesis at the last DDIM step, or an
  optional `return_modal_internals=True`), compute
  `E_k = mean_t e^{−2ρ̄_k(t)} · (‖c_k‖² + ‖b_k‖²)`, select top-E_k modes; additionally
  report the **residue-weighted effective trajectory**
  `ω_eff(t) = Σ_k E_k ω̂_k(t) / Σ_k E_k` (and ρ_eff) as the single recovered curve.
- **Acceptance:** selected modes' share of total output energy ≥ 50% (hard assert; below
  threshold the figure is stamped "selection invalid" and not used as evidence);
  contributing modes' poles land in a physically usable range (ρ ≲ O(5/H) per step, ω
  below cap).

### P4 — The recovery *metric* inherits P3 (best-of-junk RMSE)
- **Symptom:** `omega_rmse_best_mean` / `rho_rmse_best_mean` in the CSV look bad and are
  meaningless: they take the min over the top-variation (junk) modes; `best_mode_index`
  records only ω's pick.
- **Fix:** replace with (a) contribution-weighted RMSE of ω_eff/ρ_eff vs truth and
  (b) best-of-top-**contribution** modes RMSE; record per-selected-mode residue norm,
  envelope mass, E_k share in the recovery JSON.
- **Acceptance:** metric definitions in JSON; regenerated numbers used for the prereg
  Fig-2 caption RMSE.

### P5 — Benchmark design: there is almost no within-window chirp to recover
- **Symptom:** ground-truth dashed lines are ~flat (fig 2's truth rises only ~7%).
- **Root cause:** `_pole_profiles` ramps over the **whole series** (`t_norm = t/t_end`,
  series_length default 768, `freq_multiplier = 2.0`), so a 48-step window sees
  48/768 ≈ 6% of the sweep. As parameterized, "LTI fails structurally while CMD tracks"
  cannot materialize: within one window both arms face a near-constant pole. (Truth
  ranges check out: ω = 2π·f, f ∈ [1/48, 1/24]·sweep → 0.13–0.52 rad/step; truth
  ρ ∈ [0.002, 0.025] — invisible at the current y-axes, hence the "zero" line.)
- **Fix (both, they answer different questions):**
  1. **Within-window sweep:** add a `sweep_span` (or period) parameter so the profile
     completes its sweep over ~(window + horizon) steps; target within-window
     Δω/ω ≥ 30–50%. This is the panel where LTI is structurally wrong.
  2. **Cross-window figure:** keep the slow sweep and stitch correctly-selected
     per-window pole trajectories across consecutive test windows along absolute series
     time — the history-conditioned poles should step along the slow sweep, which also
     demonstrates the conditioning story.
- **Acceptance:** new within-window truth variation ≥ 30%; chirp-vs-LTI forecast CRPS
  separates on the reswept tasks (if it does not, that is now a *model* finding, not a
  design artifact).

### P6 — Only the first valid window is plotted; window provenance unknown
- **Symptom:** one arbitrary window per (task, seed); `num_recovery_windows=4` affects
  only the RMSE average; the payload comes from the first valid row of the first batch,
  so plotted/averaged windows are likely one entity and one sweep position; the title
  ("seed=0") says nothing about which part of the sweep you are looking at.
- **Fix:** stratify recovery windows across entities and window starts (early/mid/late
  t_norm — essential given P5's series-long sweep); plot small multiples (one row per
  window) or at least annotate `window_start`, entity, t_norm span; keep the seed in the
  title but add window metadata.
- **Acceptance:** figure/JSON identify each window's position; RMSE averaged over
  stratified windows.

### P7 — Plotted poles come from a static τ=1 probe, not from the generation
- **Symptom/risk:** `extract_chirp_pole_trajectories` embeds diffusion step `t_idx=1` and
  probes the pole head directly; the forecast is produced by poles along the DDIM path
  (final step ≈ but ≠ this probe, and conditioned on the evolving z_τ for the residues).
  The figure can therefore show poles that never synthesized anything.
- **Fix:** folded into P3's fix — capture poles + residues from the real `generate()`
  call at the last denoising step (hook). Keep the τ-probe only as a cheap debug mode.
- **Acceptance:** figure caption states "poles at the final denoising step of the
  evaluated forecast".

### P8 — Plot hygiene
- **Symptoms:** y-limits dictated by junk modes (truth ρ = 0.002–0.025 crushed onto the
  axis); no ω_max = π reference line; alpha-only emphasis (grayscale-unsafe); axis unit
  labels correct but unverified against Δt units at plot time.
- **Fix:** per-panel y-limits from truth ∪ selected modes (post-P3 this fixes itself, but
  keep a guard); horizontal ω_max line; distinct colors + linestyles; a one-line runtime
  assert that Δt is in native steps before labeling "[rad/step]", "[1/step]".

### P9 — No selection-validity diagnostics existed to catch this
- **Symptom:** the e2e smoke "passed" while the figure was meaningless.
- **Fix:** recovery JSON gains, per window: per-mode `E_k` histogram summary, selected
  share of output energy, residue norms, envelope mass; a unit test on a synthetic
  checkpoint asserting the selected-share threshold; the benchmark refuses to emit the
  PDF (or watermarks it "selection invalid") below threshold.

### P10 — Prereg falsification clause needs a third category
- **Symptom:** The signature Fig clause distinguishes only
  "recovery works" vs "identifiability fails"; the actual outcome was neither — the
  *tool* was invalid.
- **Fix:** amend the clause: recovery is judged **only after** the selection-validity
  gate (P9) passes; "selection invalid" triggers a tool fix + rerun, not a scientific
  conclusion. (This event does not count against the model.)

---

## 2. Implementation order (half-day of work, in this order)

1. **Diagnostic first, before any code change:** for one window, dump the per-mode
   output-energy histogram and the poles of the top-contribution modes from a hooked
   `generate()`. Two outcomes: contributing modes at ρ ≈ 0.002–0.2/step, ω ≈ 0.4–0.5 →
   the model was recovering fine and only the figure lied (expected case); contributing
   modes also wrong → a genuine modeling issue — stop and investigate before touching
   the plot.
2. `LapFormer` optional `return_modal_internals` (θ, ρ̄, ω̄, instantaneous ρ/ω) +
   `generate(..., capture_modal=True)` plumbing; unit test.
3. Rewrite selection/metrics per P3/P4; extend the JSON schema per P9; add the
   selected-share assert + test.
4. LTI extraction + overlay (P2); plotting rewrite (P1, P8); window stratification (P6).
5. `sweep_span` parameter + regenerated caches; cross-window stitched figure (P5).
6. Amend the prereg doc (P10) and regenerate all five task figures × 3 seeds.

## 3. Acceptance checklist for the regenerated figure

- [ ] Legend decodes every line (truth / chirp mode #k with E_k share / LTI constant).
- [ ] Same highlighted mode in both panels; selected-share ≥ 50% printed in the caption.
- [ ] LTI flat line present with its RMSE; chirp ω_eff tracks truth within the caption
      RMSE; truth visibly non-constant within the window (post-`sweep_span`).
- [ ] ρ y-axis on the truth's scale; ω_max = π reference visible.
- [ ] Window metadata (entity, start, t_norm span) in caption; stratified windows in the
      JSON averages.
- [ ] CSV: chirp vs LTI forecast CRPS separates on reswept tasks; recovery metrics are
      contribution-weighted.
