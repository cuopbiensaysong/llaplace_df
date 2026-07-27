# Figure 2 Production Strategy — Signature Pole-Recovery Figure (v3 campaign)

> ## STATUS 2026-07-21 — v3 campaign was INVALID for a tooling reason; Step 0 now implemented
>
> **The v3 run did not test the claim.** 11 of its 30 runs — all six `linear_chirp` and
> five of six `quadratic_chirp`, i.e. exactly the two tasks with a swept ω — were
> *evaluated* on a **07-14 checkpoint trained on the legacy near-constant cache**, while
> that run's own freshly trained model sat unused. Cause: `_best_raw.pt` is written only
> when `EMA_COMPARE_EVERY > 0` (the benchmark sets 0), the trainer reported it via a bare
> `.exists()`, and `_train_or_reuse_stack` preferred `best_checkpoint_raw` first — so a
> stale file won. Independently, both chirp tasks reused a **07-14 VAE + summarizer**
> (artifact dirs are keyed by (task, seed) with no cache identity, and
> `--recompute-artifacts` was not passed). Both are fixed: the trainer now reports only
> checkpoints written by the current run, the tool takes the trainer's own
> `loaded_checkpoint`, and an `artifact_cache.json` fingerprint makes cross-cache reuse
> fail loudly. All stale files have been deleted.
>
> **Step 0.4 is DONE, and the answer is favourable — the truth line is fair.** Measured on
> the sweep caches: the VAE latent's carrier frequency matches the data-space truth
> (0.48 vs 0.467 rad/step; 96–100% of channels within 0.08), and a single truth-pole basis
> explains **83–92%** of latent variance across all 24 channels. No footnote needed;
> recovery is well posed.
>
> **Steps 0.1–0.3 are implemented** (`--recovery-guidance`, `omega_trend_slope` over
> `--recovery-draws`, `--laplace-k`/`--chirp-num-basis` + a Nyquist warning). Also added:
> `forecast_freq_slope`, the Option-3 observable (instantaneous frequency of the forecast,
> validated to recover a known sweep at slope +1.03).
>
> **Corrections to the plan below.**
> 1. **K is missing from Step 0.3.** M controls smoothness, K controls identifiability.
>    ~8 modes already give reconstruction R²=0.99 and *which* 8 differs between training
>    runs, so at K=256 per-mode poles are not pinned by the objective. A K arm is now the
>    primary hypothesis — see `ldt/scripts/run_fig2_stage0_scan.sh`.
> 2. **The ρ-panel premise is wrong for the current generator**: `ramp_damping_up` gives
>    ρ·H ∈ [0.155, 0.388], not ~1.2, because the shared base decay drew 0.0032. Raise
>    `decay_min ≈ decay_max ≈ 0.01` or ρ stays in the weakly identifiable regime.
> 3. **Gate 3's LTI half is vacuous** — LTI poles are constant by construction, so its
>    slope is exactly 0. Only the chirp slope is evidence.
> 4. **Gate 6 should not veto the figure.** Tracking poles and winning CRPS are separate
>    claims; a valid identifiability figure shouldn't be discarded over an accuracy tie.
> 5. **Step 0.1 has an internal tension**: at w=1 the captured poles no longer belong to
>    the CRPS-scored forecast. Report the recovery arm's own w=1 CRPS, or caption the
>    default capture honestly as the conditional branch.
> 6. **A single draw is not enough for the slope.** The poles are deterministic given the
>    conditioning but the residue weights are not; measured constant-weight slope was
>    +0.032 ± 0.006 over six draws, while a time-resolved variant swung ±0.31.
>
> **Preliminary, not conclusive:** the correct sweep-trained `linear_chirp` model still
> showed slope +0.032 ± 0.006 with a carrier ~40% low — but it was trained against the
> stale summarizer, so the Stage-0 scan is the first clean test.
>
> ## UPDATE — the flat slope has a cause upstream of the denoiser
>
> Stage 1 could not reconstruct H2's targets: encode→decode of the TRUE targets reaches
> only **corr 0.62**, capping every downstream result. The encoder mean-pools across
> entities (`latent_vae.py`) while the generator gives each entity an independent
> `U(0, 2π)` phase, so the panel cancels before the latent. Ruled out by measurement:
> input dropout (0.620→0.586), latent width 24→64 (0.586→0.591), entity count
> (8 entities: 0.479). Fixed by narrowing the phase draw — **π/8 spread → corr 0.857,
> val_recon 0.0202 (3.2× better)** — now exposed as `--phase-spread`.
>
> **Consequences for this plan.**
> 1. **Step 1 must regenerate caches with `--phase-spread 0.393`** and retrain; every
>    existing H2 arm was trained through the handicapped stage 1.
> 2. **Option 3 (forecast instantaneous frequency) was dead for this reason**, not
>    because CMD cannot track: the forecast correlated with the target at only
>    +0.06…+0.32 and oscillated ~4× too fast. Re-test it after the fix before cutting it.
> 3. **Gate 0 (new, before every other gate):** stage-1 round-trip corr on the run's own
>    cache must exceed ~0.85. Below that the run cannot speak to any Fig-2 claim. Check it
>    with `llapdiff-stage1-roundtrip` (encodes the true targets, decodes, PASS/FAILs vs a
>    threshold; exits non-zero on failure). **Confirmed end-to-end 2026-07-21**: the full
>    retrained stack on the `--phase-spread 0.393` cache passes at **corr 0.857** (matching
>    the isolated probe), vs 0.62 on the historical 2*pi cache — the first sound stage 1 in
>    this benchmark's history.
> 4. Stage-0 scan results, all **provisional** (measured inside the handicapped
>    pipeline, `ldt/results/fig2_stage0/`). K=8 never ran — the shared H100 stayed
>    full; rerun it after the phase fix rather than in isolation.
>
>    | K (M=8) | chirp ω slope | ω_eff RMSE | chirp CRPS | LTI CRPS | LTI selection |
>    |---|---|---|---|---|---|
>    | 256 | +0.0035 ± 0.0052 | 0.311 | 0.2014 | 0.2232 | invalid (0.22) |
>    | 32 | +0.0537 ± 0.1413 | 0.128 | 0.2041 | 0.1873 | valid (0.53) |
>    | 8 | not run (no GPU) | — | — | — | — |
>
>    Smaller K sharply improves the *decomposition* (ω level 2.4× more accurate; LTI
>    energy concentrates enough to pass the validity gate) but the slope stays ~10x
>    below the 0.6 gate, and its spread is across-WINDOW disagreement (-0.076, +0.013,
>    +0.255, +0.023 — each tight within itself, ±0.002-0.047 over draws), not
>    averageable noise. Directionally consistent with the K hypothesis, far from
>    confirming it.
> 5. Re-run the capacity sweep after the fix: latent width was useless only because the
>    information was already destroyed upstream; it may matter now.

*Runbook for producing the paper-grade Fig. 2 ("recovered pole trajectories track the chirp;
LTI structurally fails"). Companion to  `h2_pole_recovery_problems_fixes.md` (P1–P10, all landed), and `cmd_plan_v2.md`
(§H2, third "selection invalid" outcome). Status at write-time: tooling validated, model
plausible (energy concentration 90–100% on linear seed-0; window-1 ω_eff ≈ truth), claim
untested — legacy caches contain ≤7% within-window sweep and cannot test it.*

---

## Step 0 — Blocking code changes (~half a day, before any retrain)

| # | Change | Where | Why |
|---|---|---|---|
| 0.1 | **Recovery capture at w = 1.** Add `--recovery-guidance 1.0` (or pin `guidance_strength=1.0` inside `_recover_pole_trajectories`'s `_generate_kwargs`). Update the caption string to "poles at the final denoising step, w = 1". | `tools/run_synthetic_chirp_benchmark.py` | `GUIDANCE_STRENGTH=(1.0, 2.0)` schedules to g ≈ 2.0 exactly at the final (captured) step, so current captures are the conditional *component* of a CFG blend, not the poles of the forecast. Forecast CRPS in the CSV keeps the TEST guidance — only the capture path changes. |
| 0.2 | **Trend-slope metric.** Per window and per arm, regress ω̂_eff(t) on ω_true(t); record `omega_trend_slope` in the recovery JSON/CSV (chirp target → 1, LTI target → 0). | recovery metrics block | Under a triangle sweep the window-*mean* ω can tie between arms; the slope is the structural discriminator the figure's claim rests on. |
| 0.3 | **M = 8 (≤16) + Nyquist guard.** Set `CHIRP_NUM_BASIS = 8`; warn if `num_basis > time_scale / 2` (bound = 24 at H = 48). | `configs/config.py`, `models/laptrans.py` | M = 256 is ~10× super-Nyquist → the trajectory jitter and (plausibly) quad-seed-1's diffuse energy (top-16 share 22%). Guard prevents silent regression. |
| 0.4 | **Fair-truth diagnostic.** Standalone script: Lomb–Scargle each channel of z = VAE_enc(Y_true) on a few test windows; record the dominant latent frequency per task in the recovery JSON. | new `tools/` script | The model recovers *latent-space* poles; truth is *data-space*. If the encoded latent's carrier ≈ 0.47, an ω_eff at 0.26 is a real error; if the latent oscillates at ~0.25, recovery is correct and the truth line needs a footnote. Required to interpret ω_eff either way. |

Ride-along polish (non-blocking): per-window watermark badges + valid-window fraction
instead of the figure-level veto; legend cap at 6 entries ("+N more, Σx% E"); delete the
`__init__` debug prints (`laptrans.py:410`, `lapformer.py:438`).

---

## Step 1 — Retrain on caches that contain a within-window chirp

```bash
llapdiff-synthetic-chirp \
  --tasks synthetic_linear_chirp synthetic_quadratic_chirp \
          synthetic_ramp_damping_up synthetic_freq_shift \
  --arms lti chirp --seeds 0 1 2 \
  --sweep-period 144 --gap-distribution gamma --gap-mean 1.0 --gap-shape 4.0 \
  --recovery-share-threshold 0.5 \
  --overwrite-data --recompute-artifacts \
  --output-root ldt/results/chirp_benchmark_v3
```

Design rationale:
- **`--sweep-period 144`** → Δramp up to 2·48/144 ≈ 0.67 per horizon → **30–67% within-window
  frequency change** — the "something to track" legacy caches lacked. Triangle sweep is
  computed in absolute time (gap-aware), every purged tail window sees it, and windows
  straddling a peak give free up-then-down chirps.
- **Task → panel mapping:** linear chirp = ω-tracking; quadratic = accelerating ω;
  **ramp_damping_up = the ρ panel** (ρ·H up to ~1.2, the only place ρ is identifiable —
  chirp-task ρ-RMSE stays excluded per the identifiability finding: truth ρ·H ≈ 0.14 means
  the prior floor fills the null direction, recovered ρ ∈ [0.06, 0.25] ≈ the init range);
  freq_shift = piecewise regime switch (correctly rejects `sweep-period`, keeps its
  change point).
- **Fresh caches + artifacts are mandatory** (`--overwrite-data --recompute-artifacts`):
  anything trained on legacy caches or M = 256 is dead for this figure.
- `synthetic_growth_decay` + a c_g > 0 arm is the Theorem-B′ panel — run it later,
  decoupled from this campaign.
- Scheduling: these runs are small — interleave on the GPU **without pausing the h=168
  campaign**. Fig. 2's outcome forks the prereg tree (it gates Fig. 5), so resolve it
  before spending big compute on real-data pole-trajectory figures.

---

## Step 2 — Evidence gates (check in this order, before judging prettiness)

A window's evidence counts only if all of:

1. **No watermark** — selected-mode output-energy share ≥ 0.5.
2. **Sweep confirmed** — cache meta shows within-window Δω/ω ≥ 30%.
3. **Slope separation** — chirp `omega_trend_slope` ≥ ~0.6 while LTI ≤ ~0.2.
4. **Level accuracy** — chirp ω_eff RMSE clearly below LTI's.
5. **Fair-truth consistency** — Step-0.4's latent carrier is consistent with the plotted
   truth line (else annotate/replace the truth line).
6. **Corroboration** — forecast CRPS in the CSV: chirp beats LTI on these reswept caches
   (the tracked poles must matter for forecasting, not only for the figure).

---

## Step 3 — Assemble the figure

- Four panels (one per task). Each: truth (black dashed), **ω_eff bold**, top-3
  contributing modes (faint, legend-decodable with E-shares), LTI constant (gray dashed),
  ω_max = π reference.
- Pick the **median-RMSE stratified window** per task — never the best one.
- Caption carries: RMSE and trend slope as mean ± std over all stratified windows × 3
  seeds; selected-energy share; "poles at the final denoising step of the evaluated
  forecast, w = 1"; the ρ-identifiability footnote for the chirp tasks.
- Companion/appendix: the cross-window `*_pole_recovery_series.pdf` on the **legacy**
  slow-sweep cache — conditioned poles stepping along the series-scale sweep demonstrates
  *conditioning* (a different claim from within-window *expressiveness*; keep them
  separate).

## Acceptance checklist (regenerated figure)

- [ ] All windows pass gates 1–2; slopes reported for both arms (gate 3 met).
- [ ] Legend decodes every line; same highlighted mode in both ω/ρ panels.
- [ ] ρ panel comes from ramp_damping_up only; chirp-task ρ excluded with the footnote.
- [ ] Median-window selection documented in the JSON; seeds × windows stats in caption.
- [ ] CSV shows chirp > LTI forecast CRPS on the v3 caches.
- [ ] Prereg Fig-2 clause updated with the run ID before test-split evaluation elsewhere.

---

## Failure branch (pre-registered)

If selection is valid, the sweep is confirmed ≥ 30%, and the chirp slope still sits near 0
across seeds: that is a **real negative result, cleanly obtained**. Per the prereg decision
tree: cut Fig. 2 and Fig. 5; the paper leads with certificate + matched accuracy +
analytic UQ + Theorem D (a publishable paper on its own). Decide from the slope statistic
across seeds — not from how the curves look.
