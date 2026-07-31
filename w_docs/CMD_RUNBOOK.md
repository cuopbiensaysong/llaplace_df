# CMD Experiment Runbook — operator guide for `w_plan/cmd_plan_v2.md`

This is the *how-to-run-it* companion to the experiment plan. It assumes the Phase-0
code gates are landed (branch `update_method`, 2026-06-25): `--seed`, `--output-head`,
fixed chirp time scale (`L = PRED`), ω cap, `|s| ≤ 1` clamp. Everything below is a
`llapdiff-train` / `llapdiff-checkpoint-eval` invocation unless marked
**[needs implementation]** — those items require code that does not exist yet.

Reference docs: `USAGE.md` §5.12–5.13 (flags), `DEVELOPER_GUIDE.md` §3 (config
footgun), §7.5 (chirp internals).

---

## 0. One-time setup on the GPU server

```bash
# 1. Sync the code (branch update_method) and enter the env
cd <repo-root>            # ALWAYS run from the repo root (ldt/ is CWD-relative)
conda activate llapdiff
git checkout update_method

# 2. Confirm the gates are green before spending GPU time
python -m pytest tests/ -q          # expect: all passed (290 at time of writing)

# 3. Warm the shared stage-1/2 artifacts ONCE per (dataset, horizon).
#    All arms and seeds reuse the same frozen VAE + summarizer (the parity
#    requirement). Do this BEFORE launching arms in parallel, otherwise two
#    first-runs may both try to train the VAE.
llapdiff-artifact-prep --datasets physionet --summary-json ldt/results/prep_physionet.json
# later, for the headline: --datasets noaa_uk noaa_us bms_air
```

Sanity: `ldt/vae/saved_model/<ds>/pred-<H>_ch-<C>_entity_elbo.pt` and
`ldt/summarizer/saved_model/<ds>/<H>-<C>-summarizer.pt` must exist for every
horizon you plan to run (channel table in `USAGE.md` §3.4).

> 🔴 **Do not reuse ANY chirp checkpoint trained before 2026-07-05.** Two
> generations of invalidation:
> 1. *Before 2026-06-25:* the Finding-1/-2 fixes changed the function class
>    (basis rescale, head change). Lab numbers 0.469 / pre-fix 0.367 stay out of
>    all paper tables (plan §6).
> 2. *Before 2026-07-05 (CRITICAL — found via the T1 smoke):* the pole-coefficient
>    head `to_coeffs` was zero-initialized **and squared**, and `a = 0` is a
>    stationary point (`d(a²)/dW = 2a·h = 0`) — the head received **exactly zero
>    gradient** and never moved. Every "chirp" trained before this date actually
>    had frozen, condition-independent, constant poles: the chirp arms were never
>    time-varying, so any chirp-vs-lti comparison from those runs is meaningless.
>    Fixed by eps-init (std 1e-4, ~1e-8 from LTI at init) + a permanent gradient
>    regression test (`test_chirp_coeffs_receive_gradient_at_init`).
>
>    **Affected — retrain ALL of:** G3/H1 chirp cells (c) and (d), H2 chirp arms,
>    U2/U3 UQ runs (they sit on the chirp core), and any T1/T2 inputs. lti cells
>    are unaffected (they never had the head). This bug also retroactively
>    explains why the Finding-2 basis rescale "changed nothing" on PhysioNet:
>    with the coefficients frozen at zero there was no time-varying part for the
>    rescale to act on.
>
>    **Audit any chirp checkpoint before trusting it** (dead head ⇒ absmax is
>    exactly 0.0; healthy post-fix training ⇒ nonzero):
>
>    ```bash
>    python -c "import torch; sd = torch.load('<ckpt>.pt', map_location='cpu')['model']; \
>    print('to_coeffs absmax:', sd['model.chirp_field.to_coeffs.1.weight'].abs().max().item())"
>    ```

---

## 1. Phase 0 — G2 + G3: the 2×2 factorial on PhysioNet h=12 (5 seeds)
### ✅ EXECUTED 2026-07-05 — results and branch decision below; commands kept for reruns

Four arms × seeds 0–4 = 20 stage-3 runs (stages 1–2 are reused, so each run is
diffusion training only). Cell letters follow the plan's table:

```bash
DS="--dataset-key physionet --preds 12"
for s in 0 1 2 3 4; do
  llapdiff-train $DS --seed $s                                    # (a) lti + head  = in-harness LLapDiff
  llapdiff-train $DS --output-head off --seed $s                  # (b) lti − head  = control
  llapdiff-train $DS --modal-type chirp --output-head on --seed $s  # (c) chirp + head = redundancy probe
  llapdiff-train $DS --modal-type chirp --seed $s                 # (d) chirp − head = CMD
done
```

Checkpoints land at (per arm, per seed):

```
(a) ldt/output/physionet/seed-<s>/pred-12/llapdiff_pred-12_best.pt
(b) ldt/output/physionet/head-off/seed-<s>/pred-12/…
(c) ldt/output/physionet/modal-chirp/head-on/seed-<s>/pred-12/…
(d) ldt/output/physionet/modal-chirp/seed-<s>/pred-12/…
```

Evaluate each best checkpoint (variant is rebuilt from checkpoint metadata —
no extra flags needed):

```bash
llapdiff-checkpoint-eval --dataset-key physionet --pred 12 \
  --checkpoint <path from the table above> \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/g3/physionet_h12_<cell>_s<s>.json
```

### ✅ EXECUTED 2026-07-05 — results (test CRPS, mean ± std over seeds 0–4)

| | + head | − head |
|---|---|---|
| **LTI** | (a) 0.3710 ± 0.0117 | (b) 0.3719 ± 0.0125 |
| **Chirp** | (c) 0.3706 ± 0.0024 | (d) **0.3709 ± 0.0018** |

Per-seed CRPS (note the paired-seed co-movement within each core family):

| seed | (a) | (b) | (c) | (d) |
|---|---|---|---|---|
| 0 | 0.3617 | 0.3623 | 0.3691 | 0.3694 |
| 1 | 0.3722 | 0.3728 | 0.3734 | 0.3736 |
| 2 | 0.3680 | 0.3679 | 0.3676 | 0.3693 |
| 3 | 0.3906 | 0.3929 | 0.3725 | 0.3714 |
| 4 | 0.3626 | 0.3635 | 0.3706 | 0.3707 |

**Reads** (plan §2 G3):
- **(d) ≈ (a): holds exactly** (Δ = 0.0001) — "delete the hack, keep the accuracy".
- **(c) ≈ (d): the kill-shot holds** (Δ = 0.0003) — the head is redundant once poles vary.
- **(d) > (b): not observed at h=12** (Δ = 0.001, ≪ 1σ) — the risk register's expected
  outcome at this horizon; the real test is NOAA-UK h=168.
- (a) ≈ (b): the head doesn't help the LTI core here either.
- **Unplanned finding:** chirp arms show **~6× lower seed variance** (0.0018–0.0024 vs
  0.0117–0.0125; the lti arms' std is a seed-3 excursion the chirp arms don't have).
  Attribution: both chirp cells are tight while the head-free lti control (b) is loose
  → the stability comes from the **pole parameterization**, not head removal.
  Replicate at h=168 before making it a paper claim.

Full record incl. MAE/MSE, provenance and claims-language notes:
`ldt/results/g3/G3_RESULTS.md`.

### G4 parity checklist (sign off before interpreting ANY CRPS)

Defaults already match LLapDiff's protocol — verify nothing was changed locally:

| Item | Config knob | Required value |
|---|---|---|
| CRPS samples | `NUM_EVAL_SAMPLES` | 25 |
| DDIM steps | `GEN_STEPS` | 64 |
| EMA eval | `USE_EMA_EVAL` / `EMA_DECAY` | True / 0.999 |
| Guidance | `GUIDANCE_STRENGTH` / `GUIDANCE_POWER` | (1.0, 2.0) / 1.0 — log per run |
| Dynamic thresholding | `DYNAMIC_THRESH_P` | 0.0 (off) — log per run |
| Split | ratios / policy | 0.7/0.1/0.2 chronological (loader default) |
| Upstream | VAE + summarizer | identical files across all arms (automatic via skip logic) |

### Branch decision (plan §8, decided by cell (a)) — ✅ DECIDED 2026-07-05

Cell (a) = **0.3710 ± 0.0117** → the **≈0.36 "reproduction gap" branch**:
- The paper's 0.320 anchor does NOT reproduce in-harness (>4σ away) — retired from
  the narrative; report the reproduction transparently.
- The authors' released-checkpoint reference (0.367, `training_summaries`) IS
  reproduced (<0.4σ) — the harness is faithful to the released artifacts.
- **Story: "match + certify + strictly generalize"** — headline is (d) ≈ (c) ≈ (a)
  with certification; effort shifts to the synthetic benchmark and theorem
  experiments (H2 / T1 / T2) rather than CRPS-chasing.

*(The two hypothetical branches, kept for the record: ≈0.32 → "close the gap",
Tier-1 calibration critical path; ≈0.36 → the branch above.)*

### Reproduction record (how these exact numbers were produced)

**Environment.** Single NVIDIA H100 PCIe (driver 535.309.01, CUDA 12.2), conda env
`llapdiff` (Python 3.11, torch 2.5.1+cu121), repo root as CWD, `pip install -e .`
run beforehand (registers `llapdiff-train`). Code state: the combined branch with
ALL fixes as of 2026-07-05 — crucially the ε-init fix (see the red box in §0);
`python -m pytest tests/ -q` → 290 passed immediately before launch.

**Pre-flight (all required).**
1. `nvidia-smi` + `python -c "import torch; print(torch.cuda.is_available())"` → True.
2. GPU smoke: build + forward + backward of all five model variants (chirp,
   chirp+head, lti−head, chirp+UQ, chirp+growth) on CUDA — unit tests are CPU-only,
   so this is the first CUDA exercise of the new code paths.
3. Shared frozen stage-1/2 artifacts already staged (identical files for every arm
   and seed — the parity requirement):
   `ldt/vae/saved_model/physionet/pred-12_ch-16_entity_elbo.pt` and
   `ldt/summarizer/saved_model/physionet/12-16-summarizer.pt` (both dated 2026-05-25;
   stage-3-only fixes do not touch them).
4. G4 parity checklist confirmed at defaults (table below): 25-sample CRPS, DDIM 64,
   η=0 (`GEN_ETA=0.0`), EMA 0.999, guidance (1.0, 2.0)/1.0, dynamic thresholding off.

**Launch.** Two sequential campaign scripts (kept in-repo, verbatim what ran):
`ldt/scripts/run_g3_phase1_cell_a.sh` (cell (a) × seeds 0–4 — run FIRST; it alone
decides the branch) and `ldt/scripts/run_g3_phase2_cells_dcb.sh` (cells (d), (c),
(b) × seeds 0–4, in that order). Each run is exactly the §1 command with
`--summary-json ldt/results/g3/summaries/physionet_h12_<cell>_s<s>.json` and a
per-run log under `ldt/results/g3/logs/`. Runs are sequential on one GPU.

**Observed runtime.** ~85–160 s per run (≈110 s median); the whole 20-run campaign
took ≈33 min. Every run early-stopped at epoch ≈62 (EARLY_STOP=20 with per-epoch
evals — the same selection rule in every arm, so this is parity-consistent).

**Mid-campaign audit (do not skip for chirp arms).** On the first chirp run, verify
the pole-coefficient head actually trained (§0 red box):
`to_coeffs absmax = 1.3e-2` on cell-(d) seed 0 (vs 1e-4 init) → learning confirmed.
A value of exactly 0.0 means a pre-fix binary/code state — abort and re-check.

**Metrics.** The numbers above are the trainer's own final-test `eval_stats`
(best checkpoint reloaded, 25 samples, parity defaults), read from the summary
JSONs — aggregate with mean ± std over seeds. The separate
`llapdiff-checkpoint-eval` pass (adds the imputation cases) was NOT run for the
branch decision; run it in bulk against the routed `_best.pt` checkpoints when the
imputation table is needed.

**Reproducibility caveats.** Same-seed reruns reproduce these numbers
*statistically*, not bitwise: `set_torch(seed)` fixes init and batching, but GPU
kernels are nondeterministic by default (set `DETERMINISTIC=True` in `config.py`
for bit-exact runs, at a speed cost). Using different stage-1/2 artifact files (or
retraining them) shifts all arms together — keep the same VAE/summarizer files for
any comparison against this table.

---

## 2. Phase 1 — H1 headline: NOAA-UK h=168 (10 seeds) + replications

Same four arms, `--dataset-key noaa_uk --preds 168`, seeds 0–9 (40 stage-3 runs;
these are the expensive ones — context 336, K=256, 600 epochs). Replications:
`noaa_us --preds 168` and `bms_air --preds 168` at 5 seeds each.

```bash
DS="--dataset-key noaa_uk --preds 168"
for s in $(seq 0 9); do
  llapdiff-train $DS --seed $s
  llapdiff-train $DS --output-head off --seed $s
  llapdiff-train $DS --modal-type chirp --output-head on --seed $s
  llapdiff-train $DS --modal-type chirp --seed $s
done
```

Pre-registered metric: test CRPS on NOAA-UK h=168 (secondary MSE, tertiary PIT).
Freeze `PREREG.md` (plan §1) **before** these runs; select every hyperparameter
on val, touch test once per final config per seed.

**H2 (synthetic ground-truth chirp benchmark)** — **runnable** via
`llapdiff-synthetic-chirp` (after `pip install -e .` to register the entry point):

```bash
llapdiff-synthetic-chirp \
  --tasks synthetic_linear_chirp synthetic_quadratic_chirp \
          synthetic_ramp_damping_up synthetic_ramp_damping_down synthetic_growth_decay \
  --arms lti chirp --seeds 0 1 2 \
  --sweep-period 144 \
  --output-root ldt/results/chirp_benchmark
```

Per (task, arm, seed) it generates a shared-pole cache (`shared_poles=True`, one
ground-truth pole function per joint row), trains both arms on **shared** frozen
VAE/summarizer, writes forecast CRPS/MAE/MSE (`chirp_benchmark_raw/summary.csv`),
and — for **both arms** — scores + plots recovered-vs-truth pole trajectories
(`recovery/*.json`, `figures/*_pole_recovery.pdf` + the cross-window
`*_pole_recovery_series.pdf`). Add `synthetic_freq_shift`
to `--tasks` for the piecewise regime-switch case; `--smoke` for a 1-epoch check.
Geometry note: the purged split needs `val_ratio·(L−K−H+1) > H` — the default
`--series-length 768` satisfies it (288 does NOT; the tool validates and explains).

> 🔴 **Pole-recovery rewrite (2026-07-20; see `h2_pole_recovery_problems_fixes.md`).**
> The original recovery figure ranked modes by **coefficient variation energy** and
> plotted junk: with K=256 modes and a ~1-D target, zero-residue modes get no
> gradient, their pole coefficients drift, and the criterion selects exactly those
> (the diagnosed figures showed ρ≈87–118/step, ω pinned at 0.9π — modes whose
> output-energy share is exactly 0.0). The tool now captures residues + poles from
> the **final denoising step of the evaluated generation** (`modal_capture`), ranks
> by output contribution `E_k = mean_t e^{−2ρ̄ₖ}‖θₖ‖²`, reports the E-weighted
> **effective trajectory over all modes** as the primary curve
> (`omega_eff_rmse`/`rho_eff_rmse` in CSV + JSON, both arms), stratifies recovery
> windows across the test span, and enforces a **selection-validity gate**
> (`--recovery-share-threshold`, default 0.5): below it the figure is watermarked
> "SELECTION INVALID" and must not be used as evidence (prereg third category —
> tool fix + rerun, not a scientific conclusion). Old `omega_rmse_best_*` metrics
> and pre-2026-07-20 recovery figures/JSONs are meaningless — regenerate.
> Diagnostic on the trained linear-chirp seed-0 checkpoint: top-4 contribution
> modes carry 98–99.5% of output energy at sane poles (ω_eff RMSE ≈ 0.05–0.09
> rad/step vs truth) — the model was fine, the figure lied. Genuine model
> findings now visible: ρ_eff overestimated (~0.1–0.2 vs truth 0.003) and ω_eff
> under-tracks late windows. NOTE: those checkpoints were trained with the leaked
> base-config `CHIRP_NUM_BASIS=256` (a knob-class-2 edit; 256 basis functions
> across a 48-step window is ~10× past Nyquist) — recovery JSON now records
> `chirp_num_basis`; consider 8–16 at H=48 for the paper runs.

> 🔴 **STALE-CHECKPOINT TRAP (2026-07-20) — check this before believing any H2 result.**
> The v3 sweep campaign silently evaluated **07-14 checkpoints** on 11 of 30 runs (all
> `linear_chirp`, 5/6 `quadratic_chirp`) while each run's freshly trained model went
> unused. `_best_raw.pt` is written only when `EMA_COMPARE_EVERY > 0` — the benchmark
> sets it to 0 — but the trainer reported the path via a bare `.exists()` and the tool
> preferred `_raw` first, so a 6-day-old file won. **Audit one-liner:** compare the
> `checkpoint` column's mtime in `chirp_benchmark_raw.csv` against the run time. Fixed
> 2026-07-21 (the trainer now reports only checkpoints written by the current run; the
> tool takes its `loaded_checkpoint`). ⚠️ `llapdiff-checkpoint-eval --checkpoint-kind
> auto` still searches **raw first** — the same trap for manual evals.
>
> 🔴 **Artifacts are keyed by (task, seed), NOT by cache.** Adding `--sweep-period`
> created new data but silently reused the previous cache's VAE + summarizer. An
> `artifact_cache.json` fingerprint now fails loudly on mismatch; pass
> `--recompute-artifacts` (or delete stage-1/2) when the data changes.

> 🔴 **H2's TARGETS WERE UNRECONSTRUCTIBLE (root cause, 2026-07-21) — use `--phase-spread`.**
> Every H2 result up to this point sat under a **0.62 round-trip ceiling**: encoding the
> TRUE targets and decoding them straight back (no forecasting) reproduced them at only
> corr 0.62, so no denoiser could do better. Cause: `latent_vae.py` **mean-pools across
> the entity axis** before `mu_head`, while the generator draws `phase0 ~ U(0, 2π)`
> independently per entity — 64 sinusoids at 64 phases cancel, and the carrier is gone
> before it reaches the latent. It is upstream of the bottleneck, so **latent width does
> not help** (measured: 24 → 64 channels changed corr 0.586 → 0.591), nor does input
> dropout (0.20 → 0.0: 0.620 → 0.586), nor fewer entities (8 entities: 0.479, worse).
>
> | phase spread | round-trip corr | val_recon |
> |---|---|---|
> | 2π (historical) | 0.620 | 0.0611 |
> | 0 (shared) | 0.863 | 0.0188 |
> | **π/8 ≈ 0.393 (recommended)** | **0.857** | 0.0202 |
>
> `--phase-spread 0.393` keeps entity diversity and recovers ~3.2× lower reconstruction
> error at identical architecture/capacity/schedule. It tags the cache (`_phase-0.393`)
> and is recorded per result row. **All H2 caches and every model trained on them must be
> regenerated with it** — prior chirp-vs-LTI comparisons were made through a handicapped
> stage 1. Symptoms this explains: forecast-vs-target correlation of only +0.06…+0.32 at
> h=48, a forecast oscillating ~4× too fast, and flat recovered pole trajectories.
> Residual headroom remains (0.86 vs the ~0.99 generator noise permits), and latent
> capacity may matter NOW that pooling no longer destroys the signal — re-run the
> capacity sweep on a reswept, phase-narrowed cache before concluding anything about K.

> 🔴 **HORIZON-BLIND ρ INIT (2026-07-22) — affects BOTH cores, and h=168 worst.**
> Both `LaplaceTransformEncoder` (LTI) and `ChirpModalField` (chirp) initialised
> `ρ ~ U(0.01, 0.2)` with **no horizon term**, but the envelope is `e^{-ρ̄(t)}` so survival
> depends on **ρ·H**: [0.12, 2.4] at h=12 (fine), [0.5, 9.6] at h=48, **[1.7, 33.6] at
> h=168 (NOAA-UK headline) = nearly all modes dead at init**. Measured on trained h=48
> models: every mode at a real signal frequency had ρ ≈ 0.08–0.25 (**26–76× the truth
> 0.0032**), envelope mass 0.03–0.12 — extinct within the horizon — leaving only near-DC
> modes alive (up to 92% of output energy). That is the true cause of the flat recovered
> pole trajectories and the damped "DC-hedging" forecast. ρ is a **null direction** of the
> loss here (true ρ·H = 0.15 ⇒ almost no decay to learn from), so training never corrects
> a bad init. **Both arms were affected equally — prior chirp-vs-LTI ties compared two
> equally crippled models.** Fixed: ρ init anchored to the horizon (`U(0.1, 2.0)/H`) via a
> new `pole_init_horizon` threaded to both cores from `config.PRED`; pre-fix checkpoints
> keep the legacy init. Audit: modes need **ρ ≲ 2/H** (0.042 at h=48, 0.012 at h=168);
> red flag = oscillating modes with `envelope_mass < 0.1`, or recovered ρ spanning the
> init range. **Full write-up: `w_docs/CMD_BUG_REPORT.md` (B6, B7).**

> 🔴 **THE POLE BASIS COULD NOT DRIFT (2026-07-23) — this invalidates the Fig-2 negative.**
> `basis_freqs = linspace(1, M, M)` gives every basis function a **whole number of cycles**
> across the window, so `phi_m(0) = phi_m(L) = 2` is its maximum; with squared (nonnegative)
> coefficients that forces **`omega(0) == omega(L) == sup_t omega(t)` for every mode**
> (measured: |diff| = 2e-7 over 512 modes, `argmax` always at an endpoint). The
> instantaneous frequency could only dip and return — a monotone sweep, which is what the
> chirp truth does across a window, was **outside the function class**. Fitted to the real
> truth over 120 test windows with the model's own basis and its nonnegativity constraint:
> **R² = 0.119 ± 0.266** (legacy) vs **0.990 ± 0.008** (half-integer); the net endpoint drift
> the legacy class sets to zero is **73% of the within-window range**. The five "independent"
> designs (fm-2, fm-4, sweet-spot, period-96, period-32) varied sweep steepness and period but
> never the **shape class**, so they shared one confound and their agreement looked like
> robustness. Fixed by `CHIRP_BASIS="half_integer"` (default): harmonics f = 0.5, 0.5, 1.0,
> 1.0, … paired with ± signs, so a nonnegative combination spans a *signed* multiple of each
> cosine. Every bound is unchanged (φ ∈ [0,2], |φ−1| ≤ 1), Theorem A stays closed-form, and
> each element is still zero-mean over the window (2·f_m integer) as the `centered` ρ basis
> requires. Pre-fix checkpoints `setdefault` to `"integer"`. **Any pre-2026-07-23 Fig-2
> result measured whether the model learns a sweep it could not express — re-run it.**
> Full write-up: `w_docs/CMD_BUG_REPORT.md` (B9-B14); pilot:
> `ldt/scripts/run_fig2_basis_pilot.sh`.
>
> Companion metric defect fixed with it: `omega_eff` weights modes by their **time-averaged**
> energy, so for the LTI core (constant ω_k) its trend slope is **identically 0 by
> construction** — the "LTI fails structurally" panel was a tautology, not evidence. And a
> model that sweeps by redistributing energy **across** constant-frequency modes scores 0
> too; the LTI arm demonstrably does this (instantaneous mean frequency moves 0.188 ± 0.142,
> 59% of the truth's swing, vs 0.023 for chirp). `omega_trend_slope_dyn` /
> `omega_eff_dyn_rmse` (instantaneous weights) are now recorded alongside the static ones.
> **Do not claim "LTI is spectrally static".**

**Modal budget (`--laplace-k`, `--chirp-num-basis`).** Both were previously hardcoded or
inherited from the base config — which is how `CHIRP_NUM_BASIS=256` (≈10× past Nyquist at
H=48) leaked into H2. Keep `M ≤ horizon/2` (the model warns otherwise); `K` controls
identifiability, not smoothness — the synthetic latents are ~rank 2, so at K=256 roughly
8 modes carry R²=0.99 of the output and *which* 8 changes between training runs, making
per-mode poles meaningless. Both are recorded in every result row, and non-default values
tag the denoiser directory so scan cells cannot overwrite each other.

**Recovery honesty knobs.** `--recovery-guidance 1.0` pins CFG for the capture (the TEST
schedule reaches w≈2 exactly at the captured final step, so by default the poles belong to
the conditional half of a guided blend). `--recovery-draws N` averages the E_k weighting
over N DDIM draws — the poles are deterministic given the conditioning but the residues
are not, so a single draw gives a noisy `omega_trend_slope`. The JSON/CSV now carry
`omega_trend_slope` (1 = tracks the sweep, 0 = flat, NaN when truth is constant) and
`forecast_freq_slope` (same regression on the instantaneous frequency of the **forecast**,
which is identifiable however the modes split the signal).

**Within-window sweep (`--sweep-period`, fix-plan P5).** The legacy profiles ramp
over the whole series, so one 48-step horizon sees only ~6% of the sweep —
within a window both arms face near-constant poles and "LTI fails structurally"
**cannot materialize** (confirmed: on legacy caches the LTI arm's constant ω_eff
matches truth as well as chirp's). `--sweep-period 144` (≈ window+horizon) turns
the ramp variable into a triangle wave of that period in absolute time, putting
the full pole excursion inside **every** window, including the tail test windows
the purged split uses. Applies to the four smooth-ramp tasks; piecewise
change-point tasks ignore it. The period is tagged in the cache dir
(`..._sweep-144`); omit the flag for the legacy slow ramp (bit-identical caches),
which remains the right regime for the cross-window stitched figure.

> 🔴 **Sweep AMPLITUDE (`--freq-multiplier`) — `--sweep-period` alone is not enough.**
> A constant-pole (LTI) forecast desyncs from a chirp by a phase error ≈ Δω·H/12 over
> the horizon. At the default `--freq-multiplier 2` the within-window Δω is only ~0.13
> rad/step → ~0.5 rad desync (~0.09 cycle), which a constant pole absorbs — so **LTI
> never structurally fails and the benchmark cannot separate the arms** (confirmed on the
> converged, Gate-0-passing phase-fixed run: chirp ties LTI, and LTI's own constant-ω
> recovery is *valid* and beats chirp's). Raise the multiplier to build the LTI-failure
> regime: `--freq-multiplier 4` → Δω ~0.4 rad/step, ~1.6 rad desync (~0.26 cycle), where a
> constant pole visibly breaks. Cost: ω rises to ~1.0 rad/step (~6 samples/cycle) — still
> reconstruction-safe; `--freq-multiplier 5` (ω→1.3, ~5 samples/cycle) risks the Gate-0
> round-trip, so re-check it. Tagged `..._fm-4`; recorded per result row and summary key.

**Denoiser schedule (`--epochs`, `--early-stop`).** The default (80 epochs, early-stop on
the cheap `val_diag_mse_raw` proxy) stopped the denoiser at **~epoch 19** (val v-MSE 0.60),
a weak forecast that confounds the recovery figure — **any recovery-quality run must train
to convergence.** `--epochs 400 --early-stop 30` auto-switches selection to CRPS and scores
it every 5 epochs; non-default `--epochs` tags the denoiser dir `_ep-NNN` so it does not
clobber the short-schedule checkpoints. Every result row now records `best_epoch`, so a
best_epoch far below `--epochs` (early stopping fired) is a visible undertraining tell.

> 🔴 **GATE 0 first (`llapdiff-stage1-roundtrip`).** Before trusting ANY Fig-2 number,
> confirm the stage-1 VAE reconstructs its own targets: `llapdiff-stage1-roundtrip
> --tasks … --phase-spread 0.393 [--freq-multiplier N] --artifact-root …` PASS/FAILs
> against 0.85 and exits non-zero on failure. Below the gate the pipeline cannot represent
> the signal and every downstream chirp-vs-LTI comparison is meaningless (the phase-2π
> caches sat at 0.62; the `--phase-spread 0.393` fix reaches 0.857). A steeper
> `--freq-multiplier` can re-break the gate — always recheck it after changing difficulty.

**Irregular sampling (the plan's H2 premise, added 2026-07-05).** Signals are
sampled at **Gamma renewal gaps by default**: `--gap-distribution gamma`
(default) with `--gap-mean 1.0 --gap-shape 4.0`, giving i.i.d. gaps with
`Var(Δ) = gap_mean²/gap_shape` — sweep `--gap-shape` (e.g. 16 / 4 / 1) for
low/medium/high gap-variance regimes at fixed mean (shape 1 = Poisson; use
`--gap-distribution regular` for the historical dense grid, which is
bit-identical to pre-change caches and consumes no extra RNG). One grid is
drawn per cache and **shared by all entities** — the joint-panel collate
requires a common query grid per row. The realized gap moments (the Theorem-D
quantities E[Δ], Var(Δ), E[Δ²]) are recorded in the cache `meta.json`, the gap
regime is tagged in the cache path, result rows, and summary grouping, and the
ground-truth `pole_truth/*.npz` now also stores the sample `times` (native
hours). ⚠️ Keep `--gap-mean 1.0` unless you also set `CHIRP_TIME_SCALE ≈
PRED·gap_mean` in config.py — the chirp basis window `L` resolves to `PRED` in
native units, which matches the horizon's time span only at unit mean gap.

The **Prop. A.1 figure** (companion vs normal-form integration error):
`python -m llapdiffusion.tools.plot_companion_vs_normal_form` (verified: companion
error ~4.6 vs normal-form ~2e-11). Pole *trajectory* plots for any real-data chirp
checkpoint are now emitted automatically by `llapdiff-plot-poles`
(`*_pole_trajectories.pdf`).

**H3 (boundary-crossing stress)** uses the existing `llapdiff-synthetic-regime`
tool, **but beware a pre-existing bug in that tool** (found 2026-06-25 while
smoking the chirp benchmark, NOT yet fixed):

> 🔴 **`llapdiff-synthetic-regime` crashes at its own default geometry.** With
> `boundary_crossing` defaults (series 288, window 96, horizon 48) the
> `global_purged_horizon` split has a val band of ~15 window-starts, but a val
> window needs its full 48-step target interval inside the band → val is
> *structurally empty* and the run dies with
> `ValueError: target-interval purged split produced an empty train/val/test split`.
> The `strict_unseen_regime` defaults (432/373) have the same problem
> (val band ~29 < 48). Two constraints for a working geometry:
> (i) `val_ratio · (series_length − window − horizon + 1) > horizon` (non-empty
> val), and (ii) the change point must sit a full `--lookback-steps` **inside
> the test region** (which begins around `window + 0.8·(#window starts)`), or
> the boundary-crossing eval slice is empty. **Verified workaround**
> (`--validate-split-only` passes; 12 crossing windows per asset, the full
> lookback band; test region starts at context-end 594):

```bash
llapdiff-synthetic-regime --protocol-name boundary_crossing \
  --tasks synthetic_freq_shift synthetic_decay_shift \
  --seeds 3407 3408 3409 \
  --series-length 768 --change-point 606 \
  --output-root ldt/results/h3_boundary
```

Sanity-check any other geometry first with `--validate-split-only` and confirm
`test_boundary_crossing_windows_per_asset` > 0 in the emitted
`synthetic_regime_geometry.json` before spending GPU time.

(chirp-vs-LTI arms of H3 need the trained checkpoints from above. The new
`llapdiff-synthetic-chirp` tool is not affected — it defaults to 768 and
validates the geometry with a clear error message.)

---

## 3. Hyperparameters — what to change, and WHERE (the footgun)

`apply_dataset_preset` re-stamps ~40 config attributes **twice per run**, so a
runtime `config.X = value` is silently reset for stamped values. Three classes:

**(1) CLI-tracked — change per run with flags** (survive the re-stamp):

| Knob | Flag |
|---|---|
| Prediction target (v/x0/eps) | `--predict-type` |
| Dynamical core | `--modal-type` |
| Output head | `--output-head` |
| Seed | `--seed` |
| Batch size (dates) | `--batch-size` |
| Target column(s) | `--target-col` / `--target-cols` |
| Context coverage stress | `--coverage` |
| Aux completion mixing | `--target-mask-aux-*` |

**(2) Base-config only — edit `llapdiffusion/configs/config.py`; NOT preset-stamped,
so an edit holds for every subsequent run** (record the value per run in your log):

| Knob | Default | Used for |
|---|---|---|
| `NUM_EVAL_SAMPLES` | 25 | CRPS protocol (keep 25 for reported numbers) |
| `GEN_STEPS` | 64 | DDIM steps (U1 sweep: 16/32/64) |
| `GUIDANCE_STRENGTH`, `GUIDANCE_POWER` | (1.0, 2.0), 1.0 | U1 guidance sweep |
| `DYNAMIC_THRESH_P`, `DYNAMIC_THRESH_MAX` | 0.0, 1.0 | thresholding + clip-rate checks |
| `USE_EMA_EVAL`, `EMA_DECAY` | True, 0.999 | EMA on/off ablation |
| `CHIRP_NUM_BASIS` | 8 | Tier-2 M sweep (consider 4 at h=12 — 8 cycles across 12 steps is near-Nyquist) |
| `CHIRP_RHO_MIN` | 1e-4 | Tier-2 ρ_min sweep |
| `CHIRP_TIME_SCALE` | None (→ run's PRED) | Tier-2 time-constant sensitivity |
| `CHIRP_USE_MLP_RESIDUAL` | False | keep False for certified arms |
| `CHIRP_UQ_HEAD` | False | Theorem-C analytic UQ head (U2; needs chirp + x0 + certified path) |
| `CHIRP_GROWTH_BUDGET` | 0.0 | Theorem-B′ growth budget c_g (T2 sweep {0, log 2, log 5}); 0 = Thm B exactly |
| `CHIRP_PARAMETERIZATION` | "p_exact" | Phase-4 pole-field ablation: p_exact / p_mono / p_grid |
| `CHIRP_BASIS` | "half_integer" | Pole-basis harmonics. `"integer"` = the legacy f=1..M set, which pins ω(0)=ω(L)=sup ω so the poles **cannot drift** across the window (red box below). Also `--chirp-basis`, tags the denoiser dir `_cb-<mode>` |
| `CHIRP_RHO_BASIS` | "centered" | ρ variation basis. `"centered"` (φ−1 = cos, zero mean over the window ⇒ variation adds **no net decay**) vs `"nonneg"` (legacy 1+cos, whose mean-1 basis couples ρ variation to ρ mean and extinguishes oscillatory modes when the true decay is small). Also `--chirp-rho-basis` |
| `CHIRP_RHO_MAX_SCALE` | 4.0 | ρ_max = scale/horizon — the decay analogue of the Nyquist cap on ω (a mode needs ρ·H ≲ a few to survive the window). Also `--chirp-rho-max-scale`; the Fig-2 campaign used 1 |
| `CHIRP_COEFF_L2` | 0.0 | Tier-2 L2 on the pole-variation coefficients (shrinks toward LTI); training-only, chirp-only, all three parameterizations; growth head excluded (c_g governs it) |
| `DIFF_LOSS_MODE` | "mse" | "gaussian_nll" trains mean+variance jointly (needs `CHIRP_UQ_HEAD`) |
| `TRAIN_T_SAMPLER` | "uniform" | "max_only" = the U3 one-shot (no-diffusion) regression arm |
| `DETERMINISTIC` | False | set True for bit-exact reruns (slower). Also forces `ALLOW_TF32` off |
| `ALLOW_TF32` | True | TF32 tensor cores for fp32 matmuls (~1.3x on the denoiser; 10-bit mantissa on the multiply, fp32 accumulate). Was effectively False everywhere before 2026-07-28 |
| `DATALOADER_NUM_WORKERS` | 8 | loader workers (+ `_PERSISTENT_WORKERS`, `_PREFETCH_FACTOR`). Was 0: nothing passed it, so loaders ran in the training process (16.4 -> 7.6 ms/batch on noaa_uk) |
| `EVAL_STEPS` / `EVAL_NUM_SAMPLES` / `EVAL_MAX_BATCHES` / `EVAL_SEED` | **unset** / unset / 0 / 4242 | **in-training validation only** — see the cost box below. Left unset in `config.py` on purpose: they move checkpoint *selection*, and were validated on noaa_uk h=168 only. `finetune_noaa_uk.sh` exports `LLAPDIFF_EVAL_*` (16 / 5 / 48 / 4242, `VAL_DIAG_EVERY=5`) and `run_trial.py` applies them, so the campaign gets the 61× and every other dataset keeps the full protocol |
| `DIFF_PRECOMPUTE_VERIFY_PLAN` | False | skip the 3-split rescan at trial startup on a cache hit (~25 s/trial) |

> 🟢 **THE VALIDATION PROTOCOL IS NOT THE REPORTED PROTOCOL (2026-07-28).** A full
> noaa_uk h=168 val CRPS eval is `146 batches x 25 samples x 64 DDIM steps x 2 forwards`
> (CFG runs conditional **and** unconditional) = **110 min**, every 5 epochs — **>90% of a
> trial**, which is why the campaign was a ~60-day job, not the 29 days this file used to
> claim (that estimate charged one forward per step).
>
> But the trainer's val CRPS only picks the best epoch and drives early stopping.
> `_sampling_kwargs` resolves `EVAL_*` for exactly those two call sites and `TEST_*` for
> everything that produces a **reported** number (the final test read,
> `llapdiff-checkpoint-eval`, `finetuning/eval_sampling.py`, `--phase final`). So the
> `EVAL_*` values above cut it **61x** and change nothing you publish; `final_eval.py`
> refuses to spend the test split if the TEST protocol ever disagrees with the requested
> sampling. Details in `DEVELOPER_GUIDE.md` §7.6.
>
> ⚠️ Before trusting a cheap protocol on a NEW setting, check it still **ranks** the same:
> score >=8 checkpoints under both protocols on val and require Spearman rho >= 0.9.
> Otherwise it selects the wrong hyperparameters, faster.
>
> ⚠️ Do **not** raise `DOWNSTREAM_EVAL_EVERY` without lowering `EARLY_STOP`: early stopping
> counts **evals**, so at every=10 the patience becomes 200 epochs and trials get LONGER.

**Launch order for a noaa_uk campaign.** `LAPLACE_K=256` means 512 modal tokens and is
essentially the whole forward cost (K=128 is 1.84x faster, K=64 is 3.1x), and the project's
own analysis says ~8 modes carry 99% of output energy. Probe it FIRST, because a smaller K
makes every later trial cheaper:

```bash
RUN_TAG=k_probe STAGES="laplace_k" EXTRA_ARGS="--include-tier3" bash finetune_noaa_uk.sh
# 2 trials: baseline K=256 vs K=128. If they tie on val CRPS, set LAPLACE_K in the noaa_uk
# preset's model_overrides (dataset_defaults.py -- it is preset-stamped) before the main run.
bash finetune_noaa_uk.sh        # 23 trials, selection on val
```

**(3) Preset-stamped — edit the preset row in
`llapdiffusion/configs/dataset_defaults.py` (or wrap `apply_dataset_preset`);
a `config.py` or runtime edit will NOT hold:**

| Knob | Stamped value |
|---|---|
| `EPOCHS` | 600 (per preset) |
| `BASE_LR` | 1.5e-4 |
| `MINSNR_GAMMA` | per dataset |
| `TIMESTEPS` | 1000 |
| `MODEL_WIDTH`, `LAPLACE_K` | 256, 256 (Tier-3 / K-sweep) |

For scripted sweeps of class (3), use the monkeypatch pattern from
`DEVELOPER_GUIDE.md` §3:

```python
import llapdiffusion.pipeline as P
_orig = P.apply_dataset_preset
def patched(cfg, key, *, pred=None):
    out = _orig(cfg, key, pred=pred); cfg.EPOCHS = 300; return out
P.apply_dataset_preset = patched
```

**Tier discipline (plan §7):** Tier-1 knobs (guidance w, DDIM steps, EMA,
predict-type, MinSNR γ) are tuned per-arm with identical grids and trial counts;
Tier-2 (`CHIRP_*`, K, time constant) applies to CMD only. Selection on val only.

---

## 4. Phase 2–3 — status of each experiment

| Item | Status | How to run / what's missing |
|---|---|---|
| U1 guidance/DDIM calibration sweep | **runnable** | `llapdiff-u1-sweep --dataset-key <ds> --pred <H> --checkpoint <ckpt> --guidance 1.0 1.25 1.5 2.0 --steps 16 32 64` — evaluates on **val** (pre-registration rule) and logs the dynamic-threshold clip fraction per cell. The plan's specific clip check: add `--dynamic-thresh-p 0.995` and read `clip_fraction_mean` (recorded per row together with the effective p). First data point (G3 cell-(d) ckpt, physionet h=12, val): clip fraction **0.23%** at p=0.995 — ẑ₀ barely brushes the threshold after the head removal |
| U2 Theorem-C analytic UQ (q_k, p_k⁰, Gaussian NLL, PIT) | **runnable** | train with `CHIRP_UQ_HEAD=True`, `DIFF_LOSS_MODE="gaussian_nll"` (config.py; both base-config, survive presets) + `--modal-type chirp --predict-type x0`; then `llapdiff-uq-eval --dataset-key <ds> --pred <H> --checkpoint <ckpt>` reports **latent** PIT calibration error/reliability/NLL/RMSE **and, by default, the data-space comparison**: the analytic law propagated through the decoder (latent Gaussian draws → decode, scored by the *unchanged* `evaluate_regression` — same masking/CRPS estimator/ensemble size) vs the sampled-diffusion baseline, with wall-clock for both (`analytic_speedup_x`; ~8× at 5 samples on the smoke, grows with DDIM steps). Flags: `--num-samples` (ensemble for BOTH arms; default 25), `--skip-sampled`, `--latent-only`, **`--weights {raw,ema}`**. ⚠️ **Pass `--weights ema`**: the tool goes through `_load_stack`, which reads `payload["model"]` — RAW weights even in `*_best_ema.pt` (bug B16) — so the default `raw` reports a raw-weight number while every other phase of this project uses EMA. Measured on an existing checkpoint: summed per-tensor `max\|raw − ema\|` = 0.696, i.e. the flag is not cosmetic. ⚠️ **Read the NLL warm-start warning below before launching.** |
| U3 one-shot NLL (no diffusion) arm | **runnable — use the driver** | The three arms are **two checkpoints**, not one: A1 (`data_space_sampled`) and A2 (`--mean-source ddim`) come from the diffusion-trained UQ model; A3 (`--mean-source oneshot`) needs its **own** training run with `TRAIN_T_SAMPLER="max_only"`. ⚠️ On that A3 checkpoint the sampled baseline is **invalid** (reverse DDIM walks timesteps the model never trained on) — pass `--skip-sampled`. Rather than hand-editing config.py per arm, use **`python finetuning/run_u3_uq.py {plan,train,gates,eval,report}`**, which drives all three stages through `run_trial.py` config overrides (no config.py edit), threads each seed's `DIFF_INIT_CKPT` warm start, enforces `--skip-sampled`/`--weights ema`, and aggregates to `finetuning/results/uq_u3/RESULTS.md`. Pre-registration: `w_docs/PREREG_U3.md` |
| T1 pole-invariance across gap regimes | **runnable** | `llapdiff-t1-poles --dataset-key <ds> --pred <H> --checkpoint <chirp ckpt> --coverages 0.0 0.2 0.4 0.6 0.8` — per-regime trajectory distance vs baseline + observed gap moments + Eq.-(8) implied multipliers, CSV/JSON + overlay figure. Read: distances ~flat, multipliers shift |
| T2 growth budget c_g ∈ {0, log2, log5} | **runnable** | set `CHIRP_GROWTH_BUDGET` in config.py per arm (base-config; e.g. `math.log(2)`), train chirp as usual; c_g=0 is exactly Theorem B (no γ-head built). Bound becomes `e^{c_g}·e^{-ρ_min t̃}·Σ√(…)`. The synthetic `synthetic_growth_decay` task (budget log 2) is the matched benchmark case |
| T3 imputation vs CSDI | **runnable** | `--imputation-random-mask-ratio 0.30` at eval; CSDI side via `llapdiff-baselines csdi-imputation` |
| T4 efficiency table | **runnable** | `llapdiff-t4-timing --dataset-key <ds> --checkpoints <pred-24 ckpt> <pred-48 ckpt> … --repeats 5` — DDIM wall-clock + single-forward time per horizon (real conditioning), CSV |
| Phase-4 P-exact / P-mono / P-grid ablation | **runnable** | set `CHIRP_PARAMETERIZATION` in config.py per arm: `"p_exact"` (default, closed-form antiderivative), `"p_mono"` (monotone integrated poles, closed-form derivative), `"p_grid"` (pointwise poles + trapezoid on the query grid — its integration error on wide gaps is part of the story). Recorded in checkpoint metadata; composes with growth/UQ heads |

### 🟡 Warning: do NOT train Gaussian-NLL from scratch (observed 2026-06-25)

> **Finding (1-epoch UQ smoke on physionet pred-4):** training `gaussian_nll`
> from random init produced a wildly overconfident, wrong model — latent RMSE
> **10.8** (vs ~1 for "predict zero" on standardized latents), mean predicted
> std **0.16**, PIT calibration error **0.22**, central-interval coverage ≈ **0**
> at every nominal level. Mechanism: the NLL's mean gradient is
> `(pred − target)/σ²`, and the UQ head initializes with small variance
> (`p0 = q = 0.01`), so early training gets enormous mean gradients that blow up
> the mean before the variance can adapt — a classic NLL failure mode, not a
> code bug.
>
> **Recipe (two-stage warm start, code-supported):**
> 1. Train the plain chirp MSE arm first — cell (d),
>    `llapdiff-train … --modal-type chirp --predict-type x0 --seed <s>`.
> 2. In `config.py` set `CHIRP_UQ_HEAD = True`, `DIFF_LOSS_MODE = "gaussian_nll"`,
>    and `DIFF_INIT_CKPT = "<path to the stage-1 best checkpoint>"`
>    (all three are base-config, not preset-stamped), then rerun the same
>    training command. The loader tolerates exactly the missing UQ-head keys
>    (`_load_diff_init_state`, added 2026-06-25 — before that, `DIFF_INIT_CKPT`
>    into a UQ model crashed with strict-load missing keys) and initializes the
>    variance head fresh on top of the trained mean.
>
> Confirm in the log: `[init] DIFF_INIT_CKPT lacks the UQ head; kept fresh init
> for: […]` (run with `--verbose`). Then check `llapdiff-uq-eval`: predicted std
> should be O(target std), coverage near nominal — not the ≈0 signature above.
>
> ### 🔴 The warm start is NOT sufficient on its own — also set `CHIRP_UQ_INIT_VAR` (2026-07-30)
>
> The recipe above fixes the **mean** init and leaves the **variance** init at the
> library default `CHIRP_UQ_INIT_VAR = 1e-2`. On physionet h=12 that puts the initial
> predicted variance at **0.0011** against a squared error of **2.675** — a **≈2415×**
> mismatch. The variance head cannot climb from 0.01 to the residual scale during
> training, so the arm ends up catastrophically **overconfident** (seed 0, val, EMA):
>
> | `CHIRP_UQ_INIT_VAR` | PIT cal. error | coverage @ 0.9 | mean predicted std | data CRPS |
> |---|---|---|---|---|
> | 1e-2 (library default) | 0.2210 | **0.034** | 0.096 | 0.2552 |
> | **25.0** (calibrated) | **0.0389** | **0.872** | 4.09 | 0.2543 |
>
> (target std 1.46.) ⚠️ **CRPS does not reveal this** — it moved by <0.001 while coverage
> went from 3% to 87%. Never accept an NLL arm on CRPS alone; read the PIT/coverage numbers
> from `llapdiff-uq-eval`. Set `CHIRP_UQ_INIT_VAR` near the residual scale (25.0 for
> physionet h=12; re-measure per dataset/horizon). Default stays 1e-2, so nothing
> pre-existing moves. The warm start itself was verified fine — a control without
> `DIFF_INIT_CKPT` sits **51×** further from S1.
>
> **Two measurement traps found while establishing the above — read before diagnosing:**
>
> 1. **`val_diag_mse_raw` is not always an MSE.** `evaluate_val_diagnostics` reports
>    `stats["raw_loss"]`, which is the MSE under `DIFF_LOSS_MODE="mse"` but the **NLL**
>    under `"gaussian_nll"`. Comparing the two across modes is meaningless (it briefly
>    suggested a 340× mean degradation that does not exist). Compare within a mode only.
> 2. **The latent mean is uninformative in *every* arm, including plain MSE.** Like-for-like
>    (t = T/2, identical noise, EMA): latent x0 MSE **5.72** (S1 chirp-MSE), **5.48**
>    (NLL @1e-2), **4.28** (NLL @25), **5.25** (one-shot @25) against a predict-zero
>    baseline of **2.16**, with correlation ≈ **−0.05** throughout. So the NLL does not
>    damage the mean — the mean was never informative. This is the open
>    "denoiser barely uses its conditioning" issue (§3), and it sits upstream of U2/U3.

Every experiment in the plan (Phases 0–4) is now runnable from this branch —
no [needs implementation] items remain. Bring the G2/G3 CRPS table back to
decide the §8 branch and prioritize the rest of the queue.

---

## 5. Bookkeeping rules (from the plan — do not skip)

1. **Pre-registration**: freeze `PREREG.md` (metric, seeds, selection rule,
   parity checklist) before Phase 1. Test is touched once per final config/seed.
2. **Table hygiene**: 0.469 (naive Option A) and pre-fix 0.367 never appear in
   paper tables.
3. **Symmetric architecture**: every no-head arm uses `--output-head off` (or
   chirp `auto`); every +head arm uses the original head. Never mix.
4. **Log per run**: seed, arm flags, guidance/threshold values, commit hash,
   and the routed checkpoint path. `--summary-json ldt/results/<run>.json` on
   every training command gives you most of this for free.
5. **Claims language**: 1-seed deltas are "directionally encouraging", nothing
   more, until the seeded table exists.
