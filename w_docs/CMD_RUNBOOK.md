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
  --output-root ldt/results/chirp_benchmark
```

Per (task, arm, seed) it generates a shared-pole cache (`shared_poles=True`, one
ground-truth pole function per joint row), trains both arms on **shared** frozen
VAE/summarizer, writes forecast CRPS/MAE/MSE (`chirp_benchmark_raw/summary.csv`),
and for the chirp arm scores + plots recovered-vs-truth pole trajectories
(`recovery/*.json`, `figures/*_pole_recovery.pdf`). Add `synthetic_freq_shift`
to `--tasks` for the piecewise regime-switch case; `--smoke` for a 1-epoch check.
Geometry note: the purged split needs `val_ratio·(L−K−H+1) > H` — the default
`--series-length 768` satisfies it (288 does NOT; the tool validates and explains).

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
| `CHIRP_COEFF_L2` | 0.0 | Tier-2 L2 on the pole-variation coefficients (shrinks toward LTI); training-only, chirp-only, all three parameterizations; growth head excluded (c_g governs it) |
| `DIFF_LOSS_MODE` | "mse" | "gaussian_nll" trains mean+variance jointly (needs `CHIRP_UQ_HEAD`) |
| `TRAIN_T_SAMPLER` | "uniform" | "max_only" = the U3 one-shot (no-diffusion) regression arm |
| `DETERMINISTIC` | False | set True for bit-exact reruns (slower) |

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
| U2 Theorem-C analytic UQ (q_k, p_k⁰, Gaussian NLL, PIT) | **runnable** | train with `CHIRP_UQ_HEAD=True`, `DIFF_LOSS_MODE="gaussian_nll"` (config.py; both base-config, survive presets) + `--modal-type chirp --predict-type x0`; then `llapdiff-uq-eval --dataset-key <ds> --pred <H> --checkpoint <ckpt>` reports **latent** PIT calibration error/reliability/NLL/RMSE **and, by default, the data-space comparison**: the analytic law propagated through the decoder (latent Gaussian draws → decode, scored by the *unchanged* `evaluate_regression` — same masking/CRPS estimator/ensemble size) vs the sampled-diffusion baseline, with wall-clock for both (`analytic_speedup_x`; ~8× at 5 samples on the smoke, grows with DDIM steps). Flags: `--num-samples` (ensemble for BOTH arms; default 25), `--skip-sampled`, `--latent-only`. ⚠️ **Read the NLL warm-start warning below before launching.** |
| U3 one-shot NLL (no diffusion) arm | **runnable** | same as U2 plus `TRAIN_T_SAMPLER="max_only"` in config.py (trains at the pure-noise step ⇒ conditional regression); the pre-registered three-way comparison is now one tool on the same split/ensemble/seed: sampled-diffusion (`data_space_sampled`), diffusion + analytic UQ (`--mean-source ddim` → `data_space_analytic`), one-shot NLL (`--mean-source oneshot` → `data_space_analytic`) |
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
