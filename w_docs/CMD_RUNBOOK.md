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
python -m pytest tests/ -q          # expect: all passed (241 at time of writing)

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

> **Do not reuse any chirp checkpoint trained before 2026-06-25.** The basis
> rescale (Finding 2) and head change (Finding 1) changed the function class;
> old chirp checkpoints load but are not comparable. The lab numbers 0.469 and
> pre-fix 0.367 are excluded from all paper tables (plan §6, table hygiene).

---

## 1. Phase 0 — G2 + G3: the 2×2 factorial on PhysioNet h=12 (5 seeds)

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

Collect test-CRPS into this table (mean ± std over the 5 seeds):

| | + head | − head |
|---|---|---|
| **LTI** | (a) ____ ± ____ | (b) ____ ± ____ |
| **Chirp** | (c) ____ ± ____ | (d) ____ ± ____ |

**Reads to look for** (plan §2 G3): (d) > (b) → time variation carries accuracy;
(c) ≈ (d) → the head is redundant once poles vary (the kill-shot); (d) ≈ (a) →
"delete the hack, keep the accuracy".

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

### Branch decision (plan §8, decided by cell (a))

- **(a) ≈ 0.32** (paper number reproduces) → "close the gap": Tier-1 calibration
  (guidance/DDIM sweep) is the critical path.
- **(a) ≈ 0.36** (reproduction gap) → "match + certify": retire the 0.320 anchor,
  headline is (d) ≈ (c) ≈ (a) with certification; shift effort to the synthetic
  benchmark and theorem experiments.

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
| U1 guidance/DDIM calibration sweep | **runnable** | `llapdiff-u1-sweep --dataset-key <ds> --pred <H> --checkpoint <ckpt> --guidance 1.0 1.25 1.5 2.0 --steps 16 32 64` — evaluates on **val** (pre-registration rule) and logs the dynamic-threshold clip fraction per cell |
| U2 Theorem-C analytic UQ (q_k, p_k⁰, Gaussian NLL, PIT) | **runnable** | train with `CHIRP_UQ_HEAD=True`, `DIFF_LOSS_MODE="gaussian_nll"` (config.py; both base-config, survive presets) + `--modal-type chirp --predict-type x0`; then `llapdiff-uq-eval --dataset-key <ds> --pred <H> --checkpoint <ckpt>` reports latent PIT calibration error, reliability curve, NLL, RMSE. ⚠️ **Read the NLL warm-start warning below before launching.** |
| U3 one-shot NLL (no diffusion) arm | **runnable** | same as U2 plus `TRAIN_T_SAMPLER="max_only"` in config.py (trains at the pure-noise step ⇒ conditional regression); eval with `llapdiff-uq-eval --mean-source oneshot` (the diffusion arms use `--mean-source ddim` for the same comparison) |
| T1 pole-invariance across gap regimes | **mostly runnable** | training with `--coverage 0.0/0.2/…/0.8` works today; per-checkpoint trajectories via `llapdiff-plot-poles` / `extract_chirp_pole_trajectories`; only the cross-regime distance/Eq.-(8) comparison script is [needs implementation] |
| T2 growth budget c_g ∈ {0, log2, log5} | **[needs implementation]** | γ-head (Theorem B′) + `CHIRP_GROWTH_BUDGET` config |
| T3 imputation vs CSDI | **runnable** | `--imputation-random-mask-ratio 0.30` at eval; CSDI side via `llapdiff-baselines csdi-imputation` |
| T4 efficiency table | **mostly runnable** | time `llapdiff-checkpoint-eval` across horizons; a dedicated timing script is [needs implementation] |
| Phase-4 P-mono / P-grid parameterizations | **[needs implementation]** | alternative pole fields |

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

When you have results from Phase 0/1 and want the [needs implementation] items
built, bring the CRPS table — the §8 branch decides which of them are on the
critical path.

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
