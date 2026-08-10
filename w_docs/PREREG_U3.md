# Pre-registration — U3, the matched-synthesizer UQ ablation

> 🔴 **SUPERSEDED as a live pre-registration, and its replacement is UNWRITTEN.**
>
> This file is frozen to **PhysioNet h=12**, which is now settled as a smoke-test cell only
> (13 train windows against 11.75 M parameters, B19). Its gate outcomes are PhysioNet outcomes
> and do not transfer.
>
> **The campaign now runs on noaa_uk h=168 with no frozen pre-registration.** Recovery plan
> Phase 4 requires one *before the test split is touched*, as
> `w_docs/PREREG_U3_noaa_uk_h168.md`. Until it exists, **val only**. See §5 for what the new
> file must fix.
>
> Kept because §1–§4 define the arm design that is still in force, and §6's scope limits still
> apply verbatim.

**Frozen:** 2026-07-30. **Driver:** `finetuning/run_u3_uq.py` (tag `uq_u3`), arm `d`.

---

## 1. The question

Does the diffusion process earn its cost, given that the chirp-modal synthesizer already
supplies a closed-form Gaussian predictive law (Theorem C)?

Three arms share the **identical synthesizer** and differ only in how the law is obtained:

| Arm | Mean | Uncertainty | Checkpoint |
|---|---|---|---|
| **A1** diffusion + sampled UQ | 25-draw DDIM ensemble | empirical ensemble spread | S2 |
| **A2** diffusion + analytic UQ | ~~deterministic DDIM x₀~~ → **the same 25-draw ensemble mean** (fix 1d) | Thm-C closed form, decoded | S2 (same file as A1) |
| **A3** one-shot Gaussian NLL | one forward at t = T−1 | Thm-C closed form, decoded | S3 (`TRAIN_T_SAMPLER="max_only"`) |

> **Amended 2026-08-10.** A2's mean was a single deterministic DDIM pass, which is **one draw**
> from the predictive distribution rather than its mean — so A1 and A2 differed in *location*
> before any UQ question was asked (measured: MSE 4.25× apart on the same weights). A2 now uses
> the matched ensemble mean. Consequence: **A2 costs what A1 costs**, so A1-vs-A2 is
> "calibration at matched cost" and the speedup claim moves to **A1 / A3**.

## 2. Metrics

- **Primary:** data-space **CRPS**, 25 samples, scored by the unchanged `evaluate_regression`
  (identical masking, estimator, ensemble size and `generator_seed` across all arms).
- **Secondary:** MAE, MSE, wall-clock, **denoiser passes per batch** (exact where wall-clock is
  not).
- **Calibration, all three arms:** PIT-ECE, central-interval coverage, mean interval width —
  in **data space**, by one estimator (`ensemble_pit`). Latent Gaussian NLL / PIT / predicted std
  are reported for A2/A3 as the exact-Gaussian cross-check, **not** as those arms' entry in a
  shared column.

  > **Amended 2026-08-10.** The original excluded calibration for A1 ("no closed-form law"). A
  > closed form is not required — `pit_calibration_error` and `reliability_curve` are
  > law-agnostic. Without this the A1-vs-A2 comparison rests on CRPS alone, the metric that is
  > blind to a 3 % → 87 % coverage swing.

## 3. Selection and splits

- **5 seeds** (0–4) per arm; report mean ± std.
- All hyperparameters **fixed in advance** to the `fixed_bugs` incumbent and **identical across
  arms** — no per-arm tuning.
- Gates and diagnostics run on **val**. **The test split is evaluated exactly once per arm per
  seed**, after a prereg for that cell is frozen.

## 4. Protocol parity

| Item | Value |
|---|---|
| CRPS samples | 25 (`NUM_EVAL_SAMPLES`) |
| DDIM steps | 64 (`GEN_STEPS`), η = 0 |
| Guidance | ~~ramp (1.0, 2.0), power 2.0~~ → **1.0, frozen** (monotone-harmful on noaa_uk) |
| Dynamic thresholding | `DYNAMIC_THRESH_P = 0.0` (off) |
| **Weights** | **EMA (decay 0.999)** — `--weights ema`, not the tool default |
| Upstream | identical frozen VAE + summarizer across all arms |
| Prediction type | **x0** for all three (forced: `chirp_uq_head` requires it) |

`--weights ema` is not optional: `_load_stack` reads `payload["model"]`, which holds **raw**
weights even in `*_best_ema.pt` (B16).

## 5. What the noaa_uk replacement must fix

1. **Thresholds against measured baselines.** The old G-c assumed unit-scale latents; the actual
   std was 1.46. State thresholds relative to `baseline_rmse_predict_mean` as the tool reports it,
   and name which baseline — `baseline_rmse_predict_mean` is computed on the **eval** split and
   runs 2.3× stronger than an honest train-mean baseline.
2. **Outcome branches covering the orderings actually observed.** PhysioNet produced A2 < A3 < A1,
   matching none of the four frozen branches. noaa_uk currently reads A1 < A2 < A3 on CRPS.
3. **The measured `CHIRP_UQ_INIT_VAR`** and the residual scale it came from — derived at a
   *matched ensemble mean*, since a single-draw read inflates the residual by ~2× on this cell.
4. **A conditional-calibration criterion.** Every metric above is a pooled statistic, and the law
   on this cell is marginally near-calibrated while its per-decile coverage runs 0.78 → 0.997.
   A prereg of pooled columns alone would freeze a blind spot.

## 6. Declared scope limits (not discovered post hoc)

- The Theorem-C law is **aleatoric only**, conditional on the predicted parameters — no epistemic
  component.
- `q_k` is **constant in time**; the method doc permits `q_k(t̃)`.
- `v_k` is an exponential-integrator **quadrature**, not a closed form.
- Propagation to data space is **decoder Monte Carlo**, not the delta method — deliberately, so
  the analytic arm shares the sampled arm's CRPS estimator. ⚠️ Eq. (7) is a law for `z₀`, so this
  step is **outside Theorem C's guarantee** ("exactly if linear; delta method otherwise"), and
  measured calibration degrades through it on every analytic arm.
- All arms are **x0**, so absolute CRPS sits above the v-prediction headline table. The
  three-way comparison is internally matched and **not** comparable to
  `finetuning/results/fixed_bugs/RESULTS.md`.
