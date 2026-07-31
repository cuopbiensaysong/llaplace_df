# Pre-registration — U3, the matched-synthesizer UQ ablation

**Scope.** This file freezes **only** the U3 slice of `cmd_plan_v2.md` §4. The repo-level
`PREREG.md` that `cmd_plan_v2.md:63` requires before Phase 1 is still unwritten; H1/H2/T1–T4
remain unfrozen. Do not read this as the campaign-wide pre-registration.

**Frozen on:** 2026-07-30, before any test-split evaluation of the U3 arms.
**Driver:** `finetuning/run_u3_uq.py` (run tag `uq_u3`). **Dataset:** PhysioNet, h = 12, arm `d`.

---

## 1. The question

Does the diffusion process earn its cost, given that the chirp-modal synthesizer already
supplies a closed-form Gaussian predictive law (Theorem C)?

Three arms share the **identical synthesizer** and differ only in how the predictive law is
obtained:

| Arm | Mean | Uncertainty | Checkpoint |
|---|---|---|---|
| **A1** diffusion + sampled UQ | 25-draw DDIM ensemble | empirical ensemble spread | S2 |
| **A2** diffusion + analytic UQ | deterministic DDIM $x_0$ | Thm-C closed form, decoded | S2 (same file as A1) |
| **A3** one-shot Gaussian NLL | one forward at $t = T-1$ | Thm-C closed form, decoded | S3 (`TRAIN_T_SAMPLER="max_only"`) |

## 2. Metrics

- **Primary:** data-space **CRPS**, 25 samples, scored by the unchanged `evaluate_regression`
  (identical masking, CRPS estimator, ensemble size, and `generator_seed` across all arms).
- **Secondary:** MAE, MSE, wall-clock seconds, `analytic_speedup_x`.
- **Calibration (A2, A3 only — A1 has no closed-form law):** latent Gaussian NLL, PIT
  calibration error, central-interval coverage, latent RMSE, mean predicted std.

## 3. Selection and splits

- **5 seeds** (0–4) per arm; report mean ± std.
- All hyperparameters are **fixed in advance** to the `fixed_bugs` incumbent
  (`finetuning/results/fixed_bugs/RESULTS.md`) and are **identical across all three arms** —
  no per-arm tuning, so the matched-budget rule of `cmd_plan_v2.md` §1 holds trivially.
- Gates and any diagnostics run on **val**. **The test split is evaluated exactly once per
  arm per seed**, after this file is frozen.

## 4. Protocol parity (checked before any CRPS is interpreted)

| Item | Value |
|---|---|
| CRPS samples | 25 (`NUM_EVAL_SAMPLES`) |
| DDIM steps | 64 (`GEN_STEPS`), η = 0 |
| Guidance | ramp (1.0, 2.0), power 2.0 |
| Dynamic thresholding | `DYNAMIC_THRESH_P = 0.0` (off) |
| **Weights** | **EMA (decay 0.999)** — `llapdiff-uq-eval --weights ema` |
| Split | chronological, PhysioNet patient-relative policy (loader default) |
| Upstream | identical frozen VAE + summarizer files across all arms |
| Prediction type | **x0** for all three arms (forced: `chirp_uq_head` requires it) |

The `--weights ema` flag was added for this campaign. Without it the tool scores **raw**
weights (bug B16: `_load_stack` reads `payload["model"]`, which holds raw weights even in
`*_best_ema.pt`). Verified meaningful: on an existing checkpoint the summed per-tensor
`max|raw − ema|` is 0.696, not 0.

## 5. Pre-registered outcomes — every branch is a result

- **A1 ≪ A2, A3** → the diffusion process is doing real work; the analytic law is a cheap
  approximation. Report the speedup as the price of the gap.
- **A1 ≈ A2** → analytic UQ matches sampled UQ at a fraction of the cost; the DDIM ensemble is
  redundant *for uncertainty*, and the reverse process still supplies the mean.
- **A2 ≈ A3** → the diffusion process is not earning its cost at this horizon; the honest
  conclusion is a faster one-shot model. This is a publishable outcome, not a failure.
- **A3 ≪ others** → diffusion justified, **but only if gate G-c passes** (below).

## 6. Gates that can invalidate the reading (declared in advance)

- **G-b — NLL health (S2, val, latent-only).** Gaussian-NLL training is known to collapse
  (`CMD_RUNBOOK.md` §4: latent RMSE 10.8, predicted std 0.16, PIT error 0.22, coverage ≈ 0).
  Both UQ arms warm-start from that seed's S1 MSE checkpoint via `DIFF_INIT_CKPT`. If the
  failure signature appears, the run is void — not evidence about UQ.

  > **Amendment 2026-07-30, before any test-split read.** The first seed-0 attempt failed
  > this gate, and the documented cause was ruled out: a control run without
  > `DIFF_INIT_CKPT` sits 51× further from S1 than the warm-started model, so the warm
  > start demonstrably took. The real cause is the **variance** init. At the library
  > default `CHIRP_UQ_INIT_VAR = 1e-2` the initial predicted variance is 0.0011 against a
  > squared error of 2.675 — a ≈2415× mismatch the variance head cannot close during
  > training, leaving the arm catastrophically overconfident:
  >
  > | `CHIRP_UQ_INIT_VAR` | PIT cal. error | coverage @ 0.9 | predicted std | data CRPS |
  > |---|---|---|---|---|
  > | 1e-2 | 0.2210 | 0.034 | 0.096 | 0.2552 |
  > | **25.0** | **0.0389** | **0.872** | 4.09 | 0.2543 |
  >
  > (target std 1.46.) CRPS moved <0.001 while coverage went 3% → 87%, so **CRPS is blind
  > to this failure**. Fix: `CHIRP_UQ_INIT_VAR` is now a config knob (default 1e-2, so all
  > pre-existing behaviour and checkpoints are unchanged); the campaign sets **25.0** for
  > all three arms. The value is scale-dependent — re-measure per dataset/horizon.
  >
  > **Correction to an earlier draft of this amendment.** It claimed the NLL degraded the
  > latent mean ~340× (2.76 → 92.9 → 944). That was a unit error: `val_diag_mse_raw` is
  > `stats["raw_loss"]`, an MSE under `DIFF_LOSS_MODE="mse"` but an **NLL** under
  > `"gaussian_nll"`, so those numbers are not comparable across arms. Measured
  > like-for-like (t = T/2, identical noise, EMA), latent x0 MSE is **5.72** (S1 chirp-MSE),
  > **5.48** (NLL @1e-2), **4.28** (NLL @25) against a predict-zero baseline of **2.16**.
  > The NLL does **not** damage the mean; the mean is uninformative in every arm — see G-c.
- **G-c — one-shot conditioning (S3, val, latent-only).** `CMD_BUG_REPORT.md` ("the denoiser
  barely uses its conditioning", OPEN) measured std 0.09 vs target 0.62 and corr ≈ 0 at
  `steps=1`. That was a *uniformly*-trained model; S3 trains only at $t = T-1$, so it is the
  fair test. **If S3's one-shot mean does not beat predict-the-latent-mean, any "diffusion
  wins" reading of U3 is confounded by that open bug and must be reported as confounded.**
  This is declared now, before the numbers exist.

  > **Outcome 2026-07-30 (seed 0, val): CONFOUNDED, and more broadly than this gate
  > anticipated.** S3's one-shot mean: latent RMSE **2.431** vs a predict-mean baseline of
  > **1.464**, correlation **+0.004**. But the same holds for *every* arm, including the
  > plain-MSE S1 — like-for-like latent x0 MSE 4.3–5.7 against a predict-zero baseline of
  > 2.16, correlation ≈ −0.05 throughout. So the confound is not specific to the one-shot
  > arm: **all three U3 arms sit on a denoiser whose latent prediction carries no signal.**
  >
  > The gate's threshold was also mis-specified when frozen: it assumed unit-scale latents
  > ("RMSE ≥ ~1.0"), but the latents have std 1.46. Both gates now compare against the
  > *measured* predict-mean baseline, which `llapdiff-uq-eval` reports
  > (`baseline_rmse_predict_mean`, `latent_target_std`, `latent_corr`). This is a
  > correction to the gate's arithmetic, not a relaxation of its criterion: the verdict is
  > CONFOUNDED under both the frozen and the corrected form.
  >
  > Countervailing evidence that keeps the data-space arms worth reporting: disabling
  > conditioning entirely still costs CRPS (0.2606 → 0.2990 on val), so the model does use
  > conditioning somewhere downstream of the latent read. The U3 data-space table is
  > therefore reported, but **any conclusion about diffusion per se is confounded** and is
  > labelled as such.

## 7. Declared scope limits (not discovered post hoc)

- The Theorem-C law is **aleatoric only**, conditional on the predicted parameters — no
  epistemic component.
- $q_k$ is **constant in time**; the method doc permits $q_k(\tilde t)$.
- $v_k$ is an exponential-integrator **quadrature**, not a closed form — the P-exact integrand
  $e^{2\bar\rho(s)}q(s)$ has no elementary antiderivative.
- Propagation to data space is **decoder Monte Carlo**, not the delta method named in the plan
  — deliberately, so the analytic arm shares the sampled arm's CRPS estimator.
- All arms are **x0**. `USAGE.md` §3.6 records x0 as ≈0.03 CRPS worse than v on PhysioNet
  h = 12, so absolute CRPS will sit above the v-prediction headline table. The three-way
  comparison is internally matched and is **not** comparable to
  `finetuning/results/fixed_bugs/RESULTS.md`.
