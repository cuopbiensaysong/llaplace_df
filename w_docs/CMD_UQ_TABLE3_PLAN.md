# CMD UQ — plan of record for a defensible Table 3

**The live plan.** Supersedes the Phase 0–2 *gate logic* of `CMD_UQ_RECOVERY_PLAN.md`; that
document's standing rules and U3 phase spec are carried forward by reference.
Read `CMD_CAMPAIGN_STATE.md` first.

## Why the latent gate was replaced

Four independent interventions moved the Phase-1 gate axis and val CRPS in **opposite**
directions:

| arm | stage 1 | pool norm | `ddim` resc | Δ gate | val CRPS | Δ CRPS |
|---|---|---|---|---|---|---|
| control (2 seeds) | shipped | inherit | 6.41 % | — | 0.3280 | — |
| `b23only` | shipped | global | 4.82 % | −1.59 | 0.31513 | −0.013 ✅ |
| `s1only` | entenc | inherit | 3.01 % | −3.40 | **0.30179** | −0.026 ✅ |
| both (2 seeds) | entenc | global | 2.09 % | −4.32 | 0.3197 | −0.008 ✅ |

Mechanism: a faithful stage 1 (round-trip 0.922 → 0.995) moves genuinely **unpredictable**
per-entity content into the latent, so the *fraction* of latent variance explained falls while
the decoded forecast improves.

**A good UQ result does not need a strong mean. It needs an honest mean and a law that can
express heteroscedasticity** — and, per §5, a law whose heteroscedasticity is *informative*.

---

## 🟢 Table 3 — 2026-08-10, noaa_uk h=168, val, 2 seeds

Provenance is in each eval JSON's `resolved` block and is identical across all four reported
evals: `vae_entity_encode=True`, `mean_source=ensemble25`, `variance_timestep=999`,
`variance_draws=4`, `guidance_strength=1.0`, `steps=64`, `num_samples=25`, `coverage_level=0.8`.

| Arm | CRPS | MSE | PIT-ECE | cov @ 0.8 | width | **passes/batch** | wall (s) |
|---|---|---|---|---|---|---|---|
| A1 sampled (25 draws) | **0.3119 ± 0.0020** | 0.3165 ± 0.0018 | 0.0250 ± 0.0278 | 0.7602 ± 0.0326 | 1.178 | 1600 | 1681 |
| A2 analytic, matched mean | 0.3184 ± 0.0004 | 0.3271 ± 0.0053 | 0.0446 ± 0.0020 | 0.7098 ± 0.0680 | 1.088 | 1604 | 1725 |
| A3 one-shot analytic | 0.3555 ± 0.0236 | 0.4004 ± 0.0622 | 0.1000 ± 0.0205 | 0.7225 ± 0.0702 | **1.243** | **1** | **21** |

> **Corrected 2026-08-10.** The first version of this row omitted A3's `width` cell, which
> shifted every later column left and rendered A3 as the **narrowest** arm (width "1", passes
> "21", wall blank). A3 is in fact the **widest** — 1.243 against A1's 1.178 and A2's 1.088.
> A3 is therefore **wide *and* under-covering** (0.7225 at nominal 0.8): a badly *shaped*
> predictive distribution, not a mis-scaled one.

Both pre-flight gates PASS (G-b NLL health, G-c one-shot conditioning) — the first time in this
campaign. ⚠️ **G-b passes on coverage @ 0.9 = 0.7716 against a 0.75 threshold**, a criterion
whose documented collapse signature is ~0.0. It is a *collapse detector, not a calibration bar*;
that read still under-covers by 13 points at nominal 0.9. Do not put "both gates pass" and "the
law is calibrated" in the same sentence.

⚠️ **`gates()` reads at `mean_source="ddim"`**, not the table's `ensemble25` — a cost choice.
Its `latent_rmse` = 1.0847 against baseline 0.9578 therefore says the mean *loses* to
predict-the-mean, while at the table's protocol it **wins** (0.9063 vs 0.9577, `latent_corr`
0.3785 vs 0.2621). The two are not comparable; do not quote them side by side.

### Resolved at n=2

- **A3's efficiency.** 1 denoiser pass against 1600 (**1600×**; 82× wall) for CRPS +14 % and
  PIT-ECE 0.100 vs 0.025. Holds on both seeds. This is the method-level speedup to report.
  The A1/A2 pass ratio is **0.9975** by construction after Fix 1d — A2 costs 4 passes *more*
  (its variance draws), so there is no A1/A2 speedup and none is claimed.
- **CRPS ordering.** A1 < A2 < A3 on **both** seeds (0.3105/0.3133 < 0.3181/0.3186 <
  0.3388/0.3722). Sampled beats analytic by ~2 %.

### 🔴 NOT resolved — the A1-vs-A2 calibration comparison, which is what the table is for

The arms swap ranking between seeds:

| | seed 0 | seed 1 |
|---|---|---|
| A1 coverage @ 0.8 | 0.7371 | **0.7832** |
| A2 coverage @ 0.8 | **0.7578** | 0.6617 |
| A1 PIT-ECE | 0.0446 | **0.0053** |
| A2 PIT-ECE | **0.0432** | 0.0460 |

**Both arms are seed-unstable, in different columns** — A1 on PIT-ECE (±0.0278, std exceeds the
mean), A2 on coverage (±0.0680, 2.1× A1's) and width (±0.2181, 1.9× A1's; its scale swings
1.2417 → 0.9332 between seeds). **Two seeds cannot separate these arms.** The seed-dependence of
the analytic law's *scale* is itself worth reporting.

⚠️ **Wall-clock is not a clean measurement** (1681 ± 1044 s): the seeds ran concurrently on
different GPUs with contention. Quote `denoiser_passes_per_batch`, which is exact.

---

## The four UQ-specific defects — all fixed

None was ever tested by the old gate; each corrupts Table 3 independently.

| # | fix | status |
|---|---|---|
| 1a | `COND_POOL_NORM_MODE="global"` for the U3 arms — without it `chirp_field.uq_params` sees no history and A2/A3 score a structurally homoscedastic law (B23) | ✅ arms opt in; repo default stays `"inherit"` |
| 1b | `CHIRP_UQ_INIT_VAR ≈ 1.0`, not the 1e-2 library default or PhysioNet's 25.0 — init at the residual scale (`latent_mse ≈ 0.96`) | ✅ verified derived on the entenc stack, not the legacy one |
| 1c | `ensemble_pit` for A1, so every arm is scored by the same estimator | ✅ randomized-rank form (below) |
| 1d | matched means — A2's mean is now the 25-draw ensemble mean, not one DDIM pass | ✅ A2/A1 MSE ratio 4.25× → **1.01×** |

### Three scoring defects found in review, all silent

| defect | what it did | fix |
|---|---|---|
| **wrong stage-1 VAE at eval** | `eval_one` omitted `--vae-entity-encode`, so `build_eval_config` resolved the **legacy** VAE, which exists on disk and loads `strict=True` without warning. Every latent target and every data-space CRPS would have come from a stage 1 the denoiser never saw — measured 490× difference in `latent_mse` | flags derived from `STACK_CONFIG` so training and eval stacks cannot drift |
| **guidance not frozen at eval** | sampling knobs are not in the checkpoint; `_sampling_kwargs(prefix="TEST")` resolved the shipped `(1.0, 2.0)` ramp regardless of training | `--guidance` / `--gen-steps`, recorded in `resolved` |
| **`ensemble_pit` biased at S = 25** | `(below + 0.5·ties)/S` put the PIT on S+1 atoms: on a *perfectly calibrated* ensemble it reported coverage 0.846 at nominal 0.9 and 24× the PIT-ECE of `gaussian_pit`, manufacturing the very arm difference the table measures | randomized rank `(below + V)/(S+1)`, exact at any S |

### The variance read timestep — a parity fix worth knowing

`variance = einsum(s_modal(cond), energy(theta))` where `theta` is the modal state of `x_t`, so
**the spread scales with the modal energy of whatever noise level the head is handed.** A2 read
at a hardcoded `t=1`, A3 at `t=T−1`. Measured on one checkpoint, matched mean, 322 560 elements:

| t | E[var] | cov @ 0.9 |
|---|---|---|
| 1 | 0.190 | 0.690 |
| 400 | 0.237 | 0.747 |
| **T−1 (999)** | **0.681** | **0.931** |

At `t=1` the head is handed a nearly-clean `x_t` built from its own prediction, so it reports
uncertainty about `x0` *given it already almost knows x0* — the wrong quantity. **Default is now
T−1 for every mean source**, with `--variance-timestep` to override and the value recorded.
`--variance-draws` (default 4) averages the read, which removes a small systematic bias toward
the single-draw value; the per-element noise it also removes turns out not to reach the pooled
columns (swing ~0.001 at n = 242 k).

---

## 🔴 The open finding Table 3 cannot see

**The law's heteroscedasticity is essentially uninformative.** Seed 0, A2, matched mean, val,
483 840 latent elements:

```
corr(predicted variance, squared residual) = +0.0159
Spearman rank correlation                  = +0.0568
```

| decile of predicted variance | 1 | 5 | 10 |
|---|---|---|---|
| E[var] | 0.264 | 0.654 | **3.818** |
| E[resid²] | 0.278 | 0.482 | **0.446** |
| coverage @ 0.8 | 0.778 | 0.879 | **0.997** |

Predicted variance spans **14.5×**; the actual squared error spans 1.6× and does not track it.
The pooled PIT-ECE (0.052) and coverage (0.890) look acceptable **because the over- and
under-confident deciles cancel**.

This also explains why calibration degrades through the decoder (A2 PIT-ECE 0.0129 latent →
0.0432 data; A3 0.0091 → 0.0854): `s_modal[b,t,:]` is shared across all 16 latent channels, so a
mis-scaled element is mis-scaled in *every* channel at once, and the decoder mixes channels **at
the same element** — the cancellation that flatters the pooled statistic cannot survive it.

**This is not a Theorem-C problem.** The theorem propagates whatever `q_k`, `p_k⁰` the network
predicts, and the closed form is fine. The finding is about the *learned parameters*: B23 made
the pooled path **see** the history, and it does, but seeing it has not become predicting error.

Two explanations tested and **falsified**, so do not re-run them:

| hypothesis | test | result |
|---|---|---|
| the missing term is stage-1 reconstruction error | `--add-reconstruction-variance` (estimated on train, per horizon/channel, applied to every arm) | pooled variance 0.00265 — a 0.83 % widening, recovering **+0.003 of a +0.106** coverage gap. Explains 3 %. Stage 1 is too good here (round-trip 0.995) |
| the diagonal readout ignores cross-mode covariance | impose the empirical cross-channel correlation on the draws and re-decode | correlations are large (mean \|off-diag\| 0.346) but **cancel** (mean signed +0.0001); coverage 0.568 → 0.563. No effect |

---

## Next

1. **Seeds 2–4** (~3.3 days on 2 GPUs) for the pre-registered 5.
2. **Write the noaa_uk pre-registration** before the test split is touched (`PREREG_U3.md` is
   PhysioNet-scoped). Fix the three things Phase 4 of the recovery plan names: thresholds stated
   against measured baselines, outcome branches covering non-monotone orderings, and the recorded
   `CHIRP_UQ_INIT_VAR` with the residual scale it came from.
3. **Add a conditional-calibration column** — decile ratio spread, or the rank correlation above.
   Every current column is pooled and blind to §5.
4. Optional: the off-diagonal cells (`--mean-source ddim` at matched variance protocol) isolate
   the location and spread contributions separately.

## Standing rules

- `--weights ema` on every eval (B16).
- Score CPU or GPU consistently — they differ 0.4–0.8 % relative; never mix within a table.
- ≥ 2 seeds for any decision; CRPS seed band is **0.036**.
- Concurrent stage-3 runs with different stage-1/2 artifacts need separate `DIFF_PRECOMPUTE_DIR`.
- Never overwrite a frozen artifact; new architecture ⇒ new filename.
- Register predictions before launching. *(Scorecard: 1 of 5 correct — the argument for measuring
  rather than reasoning.)*
