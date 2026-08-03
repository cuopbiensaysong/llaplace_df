> **Tracked canonical copy.** The working copy at `finetuning/results/phase1b/gate_readout.md` is under `.gitignore:46`
> (`finetuning/results/*`) and does not survive a fresh clone. Edit **this** file; mirror to the
> results path only if a tool reads it there. Copied 2026-08-03.

# Phase 1b — S1 signal gate, noaa_uk h=168 — **FAIL**, both seeds

Completed 2026-08-03. Canonical write-up: §1b of `w_docs/CMD_UQ_RECOVERY_PLAN.md` (this path is
gitignored). Raw JSON in this directory: `gate_seed{0,1}.json` (ddim),
`gate_seed0_oneshot.json`, `gate_seed1_oneshot.json`, plus the training summaries
`s1_gate_seed{0,1}.json`.

## What ran

Arm `d`, plain-MSE S1, `predict_type=x0`, `EPOCHS=600` + early stopping, campaign eval profile,
cuda:1. Verified per checkpoint from its own `model_config`: `cond_norm_mode="sample"` (default,
**not** `global`), `sum_ft_mode` absent, `pole_pool_use_raw_summary=False`, `chirp_uq_head=False`.
Matching 1a row is `cond_summary, "sample"`.

| seed | best val CRPS | best epoch | stopped | wall |
|---|---|---|---|---|
| 0 | 0.32376 | 255 | 355 | 9 h 50 m |
| 1 | 0.33229 | 210 | 310 | 8 h 52 m |

Both early-stopped on their own — not the `EPOCHS=60` truncation that made the `global` seeds
unmeasurable. CRPS spread 0.0085 against the documented 0.036 band ⇒ the seeds agree.

## Gate read (val, EMA weights, `--latent-only`); baseline mse 0.64917 / rmse 0.80571

| `--mean-source` | seed | `latent_mse` | vs baseline | RMSE | `latent_corr` |
|---|---|---|---|---|---|
| `ddim` | 0 | 0.68280 | +5.18 % worse | **−2.56 %** | 0.3702 |
| `ddim` | 1 | 0.72277 | +11.34 % worse | **−5.52 %** | 0.3395 |
| `oneshot` | 0 | 0.64369 | −0.84 % better | +0.42 % | 0.1903 |
| `oneshot` | 1 | 0.63927 | −1.53 % better | +0.77 % | 0.1931 |

Threshold 10 %. **All four reads fail; the plan-mandated `ddim` read loses to predict-mean.**

## `ddim` is the wrong statistic for a *signal* gate

`generate(eta=0.0)` from a random `x_T` is deterministic in trajectory but still **one sample**,
not a conditional mean, and it applies CFG on `guidance_strength=(1.0, 2.0)`,
`guidance_power=0.3`. Both inflate spread without adding signal. Solving
`mse/base = a − 2·corr·√a + 1` for `a = Var(ŷ)/Var(y)` separates signal from scale:

| read | seed | corr | std(ŷ)/std(y) | optimal | over-dispersion | RMSE if rescaled |
|---|---|---|---|---|---|---|
| `ddim` | 0 | 0.3702 | 0.805 | 0.370 | 2.17× | **+7.11 %** |
| `ddim` | 1 | 0.3395 | 0.818 | 0.339 | 2.41× | **+5.94 %** |
| `oneshot` | 0 | 0.1903 | 0.357 | 0.190 | 1.88× | +1.83 % |
| `oneshot` | 1 | 0.1931 | 0.342 | 0.193 | 1.77× | +1.88 % |

Mean rescaled ceiling **6.5 %** — still under 10 %, so granting the model a free optimal rescaling
it does not have would not change the verdict.

## The finding: the denoiser sits at the ridge ceiling

1a's best *linear* read of this latent from this conditioning: **C = 5.6 %** (2 048 dims,
`"sample"`) → implied corr 0.330. The trained denoisers reach **7.11 % / 5.94 %**, mean 6.5 % —
**+0.9 pt over ridge**, seed 1 within **0.3 pt**.

⇒ 10.2 M parameters, a nonlinear architecture and 355 epochs buy under one percentage point over
linear least squares on the same inputs. **The denoiser is not the bottleneck**; the conditioning
ceiling is, exactly as 1a predicted when it returned `FIX THE CONDITIONING (B21)`.

## Independent confirmation of debugging lead 2 (`block_summary_adaln = False`)

`oneshot` reads the model at `t = T−1` — x0 given pure noise plus conditioning, i.e. how much
conditioning reaches the trunk when `x_t` carries nothing. It scores **0.1903 / 0.1931**; the
trajectory reaches **0.3702 / 0.3395**. An ideal model orders these the other way
(`corr(cond. mean) = ρ` vs `corr(single sample) = ρ²`). Both models instead recover ~half their
correlation only *after* `x_t` becomes informative — lead 2's mechanism, and the two seeds land
1.5 % apart, so it is architectural, not a draw.

## Reproduce

```bash
llapdiff-uq-eval --dataset-key noaa_uk --pred 168 \
  --checkpoint ldt/tuning/phase1b/noaa_uk_h168/d/s1_gate_seed<N>/output/predict-x0/modal-chirp/seed-<N>/pred-168/llapdiff_pred-168_best_ema.pt \
  --weights ema --split val --latent-only --mean-source {ddim,oneshot}
```
