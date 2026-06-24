# LLapDiffusion — Developer / Internals Guide

This is the companion to `[USAGE.md](USAGE.md)`. Where `USAGE.md` tells you
*which commands to run*, this guide tells you **what each command actually
does, file by file**, **where the logic lives if you want to change it**, and
**which footguns silently produce wrong results**.

All line numbers refer to the source as it stands in this repo; treat them as
"look near here", since edits shift them.

Contents:

1. [The big picture](#1-the-big-picture)
2. [Execution trace: what `llapdiff-train` does, step by step](#2-execution-trace-what-llapdiff-train-does-step-by-step)
3. [The config system (and its #1 footgun)](#3-the-config-system-and-its-1-footgun)
4. [The data pipeline](#4-the-data-pipeline)
5. [Stage 1 — Latent VAE](#5-stage-1--latent-vae)
6. [Stage 2 — History summarizer](#6-stage-2--history-summarizer)
7. [Stage 3 — LLapDiff diffusion](#7-stage-3--llapdiff-diffusion)
8. [Evaluation (`llapdiff-checkpoint-eval`)](#8-evaluation-llapdiff-checkpoint-eval)
9. ["I want to change X" → file map](#9-i-want-to-change-x--file-map)
10. [Pitfalls checklist (read before you change anything)](#10-pitfalls-checklist-read-before-you-change-anything)
11. [Common errors and what they mean](#11-common-errors-and-what-they-mean)

---

## 1. The big picture

A single `llapdiff-train` run trains **three models in sequence** for one
`(dataset, horizon)` pair. Each consumes the artifact from the previous stage:

```
raw cache ──▶ [1] Latent VAE ──▶ encodes target windows into a latent code
                   │
context window ──▶ [2] Summarizer ──▶ a conditioning vector from history
                   │
            [3] LLapDiff ──▶ denoises the VAE latent, conditioned on the summary,
                              using Laplace-domain stable poles
```


| Stage         | Trainer file                       | Model file                              | Output checkpoint                                                                          |
| ------------- | ---------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1. Latent VAE | `trainers/train_val_latent.py`     | `latent_space/latent_vae.py`            | `ldt/vae/saved_model/<ds>/pred-<H>_ch-<C>_entity_elbo.pt` (+`_recon`)                      |
| 2. Summarizer | `trainers/train_val_summarizer.py` | `models/summarizer.py`                  | `ldt/summarizer/saved_model/<ds>/<H>-<C>-summarizer.pt`                                    |
| 3. LLapDiff   | `trainers/train_val_llapdiff.py`   | `models/llapdiff.py` (+ `lapformer.py`) | `ldt/output/<ds>/pred-<H>/llapdiff_pred-<H>_best.pt` (+ `_best_raw`, `_best_ema`, `_last`) |


The orchestrator that wires all of this together is
`**llapdiffusion/pipeline.py**` — start there for any "how does the run flow"
question.

---

## 2. Execution trace: what `llapdiff-train` does, step by step

The console script `llapdiff-train` maps (via `pyproject.toml`) to
`llapdiffusion.pipeline:cli_main` → `main()` (`pipeline.py:803`).

### 2.1 `main()` — argument parsing and global setup (`pipeline.py:803-863`)

1. **Parse args** (`_parse_args`, line 475). Every CLI flag in §8 of `USAGE.md`
  is defined here.
2. `**configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)`**
  (line 805) — sets the env vars that `dataset_archives.py` reads when it needs
   to extract the bundled cache. See §4.
3. `**apply_dataset_preset(config, args.dataset_key, pred=initial_pred)**`
  (line 807) — stamps every dataset-specific value onto the global `config`
   module. See §3 — **this is the function that will surprise you.**
4. Set verbosity, then copy the CLI values the preset must *not* clobber into
  the `REQUESTED_*` tracking attributes:
   `REQUESTED_BATCH_SIZE_ARG`, `REQUESTED_TARGET_COL_ARG`,
   `REQUESTED_TARGET_COLS_ARG`, `REQUESTED_PREDICT_TYPE_ARG`, plus
   `split_policy`, `exact_timestamp_batches`, `COVERAGE` (lines 808-829).
5. Build `training_overrides` from the `--target-mask-aux-*` flags
  (`_training_overrides_from_args`, line 644).
6. Resolve the horizon list (`args.preds` or the preset's full set).
7. `**_apply_predict_type_output_routing`** then
  `**_apply_modal_type_output_routing`**: for a non-default `--predict-type`
   (`x0`/`eps`) append `predict-<type>`, and for a non-default `--modal-type`
   (`chirp`) append `modal-<core>`, to `OUT_DIR`/`CKPT_DIR` so neither overwrites
   the default `v`/`lti` outputs. The two compose (`predict-x0/modal-chirp/`).
   Capture `base_out_dir` / `base_ckpt_dir`.
8. **Loop over horizons** → `run_single_pred(...)` for each (line 840).
9. Print a summary table; optionally write `--summary-json`.

### 2.2 `run_single_pred()` — one horizon, all three stages (`pipeline.py:281`)

This is the heart of the run. In order:

1. `**_update_config_for_pred(pred)`** (line 305 → def at 230). **Re-applies the
  whole preset** for this specific horizon, then *restores* the tracked
   `REQUESTED_*` values on top. See the §3 footgun.
2. **Output routing**: with `base_out_dir`/`base_ckpt_dir` set (the CLI path),
  `_apply_pred_output_dirs` puts artifacts under `…/pred-<H>/` so multiple
   horizons don't collide. (Without them, falls back to predict-type + modal-type
   routing.)
3. `**_apply_training_overrides`** (line 315): pushes the target-mask-aux knobs
  onto `config` (and flips `IMPUTATION_TRAINING=True` when `aux_p>0`).
4. `**_sync_target_shape_config**` (line 316 → `_target_policy`, line 184):
  reads `<DATA_DIR>/cache_ratio_index/meta.json`, resolves the requested target
   column(s) against the cache's columns, and sets the target dimensionality +
   the `TARGET_ARTIFACT_SUFFIX` that becomes part of stage-3 checkpoint names.
5. **Build dataloaders once** (`prepare_dataloaders`, line 323) and share them
  across stages unless `--no-shared-loaders`.
6. **Stage 1 (VAE)** — *skip-or-train* (lines 325-341):
  ```python
   if recompute_vae or not Path(config.VAE_CKPT).exists():
       latent_stats = train_val_latent.run(...)
   else:
       latent_stats = {"status": "skipped", "reason": "checkpoint_exists", ...}
   config.VAE_CKPT = str(_select_vae_checkpoint(latent_stats, vae_ckpt_path))
  ```
   So the VAE trains **only if its checkpoint file is missing** (or you pass
   `--recompute-vae`). This is exactly what lets you reuse the pretrained VAE.
7. **Stage 2 (summarizer)** — same skip-or-train pattern keyed on
  `_summarizer_ckpt_path()` (lines 343-357).
8. **Stage 3 (LLapDiff)** — **always runs** (line 359):
  `train_val_llapdiff.run(...)`. Returns `eval_stats` (test metrics) and
   `loaded_checkpoint` (the best checkpoint used for the final test eval).
9. **Optional post-train eval** (lines 374-397): only if `--run-checkpoint-eval`;
  calls `evaluate_checkpoint` (§8).
10. Returns a dict bundling every stage's stats + the data/split policy
  metadata.

> **Mental model:** `llapdiff-train` is "train stage 3, lazily building stages
> 1-2 if their files are missing." Stages 1-2 are gated purely by **file
> existence at the config-derived path**, never by content.

---

## 3. The config system (and its #1 footgun)

There is **one** config object: the module `llapdiffusion/configs/config.py`,
imported as `from llapdiffusion.configs import config` everywhere. It is a
plain module used as a mutable namespace. `config.py` holds only **generic base
defaults**; everything dataset-specific is stamped on at runtime.

### `apply_dataset_preset(cfg, dataset_key, pred=...)` (`dataset_defaults.py:214`)

This **mutates the global config in place**, overwriting ~40 attributes from
the `DatasetPreset` table (`dataset_defaults.py:84-167`): `DATA_DIR`, `PRED`,
`WINDOW` (= context length), `BATCH_SIZE`/`DATES_PER_BATCH`,
`VAE_LATENT_CHANNELS`, all the `VAE_*`/`SUM_*` paths, `OUT_DIR`, `CKPT_DIR`,
`**EPOCHS=600`**, `**PREDICT_TYPE="v"**`, `MINSNR_GAMMA`, `BASE_LR`, the
target-mask-aux defaults, `TIMESTEPS`, `MODEL_WIDTH`, `LAPLACE_K`, and (for
`physionet`/`crypto`) the `_IRREGULAR_PUBLIC_PRESET` model overrides.

> **Note (chirp core).** `DENOISER_MODAL_TYPE` and its `CHIRP_*` tunables (§7.5)
> live **only** in `config.py` base defaults — `apply_dataset_preset` does *not*
> stamp them. So unlike `EPOCHS`/`PREDICT_TYPE`, a runtime assignment (or the
> `--modal-type` flag, set once in `main()`) **survives** both preset
> applications. `_llapdiff_model_kwargs` reads them at model-build time.

### 🔴 Footgun #1: the preset is applied **twice**, and resets your edits

`apply_dataset_preset` runs once in `main()` **and again inside
`run_single_pred` via `_update_config_for_pred`** (`pipeline.py:247`). Each call
**re-stamps** the defaults. So:

```python
from llapdiffusion.configs import config
config.EPOCHS = 1          # ❌ silently reset to 600 inside run_single_pred
run_single_pred(100)
```

Only the values the pipeline *explicitly tracks and restores* survive the
re-stamp — these are the `REQUESTED_`* attributes, `split_policy`,
`split_scope`, `exact_timestamp_batches`, and the target columns
(`pipeline.py:231-265`). Everything else (`EPOCHS`, `PREDICT_TYPE`,
`BASE_LR`, `MINSNR_GAMMA`, model width, …) reverts.

**To change an untracked value programmatically**, wrap the preset:

```python
import llapdiffusion.pipeline as P
_orig = P.apply_dataset_preset
def patched(cfg, key, *, pred=None):
    out = _orig(cfg, key, pred=pred); cfg.EPOCHS = 1; return out
P.apply_dataset_preset = patched      # now EPOCHS survives both calls
```

(This is exactly the trick the VRAM-probe script in `USAGE.md` uses.) To change
it *permanently*, edit `DatasetPreset.epochs` / the relevant field in
`dataset_defaults.py`, or the base default in `config.py` — **not** a runtime
assignment.

### 🔴 Footgun #2: `context_length` must equal `2 × max(horizons)`

`validate_dataset_presets` (`dataset_defaults.py:304`) hard-asserts this and
that `epochs == 600`. If you add a dataset or horizon, honor the invariant or
validation throws.

### 🔴 Footgun #3: `pred` must be in the preset's `horizons`

`apply_dataset_preset` raises `ValueError` for an unsupported horizon
(`dataset_defaults.py:220`). Add the horizon to the preset tuple first.

---

## 4. The data pipeline

`prepare_dataloaders(config)` (`pipeline.py:39`) is the single entry to data:

```python
run_experiment = resolve_run_experiment(config.DATA_DIR)   # dataset_registry.py
return run_experiment(data_dir, date_batching, dates_per_batch=batch_size,
                      K=config.WINDOW, H=config.PRED, coverage=config.COVERAGE,
                      ratios=(0.7, 0.1, 0.2), split_policy=..., target_col=...)
```

- `**resolve_run_experiment**` (`configs/dataset_registry.py:86`) dispatches to a
per-family loader: `fin_dataset.py` (crypto/us_equity), `bms_air_dataset.py`,
`uci_air_quality_dataset.py`, `noaa_isd_dataset.py`,
`physionet_cinc_dataset.py`, `synthetic_regime_dataset.py`. **If you change
how a dataset is windowed/normalized/split, edit the matching file here.**
- **Cache resolution & auto-extraction** is in
`configs/dataset_archives.py`: `resolve_dataset_dir` returns the cache dir,
extracting the matching subtree from the bundled zip into
`~/.cache/llapdiffusion/datasets/` (or `$LLAPDIFF_DATASET_EXTRACT_DIR`) on
first use, guarded by a `.stamp` marker (`_extract_archive_once`).
- **Split** is computed **at load time** from `ratios` + `split_policy`
(`global_purged_horizon` default; `physionet` uses `contiguous` with a
patient-relative scope). It is **not** baked into the cache, so retraining and
re-evaluating stays apples-to-apples (see `USAGE.md` §3.1).

### Batch shape & the `meta` dict

Loaders yield `(xb, yb, meta)` per "date batch". `**--batch-size` is the number
of dates (`dates_per_batch`), not samples** — the realized sample count per
batch depends on how many assets/entities are observed on those dates. `meta`
carries the irregular-time bookkeeping the models need; keys consumed downstream
include `delta_t`, `delta_t_y`, `x_obs_mask` (see
`train_val_llapdiff.py:2192-2207`). If you write a new loader, you must produce
these or the stage-3 conditioning/sanitization will fail.

---

## 5. Stage 1 — Latent VAE

**Trainer:** `trainers/train_val_latent.py` · **Model:** `latent_space/latent_vae.py`
· **Entry:** `train_val_latent.run(...)` (line 739).

**What it does:** a Set-Transformer VAE (`LatentVAE`, `latent_vae.py:45`) encodes
each target window into a per-window latent (`encode_mu`, line 148) and decodes
it back (`decode_mu`, line 177). Stage 3 later diffuses in this latent space.

**Loop logic** (`run`, lines 739-895):

- Per-epoch train/val passes via `_epoch_pass` (line 252); reconstruction is a
**masked** MSE (`_masked_mse`, line 208) so missing entries don't count.
- **β (KL) schedule:** `_beta_for_epoch` (line 714) — `VAE_WARMUP_EPOCHS=5` flat,
then anneals over `VAE_KL_ANNEAL_EPOCHS=25` toward `VAE_BETA=1e-3`.
- **Two checkpoints saved** on improvement (lines 851-883): best β·ELBO →
`…_elbo.pt`, best recon → `…_recon.pt`. Stage 3 uses the **elbo** one
(`config.VAE_CKPT` points there).
- **Early stop** on β·ELBO: `VAE_MAX_PATIENCE=20`, not before `VAE_MIN_EPOCHS=40`
(lines 884-891).

**Key knobs** (`config.py:37-67` / preset): `VAE_LATENT_CHANNELS` (per-dataset,
`**C` in the paths**), `VAE_LATENT_DIM`, `VAE_LAYERS/HEADS/FF`, `VAE_BETA`,
`VAE_INPUT_DROPOUT`, `VAE_NOISE_STD`. The irregular datasets bump dropout/noise
via `_IRREGULAR_PUBLIC_PRESET` (`dataset_defaults.py:36`).

> ⚠️ The latent **channel count is baked into the filename** (`ch-<C>`). If you
> change `VAE_LATENT_CHANNELS`, you get a *different* file — the old one won't be
> picked up, and a stale one with a different `C` that happens to match the path
> would load with the wrong shape. Keep `C` consistent across stages.

---

## 6. Stage 2 — History summarizer

**Trainer:** `trainers/train_val_summarizer.py` · **Model:** `models/summarizer.py`
· **Entry:** `train_val_summarizer.run(...)` (line 408).

**What it does:** consumes the observed history (values, timestamps, gaps,
masks) and produces the conditioning representation stage 3 attends to. The
model is a Laplace/continuous-time encoder (`summarizer.py`): `Time2Vec`
(line 31), optional continuous-RoPE attention (`ContinuousRoPESelfAttention`,
line 68), and a time-value head (`TVHead`, line 11).

**Loop logic** (`run`, lines 408-573):

- Per-epoch `_run_epoch` (line 231); batch prep in `_prepare_batch` (line 155).
- **Composite loss** weighted by `SUM_LOSS_W_{X,V,T,DT,OBS}` (`_loss_weights`,
line 23). `W_X=1.0` (value) dominates; the irregular preset turns on the
`DT`/`OBS` terms.
- **Save best** val-loss checkpoint via `save_ckpt` (line 64); **early stop**
`SUM_PATIENCE` (10, or 15 for irregular), `SUM_EPOCHS=200`.
- AMP (`SUM_AMP`) is **on** for crypto/us_equity/physionet but **off** for
bms_air/uci_air/noaa_* (`sum_amp=False` in those presets) — a stability
choice; non-finite grads are tolerated up to `SUM_MAX_NONFINITE_GRAD_STEPS=8`.

**Key knobs:** `SUM_CONTEXT_DIM=256`, `SUM_POS_ENCODING`
(`learned_abs` vs `continuous_rope`), `SUM_T_TOKEN_MODE`, `SUM_LR`.

---

## 7. Stage 3 — LLapDiff diffusion

**Trainer:** `trainers/train_val_llapdiff.py` (the big one, ~3000 lines) ·
**Model:** `models/llapdiff.py` + `models/lapformer.py` · **Entry:**
`train_val_llapdiff.run(...)` (line 1786).

This is where you'll spend most tuning effort, so it's described in detail.

### 7.1 Setup (`run`, lines 1786-1985)

- Builds the diffusion model `LLapDiff` (`llapdiff.py:15`) from
`_llapdiff_model_kwargs` (line 1397): `MODEL_WIDTH=256`, `NUM_LAYERS=5`,
`NUM_HEADS=4`, `LAPLACE_K=256`, `RHO_CONDITIONING_MODE`, `SELF_COND`, and
`DENOISER_MODAL_TYPE` (+ `CHIRP_*`). The backbone runs attention-based **Laplace
pole analysis** (LapFormer) with one of two dynamical cores — constant-pole
`lti` (default) or time-varying `chirp` (§7.5); the Karras σ schedule exponent
`rho` shows up in the forward (`llapdiff.py:126,201`).
- Loads the **frozen** VAE and summarizer.
- Optionally warm-starts from `DIFF_INIT_CKPT` (lines 1913-1952), including EMA
state.
- Creates an **EMA** shadow of the model (`EMA(diff_model, decay=EMA_DECAY=0.999)`,
line 1946) when `USE_EMA_EVAL`.

### 7.2 Per-step training (`train_one_epoch`, lines 2134-2442)

For each `(xb, yb, meta)` batch:

1. `**_sanitize_batch`** (line 2169) → `(V, T), yb, mask_bn`; skip empty batches.
2. **Build conditioning** from the (eval-mode, frozen) summarizer:
  `_build_cond_summary_pair` (line 2185) returns the projected + raw summary.
   The summarizer can be **fine-tuned** late via `SUM_FT_MODE`/`SUM_FT_START_EPOCH`
   (`summary_ft_active`, line 2137).
3. **Encode latent targets** with the VAE: `_latent_targets_for_batch`
  (line 2211) → `mu_norm` (normalized latent) + `obs_any` mask. Skip if nothing
   observed.
4. **Sample diffusion timesteps** `t` (line 2242, sampler `TRAIN_T_SAMPLER`) and
  form the noised latent `x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)`
   (line 2250).
5. **Classifier-free guidance split** (lines 2229-2240): with prob `DROP_COND_P=0.18`
  a sample goes to the *unconditional* branch (`idx_u`), else conditional
   (`idx_c`); losses are weighted by the realized fractions `w_c`/`w_u`.
6. **Target-mask auxiliary** (lines 2267-2284): with prob `TARGET_MASK_AUX_P`
  (after `TARGET_MASK_AUX_START_EPOCH`, default 10) replace the conditional
   batch with a partial-observation completion task (`_maybe_apply_target_mask_aux`).
   This is what improves *imputation* at inference — the `--target-mask-aux-*`
   flags map straight here.
7. **Self-conditioning** (lines 2287-2318): only after `SELF_COND_START_EPOCH=450`
  and with prob `SELF_COND_P`, off by default.
8. **Loss** = `diffusion_loss(...)` for cond + uncond branches
  (lines 2324-2382), under `LOSS_WEIGHT_SCHEME="weighted_min_snr"` with
   `MINSNR_GAMMA` (per-dataset). Parameterization is `config.PREDICT_TYPE`
   (`v`/`x0`/`eps`).
9. **Optimize** (lines 2384-2436): AMP scaler (`DIFF_AMP=False` by default),
  non-finite-grad guards, `GRAD_CLIP=1.0`, `optimizer.step`, `**ema.update`**,
   LR scheduler step (`LR_SCHEDULE="warmup_constant"`, `WARMUP_FRAC=0.095`).

> **Non-finite policy:** the loop **raises `FloatingPointError`** on non-finite
> conditioning, latents, loss, gradients, or grad-norm (lines 2197-2430). It does
> *not* silently continue (except tolerated summarizer-FT grad skips). A crash
> here usually means a bad upstream checkpoint or AMP instability, not a code bug.

### 7.3 Checkpointing & selection (lines 2493-2896)

- Filenames use `pred_tag = f"pred-{PRED}{TARGET_ARTIFACT_SUFFIX}"` (line 2498),
written to `out_dir = config.OUT_DIR`:
`llapdiff_<pred_tag>_best.pt`, `_best_raw.pt`, `_best_ema.pt`, `_last.pt`.
- The **primary selection metric is validation CRPS** (`PRIMARY_EVAL_METRIC`);
best-raw and best-EMA are tracked separately (lines 2768-2797) and the overall
`_best.pt` follows the configured source.
- **Early stop:** `EARLY_STOP=20` evals without improvement, not before a
warmup-derived `EARLY_STOP_MIN_EPOCHS` (lines 2548-2552, 2843-2850).
- After training, the best checkpoint is reloaded and the **final test metrics**
are computed (lines 2893-2933); `loaded_checkpoint` in the return dict is that
file.

### 7.4 Optional input precompute (speed)

`DIFF_PRECOMPUTE_INPUTS=True` lets the trainer cache frozen VAE latents +
summaries to disk/memory (`DIFF_PRECOMPUTE_DIR`) to avoid recomputing them every
epoch (`train_input_cache`, line 2158). ⚠️ **If you retrain the VAE or
summarizer, a stale precompute cache will feed the diffusion model the *old*
latents.** Clear `DIFF_PRECOMPUTE_DIR` (or leave it `None`) when upstream changes.

### 7.5 Chirp-modal core (time-varying poles)

`DENOISER_MODAL_TYPE="chirp"` (or `--modal-type chirp`) swaps the LTI core's
constant poles + residual MLP for **time-varying** poles whose closed form stays
exact. The change is localized; the residue cross-attention, modal-token
refinement, conditioning, loss, and DDIM sampling are all shared with the LTI
path, and the residues `theta [B,2K,D]` (the constant cₖ/bₖ) are reused unchanged.

- **`ChirpModalField`** (`models/laptrans.py`) — the one substantive new module.
  From the *same* pole-conditioning vector the LTI core uses, it predicts, per
  mode, nonnegative coefficients over a **fixed** Fourier basis
  `φ_m(t̃)=1+cos(2π f_m t̃)` (≥0) with closed-form antiderivative `Φ_m` ("P-exact").
  This yields instantaneous `ρ_k(t̃)=ρ_floor_k+Σ_m a²·φ_m` and **exact integrated**
  `ρ̄_k(t̃)=ρ_floor_k·t̃+Σ_m a²·Φ_m` (ω analogous): `integrated(cond, t_rel)→(ρ̄,ω̄)`
  `[B,T,K]`, `seed_poles(cond)→(ρ₀,ω₀)` `[B,K]` (instantaneous at t̃=0, seeds
  residue extraction). The coeff head is **zero-initialized**, so at init the
  field reduces to constant poles and the chirp model **exactly equals** the LTI
  base (a strict generalization — and a unit test).
- **Synthesis** — `LaplaceTransformEncoder.chirp_basis_matrix(ρ̄,ω̄)` builds
  `e^{-ρ̄}[cos ω̄, sin ω̄]`; `LaplacePseudoInverse.forward` takes optional
  `rho_bar/omega_bar` and uses it in place of the constant-pole `basis_matrix`.
  With `CHIRP_USE_MLP_RESIDUAL=False` (default) the residual MLP is absent —
  stability is by construction (`‖ŷ(t̃)‖ ≤ e^{-ρ_min·t̃}·Σ_k√(‖cₖ‖²+‖bₖ‖²)`).
- **Wiring** — `LapFormer.__init__` builds `self.chirp_field` and forces the
  synthesis residual off in chirp mode; `LapFormer.forward` branches to seed
  analysis with `seed_poles` and synthesize with `integrated` poles.
- **Persistence / back-compat** — `_llapdiff_model_kwargs` writes
  `denoiser_modal_type` + `chirp_*` into the checkpoint's `model_config`;
  `_llapdiff_config_from_checkpoint` does `setdefault("denoiser_modal_type",
  "lti")`, so **pre-chirp checkpoints rebuild as LTI** and eval/plotting need no
  extra flag (the core is read from metadata). Independent of `PREDICT_TYPE`.
- **Tests** — `tests/test_chirp_modal.py`: LTI-equivalence at init, integral
  correctness (ρ̄(0)=0, d/dt ρ̄ = instantaneous ρ), the contraction bound,
  end-to-end `LapFormer` shapes, and checkpoint back-compat.

> **Output routing.** A chirp run is nested under a `modal-chirp/` segment by
> `_apply_modal_type_output_routing` (`pipeline.py`), composing with any
> `predict-<type>/` segment (`predict-x0/modal-chirp/`), so chirp and lti never
> overwrite each other. The default lti adds no segment (historical paths
> unchanged). The segment is applied in `main`/`run_preds` before `base_out_dir`
> is captured, and re-derived in the `run_single_pred` direct-call branch.

---

## 8. Evaluation (`llapdiff-checkpoint-eval`)

**File:** `tools/llapdiff_checkpoint_eval.py` · entry `main()` (line ~805) →
`build_eval_config` (line 49) + `evaluate_checkpoint` (line 610).

What it computes (lines 692-767): one **forecast** case (`forecast_test`) plus
two **target-horizon imputation** cases — a regular/structured keep-mask and a
**random** keep-mask at `--imputation-random-mask-ratio`.

### 🔴 Footgun #4: it loads the VAE/summarizer from **config paths**, not the checkpoint

```python
vae_payload = torch.load(cfg.VAE_CKPT, ...)   # line 370
sum_state   = torch.load(cfg.SUM_CKPT, ...)   # line 395
# LLapDiff itself comes from --checkpoint
```

`--checkpoint` only supplies the diffusion model. The VAE and summarizer are
loaded from `cfg.VAE_CKPT` / `cfg.SUM_CKPT`, which `apply_dataset_preset` derives
from `--dataset-key`/`--pred`. **This is why `USAGE.md` §3.5 makes you stage the
pretrained VAE+summarizer into the `ldt/` tree** — otherwise eval loads the wrong
(or missing) upstream artifacts.

### `predict_type` resolution

`_resolve_checkpoint_predict_type` (line 229) reads the parameterization from
checkpoint metadata. If the checkpoint records it, you don't pass `--predict-type`
(and passing a *conflicting* one raises). Legacy checkpoints without metadata
**require** an explicit `--predict-type`.

---

## 9. "I want to change X" → file map


| You want to…                                                                  | Edit / look at                                                                                              |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Add a dataset or change horizons/context/batch/channels                       | `configs/dataset_defaults.py` (`DATASET_PRESETS`); honor `context = 2×max(horizon)`                         |
| Change how a dataset is windowed / normalized / split                         | the per-dataset loader in `datasets/` (dispatched by `configs/dataset_registry.py`)                         |
| Change train/val/test ratios or split policy                                  | `configs/config.py` (`train_ratio`…), or `--split-policy`; logic in the loaders                             |
| Change global diffusion hyperparams (epochs, LR, layers, MinSNR γ, timesteps) | `configs/dataset_defaults.py` preset fields (they override `config.py`); see §3 footgun about runtime edits |
| Change the diffusion training loss / CFG / self-cond / aux-mask behavior      | `trainers/train_val_llapdiff.py:train_one_epoch` (~2134)                                                    |
| Change checkpoint selection metric / early stop                               | `trainers/train_val_llapdiff.py` (~2493-2850); `PRIMARY_EVAL_METRIC`, `EARLY_STOP`                          |
| Change the diffusion network / Laplace poles                                  | `models/llapdiff.py`, `models/lapformer.py`                                                                 |
| Add/modify the chirp time-varying-pole core (§7.5)                            | `models/laptrans.py` (`ChirpModalField`, `chirp_basis_matrix`), `models/lapformer.py` (chirp branch); toggle/tunables in `configs/config.py` (`DENOISER_MODAL_TYPE`, `CHIRP_*`) + `pipeline.py` (`--modal-type`) |
| Change the VAE architecture / KL schedule / recon loss                        | `latent_space/latent_vae.py`, `trainers/train_val_latent.py`                                                |
| Change the summarizer architecture / its loss weights                         | `models/summarizer.py`, `trainers/train_val_summarizer.py`                                                  |
| Change what the dataloader emits per batch                                    | the per-dataset loader; consumers expect `meta['delta_t'                                                    |
| Change which stages run / the skip logic                                      | `pipeline.py:run_single_pred` (281)                                                                         |
| Change CLI flags                                                              | `pipeline.py:_parse_args` (475) and `tools/llapdiff_checkpoint_eval.py`                                     |
| Change evaluation cases / sample counts                                       | `tools/llapdiff_checkpoint_eval.py:evaluate_checkpoint` (610)                                               |
| Change where artifacts land                                                   | `ARTIFACT_ROOT` in `config.py`; path assembly in `apply_dataset_preset`                                     |


---

## 10. Pitfalls checklist (read before you change anything)

1. **The preset is re-applied inside `run_single_pred**` and resets untracked
  config edits. Change presets/defaults in `dataset_defaults.py`/`config.py`,
   or monkeypatch `apply_dataset_preset`. (§3)
2. **Stages 1-2 skip on file existence only.** A stale or wrong-`C` VAE/summarizer
  at the expected path is silently reused → shape errors or garbage results.
   Use `--recompute-*` or delete the file when in doubt.
3. `**llapdiff-checkpoint-eval` loads VAE/summarizer from config paths, not from
  `--checkpoint`.** Stage them (`USAGE.md` §3.5). (§8)
4. `**--batch-size` counts dates, not samples.** Memory and effective batch scale
  with assets-per-date; the per-dataset preset values are tuned, change with care.
5. **Run from the repo root.** `ARTIFACT_ROOT="./ldt"` is relative to CWD; running
  elsewhere scatters/recreates `ldt/` and breaks the skip logic.
6. **fp16 cache.** The bundled cache stores features/targets as fp16 — fine for the
  benchmark, but don't expect float32 fidelity (rebuild from raw if you need it).
7. **Stale precompute cache.** If `DIFF_PRECOMPUTE_DIR` is set, clear it after
  retraining the VAE/summarizer. (§7.4)
8. `**predict_type` routing & metadata.** `x0`/`eps` runs go under
  `predict-<type>/`; eval needs metadata or an explicit `--predict-type`. (§8)
9. `**context = 2 × max(horizon)`** is asserted by `validate_dataset_presets`.
10. **Calendar features (`DOW_`*,`DOM_*`,`MOY_*`) can't be targets** — they're
  context-only (`USAGE.md` §3.4; enforced in target resolution).
11. **AMP differs by stage/dataset.** Diffusion AMP is off by default
  (`DIFF_AMP=False`); summarizer AMP is off for the four high-context air/NOAA
    datasets. Don't blanket-enable AMP expecting speedups without checking
    stability.
12. **Modal core routing.** `--modal-type chirp` nests outputs under
  `modal-chirp/` (composing with `predict-<type>/`); the default `lti` keeps the
    historical paths. Chirp and lti therefore don't collide, but a chirp
    checkpoint lives at `ldt/output/<ds>/[predict-<t>/]modal-chirp/pred-<H>/…` —
    point `--checkpoint` there for eval. The core is also recorded in the
    checkpoint, so eval rebuilds it regardless. (§7.5)

---

## 11. Common errors and what they mean


| Symptom                                                                               | Likely cause                                               | Fix                                                                                                                                   |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: Dataset cache directory is missing … set LLAPDIFF_DATASET_ZIP`    | cache not extractable and no bundled/known zip found       | run from repo root (bundled zip present), or pass `--dataset-zip` (`dataset_archives.py:62`)                                          |
| `ValueError: <ds>: pred=N not in supported horizons …`                                | horizon not in the preset                                  | use a listed horizon or add it to the preset (§3)                                                                                     |
| `FloatingPointError: non-finite cond_summary / latent / loss / gradients`             | bad/incompatible upstream checkpoint, or AMP instability   | `--recompute-`* the upstream stage; check the staged VAE/summarizer match `C`; the trainer raises rather than poison the model (§7.2) |
| Eval crashes loading VAE/summarizer, or metrics look wrong with a pretrained LLapDiff | VAE/summarizer not staged at `cfg.VAE_CKPT`/`cfg.SUM_CKPT` | stage them per `USAGE.md` §3.5 (§8)                                                                                                   |
| `Checkpoint does not record predict_type metadata; pass --predict-type …`             | legacy checkpoint                                          | add `--predict-type x0` (or the right one)                                                                                            |
| `ValueError: Checkpoint has conflicting predict_type metadata` / `… != …`             | `--predict-type` disagrees with metadata                   | drop the flag and let metadata win                                                                                                    |
| Programmatic `config.EPOCHS = N` (or LR, width…) has no effect                        | reset by the second `apply_dataset_preset`                 | edit the preset, or monkeypatch `apply_dataset_preset` (§3)                                                                           |
| Stage 1/2 "trained" when you expected "skipped" (or vice-versa)                       | checkpoint file presence at the config-derived path        | check the exact path printed by the run; mind `C` in the filename (§5)                                                                |
| Multiple horizons overwrite each other's checkpoints                                  | running stage logic without per-horizon `base_out_dir`     | the CLI handles this via `pred-<H>/`; if calling `run_single_pred` directly, pass `base_out_dir`/`base_ckpt_dir`                      |


