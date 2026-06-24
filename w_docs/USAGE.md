# LLapDiffusion — Usage Guide

This guide walks through getting LLapDiffusion running end-to-end: setting up
the environment, using the **bundled dataset cache** (no manual download
needed), training the three pipeline stages (latent VAE → summarizer →
LLapDiff), evaluating forecast and target-horizon imputation, and running
baselines.

Everything here is derived from `README.md`, `pyproject.toml`, and the CLI
entry points under `llapdiffusion/pipeline.py` and `llapdiffusion/tools/`.

> **Data path used in this guide:** the bundled
> `llapdiffusion/datasets/LLapDiff-evaluation-datasets.zip`. It contains the
> full preprocessed panels and is sufficient to **train from scratch and
> re-evaluate** — you do *not* need to download anything from Hugging Face
> unless you want pretrained checkpoints or want to rebuild caches from raw
> sources (see §3.3).

---

## 1. Repository layout

Source root: `LLapDiffusion-main/`

```
llapdiffusion/
├── pipeline.py             # llapdiff-train  (end-to-end VAE+Summ+LLapDiff)
├── target_artifacts.py
├── benchmark_protocol.py
├── diffusion_cache.py
├── logging_utils.py
├── configs/
│   ├── config.py               # global runtime config (mutated by presets)
│   ├── dataset_defaults.py     # preset table: horizons, contexts, latent dim
│   ├── dataset_registry.py
│   ├── dataset_archives.py     # zip extraction
│   └── config_utils.py
├── datasets/
│   ├── LLapDiff-evaluation-datasets.zip   # bundled full cache (train + eval)
│   ├── bms_air_dataset.py
│   ├── uci_air_quality_dataset.py
│   ├── physionet_cinc_dataset.py
│   ├── noaa_isd_dataset.py
│   ├── fin_dataset.py
│   ├── synthetic_regime_dataset.py
│   ├── target_selection.py
│   └── _normalization.py
├── latent_space/
│   └── latent_vae.py
├── models/
│   ├── llapdiff.py             # Latent-Laplace diffusion core
│   ├── lapformer.py / laptrans.py
│   ├── summarizer.py
│   └── time_utils.py
├── trainers/
│   ├── train_val_latent.py
│   ├── train_val_summarizer.py
│   └── train_val_llapdiff.py
├── tools/
│   ├── llapdiff_checkpoint_eval.py        # llapdiff-checkpoint-eval
│   ├── run_multidataset_artifact_prep.py  # llapdiff-artifact-prep
│   ├── run_synthetic_regime_shift.py      # llapdiff-synthetic-regime
│   └── run_baselines.py                   # llapdiff-baselines
├── viz/
│   └── plot_llapdiff_poles.py             # llapdiff-plot-poles
└── baselines/
    ├── runner.py / registry.py / data.py / metrics.py / sources.py
    └── adapters/   # dlinear, patchtst, mtan, neuralcde, contiformer, csdi,
                    # timegrad, t_patchgnn, mr_diff
```

Installed console scripts (from `pyproject.toml`):


| Command                     | Module entry point                                        |
| --------------------------- | --------------------------------------------------------- |
| `llapdiff-train`            | `llapdiffusion.pipeline:cli_main`                         |
| `llapdiff-checkpoint-eval`  | `llapdiffusion.tools.llapdiff_checkpoint_eval:main`       |
| `llapdiff-artifact-prep`    | `llapdiffusion.tools.run_multidataset_artifact_prep:main` |
| `llapdiff-synthetic-regime` | `llapdiffusion.tools.run_synthetic_regime_shift:main`     |
| `llapdiff-plot-poles`       | `llapdiffusion.viz.plot_llapdiff_poles:main`              |
| `llapdiff-baselines`        | `llapdiffusion.tools.run_baselines:main`                  |


---

## 2. Environment setup

LLapDiffusion requires **Python ≥ 3.11**. We use **conda** to create the
environment and **pip** to install packages (the project itself is installed
with `pip install -e .`).

```bash
cd /home/nvidia-lab/ai4life/thaind2/time_series/LLapDiffusion-main

# 1. Create and activate a conda env on Python 3.11
conda create -n llapdiff python=3.11 -y
conda activate llapdiff

# 2. Upgrade pip inside the env
python -m pip install --upgrade pip

# 3. Core install (PyTorch, NumPy, pandas, matplotlib, pyarrow, fastparquet,
#    yfinance, requests, tqdm) — all via pip
python -m pip install -e .
```

### CUDA build

If you need a specific CUDA wheel of PyTorch, install it via pip *before*
`pip install -e .` using the
[official PyTorch selector](https://pytorch.org/get-started/locally/), e.g.:

```bash
conda activate llapdiff
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

Avoid mixing `conda install pytorch ...` with the pip install — keep PyTorch
managed by pip so it stays consistent with the rest of the project deps.

### Optional extras

```bash
# Baseline adapter dependencies (torchcde, torchdiffeq, gluonts, lightning, ...)
python -m pip install -e ".[baselines]"
# Dev / test
python -m pip install -e ".[dev]"
# NOAA raw download support (only needed if regenerating NOAA caches)
python -m pip install -e ".[noaa-download]"
```

### Removing / recreating the env

```bash
conda deactivate
conda env remove -n llapdiff
```

---

## 3. Datasets and checkpoints

### 3.1 What ships in the box

A compact, ready-to-use cache ships inside the package:

```
llapdiffusion/datasets/LLapDiff-evaluation-datasets.zip
```

Despite the "evaluation" in the name, this is **not a test-only split**. Each
dataset directory holds the full preprocessed panels plus the global window
index, e.g. for `crypto`:

```
fin_dataset/crypto/cache_ratio_index/
├── features_fp16/   # one .npy per asset (full date range)
├── targets_fp16/
├── times/
├── obs_masks_bool/  # observed/missing mask per asset
├── fill_masks_bool/
├── windows/         # global_pairs.npy [asset_id, start_idx], end_times.npy
├── meta.json        # assets, feature_cols, target_col, window, horizon, freq
└── norm_stats.json  # normalization statistics
```

The train / val / test split is **computed at load time**, not baked into the
zip. `pipeline.prepare_dataloaders` calls `run_experiment(..., ratios=(...), split_policy=...)` and the windows are partitioned using the defaults in
`configs/config.py`:

```python
train_ratio = 0.7
val_ratio   = 0.1
test_ratio  = 0.2
```

So the **same cache trains and evaluates** the model. To improve the method
and re-benchmark fairly, just retrain on this cache and re-evaluate on its test
split — the split ratios and `split_policy` are fixed, so the comparison stays
apples-to-apples.

> Note: features/targets are stored as fp16 (compact, slightly lossy). This is
> what the benchmark uses; only switch to a freshly built float32 cache (§3.3)
> if you specifically need full precision.

### 3.2 Auto-extraction (nothing to do)

The first time you run any command for a preset whose cache directory is
absent, the pipeline extracts the matching dataset out of the bundled zip
automatically (`configs/dataset_archives.py`). By default it extracts to a user
cache directory **outside** the installed package:

```
$XDG_CACHE_HOME/llapdiffusion/datasets/      # or ~/.cache/llapdiffusion/datasets/
```

A `.llapdiff_dataset_archive_<hash>.stamp` marker prevents re-extraction on
subsequent runs. To pin the extraction location explicitly:

```bash
export LLAPDIFF_DATASET_EXTRACT_DIR="$PWD/ldt/data"
```

You can confirm what will be used without training:

```bash
python -c "from llapdiffusion.configs.dataset_defaults import validate_dataset_presets; \
import json; print(json.dumps(validate_dataset_presets(['crypto']), indent=2, default=str))"
```

### 3.3 Optional: Hugging Face mirrors

You only need these if you want **pretrained checkpoints** (skip training) or
want to **rebuild caches from raw sources** (change preprocessing, universe,
date range, frequency, or precision). They are *not* required for the
retrain-and-evaluate workflow in this guide.

- Pretrained checkpoints: [https://huggingface.co/pixelhero98/llapdiff-checkpoints](https://huggingface.co/pixelhero98/llapdiff-checkpoints)
- Raw datasets:           [https://huggingface.co/datasets/pixelhero98/llapdiff-raw](https://huggingface.co/datasets/pixelhero98/llapdiff-raw)

```bash
python -m pip install -U "huggingface_hub[cli]"

# Pretrained checkpoints (e.g. to evaluate without training)
hf download pixelhero98/llapdiff-checkpoints --local-dir ./ldt/checkpoints 

# Raw datasets (only if rebuilding caches from scratch)
hf download pixelhero98/llapdiff-raw --repo-type dataset --local-dir ./ldt/data
```

The checkpoint repo (`README_ckpt.md`) bundles several archives:


| Archive                                                | Contents                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llapdiffusion_longest_horizon_artifacts.zip`          | LLapDiff **v-prediction** artifacts for each dataset's longest horizon: VAE (elbo + recon), summarizer, and LLapDiff best. Trained from commit `59f4427`.                                                                                                                                                                                                                                |
| `llapdiffusion_x0_diffusion_checkpoints.zip`           | x0-parameterized LLapDiff diffusion `best`/`last` only (no VAE/summarizer — reuse them from the artifacts archive). Paths under `ldt/output/<dataset>/mode-x0/pred-<H>/`.                                                                                                                                                                                                                |
| `llapdiffusion_baseline_checkpoints.zip`               | 56 extrapolation **baseline** checkpoints (8 methods × 7 datasets); names `checkpoints/<dataset>_h<H>_<method>.pt`. Per `MANIFEST.csv` `completion_mode`: **35 `full_train_loop`** (`dlinear`, `mtan`, `patchtst`, `t_patchgnn`, `timegrad`) and **21 `one_batch_one_epoch_update`** (`contiformer`, `mr-diff`, `neuralcde`) — the latter are plumbing checks, *not* comparable results. |
| `llapdiffusion_csdi_mask30_imputation_checkpoints.zip` | 3 CSDI target-horizon imputation checkpoints (random 30 % holdout, seed 42) for PhysioNet h12, Crypto h100, NOAA UK h168; names `checkpoints/<dataset>_h<H>_csdi_mask30.pt`.                                                                                                                                                                                                             |


After unzipping into `ldt/checkpoints/`, each archive expands to a directory of
the same name (the `.zip` files themselves can be deleted). Verify a download
before unzipping with the published checksum, e.g.:

```bash
# the .sha256 sidecar holds "<hash>  <filename>"; -c recomputes and compares
sha256sum -c llapdiffusion_x0_diffusion_checkpoints.zip.sha256
```

The x0 archive also ships `metadata/sha256sums.txt` (per-file hashes of the
extracted `.pt`/`.json`), so individual files can be checked after the zip is
gone: `cd <x0-dir> && sha256sum -c metadata/sha256sums.txt`.

To point the pipeline at a custom archive / extraction dir (e.g. a rebuilt
cache), pass the flags or set the env vars:

```bash
llapdiff-train --dataset-key crypto \
  --dataset-zip /path/to/your-cache.zip \
  --dataset-extract-dir /path/to/extract

# or, once per shell:
export LLAPDIFF_DATASET_ZIP=/path/to/your-cache.zip
export LLAPDIFF_DATASET_EXTRACT_DIR=/path/to/extract
```

### 3.4 Public dataset keys and preset horizons

From `llapdiffusion/configs/dataset_defaults.py`:


| `--dataset-key` | Context length | Supported `--preds` horizons | VAE latent channels |
| --------------- | -------------- | ---------------------------- | ------------------- |
| `bms_air`       | 336            | 24, 48, 96, 168              | 24                  |
| `uci_air`       | 336            | 24, 48, 96, 168              | 16                  |
| `physionet`     | 24             | 4, 8, 10, 12                 | 16                  |
| `noaa_us`       | 336            | 24, 48, 96, 168              | 24                  |
| `noaa_uk`       | 336            | 24, 48, 96, 168              | 16                  |
| `us_equity`     | 200            | 5, 20, 60, 100               | 12                  |
| `crypto`        | 200            | 5, 20, 60, 100               | 16                  |


Calendar/temporal features (`DOW_*`, `DOM_*`, `MOY_*`) are **context-only** and
cannot be picked as targets.

### 3.5 Using the downloaded pretrained checkpoints

After unzipping `llapdiffusion_longest_horizon_artifacts.zip` you have a flat
folder of 28 checkpoints (4 roles × 7 datasets):

```
ldt/checkpoints/llapdiffusion_longest_horizon_artifacts/
├── checkpoints/
│   ├── <dataset>_h<H>_vae_best_elbo.pt
│   ├── <dataset>_h<H>_vae_best_recon.pt
│   ├── <dataset>_h<H>_summarizer_best.pt
│   └── <dataset>_h<H>_llapdiff_best.pt
└── training_summaries/<dataset>_h<H>.json   # recorded eval_stats (crps/mae/mse)
```

These cover only the **longest horizon per dataset**:


| Dataset     | Horizon `H` | VAE channels `C` |
| ----------- | ----------- | ---------------- |
| `bms_air`   | 168         | 24               |
| `uci_air`   | 168         | 16               |
| `physionet` | 12          | 16               |
| `noaa_us`   | 168         | 24               |
| `noaa_uk`   | 168         | 16               |
| `us_equity` | 100         | 12               |
| `crypto`    | 100         | 16               |


**Important:** `llapdiff-checkpoint-eval` takes the LLapDiff model from
`--checkpoint`, but it loads the **VAE and summarizer from config-derived
paths** (`cfg.VAE_CKPT`, `cfg.SUM_CKPT`). The downloaded files are flat-named,
so you must stage the VAE + summarizer into the `ldt/` tree the pipeline
expects (the LLapDiff file is passed directly and needs no staging):


| Downloaded file                | Expected location                                          |
| ------------------------------ | ---------------------------------------------------------- |
| `<ds>_h<H>_vae_best_elbo.pt`   | `ldt/vae/saved_model/<ds>/pred-<H>_ch-<C>_entity_elbo.pt`  |
| `<ds>_h<H>_vae_best_recon.pt`  | `ldt/vae/saved_model/<ds>/pred-<H>_ch-<C>_entity_recon.pt` |
| `<ds>_h<H>_summarizer_best.pt` | `ldt/summarizer/saved_model/<ds>/<H>-<C>-summarizer.pt`    |
| `<ds>_h<H>_llapdiff_best.pt`   | pass directly to `--checkpoint` (no staging)               |


Stage all seven datasets by **moving** the files into the expected locations
(relative to the repo root). Moving — rather than symlinking — means the real
files live where the pipeline looks for them, so there is no hidden link step
to forget later:

```bash
SRC=ldt/checkpoints/llapdiffusion_longest_horizon_artifacts/checkpoints
# dataset:horizon:channels
for spec in bms_air:168:24 uci_air:168:16 physionet:12:16 \
            noaa_us:168:24 noaa_uk:168:16 us_equity:100:12 crypto:100:16; do
  ds=${spec%%:*}; rest=${spec#*:}; H=${rest%%:*}; C=${rest##*:}
  mkdir -p "ldt/vae/saved_model/$ds" "ldt/summarizer/saved_model/$ds"
  mv "$SRC/${ds}_h${H}_vae_best_elbo.pt"   "ldt/vae/saved_model/$ds/pred-${H}_ch-${C}_entity_elbo.pt"
  mv "$SRC/${ds}_h${H}_vae_best_recon.pt"  "ldt/vae/saved_model/$ds/pred-${H}_ch-${C}_entity_recon.pt"
  mv "$SRC/${ds}_h${H}_summarizer_best.pt" "ldt/summarizer/saved_model/$ds/${H}-${C}-summarizer.pt"
done
```

The `<ds>_h<H>_llapdiff_best.pt` files are **left in place** in `$SRC` — they
are passed directly to `--checkpoint` (see below) and need no staging.

Then evaluate the pretrained LLapDiff checkpoint directly:

```bash
llapdiff-checkpoint-eval \
  --dataset-key crypto --pred 100 \
  --checkpoint ldt/checkpoints/llapdiffusion_longest_horizon_artifacts/checkpoints/crypto_h100_llapdiff_best.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_h100_pretrained_eval.json
```

The `training_summaries/<dataset>_h<H>.json` files record the authors' own
`eval_stats` (e.g. crypto h100: crps ≈ 0.357, mae ≈ 0.467, mse ≈ 0.527 over 25
samples) — useful reference numbers when reproducing or comparing.

### 3.6 The other checkpoint archives

These are unzipped under `ldt/checkpoints/` and keep their own layouts.

**x0 diffusion** (`llapdiffusion_x0_diffusion_checkpoints/`) — x0-parameterized
LLapDiff `best`/`last`, **no VAE/summarizer** (reuse the ones staged in §3.5).
It preserves repo-relative paths:

```
ldt/checkpoints/llapdiffusion_x0_diffusion_checkpoints/
├── ldt/output/<dataset>/mode-x0/pred-<H>/llapdiff_pred-<H>_{best,last}.pt
├── ldt/results/x0/<dataset>_h<H>.json     # authors' x0 eval_stats
└── metadata/{x0_manifest.json,sha256sums.txt,README_x0_diffusion_checkpoints.md}
```

Evaluate after staging the VAE + summarizer (§3.5). The model parameterization
is read from checkpoint metadata; pass `--predict-type x0` only for a legacy
file without it:

```bash
llapdiff-checkpoint-eval \
  --dataset-key crypto --pred 100 \
  --checkpoint ldt/checkpoints/llapdiffusion_x0_diffusion_checkpoints/ldt/output/crypto/mode-x0/pred-100/llapdiff_pred-100_best.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_h100_x0_eval.json
```

> Note: this archive's folder name is `mode-x0/`, but the **current** code
> writes new x0 runs to `ldt/output/<dataset>/predict-x0/...` (see §5.4). The
> difference is cosmetic for evaluation since `--checkpoint` is an explicit
> path; it only matters if you rely on the pipeline auto-discovering an output.

Reference test CRPS from the authors' own runs — `v` (default, from
`training_summaries/`) vs `x0` (from `ldt/results/x0/`), 25 samples, lower is
better:


| Dataset     | `H` | CRPS (`v`, default) | CRPS (`x0`) |
| ----------- | --- | ------------------- | ----------- |
| `bms_air`   | 168 | **0.552**           | 0.696       |
| `uci_air`   | 168 | **1.003**           | 1.251       |
| `physionet` | 12  | **0.367**           | 0.396       |
| `noaa_us`   | 168 | **0.540**           | 0.782       |
| `noaa_uk`   | 168 | **0.570**           | 1.011       |
| `us_equity` | 100 | **0.428**           | 0.544       |
| `crypto`    | 100 | **0.357**           | 0.461       |


`v`-prediction wins on CRPS (the primary selection metric) across all seven
datasets, so treat the x0 checkpoints as an **ablation**, not the headline
result. Point-error metrics (MAE/MSE) mostly agree; the lone exception is
`uci_air`, where x0 has slightly lower MAE/MSE despite higher CRPS.

**Extrapolation baselines** (`llapdiffusion_baseline_checkpoints/`) — flat
`checkpoints/<dataset>_h<H>_<method>.pt` plus `MANIFEST.csv`. These are produced
and consumed by the `llapdiff-baselines` adapters (§6), not by
`llapdiff-checkpoint-eval`. Remember the 35 / 21 full-vs-plumbing split from the
§3.3 table before quoting any baseline number.

**CSDI imputation** (`llapdiffusion_csdi_mask30_imputation_checkpoints/`) — three
`checkpoints/<dataset>_h<H>_csdi_mask30.pt` files plus `MANIFEST.csv`, for the
CSDI target-horizon imputation comparison (§6.2), reported separately from
forecast extrapolation.

---

## 4. Pipeline overview

A full LLapDiffusion run for a (dataset, horizon) pair has three stages, all
driven by a single `llapdiff-train` invocation. Each stage reuses the artifact
from the previous one (and skips it if a checkpoint already exists, unless you
pass `--recompute-`*):

1. **Latent VAE** (`trainers/train_val_latent.py`) — learns the compact
  latent representation of trajectories.
   → `./ldt/vae/saved_model/<dataset>/pred-<H>_ch-<C>_entity_elbo.pt`
2. **History summarizer** (`trainers/train_val_summarizer.py`) — conditions on
  observed values, timestamps, gaps, and masks.
   → `./ldt/summarizer/saved_model/<dataset>/<H>-<C>-summarizer.pt`
3. **LLapDiff** (`trainers/train_val_llapdiff.py`) — denoises latent
  trajectories using Laplace-domain stable poles.
   → `./ldt/output/<dataset>/llapdiff_pred-<H>_best.pt`
     (also `_best_raw.pt`, `_best_ema.pt`, `_last.pt`)

`<C>` is the dataset's VAE latent-channel count (table in §3.4). The artifact
root is `./ldt` (`config.ARTIFACT_ROOT`), created relative to your current
working directory — run commands from the repo root for consistent paths.

Default training length is **600 epochs per stage** (`preset.epochs`), so a
full run is GPU-heavy; expect long runtimes and budget disk for the checkpoints.

Non-default prediction parameterizations (`x0`, `eps`) are routed under
`ldt/output/<dataset>/predict-<type>/...` so they don't overwrite the default
`v`-prediction outputs.

The stage-3 denoiser also has **two interchangeable dynamical cores**, selected
with `--modal-type` (§5.12): the default `lti` (constant Laplace poles + a
residual-MLP correction) and `chirp` (time-varying poles, residual MLP dropped).
This is *orthogonal* to `--predict-type`, and like predict-type it **routes
outputs**: a `chirp` run is nested under a `modal-chirp/` segment (composing with
any `predict-<type>/`), so it never overwrites the default `lti` checkpoints.

---

## 5. Running experiments

> All commands below assume you are in the repo root with the conda env active
> (`conda activate llapdiff`). On first run the bundled dataset cache is
> extracted automatically (§3.2).

### 5.1 Quick-start preset

```bash
llapdiff-train \
  --dataset-key crypto \
  --summary-json ldt/results/crypto_pipeline_summary.json
```

Runs every supported horizon for `crypto` and writes a JSON summary.

### 5.2 Single horizon + recompute artifacts

```bash
llapdiff-train \
  --dataset-key us_equity \
  --preds 100 \
  --recompute-vae \
  --recompute-summarizer \
  --summary-json ldt/results/us_equity_pred100.json
```

`--recompute-vae` / `--recompute-summarizer` force a retrain even if the
cached checkpoint exists.

### 5.3 Multiple horizons

```bash
llapdiff-train --dataset-key noaa_us --preds 24 48 96 168
```

Omit `--preds` to use the preset's full horizon set.

### 5.4 Diffusion parameterization

Default is `v`. Use `x0` or `eps` without changing other hyperparameters:

```bash
llapdiff-train --dataset-key crypto --preds 100 --predict-type x0
llapdiff-train --dataset-key crypto --preds 100 --predict-type eps
```

This chooses the *prediction target*. To switch the denoiser's *dynamical core*
(constant vs time-varying poles) — an independent choice — see §5.12.

### 5.5 Auxiliary target-mask completion training

Mix target-horizon completion batches into LLapDiff training to improve
target-mask imputation at inference time:

```bash
llapdiff-train \
  --dataset-key crypto \
  --preds 100 \
  --target-mask-aux-p 0.30 \
  --target-mask-aux-keep-mode random \
  --target-mask-aux-keep-prob 0.50
```

Default `--target-mask-aux-p 0.0` means standard extrapolation training only.
Available keep-modes: `random | regular | prefix | mixed`. This is a
**training-time** mixing probability, separate from the evaluation-time
`--imputation-random-mask-ratio`.

### 5.6 Target column selection

```bash
# Single scalar target
llapdiff-train --dataset-key crypto --target-col RVOL20_CLOSE --preds 100

# Multi-target
llapdiff-train --dataset-key crypto --target-cols RET_CLOSE RVOL20_CLOSE --preds 100
```

### 5.7 Induced context missingness

`--coverage F` (`0 ≤ F < 1`) hides fraction `F` of observed context entries
before modeling. `--coverage 0` (default) disables it. This is independent of
the loader-internal `panel_coverage` dense-date panel filtering.

### 5.8 Prepare VAE + summarizer artifacts only

Useful for warming caches across all datasets before training LLapDiff:

```bash
llapdiff-artifact-prep \
  --datasets bms_air uci_air physionet noaa_us noaa_uk us_equity crypto \
  --summary-json ldt/results/artifact_prep_summary.json
```

### 5.9 Checkpoint evaluation (forecast + target-horizon imputation)

```bash
llapdiff-checkpoint-eval \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint ldt/output/crypto/llapdiff_pred-100_best.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_eval.json
```

The `--checkpoint` path is the LLapDiff best checkpoint produced by stage 3 of
`llapdiff-train` (see §4). `--imputation-random-mask-ratio 0.30` hides 30 % of
observed target-horizon entries at evaluation time. The parameterization
(`v` / `x0` / `eps`) is inferred from checkpoint metadata; for legacy
checkpoints without metadata, pass `--predict-type` explicitly.

### 5.10 Pole visualization

```bash
llapdiff-plot-poles \
  --dataset-key crypto \
  --pred 100 \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir ldt/results/pole_plot
```

### 5.11 Synthetic regime-shift experiments

```bash
llapdiff-synthetic-regime \
  --protocol-name boundary_crossing \
  --tasks synthetic_freq_shift synthetic_decay_shift \
  --seeds 3407 3408 3409 \
  --output-root ldt/results/synthetic_boundary_crossing
```

### 5.12 Chirp-modal dynamical core (time-varying poles)

By default the LLapDiff denoiser predicts **constant** Laplace poles and patches
the residual with an MLP (`--modal-type lti`). The **chirp** core instead
predicts *time-varying* poles ρₖ(t̃), ωₖ(t̃); its closed-form latent trajectory is
a chirped, time-warped damped sinusoid that is stable by construction, so the
**residual MLP is dropped**:

```bash
llapdiff-train --dataset-key crypto --preds 100 --modal-type chirp
```

- **Independent of `--predict-type`.** It works with `v` (default), `x0`, or
  `eps`; the closed-form trajectory is interpreted as that target. `x0` is the
  most natural reading (the modal sum *is* ẑ₀), e.g.
  `--modal-type chirp --predict-type x0`.
- **Self-describing checkpoints.** The variant is recorded in the checkpoint, so
  `llapdiff-checkpoint-eval` and `llapdiff-plot-poles` rebuild the correct core
  automatically — no extra flag at eval time. Checkpoints from before this
  feature load as `lti`.
- **Outputs are routed by core.** A `chirp` run is nested under a `modal-chirp/`
  segment of `OUT_DIR`/`CKPT_DIR` — e.g.
  `ldt/output/<ds>/modal-chirp/pred-<H>/llapdiff_pred-<H>_*.pt`, or
  `…/predict-x0/modal-chirp/…` when combined with a non-default `--predict-type`.
  The default `lti` keeps the historical paths, so the two never overwrite each
  other. Point `llapdiff-checkpoint-eval --checkpoint` at the `modal-chirp/` file.

Tunables (base config in `configs/config.py`, not per-dataset presets):
`CHIRP_NUM_BASIS` (number of nonnegative Fourier basis functions for the pole
field, default 8), `CHIRP_RHO_MIN` (minimum decay floor `ρ_min`, default 1e-4),
`CHIRP_USE_MLP_RESIDUAL` (re-enable a residual correction, default off).

---

## 6. Baselines

Baseline adapters live under `llapdiffusion/baselines/adapters/`:
`dlinear`, `patchtst`, `mtan`, `neuralcde`, `contiformer`, `csdi`,
`timegrad`, `t_patchgnn`, `mr_diff`.

Most adapters require their **upstream repositories cloned externally**
(MR-Diff is implemented first-party and does not). Point the runner at the
parent directory holding those clones via `--baseline-source-root` or
`LLAPDIFF_BASELINE_SOURCE_ROOT`.

### 6.1 Practical extrapolation suite

```bash
llapdiff-baselines practical-extrapolation \
  --baseline all \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/baseline_runs
```

Multi-target ablations are supported on DLinear / PatchTST:

```bash
llapdiff-baselines practical-extrapolation \
  --baseline dlinear \
  --dataset crypto \
  --target-cols RET_CLOSE RVOL20_CLOSE
```

`--input-policy all_features` is also available for DLinear / PatchTST. Other
extrapolation adapters remain scalar target-only.

### 6.2 CSDI target-horizon imputation

CSDI is reported separately from extrapolation:

```bash
llapdiff-baselines csdi-imputation \
  --dataset all \
  --baseline-source-root /path/to/baseline-sources \
  --imputation-random-mask-ratio 0.30 \
  --output-dir ldt/results/csdi_runs
```

Result JSONs record `comparison_type`, `input_scope`, `missingness_scope`,
`modeling_scope`, `split_note`, and `time_feature_protocol`. PhysioNet is
flagged as the patient-relative-split special case.

---

## 7. Suggested end-to-end workflow (bundled cache)

Retrain from scratch and re-evaluate on a fresh machine, using only the
bundled dataset cache. Example uses `crypto` at horizon `100`; substitute any
`--dataset-key` / `--preds` from §3.4.

```bash
# --- Step 1. Environment (conda for env, pip for packages) ---
cd /home/nvidia-lab/ai4life/thaind2/time_series/LLapDiffusion-main
conda create -n llapdiff python=3.11 -y
conda activate llapdiff
python -m pip install --upgrade pip
python -m pip install -e .                  # add ".[baselines]" to also run §6 baselines

# --- Step 2. (Optional) pin where the bundled cache extracts ---
# Defaults to ~/.cache/llapdiffusion/datasets if you skip this.
export LLAPDIFF_DATASET_EXTRACT_DIR="$PWD/ldt/data"

# --- Step 3. Train the full pipeline (VAE -> summarizer -> LLapDiff) ---
# First run auto-extracts crypto from the bundled zip. ~600 epochs/stage.
llapdiff-train \
  --dataset-key crypto \
  --preds 100 \
  --recompute-vae \
  --recompute-summarizer \
  --summary-json ldt/results/crypto_pred100.json

# Produces: ldt/output/crypto/llapdiff_pred-100_best.pt

# --- Step 4. Evaluate on the held-out test split (forecast + imputation) ---
llapdiff-checkpoint-eval \
  --dataset-key crypto --pred 100 \
  --checkpoint ldt/output/crypto/llapdiff_pred-100_best.pt \
  --imputation-random-mask-ratio 0.30 \
  --out-json ldt/results/crypto_pred100_eval.json

# --- Step 5. (Optional) visualise learned poles ---
llapdiff-plot-poles \
  --dataset-key crypto --pred 100 \
  --checkpoint ldt/output/crypto/llapdiff_pred-100_best.pt \
  --output-dir ldt/results/pole_plot

# --- Step 6. (Optional) baselines for comparison (needs ".[baselines]") ---
llapdiff-baselines practical-extrapolation \
  --baseline all --dataset crypto \
  --baseline-source-root /path/to/baseline-sources \
  --output-dir ldt/results/baseline_runs
```

To re-run all horizons for a dataset, drop `--preds`. To skip retraining the
VAE/summarizer after the first run (e.g. only re-tune LLapDiff), omit the
`--recompute-*` flags so cached stage-1/stage-2 artifacts are reused.

If you instead want to evaluate the authors' pretrained model without
training, skip steps 3 and use the downloaded checkpoints: stage the VAE +
summarizer per §3.5, then point `--checkpoint` at the matching
`<dataset>_h<H>_llapdiff_best.pt`.

---

## 8. Useful flags cheat sheet (`llapdiff-train`)


| Flag                                         | Effect                                                 |
| -------------------------------------------- | ------------------------------------------------------ |
| `--dataset-key KEY`                          | Required. Selects preset (table in §3.4).              |
| `--preds H1 [H2 ...]`                        | Subset of preset horizons; omit for all.               |
| `--predict-type {v,x0,eps}`                  | Diffusion parameterization (default `v`).              |
| `--modal-type {lti,chirp}`                   | Denoiser dynamical core: constant poles + MLP (`lti`, default) or time-varying poles (`chirp`, §5.12). |
| `--coverage F`                               | Hide `F` of observed context entries (`0 ≤ F < 1`).    |
| `--batch-size N`                             | Override preset batch size.                            |
| `--target-col COL` / `--target-cols`         | Single or multi-target forecasting.                    |
| `--recompute-vae` / `--recompute-summarizer` | Force retrain of upstream stages.                      |
| `--latent-plot-only`                         | Skip latent training, render plots only.               |
| `--no-shared-loaders`                        | Each stage builds its own dataloaders.                 |
| `--summary-json PATH`                        | Write a compact JSON summary of the run.               |
| `--run-checkpoint-eval`                      | After training, run forecast + imputation eval.        |
| `--checkpoint-eval-random-mask-ratio`        | Random-mask fraction for the optional post-train eval. |
| `--target-mask-aux-p P`                      | Mix completion batches into training with prob `P`.    |
| `--target-mask-aux-keep-mode MODE`           | `random`                                               |
| `--target-mask-aux-keep-prob P`              | Observed-target keep prob for `random` mode.           |
| `--target-mask-aux-keep-stride S`            | Keep stride for `regular` mode.                        |a
| `--target-mask-aux-start-epoch E`            | First epoch at which aux batches begin.                |
| `--dataset-zip PATH`                         | Override bundled dataset zip.                          |
| `--dataset-extract-dir PATH`                 | Override extraction location.                          |
| `--split-policy POLICY`                      | `global_purged_horizon`                                |
| `--calendar-day-batches`                     | Legacy calendar-day batching (else exact-timestamp).   |
| `--verbose` / `--debug`                      | Trainer logging verbosity.                             |


---

## 9. Citation and licensing

- Preprint: [https://arxiv.org/abs/2605.19805](https://arxiv.org/abs/2605.19805)
- License: MIT (see `LICENSE`)
- Derived dataset caches in `LLapDiff-evaluation-datasets.zip` remain governed
by each source's original terms.

