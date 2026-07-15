# CMD hyperparameter tuning harness

Finds the best hyperparameters **per setting** — per `(dataset, horizon, arm)` —
and writes the winners plus their scores to disk after every run.

- **Selection metric:** CRPS, evaluated on the **test** split with the
  checkpoint's **EMA** weights (`--select-split test --select-weights ema`,
  the defaults). Lower is better.
- **Not searched:** `predict_type` — `v` is settled as best (USAGE.md §3.6).
- **Searched:** learning rate, LR schedule, warmup fraction, Min-SNR γ (every
  arm), plus the chirp knobs (chirp arms only), plus the sampling-time knobs
  (guidance, DDIM steps) with no retraining.

> ⚠️ **Read this once.** Selecting on the test split means the test CRPS you
> report is *selection-biased* — it is no longer a clean held-out number, and
> `cmd_plan_v2.md` §1 pre-registers validation-split selection. The harness
> obeys your choice but stamps a warning into `RESULTS.md` whenever
> `--select-split test` was used. For the paper's headline table, rerun the
> same campaign with `--select-split val --run-tag v1-val`; everything else is
> identical, and the two campaigns do not overwrite each other.

---

## 1. Files in this folder

| File | Role |
|---|---|
| `tune.py` | **The orchestrator — the only script you normally run.** |
| `search_spaces.py` | **The grids. Edit here** to add/remove knobs or values. |
| `run_trial.py` | Worker: trains one stage-3 trial (handles the preset footgun). |
| `eval_sampling.py` | Worker: scores a checkpoint over sampling cells on a split. |
| `final_eval.py` | Worker: full forecast + imputation protocol on the test split. |
| `common.py` | Shared helpers (paths, trial-id hashing, state IO). |

---

## 2. Setup (once)

```bash
cd <repo-root>                 # ALWAYS run from the repo root (ldt/ is CWD-relative)
conda activate llapdiff

# Shared frozen stage-1/2 artifacts, ONCE per (dataset, horizon).
# Every trial, arm and seed reuses the same VAE + summarizer — the parity requirement.
llapdiff-artifact-prep --datasets physionet --summary-json ldt/results/prep_physionet.json
```

`tune.py` refuses to start if the shared VAE/summarizer are missing (or pass
`--allow-upstream-training` to let the first trial train them).

---

## 3. The three commands

### (a) Plan / smoke — validate before spending GPU time

```bash
# Print the trials that would run; touches nothing.
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d --dry-run

# End-to-end plumbing check: 3-epoch trials, 4-sample ensembles, isolated run-tag "smoke".
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d --smoke
```

Smoke numbers are meaningless by design; they live under `results/smoke/` and
are marked `(SMOKE)` in the report.

### (b) Tune — train sweep + sampling sweep

```bash
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d a --run-tag v1
```

Per arm, in order:

1. **Baseline trial** — current defaults (this is the plan's G3 cell as-is).
2. **Training sweep** — coordinate descent, one stage at a time, each stage
   sweeping its candidates on top of the current best config:
   `base_lr → lr_schedule → warmup_frac → minsnr_gamma` on every arm, then
   `chirp_num_basis → chirp_rho_min → growth_budget → chirp_coeff_l2` on the
   chirp arms (c, d). Each trial trains **stage-3 only** (the trainer's own
   test pass is skipped), then is scored: CRPS on the selection split with the
   selection weights, at the default sampling protocol.
3. **Sampling sweep** — on the incumbent checkpoint, no retraining. Two
   guidance families (both crossed with DDIM steps {32, 64}):
   - **constant** CFG weight ∈ {1.0, 1.25, 1.5, 2.0} — 8 cells;
   - **scheduled/ramp** CFG `(g_min, g_max)` ∈ {(1.0, 2.0), (1.0, 3.0)}, whose
     shape is set by `guidance_power` ∈ {0.3, 1.0, 2.0} — 12 cells.
   = **20 cells**, scored the same way. `guidance_power` only affects ramp
   cells: for a constant weight the sampler ignores it (`llapdiff.py:328`).

**Everything is resumable.** Rerun the exact command and finished trials/cells
are skipped; failed ones are retried. Interrupt with Ctrl-C any time.

Trial count per arm: 1 baseline + 4 (lr) + 2 (schedule) + 3 (warmup) + 3 (γ)
= **13**, plus **11** more on chirp arms = **24**. Plus 15 sampling cells (no
training). At ~2 min/trial on physionet h12 that is roughly an hour per arm;
NOAA-UK h168 trials are far slower — trim with `--stages`.

Useful flags:

| Flag | Effect |
|---|---|
| `--arms d` / `--arms a b c d` | Which factorial cells to tune (default `d` = CMD). |
| `--select-split {test,val}` | Split the selection CRPS is computed on. Default `test`. |
| `--select-weights {ema,raw}` | Weight source scored. Default `ema`. |
| `--stages base_lr minsnr_gamma` | Sweep only these stages (names from `search_spaces.py`). |
| `--include-tier3` | Also sweep capacity (`MODEL_WIDTH`, `LAPLACE_K`). |
| `--seed 0` | Seed for tuning trials (selection uses one seed). |
| `--run-tag v1` | Campaign namespace. A new tag = a fresh campaign; old results untouched. |
| `--phase train` / `--phase sampling` / `--phase report` | Run a single phase. |
| `--num-samples N` | Eval ensemble size. **Keep the default 25** for anything reported. |

### (c) Final — multi-seed test numbers with imputation

```bash
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d \
  --run-tag v1 --phase final --final-seeds 0 1 2 3 4
```

Retrains the winning configuration at each seed (the tuning seed reuses its
existing checkpoint) and runs the **full** `llapdiff-checkpoint-eval` protocol
on test — forecast + regular-keep imputation + 30 %-random-mask imputation —
with the chosen sampling knobs, then reports mean ± std across seeds.

---

## 4. Every file a run generates

### 4.1 Under `finetuning/results/<run-tag>/`

```
finetuning/results/v1/
├── RESULTS.md                                  ← human summary (all settings)
└── physionet_h12/                              ← one dir per (dataset, horizon)
    └── d/                                      ← one dir per arm
        ├── state.json                          ← full campaign state (resume lives here)
        ├── BEST.json                           ← the winning hyperparameters
        ├── trials/<trial-id>/                  ← one dir per trained configuration
        │   ├── summary.json                    ← llapdiff-train's own summary
        │   ├── train.log                       ← full trainer stdout/stderr
        │   ├── select_test_ema.json            ← the selection score for this trial
        │   └── select_eval.log
        ├── sampling/
        │   ├── <trial-id>_test_ema.json        ← the 15 sampling cells
        │   └── sweep.log
        └── final/                              ← only after --phase final
            ├── seed0_test.json                 ← full test protocol, seed 0
            ├── seed0_test.log
            └── … one pair per seed
```

`<trial-id>` is a 10-char hash of (dataset, horizon, arm, seed, overrides), so
the same configuration always maps to the same directory and is never trained
twice.

### 4.2 Under `ldt/tuning/<run-tag>/` (model weights)

```
ldt/tuning/v1/physionet_h12/d/<trial-id>/output/modal-chirp/seed-0/pred-12/
├── llapdiff_pred-12_best_ema.pt    ← best val-CRPS(EMA) epoch — the one the harness scores
├── llapdiff_pred-12_best.pt        ← best by the primary metric (same rule here)
└── llapdiff_pred-12_last.pt        ← final epoch
```

The inner `modal-chirp/seed-0/pred-12/` segments are the pipeline's own arm/seed
routing. Trials are rooted under `ldt/tuning/` so they can **never** collide
with the paper runs in `ldt/output/`. Delete `ldt/tuning/<run-tag>/` to reclaim
disk; `results/` (the numbers) is unaffected.

> A `*_emaweights.pt` sibling also appears after a `weights: ema` **final** eval:
> `llapdiff-checkpoint-eval` always loads the `model` key, so `final_eval.py`
> writes a copy whose `model` entries are the EMA shadow.

**Which checkpoint gets scored — and the config that makes it work.** Every
checkpoint payload stores *both* raw weights (`model`) and the EMA shadow
(`ema`); "best_ema" names the **epoch-selection rule**, not the stored tensors.
With `--select-weights ema` the harness loads the `ema` tensors, so the reported
number is "test CRPS of the EMA weights at the best val-CRPS(EMA) epoch".

This depends on three `config.py` values that must agree:

| Config | Value | Why |
|---|---|---|
| `PRIMARY_EVAL_METRIC` | `"crps"` | Checkpoints are chosen by CRPS, not latent MSE. |
| `VAL_METRIC_SOURCE` / `TEST_METRIC_SOURCE` | `"ema"` | That CRPS is measured on the EMA weights. |
| `DOWNSTREAM_EVAL_EVERY` | `5` | **Gates the val CRPS eval itself.** |

⚠️ If `DOWNSTREAM_EVAL_EVERY = 0` while `PRIMARY_EVAL_METRIC = "crps"`, the val
CRPS is never computed, so the trainer's `current_primary_metric` stays `None` —
which guards *both* the checkpoint save *and* the early-stop counter. The result
is silent: **no `_best*.pt` is written, early stopping never fires (every run
burns all 600 epochs), and the last epoch is scored.** `run_trial.py` refuses to
start in that state. At `5`, val CRPS runs every 5th epoch; note early stopping
counts *evals*, so `EARLY_STOP = 20` now means 100 epochs without improvement —
lower it if you want tighter stops.

---

## 5. The JSON files, field by field

### 5.1 `BEST.json` — the answer

The winning configuration for one setting. This is the file to read.

```json
{
  "dataset": "physionet",
  "pred": 12,
  "arm": "d",
  "arm_label": "chirp − head (CMD)",
  "smoke": false,
  "selection": {
    "split": "test",          // split the selection CRPS was computed on
    "weights": "ema",         // weight source scored (ema shadow vs raw)
    "crps": 0.3671,           // ← the score that won. Lower is better.
    "mae": 0.4103,
    "mse": 0.5210
  },
  "training_overrides": {     // ← APPLY THESE to reproduce the winner
    "cli":    {"predict_type": "x0"},        // llapdiff-train flags
    "config": {"BASE_LR": 0.0003,            // config attributes
               "LR_SCHEDULE": "warmup_cosine"}
  },
  "sampling": {               // ← chosen sampling knobs (no retraining needed)
    "guidance": 1.25,         // number = constant weight; [1.0, 2.0] = ramp;
                              // null = keep the config default ramp
    "steps": 32,              // DDIM steps
    "weights": "ema",
    "guidance_power": null,   // ramp exponent (null unless the winner is a ramp)
    "dynamic_thresh_p": null  // null = config default (0.0 = thresholding off)
  },
  "checkpoint": "ldt/tuning/v1/physionet_h12/d/<trial>/output/.../llapdiff_pred-12_best.pt",
  "final": { … }              // present only after --phase final; see §5.5
}
```

`training_overrides` is empty (`{}`) when the defaults won.

**To reproduce a winner outside the harness:** set each `config` key in
`llapdiffusion/configs/config.py` (for preset-stamped knobs like `BASE_LR` and
`MINSNR_GAMMA` you must edit `dataset_defaults.py` or wrap
`apply_dataset_preset` — DEVELOPER_GUIDE §3), pass each `cli` key as a
`llapdiff-train` flag, and set the `sampling` knobs (`GUIDANCE_STRENGTH`,
`GEN_STEPS`, …) before evaluating.

### 5.2 `state.json` — campaign state (resume + audit trail)

```json
{
  "dataset": "physionet", "pred": 12, "arm": "d",
  "arm_label": "chirp − head (CMD)",
  "run_tag": "v1",
  "tune_seed": 0,                    // seed used for all selection trials
  "smoke": false,
  "selection": {"split": "test", "weights": "ema"},
  "incumbent": "ee8f25436d",         // trial-id of the current best configuration
  "updated_at": "2026-07-14 05:10:22",

  "trials": {
    "ee8f25436d": {
      "overrides": {},               // {} = the defaults (the baseline trial)
      "stage": "baseline",           // which sweep stage produced this trial
      "seed": 0,
      "status": "done",              // pending | trained | done
                                     //   | failed_train | failed_score
      "created_at": "2026-07-14 04:55:01",
      "train_wall_s": 112.4,         // training wall-clock, seconds
      "train_log":    ".../trials/ee8f25436d/train.log",
      "summary_json": ".../trials/ee8f25436d/summary.json",
      "checkpoint":   "ldt/tuning/.../llapdiff_pred-12_best.pt",

      // What the TRAINER used to pick its best epoch (val CRPS on EMA weights).
      // Related to, but not the same as, the harness's selection score below:
      // this one is on VAL, the selection score is on the split you chose.
      // A null here means no best checkpoint was saved — see the ⚠️ in §4.2.
      "trainer_best_primary_metric": 0.3712,
      "trainer_best_primary_metric_name": "crps",

      // The harness's selection score for this trial:
      "select": {
        "split": "test", "weights": "ema",
        "crps": 0.3671, "mae": 0.4103, "mse": 0.5210,
        "file": ".../trials/ee8f25436d/select_test_ema.json"
      }
    }
    // … one entry per configuration tried
  },

  "sampling": {
    "checkpoint_trial": "ee8f25436d",   // sweep is tied to this trial's checkpoint
    "split": "test", "weights": "ema",
    "rows": { "<cell-key>": { …one row per cell, see §5.4… } },
    "best": { …the lowest-CRPS row… }
  },

  "final": { …see §5.5… }
}
```

If `selection` (split/weights) changes between runs, trials are **re-scored**
under the new rule; the trained checkpoints are reused, not retrained.

### 5.3 `trials/<id>/summary.json` — the trainer's own output

Written by `llapdiff-train` (not by this harness). Per-horizon results nest
under a top-level `"results"` key:

```json
{
  "created_at_utc": "…", "dataset_key": "physionet",
  "results": {
    "12": {
      "pred": 12,
      "vae":        {"status": "skipped", "reason": "checkpoint_exists"},  // shared artifact
      "summarizer": {"status": "skipped", "reason": "checkpoint_exists"},  // shared artifact
      "llapdiff": {
        "train_losses": [...], "train_history": [...],   // per-epoch diagnostics
        "val_history": [...],                            // per-eval val metrics
        "best_primary_metric": 0.3712,                   // best val CRPS (EMA) seen
        "best_primary_metric_name": "crps",              // null => no best ckpt saved (§4.2)
        "final_test_eval": {"status": "skipped"}         // the harness scores it instead
      },
      "eval_stats": {"status": "skipped", …},            // skipped during tuning by design
      "loaded_checkpoint": "ldt/tuning/.../llapdiff_pred-12_best.pt",  // ← the harness reads this
      "data_policy": {"split_policy": "contiguous", …}
    }
  }
}
```

`vae`/`summarizer` reporting `skipped (checkpoint_exists)` is **correct and
required**: every trial reuses the same frozen stage-1/2 artifacts.

### 5.4 `select_<split>_<weights>.json` and `sampling/<trial>_<split>_<weights>.json`

Same shape — a list of evaluated sampling cells. The selection file has exactly
one row (the default protocol); the sampling file has the 15 grid cells.

```json
{
  "dataset": "physionet", "pred": 12,
  "checkpoint": "ldt/tuning/.../llapdiff_pred-12_best.pt",
  "split": "test",            // which split these numbers come from
  "num_samples": 25,          // ensemble size used for CRPS (protocol value = 25)
  "rows": [
    {
      "guidance": 1.25,       // number = CONSTANT CFG weight;
                              // [1.0, 2.0] = SCHEDULED (g_min, g_max) ramp;
                              // null = keep the config default ramp
      "steps": 32,            // DDIM steps
      "weights": "ema",       // weight source scored
      "guidance_power": null, // ramp shape exponent — only meaningful when
                              // guidance is a ramp; null/ignored for a constant weight
      "dynamic_thresh_p": null,
      "status": "ok",         // "ok" | "skipped_no_ema"
      "crps": 0.3671,         // ← Continuous Ranked Probability Score (lower better).
                              //   The probabilistic metric: scores the whole
                              //   predictive distribution, not just the mean.
      "mae": 0.4103,          // mean absolute error of the sample mean
      "mse": 0.5210,          // mean squared error of the sample mean
      "clip_fraction_mean": 0.0023   // fraction of latent values hit by dynamic
                                     // thresholding (0.0 when it is off)
    }
  ]
}
```

Every cell is evaluated with the **same generator seed**, so cells differ only
by their knobs, not by sampling noise.

### 5.5 `final/seed<N>_test.json` — the full test protocol

This is `llapdiff-checkpoint-eval`'s standard payload (not harness-specific).
Three evaluation cases on the test split:

```json
{
  "label": "physionet_pred12",
  "checkpoint": "…",
  "predict_type": "v",
  "predict_type_source": "checkpoint_metadata",   // parameterization read from the ckpt

  "forecast_test": {              // CASE 1 — pure extrapolation (the headline)
    "crps": 0.3671,               //   probabilistic accuracy, lower better
    "mae": 0.4103,                //   point accuracy of the sample mean
    "mse": 0.5210,
    "pinball": {"0.1": 0.24, "0.5": 0.22, "0.9": 0.19},  // quantile loss per quantile
    "num_samples": 25,            //   ensemble size (must be 25 for reported numbers)
    "aggregation": "mean"
  },

  "regular_keep25": {             // CASE 2 — imputation, structured keep-mask (stride 4)
    "hidden_crps": 0.46,          //   ← the metric that matters: scored ONLY on the
    "hidden_mae":  0.47,          //     entries that were hidden from the model
    "hidden_mse":  1.30,
    "observed_mae": 0.39,         //   sanity check: error on the entries it could see
    "hidden_token_frac": 0.75,    //   fraction of target entries hidden (0.75 = keep 1 in 4)
    "observed_token_frac": 0.25,
    "metric_target_type": "target_horizon_imputation"
  },

  "random_mask_ratio": 0.30,
  "random_mask": {                // CASE 3 — imputation, random 30 % of entries hidden
    "hidden_crps": 0.4483, …      //   same fields as above
  },
  "random_mask30": { … },         // alias of random_mask, present when the ratio is exactly 0.30

  "balanced_summary": {
    "avg_hidden_crps": 0.4557,    // mean of the two imputation cases' hidden_crps
    "passes_forecast_guardrail": null   // only set when a baseline CRPS is supplied
  },

  "data_policy": {                // provenance — what was actually modeled
    "target_col": "HR", "target_dim": 1,
    "split_policy": "contiguous",
    "split_scope": "physionet_patient_relative_time",
    "split_note": "patient_relative_contiguous_split"
  },
  "benchmark_protocol": {"comparison_type": "extrapolation", …}
}
```

**Metric glossary.** `crps` — Continuous Ranked Probability Score, the primary
probabilistic metric (a proper scoring rule: rewards a well-calibrated
distribution, not just a good mean). `mae`/`mse` — point error of the ensemble
mean. `pinball` — quantile (check) loss at the 10th/50th/90th percentiles.
`hidden_*` — computed only on entries the model could not see (the actual
imputation task); `observed_*` — on entries it could see (a sanity check, not a
result). All are **lower-is-better**.

### 5.6 `RESULTS.md`

Auto-regenerated after every invocation (or `--phase report` alone). One table
row per setting — best training config, best sampling cell, selection CRPS,
final test CRPS mean ± std — followed by a per-setting section with the full
trial ranking and the per-seed final numbers. Carries the selection-bias warning
banner whenever `--select-split test` was used.

---

## 6. Design guarantees

- **Preset footgun handled.** `run_trial.py` wraps `apply_dataset_preset`, so
  overrides of preset-stamped knobs (`BASE_LR`, `MINSNR_GAMMA`, `MODEL_WIDTH`,
  `LAPLACE_K`) survive the double re-stamp (DEVELOPER_GUIDE §3). A plain
  `config.X = …` would be silently reverted.
- **Parity.** All trials of all arms share the identical frozen VAE/summarizer
  (those paths are never trial-routed); Tier-1 grids are identical across arms
  by construction; every eval uses the same generator seed.
- **No accidental test contact during training.** Trials train with
  `FINAL_TEST_EVAL="skip"` — the *trainer* never touches test. Test is read only
  by the harness's own scoring passes (which, with `--select-split test`, is
  where the selection bias comes from — see the banner at the top).
- **Deduplication.** Override values equal to the resolved default are dropped
  before hashing, so equivalent configs never train twice.
- **`WARMUP_FRAC` coupling.** When `EARLY_STOP_MIN_EPOCHS == 0` (the default),
  the trainer derives it as `ceil(WARMUP_FRAC × EPOCHS)` — so a larger warmup
  also delays the earliest possible early stop, i.e. those trials train longer.
  Pin `EARLY_STOP_MIN_EPOCHS` in the same override to decouple.
- ⚠️ Any chirp checkpoint must postdate the 2026-07-05 ε-init fix
  (CMD_RUNBOOK §0). Everything this harness trains does; audit anything you
  import by hand.

---

## 7. Testing a hyperparameter that isn't in the grid

Two questions decide how much work it is: **(1)** does the knob change *training*
(needs one trained trial per value) or only *sampling* (no retraining)? **(2)** is
it a plain `config.py` attribute (almost always yes)?

### 7.1 Training or sampling knob?

- **Training** — anything the stage-3 trainer reads while optimizing:
  `WEIGHT_DECAY`, `DROP_COND_P`, `EMA_DECAY`, `GRAD_CLIP`, `TIMESTEPS`,
  `SCHEDULE`, `SELF_COND`, `TARGET_MASK_AUX_*`, and the preset-stamped ones
  (`BASE_LR`, `MINSNR_GAMMA`, `MODEL_WIDTH`, `LAPLACE_K`, `EPOCHS`). → **§7.2**
- **Sampling** — anything used only at inference / DDIM: `GEN_ETA`, `KARRAS_RHO`,
  the aggregation method, and the two already-plumbed ones `guidance_power` /
  `dynamic_thresh_p`. No retraining, so much cheaper. → **§7.3**

### 7.2 Training knob → add one stage line

Add a tuple to the right tier list in `search_spaces.py`:

```python
("stage_name", "config", "CONFIG_ATTR", [candidate, values, ...])
```

- **Use `kind="config"`** for essentially every training knob — the value is
  stamped inside the wrapped `apply_dataset_preset`, so it survives the preset
  re-stamp (DEVELOPER_GUIDE §3) for both preset-stamped and base-config knobs.
  (`kind="cli"` is wired only for `predict_type`; another CLI flag needs an edit
  in `run_trial.py`, near the `--predict-type` handling.)
- **Which tier:** `TIER1_STAGES` if it applies to every arm (identical grid = the
  matched-budget parity rule); `TIER2_STAGES` for chirp-only knobs (`CHIRP_*`);
  `TIER3_STAGES` for capacity (needs `--include-tier3`).
- **`stage_name` must be unique** — it is what you pass to `--stages` and what
  labels the trial in `RESULTS.md`.
- Listing the current default among the candidates is safe: default-valued
  candidates are auto-dropped (they would duplicate the baseline trial).

Worked example — sweep weight decay on every arm. Add to `TIER1_STAGES`:

```python
("weight_decay", "config", "WEIGHT_DECAY", [5e-4, 1e-3, 3e-3]),
```

Then inspect and run just that stage into a throwaway campaign:

```bash
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d \
  --run-tag probe_wd --stages weight_decay --dry-run    # see the planned trials
python finetuning/tune.py --dataset-key physionet --preds 12 --arms d \
  --run-tag probe_wd --stages weight_decay              # baseline + 3 WD trials, scored
```

Read the effect in `results/probe_wd/physionet_h12/d/RESULTS.md` (trials ranked by
selection CRPS). Cost = one stage-3 training run per non-default value.

**Isolated vs. continued.** With a **fresh** `--run-tag`, `--stages weight_decay`
compares the values against the plain baseline (defaults). With an **existing**
campaign's `--run-tag`, the candidates build on the current *incumbent*
(coordinate-descent continuation) — use that to measure the knob on top of an
already-tuned config. Editing the grid never invalidates prior trials; re-running
an existing `--run-tag` just appends the new ones (and may move the incumbent).

**Gotcha — the attribute name must be real.** `run_trial.py` does
`setattr(config, "CONFIG_ATTR", value)`; a mistyped name sets an unused attribute
with no effect, and the tell is that every candidate in the stage scores
*identically*. Confirm the name exists in `llapdiffusion/configs/config.py` (or is
actually read by the trainer) before spending GPU. Mind coupling knobs too (e.g.
`WARMUP_FRAC` also moves the early-stop floor — §6).

### 7.3 Sampling knob → no retraining

Sampling cells are scored on the incumbent checkpoint, so these are cheap. Two cases:

**(a) Already plumbed (`guidance_power`, `dynamic_thresh_p`).** Just emit cells that
carry the key from `sampling_cells()` in `search_spaces.py`. ⚠️ Any key that
*varies* between cells **must** be in `SAMPLING_KEY_FIELDS` (`tune.py`), or cells
differing only in that key collide on resume (one overwrites the other).
`guidance_power` is already there; **`dynamic_thresh_p` is not** — add it before
sweeping it.

**(b) A genuinely new sampling knob** (e.g. `GEN_ETA` / DDIM η, currently pinned to
`0.0` in `eval_sampling.py`). Four edits, mirroring how `dynamic_thresh_p` is
wired:
1. `search_spaces.py` `sampling_cells()` — emit the new key on each cell.
2. `eval_sampling.py` — read `cell.get("my_knob")` into the `row` dict and into
   the `sampling` kwargs (the block that maps `dynamic_thresh_p`).
3. `tune.py` — add the key to `SAMPLING_KEY_FIELDS` (dedup/resume).
4. `final_eval.py` — set the matching `config.*` from the chosen cell, so the
   final test read uses the winning value.

### 7.4 Just trying a different fixed value (no sweep)

If you don't need a per-value search — you only want one different **fixed** value
for the whole campaign — skip the grid: edit `llapdiffusion/configs/config.py` for
a base-config knob, or the preset row in `configs/dataset_defaults.py` for a
preset-stamped one (a runtime `config.X = …` is reset by the preset —
DEVELOPER_GUIDE §3). Note it in your run log; it won't appear in `BEST.json`
because the harness didn't choose it.

### 7.5 Fully manual A/B (bypassing tune.py)

To probe one training knob without editing `search_spaces.py`, drive the workers
directly (throwaway `--run-tag`), then score each checkpoint on val with EMA weights:

```bash
for wd in 5e-4 1e-3; do
  python finetuning/run_trial.py --dataset-key physionet --pred 12 --arm d --seed 0 \
    --trial-id probe_wd_$wd --run-tag probe \
    --overrides-json "{\"config\": {\"WEIGHT_DECAY\": $wd}}" \
    --summary-json /tmp/wd_$wd.json
  ckpt=$(python -c "import json;print(json.load(open('/tmp/wd_$wd.json'))['results']['12']['loaded_checkpoint'])")
  python finetuning/eval_sampling.py --dataset-key physionet --pred 12 --checkpoint "$ckpt" \
    --cells-json '[{"weights":"ema"}]' --split val --out-json /tmp/wd_${wd}_score.json
done
```

Prefer the `--stages` route in §7.2 — it scores, dedups, and writes `RESULTS.md`
for you. Reach for this only when you are deliberately outside the orchestrator.
