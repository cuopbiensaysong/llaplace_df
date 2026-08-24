# USHCN sliding 24/12 — first results

Written 2026-08-22. **All numbers are the validation split; the test split has never been read.**
Raw artifacts, scripts and logs live **outside** the framework checkout, in
`fixed_bugs/_ushcn_experiments/`. Nothing in `llaplace_df` was modified to produce any of this.

---

## 1. Verdict

USHCN is **usable and worth adding**, on the strength of one result: `chirp_gp` wins the
variogram score **10/10 seeds** against both same-encoder Gaussian baselines (p = 0.0010), on a
learning-rate grid that brackets every head's optimum, with **complete per-seed separation** —
chirp's worst seed (72.15) beats the baselines' best (77.80). That reproduces the noaa_uk pattern on
**the first genuinely irregular cell in the framework**, which is what handoff §5's framing
problem needs.

Two things are **not** established and must not be quoted: the Student-t baseline is degenerate
at its selected learning rate (§6), and Gate 0's headline is not comparable to the other cells'
(§4).

---

## 2. Why this cell matters

Handoff §5 recorded the unresolved problem: *"No cell is both usable and genuinely irregular.
The paper's §10 framing problem — the title promises irregular series and the evaluation cell is
a fixed regular grid — is therefore not solved by bms_air."*

| | bms_air | noaa_uk | crypto | physionet | **USHCN 24/12** |
|---|---|---|---|---|---|
| grid regularity | 0.996 | 1.000 | 1.000 | 1.000 | **0.218** |
| observed rate | 0.992 | 1.000 | 0.937 | 1.000 | **0.220** |
| lead gap == 1 step | 99.6 % | 100 % | 100 % | 100 % | **22.1 %** |
| lead gap p50 / p95 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | **3 / 13 days** |

Measured independently on a live batch: first lead gap median **4 days** (max 17), query step
gaps median **3 days** (max 27), observed rate 0.219 (context) / 0.221 (target). Confirms the
dataset guide.

Note the guide's own correction: **PhysioNet as shipped is fully dense** — its cache is resampled
to an hourly grid and the bundled zip contains no obs-mask files, so its "irregularity" is
variable record length only. USHCN is therefore the only irregular cell available, not merely the
most irregular.

**`H = 12` counts OBSERVED STEPS, not hours or days.** 12 steps span a median of 53 days; K = 24
spans ~104 days. Never describe this horizon as "12 hours".

---

## 3. Setup — no `DatasetPreset` exists, and none was created

`apply_dataset_preset(cfg, "ushcn")` raises `KeyError` and `resolve_run_experiment(DATA_DIR)`
raises `ValueError` (`meta["dataset"] == "ushcn_sliding"` is not in `_IMPORTERS`). Every stage was
therefore driven programmatically against an **unmodified** framework, using the dataset's own
shim (`integration/ushcn_llapdiff.py`) plus three scripts in `_ushcn_experiments/scripts/`:

| script | replaces | why the shipped tool cannot run |
|---|---|---|
| `ushcn_cfg.py` | `run_stage_pretrain._build_config/_build_loaders` | both raise for `ushcn` |
| `pretrain_ushcn.py` | `llapdiff-stage-pretrain` | ditto |
| `gate0_ushcn.py` | `llapdiff-stage1-roundtrip` | ditto |
| `family_a_ushcn.py` | `llapdiff-option-a-probe` | ditto |

`family_a_ushcn.py` imports `collect`, `train_head`, `evaluate_head`, `lag_covariance`, `HEADS`
and `_build_frozen_stack` **verbatim** from the shipped probe. Only config and loader construction
are substituted, so the arms are the same code that produced the noaa_uk and bms_air numbers —
which is the only reason the three cells are comparable.

`_build_frozen_stack` builds the denoiser from config and does **not** load a diffusion
checkpoint, so Family A needed no diffusion training.

### 3.1 Two defects in the dataset handover

Both were unreachable at preparation time — the guide states no training of any kind had been run.

1. **`apply_ushcn_config` sets `VAE_DIR` but not `SUM_DIR`.** `summarizer_ckpt_path` therefore
   falls back to `base_config.SUM_DIR` = `ldt/summarizer/saved_model/dataset` — **relative**, and
   named `dataset`. Run from the checkout's cwd that writes the stage-2 artifact *inside*
   `llaplace_df`, under a name that collides with any other dataset's summarizer.
   `ushcn_cfg.build_cfg` pins it and refuses any artifact path that resolves inside the checkout.
2. **Stage 2 cannot complete under AMP.** `apply_ushcn_config` copies the irregular preset but
   leaves `SUM_AMP = True`; the run exhausts the **cumulative** `SUM_MAX_NONFINITE_GRAD_STEPS = 8`
   budget and aborts with `FloatingPointError: non-finite summarizer gradients`.
   **`SUM_AMP = False` is required.** Two candidate explanations were tested and *rejected* by
   measurement — fp16 range on the values (max |V| = 56.6, squared 3204, well inside fp16's
   65504) and zero-count division in the channel-balanced loss (it sums over the whole batch,
   filters `>0`, and clamps). The overflow is in intermediate activations or gradients and was
   not localised further.

---

## 4. Gate 0 (stage-1 round trip) — **PASS 0.9610**, with a comparability caveat

Threshold 0.85. Against 0.624 (crypto) and 0.690 (us_equity), both of which failed and are parked.

| channel | n | corr | std_ratio | gate |
|---|---|---|---|---|
| PRCP | 557 | **0.8147** | 0.913 | **FAIL** |
| SNOW | 54 | 0.9990 | 0.989 | PASS |
| SNWD | 85 | 0.9975 | 0.956 | PASS |
| TMAX | 1088 | 0.9996 | 0.995 | PASS |
| TMIN | 894 | 0.9995 | 0.987 | PASS |
| **MEAN** | 2678 | **0.9610** | | **PASS** |

PRCP is the channel the guide flags as 73–96 % zero-inflated, so its failure is expected.

**Two deviations from the shipped Gate 0, both forced by this dataset and both documented in
`gate0_ushcn.py`:** the shipped tool correlates `y_true[row,:,e,0]` — **channel 0 only**, correct
for single-target cells but here that is PRCP alone — and it correlates over all H steps including
the 78 % that are unobserved and zero-filled, which would largely measure the reconstruction of
zeros. This version reports all five channels and correlates on **observed cells only**, requiring
≥ 4 of them.

⚠️ **This 0.9610 is NOT comparable to noaa_uk's 0.9952 or bms_air's 0.989.** Those are
correlations over fully-observed 168-step windows; this is over 3–5 observed points, and a
short-series correlation is biased upward. Visible directly in the sensitivity sweep — requiring
*more* observations *lowers* the score:

| `min_obs` | n scored | share | MEAN corr | PRCP |
|---|---|---|---|---|
| 3 | 5055 | 26 % | 0.9680 | 0.8383 FAIL |
| **4** | **2678** | **14 %** | **0.9610** | **0.8147 FAIL** |
| 6 | 370 | 2 % | 0.9300 | 0.7374 FAIL |

The verdict (PASS on the mean, FAIL on PRCP) is stable at every usable threshold. The gate is
scored on 14 % of window-channels — inherent, since 12 steps at 22 % observed leaves ~2.6
observed cells per window-channel.

---

## 5. Family A — `chirp_gp` wins the variogram score 10/10

### 5.1 The learning-rate grid had to be widened UPWARD

Opposite to both existing cells. `grid7` (3e-4 … 1e-7, extended *downward* because that is where
noaa_uk and bms_air pinned) left **Student-t at the maximum on 7/7 seeds**, low-rank 3/7,
chirp_gp 2/7. `grid8` (1e-2 … 3e-6) brackets every head:

| head | grid7 selection | grid8 selection | bracketed? |
|---|---|---|---|
| chirp_gp | 3e-4 ×2, 1e-4 ×1, 1e-5 ×3, 3e-6 ×1 | 1e-4 ×1, 3e-5 ×5, 1e-5 ×3, 3e-6 ×1 | yes (1/10 at low bound) |
| diagonal | 3e-4 ×1, 1e-4 ×5, 3e-5 ×1 | 3e-3 ×2, 1e-3 ×1, 3e-4 ×2, 1e-4 ×4, 3e-5 ×1 | **yes** |
| low-rank | 3e-4 ×3, 1e-4 ×2, 3e-5 ×2 | 3e-3 ×2, 1e-3 ×3, 3e-4 ×2, 1e-4 ×2, 3e-5 ×1 | **yes** |
| Student-t | **3e-4 ×7 (at bound)** | 3e-3 ×10 (declined 1e-2) | interior, but see §6 |

(grid7 counts are over its 7 seeds, grid8 over all 10.)

The grids overlap at 3e-4 … 3e-6 deliberately, so they cross-check. chirp_gp, diagonal and
low-rank drift ≤ 1 % between grids — those three are converged.

### 5.2 Per-head means, grid8, **n = 10 seeds**

| head | VE % | ES ↓ | VS ↓ | CRPS ↓ | PIT-ECE ↓ | cov@80 |
|---|---|---|---|---|---|---|
| **chirp_gp** | 9.753 | 9.906 | **70.01** | 0.5074 | **0.0606** | 0.867 |
| diagonal | **10.231** | 9.894 | 81.45 | 0.5084 | 0.0622 | 0.870 |
| low-rank | 10.201 | **9.882** | 80.68 | **0.5071** | 0.0614 | 0.869 |
| Student-t | 2.802 | 27.44 | 631.02 | 1.3708 | 0.1465 | 0.976 |

### 5.3 Paired reads, grid8, **n = 10**, one-sided Wilcoxon by seed

| metric | vs diagonal | vs low-rank |
|---|---|---|
| **VS ↓** | **wins 10/10, p = 0.0010, d_z −5.14** | **wins 10/10, p = 0.0010, d_z −4.30** |
| ES ↓ | 5/10, p = 0.75 | 3/10, p = 0.93 |
| CRPS ↓ | 7/10, p = 0.25 | 3/10, p = 0.75 |
| PIT-ECE ↓ | 7/10, p = 0.053 | 6/10, p = 0.31 |
| VE ↑ | 3/10, p = 0.98 | 3/10, p = 0.97 |

Per-seed VS. **The distributions do not overlap**: chirp's worst seed (72.15) is better than the
best baseline seed (77.80), on all 10.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| chirp_gp | 70.23 | 68.98 | 71.00 | 72.15 | 69.90 | 68.78 | 68.19 | 71.68 | 68.73 | 70.45 |
| diagonal | 83.06 | 84.14 | 82.51 | 80.83 | 78.52 | 80.83 | 82.13 | 82.38 | 80.66 | 79.47 |
| low-rank | 86.01 | 80.06 | 80.91 | 81.17 | 79.59 | 79.59 | 81.85 | 78.67 | 77.80 | 81.12 |

**chirp_gp wins the variogram score unanimously and loses the marginal axis** (VE 2/8, ES 4/8 and
2/8, CRPS a wash). PIT-ECE is directionally favourable but not significant.

---

## 6. ⚠️ The Student-t arm is degenerate and must not be reported as a matched baseline

Under the shared protocol the learning rate is selected on the final phase's **val NLL**, then the
arm is reported on path scores. For Student-t those disagree violently. It selects 3e-3 on **10/10**
seeds, and at that LR:

| | grid7 (pinned at bound, n=7) | grid8 (interior, n=10) |
|---|---|---|
| VS | 76.7 | **631.0** |
| ES | 10.35 | **27.44** |
| CRPS | 0.5051 | **1.3708** |
| PIT-ECE | 0.0636 | **0.1465** |
| VE % | 3.44 | 2.80 |

Neither grid yields a usable Student-t: grid7's is under-trained at the bound, grid8's optimises
likelihood at catastrophic cost to every path score — a heavy-tailed law wins NLL while having
enormous spread. **`chirp_gp` beating it 10/10 on everything is beating a broken arm and means
nothing.** Report it as an observation about NLL-only model selection, not as a baseline.
Hand-picking a learning rate to make it look reasonable would be selection on the outcome.

---

## 7. Cross-cell placement

| read | noaa_uk h168 | bms_air h168 | **USHCN 24/12** |
|---|---|---|---|
| **VS (chirp vs baselines)** | **10/10** | **10/10** | **10/10** |
| ES | loses 0/10 | wins 10/10 | 5/10, 3/10 |
| CRPS | loses 0/10 | wins 10/10 | 7/10, 3/10 |
| PIT-ECE | wins 9/10 | wins 7–8/10 | 7/10 (p=0.053), 6/10 |
| VE | loses 0/10 | wins 10/10 | loses 3/10 |
| chirp_gp VE | +8.52 % | +2.63 % | **+9.75 %** |

**Variogram-score path coherence is the only score that wins on all three cells** — three
datasets, three different learning-rate grids, every baseline. ES, CRPS and VE remain
cell-dependent and go opposite ways between bms_air and the other two.

USHCN also has the *most* forecastable signal of the three by VE (+9.75 %), which answers a
weakness of bms_air, whose s1 conditional mean is nearly uninformative (latent corr ≈ 0.02).

⚠️ **ES, VS and CRPS magnitudes are NOT comparable across cells** (USHCN is H=12/d_z=16 against
H=168/d_z=24). Only VE, corr, PIT-ECE and coverage are.

---

## 7b. `CHIRP_UQ_INIT_VAR` = 2.2086 (measured, gates s2)

Measured on seed 2's finished `s1_mean` checkpoint, full val split (83 batches, 4 054 656 latent
elements), EMA weights, `--mean-source ddim` at guidance 1.0 — the protocol that produced both
other cells' values of record. **Two independent runs returned bit-identical values.**

| cell | latent_mse | value used | latent_corr |
|---|---|---|---|
| noaa_uk | 0.8236 | 1.0 | **+0.363** |
| bms_air | 1.6387 | 1.64 | +0.014 |
| **USHCN** | **2.2086** | **2.21** | **+0.081** |

All three within 2.7× — no sign of the 26× physionet/noaa_uk spread the per-cell guard exists
to prevent.

⚠️ **The diffusion s1 mean on this cell is weak.** `latent_mse` 2.209 is **50 % worse than
predicting the val mean** (baseline 1.469), and `latent_corr` is +0.081 — nearer bms_air's
+0.014 than noaa_uk's +0.363. Yet the *one-pass* Family A head on the same cell reaches
VE +9.75 % with corr 0.32 (§5). The one-pass head substantially outperforms the diffusion
conditional mean here, as it does on bms_air. Any Table 3 / Table 1 row for USHCN must be read
with that in mind.

---

## 8. Cost (measured)

| stage | wall | note |
|---|---|---|
| stage 1 (VAE) | **7 215 s = 2.0 h** | one artifact, shared by all seeds |
| stage 2 (summarizer) | **21 801 s = 6.06 h** | fp32; early stopped at epoch 140 |
| Family A `collect` | ~2 min | 3.96 GB cache, shared by all seeds/grids |
| Family A per seed | **~7.8 h** | 14.7 min/fit × 32 fits (4 heads × 8 LRs) |

---

## 9. Reproduction

```bash
cd /vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/_ushcn_experiments
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC:$PWD/scripts
export PYTHONDONTWRITEBYTECODE=1

python scripts/pretrain_ushcn.py --stage vae --seed 0
python scripts/pretrain_ushcn.py --stage summarizer --seed 0 --no-sum-amp --artifact-tag fp32
python scripts/gate0_ushcn.py --split val --max-batches 60
python scripts/family_a_ushcn.py --seed $S \
  --lr-grid 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 \
  --vae-ckpt $PWD/ldt/vae/saved_model/ushcn/pred-12_ch-16_elbo.pt \
  --sum-ckpt $PWD/ldt/summarizer/saved_model/ushcn/12-16-summarizer_fp32.pt \
  --out-json $PWD/results/R10_option_a_s${S}_grid8.json
```

Source provenance: `small_chunked_sporadic.csv`, SHA-256
`671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf`, identical to `RAWDATA_SHA256`
in GraFITi's `tsdm/datasets/ushcn_debrouwer2019.py`. Preserved unchanged in the dataset's
`source/`. Splits: 206 466 / 21 118 / 58 925 windows, `global_purged_horizon`, 24 167 purged,
`norm_scope="train_only"`.

---

## 10. Outstanding

1. ~~grid8 seeds 2 and 3~~ — **done, grid8 is complete at 10 seeds.**
2. **Student-t** needs a decision (§6). It is currently unusable either way.
3. **No diffusion family on this cell.** Family B (s1/s2), Table 3's R9 gate and Table 1's mixture
   frontier would all need diffusion training, which has not been attempted here.
4. **The nugget ablation** has not been run on USHCN. The nugget hurts VS on both other cells
   (9/10 and 10/10 against); a third cell would strengthen that.
5. **Test split untouched**, correctly — everything above is val.
