# CMD UQ campaign — recovery plan

**Purpose.** The U3 UQ ablation was run on PhysioNet h=12 and is not interpretable. This file
is the ordered plan to get a defensible Table 3. Execute phases in order. Each phase has an
explicit stop condition — **do not proceed past a failed stop condition**; report and wait.

> 🛑 **STOPPED at Phase 1 as of 2026-08-03. Read `w_docs/CMD_CAMPAIGN_STATE.md` first** — it
> carries the branch state, what is settled, and what the next task actually is. Phases 2–6 are on
> hold; the 1b gate failed on both seeds and the cause is the conditioning (B21), so the next
> action is the conditioning track, **not** the next phase in this file.

---

## Background (read once, do not re-litigate)

Two findings invalidate the current U3 results:

1. **B19 — PhysioNet h=12 gives the diffusion stage 13 training windows** (5 val, 5 test),
   against 11.5 M parameters. The loader yields the identical 13 windows every epoch
   (batch checksums bit-identical). Root cause is dataset geometry: patient-relative
   splitting, records spanning ~48 h, `WINDOW + PRED = 36` leaving ~23 valid start offsets.
   Not a code defect.

2. **Gate G-c returned CONFOUNDED.** The denoiser's latent prediction carries no signal in
   *any* arm, including plain-MSE S1: latent x0 MSE 4.3–5.7 against a predict-zero baseline
   of 2.16, correlation ≈ −0.05. S3's one-shot mean: latent RMSE 2.431 vs predict-mean
   baseline 1.464, correlation +0.004.

Finding 2 is most likely *caused by* finding 1, but that cannot be confirmed on PhysioNet,
where "too little data" and "architecture cannot use conditioning" predict identical
observations. A conditioning-shuffle probe shows the denoiser **does** respond to its input
(39 % output change under shuffled summary, 67 % with none), which favours the data-size
explanation.

**Data-space CRPS did not reveal any of this** (~0.25 val, looks healthy). Disabling
conditioning entirely costs only 0.2606 → 0.2990, because the set-VAE decoder reconstructs
each of the 445 entities largely from its own embedding. CRPS on this pipeline is dominated
by a per-entity prior, not by the forecast. Do not use CRPS as a health check.

---

## Standing rules (apply to every run in every phase)

- [ ] **`--weights ema`** must be passed to **`llapdiff-uq-eval` *and* `llapdiff-u1-sweep`**.
      Without it both score **raw** weights (bug B16: `_load_stack` reads
      `payload["model"]`, which holds raw weights even in `*_best_ema.pt`). Verified
      non-trivial: summed per-tensor `max|raw − ema|` = 0.696, and measured directly on
      the `fixed_bugs` physionet incumbent at one fixed sampling cell, **val CRPS 0.2673
      (raw) vs 0.2593 (EMA)** — a 0.008 gap, larger than the entire 0.0014 tuning gain
      that campaign was decided on. `--weights` was added to `llapdiff-u1-sweep`
      2026-08-01; it defaults to `raw` on both tools so previously recorded cells stay
      reproducible, and the choice is now recorded in every output row. Raw and EMA
      sweeps of the same checkpoint write to different files and no longer overwrite
      each other.
- [ ] **`CHIRP_UQ_INIT_VAR` must be set explicitly** for every UQ run. The library default
      (1e-2) is left in place for checkpoint compatibility and is wrong for every new dataset.
      See Phase 3 for how to choose it. Never copy 25.0 across datasets.
- [ ] **Print train/val/test window counts** before trusting any run on a new
      dataset × horizon cell.
- [ ] **Never compare `val_diag_mse_raw` across loss modes.** It reports `stats["raw_loss"]`,
      which is an MSE under `DIFF_LOSS_MODE="mse"` but an **NLL** under `"gaussian_nll"`.
      This trap already produced one phantom "340× degradation" result.
- [ ] **Test split is touched once per arm per seed**, and only after a pre-registration file
      for that dataset is frozen. Gates and diagnostics run on **val**.
- [ ] Health checks are read on **latent** metrics (`baseline_rmse_predict_mean`,
      `latent_corr`, `latent_target_std`), not data-space CRPS.
- [ ] **Record `cond_norm_mode` for every checkpoint used.** B15 persists it per checkpoint and
      `_cond_norm_mode_from_checkpoint` returns the legacy `"batch"` when the key is absent, so
      any reused pre-fix checkpoint is silently running the train/eval mismatch B15 removed.
      New runs should read `"sample"`.
- [ ] **Record `ALLOW_TF32`** (now `True` by default after the `speed_up` merge). It moves CRPS
      in the 4th–5th decimal, and `DETERMINISTIC=True` forces it off. Harmless for calibration
      work, but it must be a recorded decision, not an inherited default — the same discipline
      the runbook already applies to guidance and thresholding.
- [ ] **Gate/decision runs use ≥ 2 seeds.** `cudnn.benchmark` nondeterminism alone produced a
      0.036 CRPS swing between byte-identical configs (`CMD_BUG_REPORT.md` §4, retraction 1).

---

## Phase 0 — Window-count audit ✅ DONE 2026-07-31

Full results: **`w_docs/CMD_PHASE0_WINDOW_AUDIT.md`** (tracked; the `finetuning/results/` copy is gitignored). Reproduced here because every later phase
depends on them.

| cell | WINDOW | PRED | **train_w** | **val_w** | **test_w** | ent/win | supervision scalars | `LAPLACE_K` → params | vs params |
|---|---|---|---|---|---|---|---|---|---|
| physionet h12 | 24 | 12 | **13** | 5 | 5 | 445 | 2 496 | 256 → 11.75 M | 4 708× over |
| crypto h100 | 200 | 100 | 1 225 | **90** | 379 | 118 | 1.96 M | 256 → 11.75 M | 6.0× over |
| us_equity h100 | 200 | 100 | 1 099 | **72** | 343 | 221 | 1.32 M | 256 → 11.75 M | 8.9× over |
| noaa_uk h168 | 336 | 168 | 16 286 | 2 183 | 4 702 | 4 | 43.8 M | **128 → 10.17 M** | **4.3× under** |
| bms_air h168 | 336 | 168 | 95 099 | 17 598 | 33 449 | 12 | 383 M | 256 → 11.75 M | **32.6× under** |

> **Corrected 2026-08-02** (`w_docs/CMD_PHASE0_WINDOW_AUDIT.md` has the detail). The audit
> applied a single "10.2 M" to all five cells, but `LAPLACE_K` is a per-dataset preset field
> and **only noaa_uk is 128** — the rest default to 256. Verdicts are unchanged; the ratios
> move. The `k_probe` behind `laplace_k=128` **was run on noaa_uk**, so the gate cell's number
> is correctly evidenced and Phase 1's "under-parameterized ⇒ modelling attribution" argument
> stands. (Commit `f52f843`'s message attributes that probe to physionet; the message is
> wrong, its diff is right, and physionet still runs at K=256.)

### 🔴 Count the right quantity — the obvious one is wrong

`run_experiment` returns `sizes = (len(ds_tr), len(ds_va), len(ds_te))`
(`datasets/fin_dataset.py:2166`). Those are **dataset items, not window starts.** They run ~4×
larger and the ratio is not constant across datasets:

| cell | true train windows | `sizes[0]` |
|---|---|---|
| **physionet h12** | **13** | **2 087** |
| noaa_uk h168 | 16 286 | 63 136 |

**PhysioNet's `sizes[0]` = 2087 passes the < 500 stop condition below.** Printing the obvious
quantity green-lights the one cell this audit exists to reject — silently. Count window starts:

```python
w = 0
for batch in loader:
    w += batch[0][0].shape[0]      # axis 0 = window starts, axis 1 = entities
```

`batch[0]` is a `(V, T)` tuple, not a tensor — `torch.as_tensor(batch[0])` raises
`ValueError: only one element tensors can be converted to Python scalars`.

### Two axes matter, not one

**Windows** set how much the denoiser can learn. **Entities per window** set how much survives
the set-VAE, which mean-pools the entity axis into ONE latent per window — that is **B5**, the
defect that capped H2 at a 0.62 round-trip. Supervision is `windows × PRED × latent_channels`
regardless of entity count, which is why 445 entities did not rescue PhysioNet.

Crypto and us_equity are the worst cells on the pooling axis (118, 221); noaa_uk and bms_air the
best (4, 12).

### Stop conditions

- **< ~500 train windows** → smoke-test cell only; no statistical claim. *(physionet h12 fails.)*
- **< ~500 val windows** → the Phase-1 gate is read on val, so a thin val split makes the gate
  itself under-powered. *(crypto h100 = 90 and us_equity h100 = 72 both fail; noaa_uk and
  bms_air pass.)*
- **Over-parameterized cells cannot support a "modelling defect" verdict** — see Phase 1.

---

## Phase 1 — Signal gate (the decision point)

**Dataset: noaa_uk h=168** (fallback: bms_air h=168). Revised 2026-07-31 — this was crypto
h=100, and the Phase-0 audit shows crypto cannot carry the decision:

- Crypto is **still 5.2× over-parameterized** (1.96 M supervision scalars vs 10.2 M params).
  It is ~785× better than PhysioNet, so it *does* rule out 13-window memorization — but a
  "ties baseline" result there still admits a data/capacity explanation, which is exactly the
  ambiguity the STOP branch below claims to resolve. **noaa_uk is 4.3× *under*-parameterized**,
  so a failure there is attributable to modelling.
- Crypto has **90 val windows** and the gate is read on val.
- Crypto pools **118 entities** per latent against noaa_uk's 4 — the B5 axis.
- The old fallback, us_equity, is worse than crypto on both (72 val windows, 221 entities).
  Use **bms_air h=168** instead (95 k train windows, 12 entities).

**On cost.** The original rationale was speed, written before the `speed_up` merge. The
`EVAL_*` campaign profile now makes in-training validation ~61× cheaper on noaa_uk
(`DEVELOPER_GUIDE.md` §7.6), and stage-1/2 artifacts already exist for every cell above, so a
single S1 gate seed on noaa_uk is no longer the obstacle it was. Run crypto first as a cheap
plumbing pre-check if you like — but do **not** take the decision there.

**Prediction type.** Train the gate at `predict_type=x0`, matching the arms
(`chirp_uq_head` forces x0, `models/llapdiff.py:61`). USAGE.md §3.6 records x0 as ≈0.03 CRPS
worse than v, so a gate run at the default `v` does not test the arms' configuration.

### 1·0. Gate 0 — stage-1 round-trip (run first, minutes)

```bash
llapdiff-stage1-roundtrip --dataset-key <ds> --pred <h>    # threshold ~0.85
```

Encode the TRUE targets and decode them straight back. Reads **val** by default in this
mode (the standing rule puts gates on val); `--split` overrides. The stage-1 VAE must
already exist — build it with `llapdiff-artifact-prep --datasets <ds>` first. Seeds do not
apply: the VAE is frozen, shared across arms and seeds, and the round-trip reads the
posterior mean, so the result is deterministic.

> The `--dataset-key/--pred` mode was added 2026-08-01. Before that the tool was
> synthetic-benchmark-only (`--tasks`, config built through `bench._configure`), so this
> command was rejected outright and Gate 0 could not be run on the cell it gates.

### ✅ Gate 0 measured 2026-08-01 — the gate and cross-check datasets pass

| cell | round-trip corr | std ratio | n | gate |
|---|---|---|---|---|
| **noaa_uk h168** (Phase 1) | **0.922** | 0.90 | 120 | **PASS** |
| **bms_air h168** (Phase 2) | **0.941** | 0.98 | 235 | **PASS** |
| physionet h12 | 0.059 | 0.05 | 110 (266 constant windows skipped) | FAIL |
| crypto h100 | 0.657 | 0.75 | 1 124 | FAIL |
| us_equity h100 | 0.674 | 0.70 | 2 210 | FAIL |

Val split, threshold 0.85, 2 batches per cell. **The two datasets the plan routes the
campaign to are the two that pass**, so 1a/1b are well posed on noaa_uk and the Phase-2
cross-check is well posed on bms_air.

The three failing cells are the three Phase 0 already rejected on window counts — an
independent corroboration from a different direction (stage-1 representational capacity
rather than supervision volume). Note this is a *stronger* statement than Phase 0's for
physionet: at corr 0.059 its VAE cannot reproduce its own targets, so **no** denoiser on
that stack could, independently of the 13-window problem. Treat it as further support for
retiring PhysioNet rather than as a new result — it has not been chased down.

> ⚠️ **PhysioNet's number needed a tool fix to be readable at all.** 71 % of its val
> entity-windows are *constant* (median true target std exactly 0 — sparse clinical series
> carried forward), `np.corrcoef` on a constant series is undefined, and a single such
> window turned the aggregate into `nan` while inflating the std ratio to ~1e6. Constant
> and undefined windows are now dropped and counted (`n_degenerate`). The filter is a no-op
> where nothing is degenerate — noaa_uk and bms_air score identically before and after.
> **Read `n` and the skipped count, not just `corr`:** a cell scored on a minority of its
> split is telling you about that minority.

Encode the TRUE targets and decode them straight back. If stage 1 cannot reproduce its own
targets, no denoiser can do better and neither 1a nor 1b means anything. This is the B5
precedent (phase-2π caches sat at 0.62; the fix reached 0.857). The tool already exists and
exits non-zero on failure.

### 1a. Positive control — linear probe (run first, seconds)

Finance targets are near-random-walk, so predict-the-mean is a genuinely strong baseline
there. Before reading the denoiser, establish whether the target is forecastable **at all**
on this dataset.

**Revised 2026-08-02 — this is now a PAIRED probe.** The original rule fit one ridge (frozen
summarizer tokens → latent) and branched on it. That cannot tell "this dataset is
unforecastable" from "the conditioning loses the forecastable part", and on noaa_uk h=168 it
gave the wrong answer: it said *move datasets*, while the raw history was forecastable to
28.6 % and the entire loss was in the conditioning (**B21**). Fit all three:

| | probe | question it answers |
|---|---|---|
| **A** | raw history → raw target | is this dataset forecastable at all? |
| **B** | `cond_summary` → raw target | does the conditioning preserve it? |
| **C** | `cond_summary` → latent `mu_norm` | can the 1b gate see anything? |

```bash
llapdiff-ridge-probe --dataset-key <ds> --pred <h>          # no checkpoint needed
```

Needs no denoiser — 1a precedes 1b, so the tool builds the frozen stack from the
config-derived artifacts. Pass `--checkpoint` only to probe a specific trained
configuration's conditioning settings.

### Decision table (all thresholds are % relative RMSE reduction vs `baseline_rmse_predict_mean`)

| condition | verdict | action |
|---|---|---|
| **A < 10 %** | dataset cannot answer the gate | Move to a dataset with genuine temporal structure (bms_air h=168) and report the extra cost. **This is the only branch that justifies changing datasets.** |
| A ≥ 10 %, **B < 0.7·A** | the conditioning is the bottleneck | **Fix the conditioning (B21). Do NOT move datasets** — the same summarizer would travel to the new cell and reproduce the verdict. |
| A ≥ 10 %, B ≥ 0.7·A, **C < 10 %** | the latent is the bottleneck | Fix stage 1, or re-target what 1b reads. The conditioning delivers the signal but the VAE's latent does not expose it. |
| A ≥ 10 %, B ≥ 0.7·A, C ≥ 10 % | **PROCEED to 1b** | the gate can discriminate |

State the margins, not just the direction — on a near-random-walk target a statistically real
0.5 % win is not usable headroom.

### Three traps the tool encodes (each produced a wrong number when absent)

1. **Strided tokens, never mean-pooling.** The original wording said "pooled `E_ti`".
   Averaging the 336 summary tokens into one vector scores **−0.2 %** where the strided view
   scores **14.2 %** — the pooling manufactures a FAIL out of a working conditioning.
2. **Alpha on a chronological train holdout, never on val.** Selecting on val moved the
   headline 6.2 % → 9.8 %, i.e. straight across the 10 % threshold the branch turns on.
3. **Report both targets.** The latent is the least linearly decodable object in the pipeline
   (B21): `COND_NORM_MODE="global"` raises B from 7.7 % to 12.3 % and lowers val CRPS
   0.3627 → 0.3344 while C stays flat. An intervention that measurably improves the forecast
   can lower the latent readout, so C alone must never drive a dataset decision.

### 🔴 EXECUTED 2026-08-01/02 on noaa_uk h=168 — this is what forced the rule above

14 446 train / 2 183 val windows at 4 entities, alpha on a chronological train holdout (val
never touched), baseline exactly as `llapdiff-uq-eval` computes it.

| input to ridge | dims | **→ latent (1a's target)** | → raw target |
|---|---|---|---|
| `cond_summary`, `COND_NORM_MODE="sample"` (current default) | 2 048 | 5.6 % | 7.7 % |
| `cond_summary`, `"sample"` | 8 192 | 4.2 % | 14.2 % |
| `cond_summary`, `"global"` (new, B21) | 2 048 | 4.9 % | 12.3 % |
| `cond_summary`, `"global"` | 8 192 | **6.8 %** | 16.4 % |
| **raw history, temperature channel only** | **128** | **10.0 %** | **28.6 %** |

**A = 28.6 %** (dataset is forecastable, comfortably) · **B = 7.7–16.4 %** (roughly half of A
survives the conditioning) · **C = 4.9–6.8 %** (never clears 10 %).

⇒ **Verdict: FIX THE CONDITIONING (B21).** Under the *old* single-probe rule this cell read
as "move datasets"; under the paired rule it reads correctly, because the raw history clears
the threshold on 1a's own target (10.0 %, marginally) from 64× fewer features while no conditioning
variant does.

**Status: Phase 1 is blocked on B21, not on the dataset choice.** noaa_uk remains the right
gate cell — it passes Gate 0 at 0.922 and is forecastable. The conditioning has to improve
before 1b can discriminate anything, and moving to bms_air would carry the same summarizer
along and reproduce this result.

Verified: `llapdiff-ridge-probe --dataset-key noaa_uk --pred 168 --tokens 8` reproduces the
scratch measurements (A 28.6 %/10.0 %, B 7.7 %, C 5.7 %) and returns
`FIX THE CONDITIONING (B21)`.

### 1b. The gate — two S1 seeds, trained to convergence

Train **two** S1 (plain MSE, arm `d`, `predict_type=x0`) seeds on the chosen dataset. Two, not
one: `CMD_BUG_REPORT.md` §4 retraction 1 records two byte-identical configs producing CRPS
0.2243 vs 0.1886 — a 0.036 swing from `cudnn.benchmark` nondeterminism alone — and concludes
*"any CRPS claim needs multi-seed."* This gate reads latent MSE, which is less noisy than CRPS,
but it is a STOP/GO decision for the whole campaign and a single seed cannot distinguish a real
tie from a bad draw. If both seeds agree, one is enough evidence; if they disagree, that is
itself the finding.

Then read the gate on **val**, per seed:

```bash
llapdiff-uq-eval --dataset-key <ds> --pred <h> --checkpoint <S1 ckpt> \
  --weights ema --split val --latent-only --mean-source ddim
```

- `latent_mse` (the plan's "latent x0 MSE") and `latent_rmse`
- `baseline_rmse_predict_mean` / `baseline_mse_predict_mean`
- `latent_corr`
- `latent_target_std`

Compare like-for-like (fixed t, identical noise, EMA weights).

> **`--latent-only` is required here.** ~~and so is `--mean-source ddim`~~ (fixed 2026-08-01;
> the `ddim` half **superseded 2026-08-03**, see the correction after this block).
> The gate arm is plain-MSE S1, which has **no UQ head**, and the tool used to refuse such
> a checkpoint outright — while these four metrics exist nowhere else in the tree, so the
> campaign's decision point was unmeasurable. It now reports the mean-only metrics and
> omits everything needing a variance (`latent_gaussian_nll`, `pit_calibration_error`,
> `reliability`, `mean_predicted_std`); the report carries `has_uq_head: false` to mark
> that. The data-space arms still refuse a no-UQ checkpoint — they propagate the law
> through the decoder, and there is no law. `--mean-source oneshot` additionally refuses
> any non-`x0` checkpoint, because it returns the raw network output, which is an x0
> estimate only under `predict_type='x0'`; `ddim` converts through `scheduler.to_x0` and
> is safe for any parameterization. Train the gate at x0 as §1 says and either works.

> 🔴 **Correction 2026-08-03 — read BOTH sources, and do not read `ddim` alone.** The paragraph
> above chose `ddim` for *parameterization safety*, which is a real property, and silently
> inherited a false one: that `ddim` estimates a conditional mean. It does not.
> `generate(eta=0.0)` is deterministic in its trajectory but starts from a **random `x_T`**, so it
> returns **one sample**; and it applies CFG on `guidance_strength=(1.0, 2.0)`,
> `guidance_power=0.3`, which pushes samples away from the unconditional. Both inflate MSE without
> touching signal, so a single-sample MSE **understates a signal gate by construction** — measured
> at 2.2–2.4× over-dispersion in §1b, enough to turn a +6 % model into a −3 % reading.
> `--mean-source oneshot` at `t = T−1` *is* the conditional-mean estimator and is legal on the x0
> gate arm. **Run both**: `oneshot` for the honest signal number, `ddim` for what sampling
> actually delivers, and the gap between them is itself diagnostic (see lead 2 in §1b). Reading
> only `ddim` would have scored this campaign's decision point wrong in the pessimistic direction.

### Decision table

| Outcome | Meaning | Action |
|---|---|---|
| Beats baseline, `latent_corr` clearly positive | Pipeline works. G-c's failure was B19. | Proceed to Phase 2. |
| Ties/loses baseline, **ridge passed 1a with margin**, cell is under-parameterized | Real modelling defect, independent of data size. | **STOP.** U3 is unanswerable on any dataset until fixed. Report before continuing, and open a debugging track — the two named leads first (below), then VAE latent learnability and x0 vs v. |
| Ties/loses baseline, **cell is over-parameterized** (crypto, us_equity) | Not diagnostic — data and modelling are confounded here. | Re-run 1b on noaa_uk or bms_air before concluding anything. |
| Ties/loses baseline, **ridge failed or was marginal in 1a** | Dataset uninformative. | ~~Re-run 1a/1b on bms_air h=168.~~ ⚠️ **Stale row, superseded 2026-08-03.** It predates the paired 1a rule, so "ridge failed" conflated *A* failing (dataset really is uninformative) with *C* failing (conditioning is), and its action is now known-wrong: bms_air was ruled VOID and Phase 2 moved to noaa_us. Use the row below. |
| Ties/loses baseline, **1a returned `FIX THE CONDITIONING (B21)`** — A ≥ 10 % so the cell is forecastable, C < 10 % — and the cell is under-parameterized | The conditioning ceiling is the binding constraint, now confirmed **on the trained model** and not merely inferred from probes. Not a denoiser defect, not a dataset defect. | **STOP the campaign at Phase 1** and open the B21 conditioning track. Re-running 1b on another cell tests nothing: it carries the same summarizer. Re-enter 1b only after a conditioning change moves **C** past 10 % in 1a. |

### 🔴 EXECUTED 2026-08-03 on noaa_uk h=168 — **THE GATE FAILS**, both seeds

Full readout and reproduce commands: **`w_docs/CMD_PHASE1B_GATE.md`** (tracked copy of
`finetuning/results/phase1b/gate_readout.md`, which is gitignored). Checkpoints:
`ldt/tuning/phase1b/noaa_uk_h168/d/s1_gate_seed{0,1}/output/predict-x0/modal-chirp/seed-{0,1}/pred-168/llapdiff_pred-168_best_ema.pt`.

Both seeds converged and early-stopped on their own — **not** the `EPOCHS=60` truncation that made
the `global` seeds unmeasurable:

| seed | best val CRPS | best epoch | stopped |
|---|---|---|---|
| 0 | 0.32376 | 255 | 355 |
| 1 | 0.33229 | 210 | 310 |

CRPS spread **0.0085**, well inside the documented 0.036 band, so the two seeds agree and the plan's
"if both seeds agree, one is enough evidence" clause is satisfied. Confirmed from each checkpoint's
own `model_config`: `cond_norm_mode="sample"` (the default, **not** `global`), `sum_ft_mode` absent,
`pole_pool_use_raw_summary=False`, `chirp_uq_head=False`, `predict_type="x0"`. The matching 1a row
is therefore `cond_summary, "sample"`, not either `global` row.

| `--mean-source` | seed | `latent_mse` vs baseline 0.64917 | RMSE vs 0.80571 | `latent_corr` |
|---|---|---|---|---|
| `ddim` (this section mandates it) | 0 | 0.68280 → **+5.18 % worse** | **−2.56 %** | 0.3702 |
| `ddim` | 1 | 0.72277 → **+11.34 % worse** | **−5.52 %** | 0.3395 |
| `oneshot` (conditional mean) | 0 | 0.64369 → −0.84 % better | +0.42 % | 0.1903 |
| `oneshot` | 1 | 0.63927 → −1.53 % better | +0.77 % | 0.1931 |

**Both seeds, both reads, fail the 10 % threshold; the mandated read loses to predict-mean
outright.** The `oneshot` correlations reproduce to **1.5 %** across seeds (0.1903 vs 0.1931) —
the conditioning-only signal at max noise is essentially seed-independent, which is what makes the
lead-2 reading below more than a one-seed curiosity.

**`ddim` is the wrong statistic for a *signal* gate, and this section should stop mandating it.**
It runs `generate(eta=0.0)` from a random `x_T` — deterministic in trajectory but still **one
sample**, not a conditional mean — and applies CFG on a `guidance_strength=(1.0, 2.0)`,
`guidance_power=0.3` schedule. Both inflate spread without touching signal. Solving
`mse/base = a − 2·corr·√a + 1` for `a = Var(ŷ)/Var(y)` separates the two:

| read | seed | corr | std(ŷ)/std(y) | optimal (= corr) | over-dispersion | **RMSE reduction if rescaled** |
|---|---|---|---|---|---|---|
| `ddim` | 0 | 0.3702 | 0.805 | 0.370 | **2.17×** | **+7.11 %** |
| `ddim` | 1 | 0.3395 | 0.818 | 0.339 | **2.41×** | **+5.94 %** |
| `oneshot` | 0 | 0.1903 | 0.357 | 0.190 | 1.88× | +1.83 % |
| `oneshot` | 1 | 0.1931 | 0.342 | 0.193 | 1.77× | +1.88 % |

So the model does carry signal — enough for **+6.5 % on average** with its scale fixed — and loses
only because it is ~2.3× too wide. **6.5 % is still well under 10 %, so the verdict is unchanged
even after granting the model a free optimal rescaling it does not have**; what changes is the
diagnosis.

**The finding: the denoiser is at the ridge ceiling, so it is not the bottleneck.** 1a measured the
best *linear* prediction of this same latent from this same conditioning at **C = 5.6 %** (2 048
dims, `"sample"`) → implied corr **0.330**. The two trained 10.2 M-parameter denoisers, rescaled,
reach **7.11 % and 5.94 %** — mean **6.5 %, i.e. +0.9 pt over a ridge regression**, with seed 1
landing within **0.3 pt** of it. They extract essentially everything the conditioning linearly
exposes plus a small nonlinear margin, and the total still misses the gate. Ten million parameters,
a nonlinear architecture, and 355 epochs buy under one percentage point over a linear least-squares
fit on the same inputs.

⇒ The decision-table row *"ties/loses baseline + ridge passed 1a with margin ⇒ real modelling
defect ⇒ STOP"* **does not apply**: 1a did not pass, it returned `FIX THE CONDITIONING (B21)`, and
1b now confirms the ceiling 1a predicted. This converts B21 from an inference about probes into a
measurement on the trained model.

**Independent confirmation of lead 2 below (`block_summary_adaln = False`).** `oneshot` reads the
model at `t = T−1` — its x0 estimate given pure noise plus the conditioning, i.e. the cleanest
measure of how much conditioning reaches the trunk when `x_t` carries nothing. It scores corr
**0.1903 / 0.1931**; the reverse trajectory, where `x_t` becomes progressively informative, reaches
**0.3702 / 0.3395**. Under an ideal model the ordering is the opposite —
`corr(conditional mean) = ρ` but `corr(single sample) = ρ²`, so `oneshot` should score *higher*,
not half as much. Instead both models recover about **half** their final correlation only *after*
`x_t` starts carrying signal, which is exactly lead 2's mechanism: conditioning enters only via
cross-attention into K modal tokens whose residues come from `x_t`, so at high noise those tokens
are noise-dominated. That the two seeds land 1.5 % apart on this number makes it a property of the
architecture, not of a draw.

**Start the debugging track with the two leads already in the bug report**, not from scratch:

1. **B15's residual gap.** Fixing per-batch conditioning normalization moved R²_cond from
   −0.001 to **+0.210**, against a ridge-probe ceiling of **0.747** on the model's own
   `cond_summary`. The gap is unexplained and is upstream of everything in this plan.
2. **`block_summary_adaln = False`.** Conditioning currently reaches the trunk only via
   cross-attention into K modal tokens whose residues come from `x_t`; at high noise those
   tokens are noise-dominated, so conditioning must overwrite them through residual updates —
   a weak path by construction. `CMD_BUG_REPORT.md` §3 calls enabling AdaLN injection
   "the natural next experiment."

**Do not run any UQ arms until this gate passes.** All three U3 arms sit on this denoiser; if
it carries no signal, the table measures nothing.

---

## Phase 2 — Cross-check on a second dataset

Only after Phase 1 passes. Phase 1 now runs on noaa_uk h=168 — the dataset Appendix G
pre-registers as the headline — so Phase 2 is no longer "confirm on the headline" but
"confirm the result is not dataset-specific."

### 🔴 bms_air h=168 CANNOT serve as the cross-check cell — measured 2026-08-03

The plan named bms_air h=168 here. **Run 1a on it before anything else and it fails the A
branch**: the raw history carries essentially **no forecastable signal at that horizon**.

| cell | n train/val | A raw history | B cond (best) | C latent (best) | verdict |
|---|---|---|---|---|---|
| bms_air, n=1 (modal) | 30 359 / 5 855 | **−0.3 %** | 0.6 % | 3.7 % | **MOVE DATASET** |
| bms_air, n=8 | 2 596 / 747 | **−4.6 %** | −3.5 % | 1.2 % | **MOVE DATASET** |

**The cause is physical, not a pipeline defect.** bms_air's target is **PM2.5**; noaa_uk's is
**temperature** (verified in the cache metadata). Seven-day-ahead temperature carries seasonal
and diurnal structure; seven-day-ahead PM2.5 largely does not. Ruled out as a level-shift
artefact three ways — train→val shift 0.03 val-std (noaa_uk's was 0.14 and scored +27.3 %),
−0.15 % against a *train*-mean baseline, level-free corr(pred, target) r = −0.025 — and oracle
alpha pins at 1e8, the top of the grid, i.e. ridge shrinks to nothing because there is no
direction to find.

**Replacement: `noaa_us` h=168.** Same target variable (**temperature**), same five feature
columns, same context/horizon geometry, stage-1/2 artifacts already present, and a genuinely
different sensor network from noaa_uk — so it is a real cross-check rather than a near-duplicate,
while keeping the target physically forecastable at this horizon. **Run 1a there before
committing any training.** Fallback if it also fails A: **crypto h=100**, accepting that a
*failure* there is not diagnostic (over-parameterized) while a *pass* still corroborates.

### 🔴 Standing rule: Gate 0 does not license a cell

**bms_air has the highest Gate 0 of the five cells (0.941) and no forecastable signal at all.**
So Gate 0 and 1a's **A** branch are independent, and the two must both be read:

- **Gate 0** answers "can stage 1 reconstruct its own targets?" — a property of the VAE.
- **1a's A** answers "is the target forecastable from its history at all?" — a property of the
  dataset and horizon.

A cell can pass the first and be useless on the second. Never promote a cell to a decision role
on Gate 0 alone. *(Found by measurement, 2026-08-03, after the paired-probe fix — the second
gate-design defect this campaign has caught the same way.)*

Purpose: avoid committing a 3-arm × 5-seed campaign to a result that only holds on one cache.

**Stop condition.** If the two datasets disagree, report before proceeding; do not average over
the disagreement. A pass on the under-parameterized cell plus a failure on an
over-parameterized one is *not* a disagreement — it is the expected pattern and should be
recorded as such.

---

## Phase 3 — Calibrate `CHIRP_UQ_INIT_VAR` per dataset

Do this **once per dataset × horizon**, before any UQ arm runs there.

The rule that transfers is *initialize predicted variance near the residual scale*. The
specific value does not transfer. On PhysioNet h=12 the default 1e-2 gave initial predicted
variance 0.0011 against a squared error of 2.675 — a ≈2415× mismatch the variance head could
not close, leaving the arm catastrophically overconfident:

| `CHIRP_UQ_INIT_VAR` | PIT cal. error | coverage @ 0.9 | predicted std | data CRPS |
|---|---|---|---|---|
| 1e-2 | 0.2210 | 0.034 | 0.096 | 0.2552 |
| 25.0 | 0.0389 | 0.872 | 4.09 | 0.2543 |

(target std 1.46.) **CRPS moved < 0.001 while coverage went 3 % → 87 %.** CRPS cannot detect
this failure.

Procedure:

1. **Evaluate the existing S1 checkpoint** on val and read its latent squared error. No extra
   warm-up run is needed — by this phase S1 exists (it is the warm-start source for S2/S3), and
   its residual *is* the scale the variance head has to reach. Same command as Phase 1b:

   ```bash
   llapdiff-uq-eval --dataset-key <ds> --pred <h> --checkpoint <S1 ckpt> \
     --weights ema --split val --latent-only --mean-source ddim
   ```

   Read **`latent_mse`**, not `latent_rmse`. `CHIRP_UQ_INIT_VAR` is a *variance*, and
   confusing it with a std is a factor the head cannot recover from — which is the exact
   failure this phase exists to prevent. Both squared forms (`latent_mse`,
   `baseline_mse_predict_mean`) are reported alongside the RMSEs for that reason.
2. Set `CHIRP_UQ_INIT_VAR` to roughly that magnitude.
3. Verify: coverage @ 0.9 should land near the **nominal 0.9**, not near 0. *(0.87 was the
   PhysioNet measurement on 13 memorized windows — a symptom that the fix worked there, not a
   target to hit elsewhere.)*
4. Record the chosen value and the measured residual scale in the dataset's prereg file.

Note the PhysioNet value 25.0 was itself fit on 13 memorized windows and is provisional even
there.

---

## Phase 4 — Freeze a new pre-registration

The existing prereg is explicitly scoped to PhysioNet h=12. Write a **new file**, not an
amendment: `finetuning/prereg/U3_<dataset>_h<h>.md`.

Carry over unchanged: the three-arm design, matched hyperparameters across arms, 5 seeds,
test-split-once discipline, and the declared scope limits (§7 of the old file — aleatoric-only
law, `q_k` constant in time, `v_k` is quadrature not closed form, decoder Monte Carlo rather
than delta method, x0 for all arms).

Fix three things while writing it:

1. **Gate thresholds against measured baselines.** The old G-c assumed unit-scale latents;
   actual latent std was 1.46. State thresholds relative to `baseline_rmse_predict_mean` as
   reported by the tool, not against an assumed scale.
2. **Outcome branches must cover the orderings actually observed.** The PhysioNet run produced
   A2 < A3 < A1, which matches none of the four pre-registered branches. Add branches for
   "analytic beats sampled on both CRPS and calibration" and for non-monotone orderings.
3. **Record the Phase-3 `CHIRP_UQ_INIT_VAR`** and the measured residual scale it came from.

---

## Phase 5 — U1, the guidance audit (prerequisite for any calibration reading)

Table 3's caption and §5.5 both state the guidance audit precedes any calibration conclusion:
guidance w > 1 sharpens the predictive distribution and is indistinguishable from
miscalibration. The PhysioNet U3 run violated this — it used ramp (1.0, 2.0) power 2.0, which
is not in U1's grid.

> **⚠️ Ordering.** This phase is numbered before Phase 6 but **cannot run before Phase 6's
> training**. `llapdiff-u1-sweep` scores a trained checkpoint, and the per-arm checkpoints (S2
> for A1/A2, S3 for A3) are produced by the Phase-6 training runs. Execute as:
> **Phase 6 training → Phase 5 sweep → freeze guidance → Phase 6 scoring.** The guidance choice
> must be frozen before any arm is *scored*, not before it is trained. (Running the sweep on the
> Phase-1/2 S1 checkpoint instead is acceptable only if you state that the choice is assumed to
> transfer from the MSE mean to the NLL arms — it is not obviously true, since the NLL arms
> differ in exactly the quantity guidance interacts with.)

On **val only**, per arm: sweep `w ∈ {1.0, 1.25, 1.5, 2.0}` × `GEN_STEPS ∈ {16, 32, 64}`,
samples fixed at 25. Log dynamic-thresholding clip rates (check whether `ẑ₀` brushes the
threshold after head removal/rescale).

```bash
llapdiff-u1-sweep --dataset-key <ds> --pred <h> --checkpoint <per-arm ckpt> \
  --guidance 1.0 1.25 1.5 2.0 --steps 16 32 64 --split val --weights ema
```

> **`--weights ema` is not optional** (flag added 2026-08-01). This tool loads through
> `_load_stack` like every other B16 consumer, so its historical default freezes the
> guidance choice on **raw** weights — for a model no other phase of the campaign ever
> evaluates. Measured cost of getting this wrong: 0.008 val CRPS on one fixed cell (see
> the standing rules).

Pick and **freeze** the guidance setting. Record it in the Phase-4 prereg. Note the PhysioNet
run used `DYNAMIC_THRESH_P = 0.0` (off) — that is defensible for calibration work, but make it
a recorded decision rather than an inherited default.

---

## Phase 6 — Run U3, with two design fixes

Three arms, identical synthesizer, 5 seeds, driver `finetuning/run_u3_uq.py` (tag `uq_u3`).

### Fix 1 — match the means across arms

Currently A1 uses a 25-draw ensemble mean and A2 a single deterministic DDIM pass, so the arms
differ in **two** ways at once (mean source *and* spread source). Evidence this bites: the
PhysioNet run produced three different MSEs (0.6286 / 0.6033 / 0.6226), and MSE is a property
of the point forecast alone — so the CRPS gaps are partly location differences wearing a UQ
costume.

Give A2 the **same ensemble mean as A1** and vary only the spread. Then any CRPS difference is
attributable to uncertainty, which is the point of the table.

**Requires code.** `AnalyticLawSampler` takes `mean_source ∈ {oneshot, ddim}`
(`tools/run_analytic_uq_eval.py:135`); an ensemble mean is a third source and must be added.

> 🔴 **This fix dissolves the efficiency headline — say so in the table caption.** An ensemble
> mean costs the 25 reverse-diffusion draws, so after Fix 1 **A2 costs what A1 costs and
> `analytic_speedup_x` for that pair goes to ≈1.** That is the correct trade — it buys clean
> attribution — but it means:
>
> - **A1 vs A2 is now purely a calibration comparison.** Do not quote a speedup from it.
> - **The efficiency claim moves to A3** (one-shot mean + analytic spread), or to the
>   off-diagonal cell below. Say which cell carries it.
>
> The "Wall-clock caveat" further down still describes the A1/A2 ratio as "the method-level
> number" — that sentence is written for the *pre*-Fix-1 design and must be updated together
> with this one, or the table will contradict itself.

Optionally run the two off-diagonal cells (ensemble mean + analytic spread; deterministic mean
+ sampled spread) to isolate the mean and spread contributions separately.

### Fix 2 — PIT and coverage for A1

The old prereg excluded calibration metrics for A1 ("A1 has no closed-form law"). A closed-form
law is not required: compute **rank histograms** and **empirical quantile coverage** from the
25 draws. The campaign plan (`cmd_plan_v2.md` §4, U2) already specifies reliability diagrams
*versus sampled UQ*, so this restores the original spec rather than adding scope.

**Cheaper than it looks — one new function.** In `models/uq_metrics.py`, only `gaussian_pit`
assumes a Gaussian law; `pit_calibration_error(u)` and `reliability_curve(u)` take PIT values
and are **law-agnostic**. So A1 needs a single helper —

```python
def ensemble_pit(draws, y, mask=None):   # draws [S,...], y [...]
    u = (draws < y).float().mean(0)      # rank/(S+1) form; break ties consistently
```

— and both existing metrics apply unchanged, which also guarantees A1 and A2/A3 are scored by
the *same* estimator rather than two lookalike ones.

Without it, the A1-vs-A2 comparison rests on CRPS alone — the metric Phase 3 proves blind to a
3 % → 87 % coverage swing.

### Table 3 columns (all arms)

`CRPS ± std | MSE ± std | PIT-ECE | coverage @ 0.8 | mean interval width | wall (s) | denoiser passes`

Report dispersion on MSE and wall-clock, not only CRPS.

**Wall-clock caveat.** Fitting cost = a + b·passes to the PhysioNet numbers (1, 64, 1600
passes → 0.5, 1.0, 12.6 s) gives ≈8 ms/pass and ≈0.49 s fixed overhead. A3's timing is ~98 %
harness overhead and A2's ~50 %, so `analytic_speedup_x` for the A2/A3 pair is nearly
meaningless at short horizons. Re-measure at the reporting horizon, where the fixed cost is a
far smaller share.

⚠️ **Updated for Fix 1.** The original text said "report the A1/A2 ratio as the method-level
number." That is no longer valid: with means matched, A2 runs the same 25 draws as A1 and the
ratio collapses to ≈1 by construction. **Report `A1 / A3` as the method-level speedup** — the
one pair that still differs in denoiser passes (1600 vs 1) — and label the A1/A2 comparison
"calibration at matched cost". Report `denoiser passes` alongside seconds in every row, so the
reader can separate method cost from harness overhead without re-deriving the fit.

---

## What to do with the PhysioNet results

Do not delete them. Relabel them.

1. **Report PhysioNet h=12 as a plumbing smoke test.** It is genuinely good at that — a full
   3-stage U3 seed costs ~20 min, and it surfaced B16, the variance-init failure, and B19.
2. **Remove PhysioNet from Table 1 and Table 2.** It is currently the only filled cell in
   Table 1 and the load-bearing column in Table 2. Everything else in those tables is already
   `[TBD]`, so the accurate status becomes "the empirical campaign starts at the Phase-1/2
   dataset" — and no unsupported claim ships.
3. **Add B19 to Appendix F (reproduction traps).** It belongs with the zero-gradient squared
   coefficients and per-dataset basis scaling. Draft entry:

   > *PhysioNet at short horizons yields only ~13 diffusion training windows under
   > patient-relative splitting (`WINDOW + PRED = 36` against ~48 h records). Data-space CRPS
   > nevertheless appears healthy, because the set-VAE decoder reconstructs each entity
   > largely from its own embedding — CRPS is dominated by a per-entity prior rather than by
   > the forecast. Audit window counts, not row counts.*

4. **Also worth an appendix line:** variance initialization must be set near the residual
   scale, and CRPS is blind to the resulting miscalibration.

### 🔴 B19 reaches further than this plan does — known blocker, out of scope here

Per `CMD_BUG_REPORT.md` B19, the five-window estimate invalidates more than Table 3:

- the **G2/G3 2×2 factorial** (cells (a)–(d) at 0.3706–0.3719, i.e. deltas of 0.0001–0.001);
- the **§8 branch decision**, taken on cell (a) = 0.3710, which set the paper's narrative to
  "match + certify + strictly generalize";
- the **22-trial `fixed_bugs` tuning campaign**, whose incumbent was chosen on a 0.0014 gap;
- the **"chirp arms show ~6× lower seed variance"** finding.

This plan recovers the UQ table only. Executed end to end it produces a defensible Table 3
attached to a paper whose **headline claim still rests on five test windows**. Fixing that is a
separate track — re-run the G3 factorial on noaa_uk h=168 — but it is a prerequisite for the
paper, not for Table 3, and it must not be lost because this document did not cover it.

---

## Execution order summary

```
Phase 0  window audit (all cells)                      ✅ DONE — w_docs/CMD_PHASE0_WINDOW_AUDIT.md
Phase 1  Gate 0 round-trip + ridge probe + S1 gate
         on noaa_uk h=168, 2 seeds, x0               → DECISION POINT
Phase 2  cross-check on bms_air h=168 (or crypto)
Phase 3  calibrate CHIRP_UQ_INIT_VAR (read S1's val residual)
Phase 4  freeze new prereg
Phase 6a U3 training (S1 -> S2 -> S3, 5 seeds)         <-- before Phase 5
Phase 5  U1 guidance audit on val, per arm; FREEZE
Phase 6b U3 scoring, means matched, PIT/coverage all arms
```

**Note the 6a/5/6b ordering** — Phase 5 needs the per-arm checkpoints that Phase 6 trains, but
the guidance choice must be frozen before any arm is *scored*. The linear 0→6 numbering is kept
for cross-references; the execution order is the one above.

Stop and report at Phase 1 if the gate fails on an **under-parameterized** cell with the ridge
probe passing by a clear margin. Everything downstream depends on there being a forecast to put
error bars around.

> 🔴 **TRIGGERED 2026-08-03.** noaa_uk h=168, both S1 seeds, both mean sources — see §1b. The
> trigger fires on *A* = 28.6 % (the cell is forecastable by a clear margin) while *C* < 10 %, so
> the intent of the rule is met exactly: there is a forecast to be had and the pipeline does not
> deliver it. **Phases 2–6 are on hold.** They are not merely unreliable, they are unanswerable —
> every one of them scores the calibration of a predictive law around a mean the gate has just
> shown carries ~6.5 % against a 10 % bar. The next action is the B21 conditioning track below,
> not any phase in this plan.
