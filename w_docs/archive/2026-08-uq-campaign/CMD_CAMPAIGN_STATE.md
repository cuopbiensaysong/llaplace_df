# co — state of play and handoff

> # 🛑 2026-08-07 — READ THIS BEFORE ANYTHING ELSE
>
> **Classifier-free guidance is destroying the forecast, and turning it off is worth ~5 % RMSE
> and 0.012 CRPS.** Guidance is strictly monotone-harmful on this cell (300 val windows, 12
> draws, ensemble mean): `w=1.0` gives RMSE **0.4296** / CRPS **0.2565**; the shipped
> `(1.0, 2.0)` ramp gives 0.4513 / 0.2688; `w=2.0` gives 0.4754 / 0.2826. **Freeze `w = 1.0`**
> for noaa_uk h=168 — it is also a *validity* requirement, since w > 1 sharpens the predictive
> distribution and is indistinguishable from miscalibration.
>
> With that fixed, the model **beats persistence by 17.8 %** (0.4296 vs 0.5228). It still loses
> to the eval-split mean (−7.7 %) — but that baseline is effectively **climatology**, which
> beats persistence by 31 %.
>
> ⚠️ **Three metric traps found here, each of which produced a wrong verdict of mine before it
> was caught.** (i) `generate(eta=0)` returns ONE sample; the point forecast is `E[decode(z)]`
> over draws — scoring a single draw cost 50 points. (ii) The "oracle-rescaled" reading collapses
> to the val-mean oracle (`c ≈ 0`) and is a tautology, not a measurement. (iii)
> `baseline_rmse_predict_mean` is computed on the **eval** split, making it 2.3× stronger than an
> honest train-mean baseline. **Name the baseline, average in data space, and never score a
> single draw.**
>
> **A ridge regression on 128 raw-history numbers scores 28.6 % on this cell. The pipeline
> delivers 0 %.**
>
> ⇒ **No UQ table is obtainable on noaa_uk h=168 with this pipeline**, and the whole
> conditioning track — B21, B23, B20, and every gate reading below — was measuring the
> distribution of a quantity that is ~zero at the end. The defect localises to **stage 3**: a
> ridge from `cond_summary → latent` beats the trained 10.2 M-parameter denoiser on the same
> input. See `w_docs/CMD_UQ_TABLE3_PLAN.md` Phase 0 for the numbers and the cheapest follow-up.
>
> Everything below remains accurate as *representation* measurement, and the three bug fixes are
> real (B20 lifts the round-trip 0.922 → 0.995, worth 65.5 % → 89.6 % in data-space
> reconstruction). None of it changes the line above.

**Written 2026-08-03, end of session.** Read this first, then `CMD_UQ_RECOVERY_PLAN.md`.
This file says where things stand and what to do next; the plan says how.

---

## 1. Status in one line

**The campaign is STOPPED at Phase 1 by its own stop rule.** The Phase-1b signal gate failed on
noaa_uk h=168 on both seeds and both mean sources, and the cause is pinned: the **conditioning**,
not the denoiser and not the data. **Phases 2–6 are on hold** — they score the calibration of a
predictive law around a mean the gate has just shown carries ~6.5 % against a 10 % bar.

**Next action is not a phase in the plan.** It is the B21 conditioning track (§5 below).

🔴 **Update 2026-08-04/05 — B23 found, measured, and closed out.** At the shipped defaults the
conditioning never reaches three of the denoiser's four consumers; the pole/chirp path that sets
ρ(t), ω(t) and the UQ law is fed an **identically zero** vector (verified on 120 real val
windows: across-window std **7.5e-08**, against 0.533 under `"global"`). Unlike B21 §6 and B5,
B23 is reachable from **stage-3 config knobs alone**, so it was the cheapest live lead.

**Six converged runs later (3 arms × 2 seeds, x0, `EPOCHS=600`, vs the Phase-1b seeds as matched
controls) the answer is: the mechanism is real and repairing it is not sufficient.**

- **Delivery moves reproducibly.** `COND_POOL_MODE="attn"` lifts `oneshot latent_corr` from
  0.18877 to **0.21400 / 0.21401 — +13.4 % on both seeds, 18× the control seed spread**, the two
  seeds agreeing to 0.006 %.
- 🔴 **The gate still fails.** Best rescaled `ddim` anywhere **+8.05 %**, best two-seed mean
  **+7.09 %**, against **10 %** (control +6.40 %). **Phases 2–6 remain on hold.**
- 🔴 **Debugging lead 2 is closed — it is harmful, not merely inert.** Run live for the first
  time (paired with `attn` so its AdaLN half is not zeros), it pushed `oneshot` *below* control
  on both seeds (0.12648 / 0.17021) and erased the CRPS gain. Do not re-open it.
- **`COND_NORM_MODE="global"` now has its first non-truncated CRPS evidence**: 0.30325 / 0.30405
  vs controls 0.32376 / 0.33229, treated-pair spread 0.0008. Two seeds, one cell — short of the
  ≥5 seeds and second dataset B21 requires, but no longer confounded by truncation *or* by
  `predict_type="v"`.

Full table, protocol notes and the fourth dissociation: **B23 in `CMD_BUG_REPORT.md`**; working
detail in `finetuning/results/b21_delivery/RUNS.md`.

**So the next action is no longer B23.** The remaining conditioning defects are **B21 §6**
(stage-2 supervises 1 of 336 token directions) and **B5** (entity mean-pooling, dominant on wide
panels). Both need a retrain and **break the shared frozen stage-1/2 parity** every cross-arm
comparison depends on, so neither is a config change — each needs its own design and its own
decision about what parity means.

---

## 1b. Update 2026-08-06 — all three defects are now FIXED IN CODE; one of them moves the gate metric the wrong way

All three are repaired, tested and measured where measurement was affordable. **Every new
artifact lands on its own filename; nothing shipped was overwritten**, so every recorded number
stays reproducible.

| defect | fix | status |
|---|---|---|
| **B23** delivery | `COND_POOL_NORM_MODE="global"` (new, default): the pooled consumers normalise `cond_summary_raw` with fixed train statistics; cross-attention keeps `COND_NORM_MODE="sample"` | ✅ **verified on real data** — pooled across-window std **7.5e-08 → 0.5329** on the same 120 val windows, reproducing B23's own predicted row (2.07 / 0.533) |
| **B21 §6** objective | `SUM_DECODE_MODE="token"` (new, off by default): cross-attention unpool decoder, per-time-step reconstruction over all S tokens | ✅ repaired, ⚠️ **does not help the gate** |
| **B20/B5** pooling | `VAE_ENTITY_ENCODE` (new, off by default): entity embedding on the *encoder* input, so the pooled latent stops being permutation-invariant | ✅ **the one that creates headroom** — Gate 0 **0.9222 → 0.9952**, above the panel-mean ceiling; the cap on `C` rises **10.0 % → 13.6 %** |

**The B21 result is the one that matters for the campaign, and it is negative on the gate axis.**
Retrained on noaa_uk h=168 with a matched control (the shipped artifact was *copied in*, not
trained here, so the legacy objective was retrained under the identical budget):

| stage-2 arm | val recon | B best view | **C** |
|---|---|---|---|
| shipped | 0.41499 | 14.2 % | 5.7 % |
| ctrl (legacy objective, retrained here) | 0.42953 | 13.9 % | 7.3 % |
| **token (the fix)** | **0.29316 (−32 %)** | 14.4 % | **2.5 %** |

The objective defect is real and the repair works *on stage-2's own loss* — a third less
reconstruction error, in fewer epochs, with 5.3× fewer parameters. But **B's ceiling does not
move and C falls**, which fires the stop condition registered before the numbers existed. This
is the **fifth** instance of this pipeline's probe/forecast dissociation, and the tightest:
`SUM_FT_MODE="all"` (best CRPS of any run) scored C = 1.4 %, the token objective (best
reconstruction of any summarizer) scores 2.5 %. **Two interventions that demonstrably add
training pressure to the 335 unsupervised directions both lower the linear latent readout.**

⇒ **Do not read this as "B21 is fixed, so the campaign is unblocked."** What is settled is that
the *objective* defect is repaired and that repairing it does not, by itself, clear the C-gate.
Whether the denoiser can use the reformatted conditioning is a stage-3 question (~20 GPU-h,
2 seeds) that the pre-registered rule says not to spend and the `SUM_FT_MODE="all"` precedent
says is the single most informative run available. **That decision is open and is the owner's
to make.**

### 🔑 The result that actually changes the campaign's position is B20's

**The Phase-1 gate was unreachable by construction, and now it is not.** `C`'s cap is the
*raw history → latent* probe — the best any conditioning could do on that latent. On the shipped
stage 1 that cap is **10.0 %** against a **10 %** threshold: a perfect summarizer would have tied
the bar. Every conditioning intervention this campaign has run (`global`, `attn`, `SUM_FT`,
B21 §6) was competing for a ceiling sitting exactly on the threshold, which is the simplest
explanation of why all of them failed to open the gate.

| stage 1 | Gate 0 round-trip | **cap on `C`** | best `C` achieved |
|---|---|---|---|
| shipped | 0.9222 (= the panel-mean share, a *dataset* property) | **10.0 %** | 5.7 % |
| **entity-encoded** | **0.9952** | **13.6 %** | **7.3 %** |

Clearing 0.922 is the proof the ceiling was architectural: a dataset property cannot be
exceeded. **The gate is still not open (7.3 % against 10 %), but there is now 3.6 pt of real
headroom above the bar where before there was none.**

**Next action.** Not another conditioning knob — B is stuck at ~14.2 % under every intervention
tried. The two open moves, in order: (i) re-read the gate on the repaired stack, since `C` is a
probe and this pipeline has five recorded probe/forecast dissociations; (ii) retrain stage 1 on
**noaa_us** (40 entities, round-trip 0.669, where pooling costs −17.4 of 20.8 pt) to see whether
the headroom generalises off the 4-station cell. (ii) was deliberately skipped here: its VAE runs
5–10× slower per epoch (15–30 h) and would have delayed every number above by a day.

### 🔴 Withdrawn a few hours later — "raise the cap with one more stage-1 fix" is not achievable

**B24 (new).** The cap is not a property of stage 1. Measured decomposition on noaa_uk: a
**lossless** rank-4 linear code of the target scores **29.0 %** at natural scale and **13.3 %**
once per-channel whitened — the same information, −15.7 pt from the normalisation alone. The
learned VAE latent scores **13.5 %**, i.e. **already above the lossless-linear reference**.

⇒ ~~There is no stage-1 headroom left to recover, and the plan "cap 13.6 → 16.5 %" is withdrawn
as unachievable.~~ 🔴 **That inference was wrong and was falsified the same hour by a registered
test:** `ch=32` reaches a cap of **18.1 %**, past the 16.5 % target. What is true is narrower —
whitening costs ~15 pt for a *given* code, and the whitened score depends on how the code spreads
variance across channels, which is a design variable.

⇒ **But `C` went the other way: 7.3 % → 4.9 %** at `ch=32` (delivery 54 % → 27 %). The extra
headroom is unreachable from `cond_summary`. So the surviving relationship is not "cap ⇒ C" but
**`B` pinned at ~14.2 % under every intervention tried**, with `C` bounded by delivery rather
than by what the latent exposes. Best stack for the gate stays **`ch=16` entity-encoded +
shipped summarizer, `C` = 7.3 %**.

⇒ The live question is no longer "how do we raise `C`" but **"what should the Phase-1 threshold
be, in the space it is actually read in?"** 10 % was imported from a data-space intuition
(the target is 28.6 % forecastable) and applied to a whitened latent whose lossless reference is
13.3 %. Re-derive it *before* looking at a new gate number. In that framing the existing readings
are: control **6.5 % ⇒ 49 %** of the lossless reference, repaired stack predicted **62–66 %**.

Full detail and the caveats — in particular that this does **not** mean "the gate is wrong so we
pass" — in **B24** of `CMD_BUG_REPORT.md`.

---

## 1c. 🔴 THE GATE WAS RE-READ 2026-08-06/07 — it FAILS, and it fails *worse* than the control while data-space CRPS IMPROVES

Two seeds, arm `d`, `predict_type=x0`, `EPOCHS=600` + early stop, campaign eval profile (six
keys verified in the log), on the repaired stack: **entity-encoded stage 1 + shipped summarizer +
B23 pooled repair**. Scored CPU throughout, like the controls. Threshold frozen in writing
(B24: 6.2 % latent ⇔ ρ ≥ 0.344) **before** the read.

| | seed 0 | seed 1 | **mean** | control mean |
|---|---|---|---|---|
| `ddim` rescaled | +2.77 % | +1.41 % | **+2.09 %** | **+6.41 %** |
| `oneshot` rescaled | +0.06 % | +0.25 % | **+0.16 %** | +1.80 % |
| best val **CRPS** | **0.30836** | 0.33104 | **0.3197** | 0.3280 |

**Both seeds agree, so this is not a draw.** The gate axis is roughly **a third of the control**
and fails both the original 10 % threshold and B24's re-derived 6.2 %. Meanwhile **data-space
CRPS improved** (−0.008 mean; seed 0 at 0.30836 is among the best recorded on this cell).

### The two axes moved in OPPOSITE directions, and that is the finding

Every probe said this stack should be better: Gate 0 **0.922 → 0.995**, the cap **10.0 → 13.6 %**,
`C` **5.7 → 7.3 %**. The trained model says worse. The coherent reading:

**Making stage 1 more faithful moves *unpredictable* content into the latent.** The old VAE
emitted the panel mean and discarded each entity's deviation — the least predictable component.
The repaired one keeps it (round-trip 0.995). The gate scores the *fraction* of latent variance
the model explains, so adding genuinely unpredictable variance to the denominator **lowers the
gate score even as the forecast improves**. CRPS rises because the decoder now reproduces
per-entity structure instead of the panel average.

⇒ **The Phase-1 gate is not merely mis-calibrated (B24); across this intervention it is
anti-correlated with the campaign's actual objective.** A criterion that penalises a stage-1
improvement which demonstrably improves the forecast cannot gate this campaign. That is now the
central open problem — bigger than any remaining conditioning defect.

### ✅ Attribution COMPLETE 2026-08-07 — the 2×2, and every arm shows the same inversion

Single-factor arms, seed 0, identical protocol, CPU-scored. Predictions were registered in
`finetuning/results/b21_objective/RUNS.md` before launch.

| arm | stage 1 | pool norm | `ddim` resc | **Δ gate** | val CRPS | **Δ CRPS** |
|---|---|---|---|---|---|---|
| control (2 seeds) | shipped | inherit | 6.41 % | — | 0.3280 | — |
| **`b23only`** | shipped | **global** | 4.82 % | **−1.59** | 0.31513 | **−0.013** ✅ |
| **`s1only`** | **entenc** | inherit | 3.01 % | **−3.40** | **0.30179** | **−0.026** ✅ |
| both (2 seeds) | entenc | global | 2.09 % | −4.32 | 0.3197 | −0.008 ✅ |

**Both factors degrade the gate, roughly additively** (−1.59 − 3.40 = −4.99 against the combined
−4.32), with stage 1 the larger term. **And every single arm improves val CRPS.**

⇒ 🔴 **Across four independent interventions, the Phase-1 gate axis and the campaign's
deliverable metric move in OPPOSITE directions, without exception.** This is no longer an
anomaly to explain away — it is a measured property of the gate. The mechanism stands: a more
faithful stage 1 (round-trip 0.922 → 0.995) moves genuinely unpredictable per-entity content into
the latent, so the *fraction* of latent variance explained falls while the decoded forecast
improves. **A criterion that is anti-correlated with the objective cannot gate this campaign.**

Prediction scorecard: `s1only` predicted +1.5–3.5 %, measured **3.01 % ✅**; `b23only` predicted
+6–8 %, measured **4.82 % ❌**.

### 🔴 `COND_POOL_NORM_MODE="global"` was shipped as the default and has been REVERTED

The registered rule was "if `b23only` lands materially below control, the default is harmful and
must be reverted". It landed −1.59 pt. **Reverted to `"inherit"`** rather than reasoned around,
even though the same arm improved CRPS. The knob stays, fully documented.

⚠️ **The U3 arms must switch it back ON deliberately.** There it is a correctness requirement,
not tuning: with the pooled path dead, `chirp_field.uq_params` sees no history, so the Theorem-C
law's `p0`/`q`/ρ̄ are identical for every window and arms A2/A3 would score a structurally
homoscedastic law (B23). Pinned by `tests/test_pole_conditioning.py`.

### ~~Attribution is NOT established~~ — superseded by the table above

New stage 1 **and** B23's pooled repair (`COND_POOL_NORM_MODE="global"`, which this session made
the default). **The B23 default is a live suspect for part of the regression** and must not be
treated as settled. The two runs that decompose it are specified and unlaunched:

| arm | overrides | isolates |
|---|---|---|
| `s1only` | entenc VAE + `COND_POOL_NORM_MODE="inherit"` | stage 1 alone |
| `b23only` | shipped VAE + `COND_POOL_NORM_MODE="global"` | the B23 default alone |

Override files are written at `finetuning/results/b24_gate/overrides_{s1only,b23only}.json`;
each is ~10 h on one GPU. **Until they run, do not attribute the regression, and do not assume
the B23 default is safe.**

### B20 generalises — noaa_us confirms it

| cell | entities | shipped round-trip | **entity-encoded** |
|---|---|---|---|
| noaa_uk | 4 | 0.9222 | **0.9952** |
| **noaa_us** | **40** | 0.669 | **0.800** |

Both clear their panel-mean ceiling, and the gain is **larger on the wide panel** (+0.131 vs
+0.073) — exactly where B20 predicted pooling costs most (−17.4 of 20.8 pt). noaa_us still misses
the 0.85 Gate-0 bar, so it is not yet a usable Phase-2 cross-check cell.

Working detail and reproduce commands: `finetuning/results/b21_objective/` (gitignored);
durable record in B21 §6, B23 and B20 of `CMD_BUG_REPORT.md`.

⚠️ **A retrained stage 1 or 2 breaks parity with every recorded stage-3 result.** All of them
were trained against the shipped artifacts. Any comparison across the change needs its own
controls.

---

## 2. Where the work is — read this before touching git

| | |
|---|---|
| branch | **`cmd-uq-tooling-and-b21`** — **not** `main` |
| ahead of `main` by | **20 commits** |
| pushed | yes, `origin/cmd-uq-tooling-and-b21` |
| merged to `main` | **no** |

⚠️ `main` is at `f1788e3` and contains **none** of this campaign. A fresh agent that checks out
`main` will see none of the tooling fixes, none of B20–B22, and will re-derive results that are
already recorded. Start from `cmd-uq-tooling-and-b21`.

⚠️ **`.gitignore:46` ignores `finetuning/results/*`.** Two campaign records lived only there and
would not survive a clone; they now have tracked copies. **Edit the tracked copy:**

| tracked (canonical) | ignored working copy |
|---|---|
| `w_docs/CMD_PHASE0_WINDOW_AUDIT.md` | `finetuning/results/window_audit.md` |
| `w_docs/CMD_PHASE1B_GATE.md` | `finetuning/results/phase1b/gate_readout.md` |

Raw JSON (`gate_seed*.json`, `s1_gate_seed*.json`) and all checkpoints under `ldt/` remain
untracked by design — they are large and reproducible from the commands in the tracked copies.

🔑 **`.git/config` contains a GitHub personal access token in plaintext** in the `origin` URL. It
is not pushed, but it is readable by anyone with filesystem access — which on this machine
includes other agents. **Rotate it** and switch to SSH or a credential helper. Left as-is because
changing a remote is the owner's call.

---

## 3. Settled — do not re-litigate, do not re-run

| finding | evidence |
|---|---|
| PhysioNet h=12 is a smoke-test cell only | 13 train windows vs 11.75 M params (B19, Phase 0) |
| crypto / us_equity cannot serve the gate | 90 / 72 val windows; over-parameterized 6.0× / 8.9× |
| bms_air h=168 is **VOID** as a cross-check | the h=168 eligibility filter leaves mostly *single-entity* windows; 23 full-panel val windows |
| Gate-0 failures are **not** a capacity problem | the Gate-0 score equals the cell's own cross-entity mean share (B20) |
| noaa_uk h=168 is the right gate cell | Gate 0 = 0.922, forecastable at A = 28.6 %, 4.3× **under**-parameterized |
| the summarizer, not the VAE, is the bottleneck | 1a: A = 28.6 % raw history vs B = 7.7–16.4 % through the conditioning |
| `COND_NORM_MODE="global"` is **not adopted and not measured** | 5 seeds, mean −0.0086 CRPS, 4 of 5 inside the 0.036 band, and `EPOCHS=60` truncated 8 of 10 runs at their final eval. 🔴 **And all of it ran at `predict_type="v"`** — arm `d` is `("--modal-type","chirp")` only (`finetuning/common.py:29`) and does not force x0; those overrides carry no `cli` key, and `ldt/tuning/rawcond_probe/.../output/modal-chirp/` has **no `predict-x0/` segment** where `phase1b/.../output/predict-x0/` does. So it does not transfer to the gate or to the U3 arms, which are all x0 |
| **the denoiser is not the bottleneck** | Phase 1b: rescaled 6.5 % mean vs a ridge ceiling of 5.6 % — **+0.9 pt** for 10.2 M params |
| seed noise band on CRPS is **0.036** | §4 retraction 1; every decision run needs ≥ 2 seeds |
| **3 of 4 conditioning paths are dead at the defaults** | B23: `"sample"` × `"mean"` makes the pole/chirp `cond_vec` summary half **identically zero** (absmax 7.9e-08); AdaLN inert; analysis-QK off. Verified on both gate checkpoints' own `model_config` |

Two corrections that cost real time and are easy to repeat — both are recorded in
`CMD_BUG_REPORT.md`, read them before designing a probe:

- **Never mean-pool the summary tokens** in a probe. Averaging 336 tokens scores −0.2 % where the
  strided view scores 14.2 % — pooling manufactures a FAIL out of working conditioning.
- **Select ridge alpha on a chronological train holdout, never on val.** Selecting on val moved a
  headline 6.2 % → 9.8 %, straight across the 10 % threshold a branch turns on.

---

## 4. What Phase 1b actually measured

Full detail in `w_docs/CMD_PHASE1B_GATE.md` and §1b of the plan. The three things that matter:

1. **The gate fails.** Both seeds (CRPS 0.32376 / 0.33229, spread 0.0085 ≪ 0.036 band), all four
   reads. Even granting a free optimal rescaling the model does not have, 6.5 % < 10 %.
2. **The denoiser sits at the ridge ceiling.** 10.2 M parameters, a nonlinear architecture and 355
   epochs buy **under one percentage point** over linear least squares on the same inputs. This
   converts B21 from an inference about probes into a measurement on the trained model.
3. **`--mean-source ddim` is the wrong statistic for a signal gate** and the plan no longer
   mandates it alone. `generate(eta=0.0)` returns **one sample** from a random `x_T`, with CFG on a
   `(1.0, 2.0)` schedule — 2.2–2.4× over-dispersion, enough to turn a +6 % model into a −3 %
   reading. Run **both** sources; the gap between them is diagnostic.

---

## 5. The next task: the B21 conditioning track

The gate will not move until **C** in Phase 1a (`cond_summary` → latent) clears 10 %. It currently
reads 4.9–6.8 %. **Re-entering 1b on any other cell tests nothing** — every cell carries the same
summarizer.

**Root mechanism (B21 §6, found with agent-B).** Every `LaplaceAE` pretraining head reads
`ctx_mean = context.mean(dim=1)` — `llapdiffusion/models/summarizer.py:637`, and lines 639–642 for
the four decoders. So of 336 token directions, **335 are unsupervised**; the objective only ever
constrains their mean. `COND_NORM_MODE="sample"` then removes exactly the one direction that *was*
supervised. That is the defect to fix, and **it has not been attempted** — everything tried so far
works downstream of it.

**Already tried, so do not repeat as if new:** `COND_POOL_USE_RAW=True`, `SUM_FT_MODE=all`,
`COND_NORM_MODE="global"` (5 seeds, inconclusive — see §3).

**Two named leads, both still open**, in the plan's debugging-track section:

1. **B15's residual gap.** Fixing per-batch conditioning normalization moved R²_cond from −0.001
   to **+0.210**, against a ridge ceiling of **+0.747** on the model's own `cond_summary`. The gap
   is unexplained and upstream of everything.
2. **`block_summary_adaln = False`.** Conditioning reaches the trunk only via cross-attention into
   K modal tokens whose residues come from `x_t`; at high noise those tokens are noise-dominated.
   **Phase 1b confirmed this independently**: `oneshot` at `t = T−1` scores corr 0.1903 / 0.1931
   against the trajectory's 0.3702 / 0.3395 — the model recovers half its correlation only *after*
   `x_t` becomes informative, and an ideal model would order these the other way. The two seeds
   land 1.5 % apart, so it is architectural, not a draw.

   🔴 **Amended 2026-08-04 (B23) — do NOT run lead 2 alone; it is provably inert.**
   `make_block_cond` pools the summary through the same `pool_summary_tokens` as the pole path,
   and at the shipped defaults (`COND_NORM_MODE="sample"` × `COND_POOL_MODE="mean"`) that pool is
   **exactly the zero vector**. Setting `BLOCK_SUMMARY_ADALN=True` would widen the AdaLN
   conditioning to `2H` and fill the new half with zeros — measured absmax **7.9e-08**. It must be
   paired with `COND_NORM_MODE="global"` or `COND_POOL_MODE="attn"`. B23 also shows lead 2's
   premise is understated: **three** of the four conditioning paths are dead in the gate runs,
   including the pole/chirp path that sets ρ(t), ω(t) and the UQ law.

**How to know it worked — three axes, and the old criterion only covers one.** `C` measures what
is *in* `cond_summary`; it is computed by a probe that never enters the denoiser, so it is blind
to B23's delivery defect by construction. B21 separately concludes that probe score does not track
forecast quality here. Report all three:

| axis | question | readout | current value |
|---|---|---|---|
| **content** | is the signal in `cond_summary`? | `C` from `llapdiff-ridge-probe --tokens 8` | 4.9–6.8 % |
| **delivery** | does it reach the trunk? | `latent_corr` @ `llapdiff-uq-eval --latent-only --mean-source oneshot` | **0.1903 / 0.1931** |
| **forecast** | does it help? | best val CRPS, matched seeds | 0.32376 / 0.33229 |

**Delivery is the primary readout for a B23 fix.** It is a measurement on the trained model rather
than a probe, so it sidesteps the probe/forecast dissociation; and its seed reproducibility is
**1.5 %** against CRPS's 0.036 band (≈ 11 % relative), so **2 seeds decide it** where 5 are not
enough for CRPS. A content-only criterion (`C ≥ 10 %`) would score a successful delivery fix as a
null.

---

## 6. Operational hazards specific to this repo

- **`PYTHONPATH=$LLAPDIFF_SRC` is not optional.** A sibling checkout is installed editable in the
  venv; without it, `import llapdiffusion` silently resolves elsewhere and edits appear to do
  nothing. See `CLAUDE.md`. Verify with
  `python -c "import llapdiffusion; print(llapdiffusion.__file__)"`.
- **`EARLY_STOP` counts evals, not epochs.** Patience is `EARLY_STOP × DOWNSTREAM_EVAL_EVERY`.
- **Do not launch a training run without setting `EPOCHS` explicitly.** The default is 600; a
  forgotten override once queued ~32 h of work. Validate override JSON with `json.load` *before*
  launching.
- **`*.json` globs over a results directory will count override files**, not just results.
- **Campaign eval profile** (~61× cheaper val, and what every recorded number above used).
  🔴 **Corrected 2026-08-04 — every variable carries the `LLAPDIFF_` prefix.** The earlier
  version of this line prefixed only the first, and `_EVAL_PROFILE_ENV`
  (`finetuning/run_trial.py:59-66`) reads all six prefixed — so copying it verbatim silently
  kept the **full** validation protocol, ~60 h per trial instead of ~2.8 h:
  ```bash
  export LLAPDIFF_EVAL_STEPS=16 LLAPDIFF_EVAL_NUM_SAMPLES=5 LLAPDIFF_EVAL_MAX_BATCHES=48 \
         LLAPDIFF_EVAL_SUBSET_MODE=stride LLAPDIFF_EVAL_SEED=4242 LLAPDIFF_VAL_DIAG_EVERY=5
  ```
  Confirm it applied: the run must log
  `[run_trial] validation-protocol profile from env: {...}` with **six** keys.
- **`--smoke` uses `setdefault("EPOCHS", 3)`** (`run_trial.py:104`), so an `EPOCHS` in
  `--overrides-json` **wins over it** — passing `--smoke` with the real override file launches a
  full 600-epoch run. Give the smoke its own override file with `EPOCHS: 3`.
- **A second agent (agent-B) shared this filesystem.** Its channel is `w_docs/agents_communication.md`
  and its results are in `w_docs/results_nonlinear_probe.md`. Its shared-variance column supersedes
  an earlier one of mine; the leave-one-out figures (noaa_uk 76.5 %, noaa_us 46.8 %, crypto 42.4 %,
  us_equity 34.2 %) are the mediator of record. If another agent is run again, give it the same
  write restrictions recorded in `w_docs/HANDOFF_nonlinear_probe.md`.
- **415 tests pass** (`python -m pytest tests/ -q`). Keep them passing; several encode bugs that
  recurred (B16 EMA/raw, the `blocked_purged_split` collapse, the degenerate-window filter).

---

## 7. Open questions nobody has answered

1. Should `finetuning/results/window_audit.md` and the other results files be force-added to git,
   or is the tracked-copy arrangement in §2 the permanent answer? Asked twice, unanswered.
2. Should `cmd-uq-tooling-and-b21` be merged to `main`, or stay a long-lived branch?
3. The shared-variance reproduction discrepancy (my noaa_us 42 % vs agent-B's 49.0 %, drop rates
   11.4 % vs 1.7 %) was resolved by adopting agent-B's column wholesale, **not** by finding the
   bug. The two scripts filter differently and the cause was never identified.
