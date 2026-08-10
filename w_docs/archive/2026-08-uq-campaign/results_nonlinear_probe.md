# Is the B21 conditioning gap information LOSS, or nonlinear re-encoding?

**Author:** agent-B · **Written:** 2026-08-02 · **Task:** `w_docs/HANDOFF_nonlinear_probe.md`
**Cell:** noaa_uk h=168, val split, `COND_NORM_MODE="sample"` (config default, no checkpoint)
**Protocol status:** §2 documents a deviation from the handoff's literal selection rule.
**agent-A ruled on it (`agents_communication.md`, 2026-08-02 06:05): the blocked+purged
holdout is accepted as primary**, the literal last-20 % is kept as a secondary row, and
agent-A has since **fixed `ridge_reduction` itself** to use it (commit `ef0d679`).

> **One-line answer — §2.2 branch 2, genuine LOSS.** An **oracle-selected** MLP extracts no
> more from `cond_summary` than ridge does (**14.15 %** vs **14.23 %**), and the cond/hist
> ratio is unchanged: **50.0 %** (oracle MLP) against **49.7 %** (fixed ridge) — a 0.3 pt
> difference, versus the ≥ 80 % that "nonlinearly encoded" would require. The gap is **not**
> an artefact of linearity. Two
> qualifications that matter: `cond_summary`'s honest figure is **~14 %**, not 7.7 % (the
> 2 048-dim view is impoverished and nonlinearity partly compensates for it — 12.23 vs 7.74);
> and this bounds what a **probe** can read, **not** what the denoiser can use, so it does
> **not** reinstate B21's retracted "discards ~½ the signal" headline.
>
> **§8 follow-up (`COND_NORM_MODE="global"`):** level-restoration and nonlinearity turn out to
> be **substitutes**, not additive — my registered prediction was wrong. Across the full
> view × normalisation × probe-class grid the best raw-target number anywhere is **14.23 %**,
> plain ridge on the `"sample"` 8 192-dim view; nothing beats it. Includes a correction to a
> B21 figure: `"global"` @8192 → raw target is **13.69 %, not 16.4 %**.

---

## 0. What was run, and where

Everything is a standalone script **outside the repo**, importing `llapdiffusion` read-only
(`/vol/dl-nguyenb5-solar/users/cuopbiensaysong/_agentB_nonlinear_probe/`). No file under
`llapdiffusion/`, `tests/`, `ldt/diffusion_cache/`, `ldt/tuning/` or `finetuning/results/`
was written. GPU: **cuda:2 only**, pinned with `CUDA_VISIBLE_DEVICES=2` (agent-A's campaign
holds GPU 1 at ~86 % throughout).

| script | role |
|---|---|
| `collect_features.py` | one frozen forward pass over train+val, caches features |
| `ridge_check.py` | reproduction of agent-A's six ridge cells |
| `mlp_probe.py` | honest MLP probe (selection strictly inside train) |
| `mlp_oracle.py` | leaky oracle-val **upper bound**, incl. depth-0 linear controls |
| `token_mean_check.py` | supplementary structural check (§6) |

---

## 1. Reproduction check — PASSES, so the comparison is valid

Handoff §3.5 requires this before anything is interpreted. Data collection reuses
`llapdiffusion/tools/run_ridge_probe.py::_collect` **verbatim**, so the eligible-window set
is by construction identical to agent-A's.

**Window counts match exactly**: **14 446 train / 2 183 val at 4 entities** (the modal
count), as §2.3 predicts. Feature dims: raw history 128 (4 entities × 32 strided steps),
`cond_summary` 2 048 / 8 192 (8 / 32 tokens strided over the 336-token axis, **never
mean-pooled**), latent `mu_norm` 2 688 (168 × 16), raw target 672 (168 × 4).

Against agent-A's published values, using the **then-current** `ridge_reduction` (last-20 %
alpha selection), all six cells reproduce to within **0.07 pt** — metric: % RMSE reduction vs
the scalar grand-mean baseline `sqrt(((y - y.mean())**2).mean())`, val never used for fitting:

| ridge probe | dims | → raw target | → latent |
|---|---|---|---|
| raw history | 128 | **27.3 %** (ref 27.3) | **11.6 %** (ref 11.6) |
| `cond_summary`, 8 tok | 2 048 | **7.7 %** (ref 7.7) | **5.7 %** (ref 5.6) |
| `cond_summary`, 32 tok | 8 192 | **14.2 %** (ref 14.2) | **4.2 %** (ref 4.2) |

That is the gate the handoff asks for, and it passes: **the windows and the metric match, so
the comparison is valid.** These are the *pre-fix* numbers; §2b explains why the raw-history
rows have since moved, and §4 reports both.

I also verified the cached row order is genuinely chronological (adjacent-window corr
**+1.0000**, lag-500 **+0.798**, shuffled **+0.0008**), and that train/val level
distributions differ as a chronological split implies (train grand mean −0.036 / std 0.912;
val −0.140 / 0.752). So "the last 20 % of train" really is a *later* period.

---

## 2. 🔴 Protocol deviation, and why — the handoff's selection holdout is pathological here

**This is the one place I did not follow the handoff literally.** It is raised as a QUESTION
in `agents_communication.md`; both variants are reported below so agent-A can take either.

§2.3 says to select hyperparameters on "a chronological holdout inside train (the last
20 %)". On this cell that band is a **shifted seasonal regime**:

| | grand mean | grand std |
|---|---|---|
| fit portion (first 80 % of train) | **+0.183** | 0.986 |
| holdout (last 20 % of train) | **−0.911** | 0.656 |
| val | −0.140 | 0.752 |

Predicting the fit mean there costs **+94 %** on that band's own baseline. The consequence is
measurable on ridge itself: fitted on the first 80 %, **ridge scores −8.3 % on that holdout
while scoring +28.8 % on val**. Centering the holdout metric does not repair it (ridge
−7.4 %) — the band differs in variance structure, not merely in level.

**Ridge survives this; early stopping does not.** A common offset cancels when *ranking*
alphas, so agent-A's alpha selection is essentially unharmed. But an MLP selected by early
stopping on that band halts at **epoch 4** and lands at **22.0 % ± 1.0** on val — *below*
ridge's 27.3 %, i.e. failing the handoff's own §2.3 sanity gate.

**The pathology is target-specific, which is a useful tell.** Under the last-20 % holdout the
holdout scores for the three **raw-target** cells are −9.71 %, −1.40 % and +0.12 %, while the
three **latent** cells look perfectly healthy (+28.4 %, +30.0 %, +32.7 %). `mu_norm` is
globally z-scored and so carries no seasonal level, whereas the raw temperature target does.
This is exactly why the two holdouts agree on `hist→latent` (9.56 vs 9.99) and diverge on
`hist→target` (22.00 vs 20.78) and `cond32→target` (**8.85 vs 13.33**) — on the last cell the
blocked selector recovers 4.5 pt that the shifted band throws away.

**Substitute** (still strictly inside train, val never touched): a **blocked, purged**
holdout — 3 contiguous blocks (3/8/13 of 15) spread across the train span, with a
**336-window purge** either side so no fit window shares context with a holdout window
(336 = `WINDOW`; the same idea as the repo's `global_purged_horizon`). Ridge reads
**+38.4 %** there against **+28.4 %** on val — same sign, same ballpark, so it tracks.

### 2b. This propagated into agent-A's own tooling — now fixed

`ridge_reduction` selected alpha on that same shifted band. On the raw-history cell it picked
**α = 1e4 → val 27.28 %**, where **α = 1e3 → val 28.61 %**, so the published **27.3 % was
itself depressed ≈ 1.3 pt**.

**agent-A has fixed this** (`run_ridge_probe.py`, commit `ef0d679`): `ridge_reduction` now
takes a `purge` kwarg and selects on `blocked_purged_split(n, n_blocks=15, holdout=(3,8,13),
purge=WINDOW)`. Re-running my six cells against the fixed function reproduces agent-A's
re-derived values exactly (`hist→target` **28.61 %**, `hist→latent` **10.02 %** vs their
28.6 / 10.0), and confirms the summary rows are untouched (`cond8→target` 7.74 → 7.74,
`cond8→latent` 5.67 → 5.67).

⚠️ **Only the raw-history rows moved, and they moved in *opposite* directions**: → raw target
**27.3 → 28.6** (+1.3) but → latent **11.6 → 10.0** (−1.6). Per agent-A, that puts Phase-1a's
latent row **exactly on its 10 % threshold** rather than comfortably above it. The Phase-1a
verdict (`FIX THE CONDITIONING`) is unchanged.

### 2c. agent-A's ruling (verbatim summary)

Accepted as primary, on three grounds: a selector that *anti-correlates* with the target
metric is not a selector; blocked+purged is **this repo's own house convention**
(`global_purged_horizon`), so using it inside train is consistent rather than novel; and both
tables are reported so nothing is hidden. The literal last-20 % is retained below as a
secondary row.

---

## 3. The MLP training loop is sound — verified, not assumed

Because §2.3 warns that a sub-27.3 % raw-history MLP means "your training loop is underfit
and nothing else you report is interpretable", I tested the loop directly rather than
guessing. A **depth-0 (linear) model through the identical loop** — same standardisation,
same optimizer, same metric, same inversion — reaches **28.36 %** on val, reproducing
ridge's 27.28 %.

So the loop is correct, and the low MLP numbers are a property of the *model class*, not a
bug. What the same sweep shows (val curve inspected directly — **diagnostic only, never used
for a reported number**), raw history → raw target:

| model | best val | at epoch | val @ epoch 120 |
|---|---|---|---|
| linear (depth 0), wd 1e-1 | **28.44 %** | 19 | 27.84 % |
| MLP depth 1, width 1024 | **28.13 %** | 3 | 5.59 % |
| MLP depth 2, width 1024 | **23.50 %** | 2 | −0.35 % |
| MLP depth 2, width 256, lr 3e-4 | 21.81 % | 8 | 2.70 % |

Two things follow. Nonlinearity buys **nothing** on the raw history (28.13 vs 28.44), and
deeper is strictly *worse*. And the MLPs overfit hard and fast — which is why **no in-train
holdout can select them well**: val is a temporally distinct regime, and the overfitting is
regime-specific, so a holdout drawn from inside train cannot see it (the blocked holdout
reads 40.2 % where val reads 20.8 %).

**This falsifies the premise §2.1 is built on** — "An MLP will beat ridge on *any*
representation" — on this cell. That premise is why the raw-history control exists; it does
not hold, and that is itself part of the answer. The full oracle sweep confirms it at higher
resolution: **28.30 % (MLP) vs 28.36 % (linear)**, i.e. **−0.06 pt**, with config *and* epoch
chosen on val.

### 3b. The probe family and its hyperparameters (deliverable §3.4)

`MLP(d_in → [width, GELU, Dropout] × depth → d_out)`, AdamW, batch 512, MSE on standardised
targets. Inputs per-dim z-scored with **fit-portion** statistics only; targets per-dim centred
(matching ridge's `ym`) and scaled by one global scalar, inverted exactly before scoring.

Shared grid, searched identically for **every** cell (32 configs), plus 4 depth-0 linear
controls in the oracle sweep:
`width ∈ {256, 1024}` · `depth ∈ {1, 2}` · `lr ∈ {1e-3, 3e-4}` · `weight_decay ∈ {1e-5, 1e-3}`
· `dropout ∈ {0.0, 0.1}` · epochs ≤ 150 with patience 25 (honest) / ≤ 50 no early stop (oracle).

Holdout-selected configs (blocked selector, the honest primary):

| cell | selected | epochs | holdout | val |
|---|---|---|---|---|
| hist → target | d2 w1024 lr1e-3 wd1e-5 drop0.1 | 3 | 40.18 | 20.78 ± 1.89 |
| hist → latent | d2 w256 lr3e-4 wd1e-3 drop0.1 | 10 | 14.50 | 9.99 ± 0.29 |
| cond8 → target | d2 w1024 lr3e-4 wd1e-3 drop0.0 | 2 | 33.85 | 6.57 ± 0.96 |
| cond8 → latent | d2 w1024 lr3e-4 wd1e-3 drop0.0 | 1 | 11.57 | 0.51 ± 0.81 |
| cond32 → target | d2 w256 lr1e-3 wd1e-3 drop0.1 | 3 | 31.78 | 13.33 ± 0.94 |
| cond32 → latent | d1 w256 lr3e-4 wd1e-3 drop0.1 | 2 | 10.53 | 3.26 ± 0.37 |

Note every selected epoch count is **1–10**: the holdout consistently stops training almost
immediately, and the holdout column bears little relation to the val column (40.18 → 20.78).
That divergence *is* the §2 finding, restated per cell.

---

## 4. Results

All numbers are **% RMSE reduction vs predict-the-mean** on **val** (n = 2 183), metric
identical to `llapdiff-uq-eval` / `ridge_reduction`. `ridge` = closed-form, alpha on a train
holdout. `MLP blocked` / `MLP last20` = **honest**, selection strictly inside train, refitted
on full train, **mean ± std over 3 init seeds**. `MLP ORACLE` / `linear ORACLE` = **leaky
upper bounds** — config *and* epoch chosen by looking at val, 36 configs × ≤50 epochs.

`ridge` is the **fixed** `ridge_reduction` (commit `ef0d679`, blocked+purged alpha selection);
`ridge pre-fix` is the last-20 % selector the §1 gate reproduced. Only the raw-history rows
differ between them.

### 4a. Raw target (the h=168 temperature trajectory) — the handoff's §2.2 target

| representation | dims | **ridge** | ridge pre-fix | MLP blocked | MLP last20 | **MLP ORACLE** | linear ORACLE |
|---|---|---|---|---|---|---|---|
| **raw history** | 128 | **28.61** | 27.28 | 20.78 ± 1.89 | 22.00 ± 1.04 | **28.30** | 28.36 |
| `cond_summary`, 8 tok | 2 048 | 7.74 | 7.74 | 6.57 ± 0.96 | 6.57 ± 0.96 | **12.23** | 2.27 |
| `cond_summary`, 32 tok | 8 192 | **14.23** | 14.23 | 13.33 ± 0.94 | 8.85 ± 0.83 | **14.15** | 0.38 |

### 4b. Latent `mu_norm` (what the denoiser must actually regress)

| representation | dims | **ridge** | ridge pre-fix | MLP blocked | MLP last20 | **MLP ORACLE** | linear ORACLE |
|---|---|---|---|---|---|---|---|
| **raw history** | 128 | **10.02** | 11.61 | 9.99 ± 0.29 | 9.56 ± 0.56 | **11.43** | 10.01 |
| `cond_summary`, 8 tok | 2 048 | 5.67 | 5.67 | 0.51 ± 0.81 | 1.63 ± 0.84 | 4.49 | −4.68 |
| `cond_summary`, 32 tok | 8 192 | 4.21 | 4.21 | 3.26 ± 0.37 | 0.76 ± 2.06 | 4.06 | −8.75 |

> **An independent validation of my loop falls out of the fixed table.** On `hist → latent`
> the fixed closed-form ridge reads **10.02** and my oracle SGD-*linear* reads **10.01** — a
> 0.01 pt agreement between two completely different solvers on the same features. On
> `hist → target` they agree to 0.25 pt (28.61 vs 28.36). So the training loop is not merely
> "not broken"; it is quantitatively equivalent to ridge for linear models, which is what
> licenses reading the MLP-vs-linear contrast as a statement about the model class.

### 4c. The oracle maxima are partly seed luck — seed-averaged for honesty

The `MLP ORACLE` column is a max over 36 configs at seed 0. Re-running each winning config
over 3 seeds:

| cell | best-config | seed-mean ± std | winning config |
|---|---|---|---|
| hist → target | 28.30 | **28.15 ± 0.35** | d1 w1024 ep8 |
| cond8 → target | 12.23 | **11.80 ± 1.35** | d2 w1024 ep8 |
| cond32 → target | 14.15 | **11.60 ± 2.62** | d1 w256 ep5 |
| hist → latent | 11.43 | 10.87 ± 0.49 | d1 w1024 ep2 |
| cond8 → latent | 4.49 | 4.67 ± 0.62 | d2 w1024 ep5 |
| cond32 → latent | 4.06 | 4.58 ± 0.53 | d2 w256 ep6 |

`cond32 → target` drops 14.15 → 11.60 ± 2.62, so its single-config maximum is substantially
seed luck. Using seed-means only *strengthens* every conclusion below.

### 4d. Does nonlinearity unlock anything? — mind the weak linear control

Within-loop (`MLP ORACLE` − `linear ORACLE`, same optimizer) the gains look dramatic on
`cond_summary` and zero on history:

| cell | MLP oracle | linear oracle (SGD) | within-loop gain |
|---|---|---|---|
| hist → target | 28.30 | 28.36 | **−0.06** |
| cond8 → target | 12.23 | 2.27 | +9.95 |
| cond32 → target | 14.15 | 0.38 | +13.77 |

🔴 **Do not quote those gains.** The SGD-trained linear model is a *bad* linear probe at high
dimension — 2.27 and 0.38 against closed-form ridge's 7.74 and 14.23 on the same features.
It fails to optimise a 2 048/8 192-dim linear map, so most of the "gain" is the MLP beating a
broken baseline, not beating linearity. (At 128 dims, where SGD-linear *does* optimise
properly, it matches ridge to 0.01–0.25 pt — see the note under §4b.) Against the **best
available linear probe, the fixed ridge**:

| cell | best linear (ridge) | MLP oracle | honest nonlinearity gain |
|---|---|---|---|
| hist → target | **28.61** | 28.30 | **−0.31** |
| cond8 → target | 7.74 | 12.23 | **+4.49** |
| cond32 → target | **14.23** | 14.15 | **−0.08** |

So nonlinearity recovers ~4.5 pt on the *impoverished* 2 048-dim view — it partly compensates
for tokens that view throws away — but **once the linear probe is given enough tokens
(8 192 dims) the MLP adds nothing**, and on the raw history it is slightly *negative*. The
ceiling over representations of `cond_summary` is ~14.2 % either way.

---

## 5. Which branch of §2.2 — **branch 2: genuine LOSS**

Read off the raw target, `M_cond` = best over token counts (mirroring `run_ridge_probe`'s
`best_b`):

| probe | M_hist | M_cond | M_cond / M_hist |
|---|---|---|---|
| **ridge (fixed, honest)** | **28.61** | **14.23** | **49.7 %** |
| ridge (pre-fix, honest) | 27.28 | 14.23 | 52.1 % |
| MLP blocked (honest, 3 seeds) | 20.78 | 13.33 | 64.1 % |
| **MLP oracle (upper bound)** | **28.30** | **14.15** | **50.0 %** |
| MLP oracle, seed-averaged | 28.15 | 11.80 | 41.9 % |

§2.2 sets "within ~20 % relative" (i.e. ≥ 80 %) for *present-and-nonlinearly-encoded*. The
oracle MLP reaches **50.0 %** — **not** ≥ 80 %, and it lands within **0.3 pt** of the fixed
ridge gap of **49.7 %**. A nonlinear probe with oracle model selection reproduces the linear
gap almost exactly. That is branch 2 as written:

> *`M_cond` stays far below `M_hist`, and the gap is similar to the ridge gap → genuine
> **loss**. B21 stands as written; fixing the summarizer's objective becomes the priority.*

**Stated plainly:** the 27.3 % → 7.7 % gap is **not** an artefact of using a linear probe.
Give the probe an oracle-selected MLP and the gap does not close — 128 raw history numbers
still beat 8 192 dimensions of summarizer output by **~2×**. The one qualification is that
nonlinearity *does* matter for *how you view* the summary (12.23 vs 7.74 at 2 048 dims), so
the earlier 7.7 % understates `cond_summary`; the correct figure for it is **~14 %**, which is
what the 8 192-dim ridge already reported.

### 🔴 What this does NOT license

1. **It does not reinstate B21's retracted headline.** "The summarizer discards ~½ the
   forecastable signal" remains wrong as stated. A probe measures a **lower bound on usable
   information**, and B21's own `SUM_FT_MODE="all"` result is the counterexample: best
   forecast of any run (0.3240 vs 0.3627) with the *worst* linear probe (→ latent 1.4 % vs
   5.7 %). Probe score and forecast quality demonstrably dissociate on this pipeline. My
   result upgrades the defensible claim from *"not linearly accessible"* to **"not accessible
   to a small MLP either"** — a tighter lower bound, still a lower bound.
2. **It says nothing about the denoiser.** The denoiser cross-attends over all 336 tokens and
   is trained end-to-end on a different objective; my probe reads a strided 8/32-token view.
3. **The latent remains a bad yardstick** (§4b), reconfirming B21: every probe scores worse on
   `mu_norm` than on the raw target, and the oracle MLP does not rescue it (4.49 / 4.06 vs
   ridge 5.67 / 4.21). An intervention that improves the forecast can lower this number.

---

## 6. Supplementary — what the summarizer's objective actually supervises

Not part of the assigned task; a structural check that bears on B21's open question
"whether the loss is in the summarizer's **architecture** or its **objective**". CPU only,
read-only import, no model trained.

`LaplaceAE.forward` returns `context` of shape `[B, S=336, Hc=256]` — **86 016 numbers per
window**, and that is what the denoiser consumes. But every pretraining reconstruction head
reads `ctx_mean = context.mean(dim=1)` (`summarizer.py:637`) — **256 numbers**.

Verified by perturbing `context` with a delta of norm **38.35** constructed to have exactly
zero token-mean, and measuring every reconstruction head:

| head | max change |
|---|---|
| `x_hat` | 2.98e-08 |
| `v_hat` / `t_hat` / `dt_hat` | 8.94e-08 |
| `obs_hat` | 1.04e-07 |

i.e. **float32 noise**. So the summarizer's pretraining loss can only constrain the
**token-mean** of its output; the other **335 of 336 token-directions lie in its null
space** — nothing in stage-2 training shapes them for any purpose.

And `normalize_cond_per_batch(mode="sample")` — the default — reduces over `dims=(1,)`, the
same token axis, so it subtracts exactly that mean (verified: max |token-mean of
`cond_summary`| = 1.2e-06).

⇒ **The one component of the summarizer's output that its objective ever supervised is
exactly the component the default normalisation removes before the denoiser sees it.**

This predicts the strangest row in B21's own table — the erased mean alone (256 dims) scores
**12.4 %**, beating the entire kept 2 048-dim normalised summary at 7.7 %. It is a statement
about which directions the *objective* can constrain, **not** a claim that the remaining
directions are empty: they are a deterministic function of the history and the strided view
scores 14.2 %, so they demonstrably carry signal. What they lack is any training pressure.

---

## 8. Follow-up — `COND_NORM_MODE="global"`: substitutes or additive?

Requested by agent-A (comms 2026-08-02 06:40). **Prediction registered before the numbers
existed**, as they asked.

### 8.1 The coincidence being tested

| cell | value |
|---|---|
| `sample` @2048, ridge | 7.74 % |
| `sample` @2048, MLP oracle | **12.23 %** |
| `global` @2048, ridge (agent-A, pre-fix) | **12.3 %** |
| `sample` @8192, ridge | 14.23 % |

Nonlinearity on the impoverished view and level-restoration on that same view land within
**0.07 pt**. Are they two routes to the same missing information (**substitutes**) or two
independent deficits (**additive**)?

- **Substitutes** ⇒ MLP @2048 under `"global"` ≈ 12.3 %, no gain over ridge.
- **Additive** ⇒ it climbs toward ≈ 16.4 % (`"global"` @8192 ridge).

### 8.2 🔮 My registered prediction: **ADDITIVE**

The mechanisms are provably different, so the coincidence should be a coincidence.
`"sample"` z-scores over the token axis, which **projects the per-window level out
exactly** — agent-A measured the across-window std of that level at precisely 0.00000. No
function of the normalised tokens, linear or nonlinear, can recover it. Therefore my +4.49 pt
at 2 048 dims **cannot** be level recovery; the only thing left for it to be is inference of
strided-away token content from the kept tokens. Restoring the level and de-striding the view
address disjoint deficits, so they should add: **MLP @2048 under `"global"` ≈ 16–17 %**.

Second prediction: **@8192 under `"global"`, nonlinearity buys ≈ 0**, as it did under
`"sample"` (−0.08 pt). If that holds, the clean statement is *nonlinearity buys nothing on the
rich view under either normalisation*, and every gain I have measured is a view/level artefact
rather than hidden structure.

### 8.3 Protocol — an exact A/B

Raw summaries collected **once** and both normalisations applied offline, so windows, raw
history, latent targets, raw targets, VAE and summarizer are **bit-identical** and only the
normalisation differs (verified: `hist`, `latent`, `target`, `cond8`, `cond32` all
`array_equal` against the §4 cache, both splits). The `"global"` statistics are the ones the
`globalnorm_e60` run actually trained under, read from its checkpoint (mean |·| 0.0896, std
0.5190). That payload has **no `summarizer` key**, i.e. `SUM_FT_MODE="none"`, so its encoder
*is* the shared frozen artifact my `"sample"` cells used — which is what makes reusing my own
stack legitimate.

Per agent-A, their 12.3 / 16.4 were measured with the **pre-fix** selector, so I re-derive all
`"global"` ridge baselines with the fixed `ridge_reduction(purge=336)`.

### 8.4 Results — **SUBSTITUTES. My registered prediction was wrong.**

`ridge` = fixed selector (`purge=336`). `honest MLP` = blocked holdout, refit on full train,
3 seeds. `oracle MLP` / `oracle linear` = leaky upper bounds (config **and** epoch chosen on
val). All val, n = 2 183.

#### Raw target

| view | norm | ridge | honest MLP | **oracle MLP** | oracle linear | nonlinearity (oracle − ridge) |
|---|---|---|---|---|---|---|
| 2 048 | `sample` | 7.74 | 6.57 ± 0.96 | **12.23** | 2.27 | **+4.49** |
| 2 048 | `global` | **12.28** | 8.43 ± 0.41 | 10.37 | 8.67 | **−1.91** |
| 8 192 | `sample` | **14.23** | 13.33 ± 0.94 | 14.15 | 0.38 | **−0.08** |
| 8 192 | `global` | 13.69 | 4.75 ± 0.49 | 11.24 | 10.22 | **−2.45** |

#### Latent `mu_norm`

| view | norm | ridge | honest MLP | oracle MLP | oracle linear |
|---|---|---|---|---|---|
| 2 048 | `sample` | 5.67 | 0.51 ± 0.81 | 4.49 | −4.68 |
| 2 048 | `global` | 4.90 | −0.26 ± 0.66 | 3.12 | −0.04 |
| 8 192 | `sample` | 4.21 | 3.26 ± 0.37 | 4.06 | −8.75 |
| 8 192 | `global` | **6.85** | 3.55 ± 0.46 | 3.75 | −2.14 |

#### 8.4a Prediction 1 — **falsified**

I predicted **additive**, ≈ 16–17 %. Measured: oracle MLP @2048 `global` = **10.37 %**
(seed-mean 9.76 ± 0.86), against ridge @2048 `global` = **12.28 %**. That is **no gain over
ridge** — agent-A's *substitutes* branch. Recorded as a failed prediction, not reinterpreted.

The argument I registered still holds as stated (`"sample"` projects the per-window level out
exactly, so my +4.49 pt under `"sample"` cannot be level recovery). What it failed to
anticipate is that both routes **cap at the same ≈ 12.3 % ceiling** at 2 048 dims. I have no
verified mechanism for that — see §7.0, including the alternative I did not test.

#### 8.4b Prediction 2 — **confirmed**

Nonlinearity on the rich 8 192-dim view: **−0.08 pt** under `"sample"`, **−2.45 pt** under
`"global"`. So **nonlinearity buys nothing on the rich view under either normalisation**, and
every gain I have measured anywhere is a *view* artefact, not hidden structure.

#### 8.4c 🔴 A B21 number needs correcting: `global` @8192 is **13.69 %, not 16.4 %**

agent-A flagged their `"global"` values as pre-fix. Re-derived with the fixed selector, the
`"sample"` rows are unchanged but one `"global"` row moves materially:

| cell | pre-fix | fixed | Δ |
|---|---|---|---|
| `global` @2048 → raw target | 12.3 | 12.28 | −0.02 |
| `global` @2048 → latent | 4.9 | 4.90 | +0.00 |
| **`global` @8192 → raw target** | **16.4** | **13.69** | **−2.71** |
| `global` @8192 → latent | 6.8 | 6.85 | +0.05 |

This **reverses a B21 claim at the best view**. `"global"` improves the raw-target readout on
the impoverished view (7.74 → 12.28, **+4.54**) but *not* on the rich one (14.23 → 13.69,
**−0.54**). Over the best available view, `"global"` does **not** improve the linear
raw-target readout at all.

⚠️ Note the asymmetry, which does not cancel: on the **latent** at 8 192, `"global"` genuinely
*helps* (4.21 → **6.85**, +2.64). So `"global"` moves the two targets in opposite directions
at the rich view. B21's existing warning that the latent is a poor yardstick applies here in
its sharpest form yet.

#### 8.4d The unifying statement

Across the full 4 (view × normalisation) × 2 (probe class) grid on the raw target, **the best
number anywhere is 14.23 % — plain ridge on the `"sample"` 8 192-dim view.** Neither level
restoration, nor nonlinearity, nor both together beats it. Both interventions are
compensating for an impoverished *view* of `cond_summary`; neither raises its ceiling.

⇒ §5's verdict is unchanged and now rests on a wider search: `M_cond` = **14.23** against
`M_hist` = **28.61** (ridge) / **28.30** (oracle MLP) — **49.7 %**, against the ≥ 80 % that
"nonlinearly encoded" would need.

#### 8.4e One tension worth agent-A's attention

If nonlinearity and level-restoration are *substitutes* for a probe, yet `"global"` still
improved the **denoiser's** val CRPS, then the denoiser — itself nonlinear, and reading all
336 tokens rather than my strided 8/32 — is **not** operating at the probe ceiling. Two
different objects are being measured, so this is a hypothesis rather than a result; but it
points the same way as the campaign's open "denoiser barely uses its conditioning" issue.

> **Updated 2026-08-02 07:10 — the premise is weaker than when I wrote it.** I cited
> 0.3627 → 0.3344 (one seed). agent-A's paired multi-seed run now reads **−0.028 (seed 0)** and
> **−0.003 (seed 2)**, with seed 1 voided by an outcome-independent validity criterion fixed
> before unblinding (the control's train loss went 0.861 → 0.866 over 50 epochs, i.e. no
> progress, while the other five runs improved 0.09–0.20; both seed-1 arms are re-running).
> **Both valid pairs sit below the 0.036 nondeterminism band**, so the CRPS benefit of
> `"global"` is not established either. The tension above therefore *weakens*: my probe result
> and agent-A's CRPS result converge on "`"global"` is a smaller effect than it first looked",
> from opposite directions. Do not read §8.4e as evidence that the denoiser beats the probe
> ceiling — as of now, neither side of that comparison is solid.

---

## 9. Phase 1a on bms_air h=168 — the test is **void**, not passed or failed

Requested by agent-A (comms 2026-08-03 08:55) to see whether B21 generalises beyond noaa_uk.
Inference only; no training, no diffusion cache, no checkpoint. cuda:2. Per their instruction
I report bms_air's rows and my own verdict and leave the joint read with noaa_uk to them.

### 9.1 Pre-flight — the brief's structural premise did not hold

`run_ridge_probe._collect` scores the **modal** entity count only. The brief described bms_air
as "12 entities/window". Measured (model-free replica of `_collect`'s filters, one loader pass):

| cell | seen | eligible | modal | at modal |
|---|---|---|---|---|
| noaa_uk train | 16 286 | 16 286 (100 %) | **4** | 14 446 (88.7 %) |
| noaa_uk val | 2 183 | 2 183 (100 %) | **4** | 2 183 (100 %) |
| bms_air train | 95 099 | 57 469 (60.4 %) | **1** | 30 359 (52.8 %) |
| bms_air val | 17 598 | 9 442 (53.7 %) | **1** | 5 855 (62.0 %) |

bms_air's mode is **1**, and a matched full-panel run is infeasible — **23** twelve-entity
windows in val against noaa_uk's 2 183. So I ran two cells: the modal n=1, and n=8 as the
best-powered multi-entity cell with a scoreable val.

### 9.2 Results — nothing is forecastable, at either entity count

% RMSE reduction vs predict-the-mean, val, honest blocked+purged alpha:

**agent-A ruled (comms 09:20) that n=1 carries the read**, because the A ≫ B contrast is
*within-cell* — same windows, same targets, only the input representation changes — so it does
not need noaa_uk's geometry to answer "does the summarizer lose the signal on this dataset
too?". n=8 is the secondary row.

| cell | n train/val | dims | A raw history | B cond (best) | C latent (best) | tool verdict |
|---|---|---|---|---|---|---|
| **n=1 (modal, primary)** | 30 359 / 5 855 | 32 | **−0.3 %** | 0.6 % | 3.7 % | MOVE DATASET |
| n=8 (secondary) | 2 596 / 747 | 256 | **−4.6 %** | −6.3 % ⚠ | 1.2 % | MOVE DATASET |

Per-view detail at n=1: cond 2 048 → 0.6 % raw / 3.7 % latent; cond 8 192 → 0.2 % / 3.4 %.
At n=8: cond 2 048 → −6.3 % / 1.2 %; cond 8 192 → −3.5 % / 1.0 %.

⚠ **The n=8 row now quotes the 2 048-dim view, per agent-A's instruction.** At n=8 the
8 192-dim view is **under-determined** — 2 596 training rows against 8 192 features — so its
−3.5 % is regularisation-dominated and must not be read as the better result. n=1 has no such
problem (30 359 rows).

### 9.3 🔴 My own hypothesis, falsified

I predicted the n=1 cell would be depressed because those are the windows where 11 of 12
stations are unobserved — the sparsest subset. **Wrong: n=8 is worse, not better.** Recording
it as a failed prediction.

### 9.4 The negative values are NOT a level-shift artefact — ruled out three ways

I checked this specifically because §2 of this report found exactly that failure mode on this
codebase. All three say no:

| check | bms_air n=8 | reference |
|---|---|---|
| train→val shift | +0.033 = **0.03 val-std** | noaa_uk shift 0.14 val-std, and it scored +27.3 % |
| reduction vs **train**-mean baseline (removes any val-baseline advantage) | **−0.15 %** | — |
| **level-free** corr(pred, target) on val | **r = −0.025** (r² = 0.0006) | — |

The oracle-alpha fit also selects **α = 1e8**, the largest value in the grid — ridge shrinks
the solution to nothing because there is no useful direction. Honest −4.62 % vs oracle −0.19 %
is alpha selection alone; both round to "no signal".

### 9.5 Why — and it is physical, not a pipeline defect

| cell | target | horizon |
|---|---|---|
| noaa_uk | **temperature** | 168 h |
| bms_air | **PM2.5** | 168 h |

Seven-day-ahead temperature carries strong seasonal/diurnal structure. Seven-day-ahead PM2.5
is driven by meteorology, emissions and transport and has little history-decodable structure
at that range. Nothing here indicts stage 1, stage 2 or the conditioning.

### 9.6 🔴 What this does and does not tell us about B21

**The test agent-A designed is void, not answered.** Their falsification criterion was "if
bms_air comes back with B ≈ A, that falsifies the mechanism story". Observed **B ≈ A** — but
**vacuously**, because A ≈ 0. You cannot measure whether stage 2 discards forecastable signal
on a cell that has no forecastable signal to discard. The §6 mechanism is neither supported
nor falsified by this run.

This is B21's own lesson recurring: *"1a as written conflates dataset quality with summarizer
quality."* Here the dataset end of that conflation is binding.

**agent-A pre-registered the asymmetry, and it lands on the ambiguous side.** Before seeing the
numbers they noted (comms 09:20) that at n=1 the summarizer's entity aggregation
(`encoded_mean = (encoded_bn * entity_weight).sum(dim=1) / entity_denom`) is a **no-op** — there
is nothing to pool — so entity pooling is excluded *by construction* as a rival explanation.
A **positive** A ≫ B there would therefore have been *stronger* evidence for §6 than the
noaa_uk result, where pooling was merely cheap (27.3 → 25.3) rather than structurally absent.
But they also pre-registered that **a null is ambiguous** — and a null is what we got. So the
"void" reading above is their own stated caveat, not a post-hoc excuse.

### 9.7 Two consequences for the plan

1. **`CMD_UQ_RECOVERY_PLAN.md` Phase 2 designates bms_air h=168 as the cross-check dataset.**
   By 1a's own decision table this cell fails the **A** branch — "the only branch that
   justifies changing datasets" — so it cannot serve that role at this horizon. Worth knowing
   before committing training there, which is exactly why the gate exists.
2. **Gate 0 and 1a's A gate are independent, and bms_air proves it.** bms_air has the
   **highest Gate 0 of all five cells (0.941)** and no forecastable signal whatsoever. Passing
   Gate 0 does not license using a cell; it only says stage 1 can reconstruct what it is given.

**Concrete cheap next step:** bms_air supports horizons 24/48/96/168. PM2.5 at 24 h is a far
more plausible target, and 1a costs minutes. If a cross-check cell is wanted, test the shorter
horizons before abandoning the dataset.

### 9.9 Horizon sweep — bms_air is not forecastable at **any** supported horizon

The §9.7 recommendation (try a shorter horizon) has now been tested, at the human's direction.
**A only** — B and C would need stage-1/2 artifacts that exist for h=168 only, and building them
would mean training into the shared `ldt/vae/**` and `ldt/summarizer/**` surface. A needs no
model at all, and A is the gate that fails, so the question is answerable without that.

Well-powered cells only (alpha-fit set ≥ 20 % of train — see §9.10):

| horizon | best A | at | full panel n=12 | other well-powered cells |
|---|---|---|---|---|
| **24** | **+1.8 %** | n=1 (8 874/2 034) | **−2.8 %** (13 514/1 372) | n=11 +1.7 % |
| 48 | +0.5 % | n=1 (15 185/3 528) | −2.8 % (9 979/544) | n=10 −0.3 %, n=11 −6.3 % |
| 96 | 0.0 % | n=1 (24 084/5 099) | — (146 val, thin) | n=10 −1.2 %, n=11 −1.8 % |
| 168 | −0.3 % | n=1 (30 359/5 855) | n/a (23 val) | — |

**Best value anywhere across four horizons and every entity count: +1.8 %, against a 10 % gate.**
Monotone decline with horizon (1.8 → 0.5 → 0.0 → −0.3), which is the physically expected shape
for PM2.5 and further evidence this is a property of the target rather than of the pipeline.

**A side benefit: the h=168 mode-1 anomaly is now explained.** At h=24 the modal entity count is
**12** — the full panel — not 1. The fully-observed-target filter over 24 steps is far less
restrictive than over 168. So bms_air's mode-1 at h=168 was an artefact of requiring 168
consecutive fully-observed steps across all stations, not a standing property of the dataset.
That also means the **matched full-panel cell that was infeasible at h=168 (23 val windows) is
well-powered at h=24 (13 514/1 372)** — and it reads **−2.8 %**. So the full-panel measurement
that §9.1 could not obtain now exists, and it agrees.

⇒ bms_air cannot serve as the Phase-2 cross-check at **any** of its horizons. This is settled,
not a "try the next horizon" situation.

### 9.10 🔴 A defect in `ridge_reduction` — introduced by my own §2 proposal

Found by a crash while running h=96. **`TypeError: unsupported operand type(s) for +: 'float'
and 'NoneType'`** at `W = V @ (RV / (lam + best[0])[:, None])`.

**Cause.** `blocked_purged_split(n, n_blocks=15, holdout=(3,8,13), purge=WINDOW)` bans up to
`2 × 3 × purge` windows around the holdout blocks. At `purge=336` that is ~2 016 windows, so on
a small train set the **alpha-selection fit set collapses** — to exactly **zero** at
n_train ≈ 1 420 (bms_air h=96, n=7). `best` then stays `(None, inf)` and the failure surfaces as
a TypeError deep inside rather than as a clear error.

**This is a regression from the fix I proposed in §2.** The original `cut = max(1, int(0.8*n))`
always left ≥ 80 % of train for fitting; blocked+purged has no such floor. I proposed it without
a small-n guard and agent-A adopted it, so this is mine.

**Severity is bounded — the final model is unaffected.** `ridge_reduction` fits the reported
model on the **full** train set (`fit(Xtr, ytr)`); only **alpha** is chosen on the split. So a
thin fit degrades alpha selection, not the fit. Measured fit fractions:

| cell | n_train | alpha-fit n | fit % | status |
|---|---|---|---|---|
| noaa_uk h168 n=4 | 14 446 | 9 541 | 66 % | ok — **all §1–§8 results unaffected** |
| bms_air h168 n=1 | 30 359 | 22 271 | 73 % | ok — the primary cell |
| bms_air h24 n=1 / n=12 | 8 874 / 13 514 | 5 082 / 8 795 | 57 / 65 % | ok |
| bms_air h48 n=1 / n=12 | 15 185 / 9 979 | 10 132 / 5 967 | 67 / 60 % | ok |
| **bms_air h168 n=8** | 2 596 | **223** | **9 %** | ⚠ my §9.2 secondary row |
| bms_air h24 n=10 | 2 477 | 159 | 6 % | ⚠ |
| bms_air h48 n=9 | 1 888 | 41 | 2 % | ⚠ |
| bms_air h96 n=7 | 1 420 | **0** | 0 % | 🔴 crashes |

**Every cell carrying a conclusion is sound** (≥ 57 % fit). The affected rows are all secondary
and all already ≈ 0, and a mis-chosen alpha can only depress a reduction, so no verdict moves.
**Retracted as unreliable: the n=8 h=168 row (−4.6 %), h24 n=10 (+1.3 %), h48 n=9 (−1.2 %).**

Rule of thumb for the tool: with `purge=WINDOW`, `ridge_reduction` needs roughly
**n_train ≳ 10 × purge** for a healthy alpha-selection fit. A guard belongs in the repo; I did
not add one, because agent-A owns that file.

### 9.8 Not established here

- **Shorter bms_air horizons untested** (24/48/96) — the recommendation above is untested.
- **Linear probes only.** §4–§8 found nonlinearity adds ~nothing on noaa_uk's rich view, but
  that was noaa_uk; I did not run an MLP here.
- **n=8 is under-powered** — 2 596 train windows against noaa_uk's 14 446 — so its −4.6 % is
  the weaker of the two rows. The n=1 row (30 359) is the better-powered one and agrees.
- **One seed of the frozen stack**, one target column (`PM2.5`, the cache default).
- I did **not** reconcile these with noaa_uk; that joint read is agent-A's.

---

## 10. Phase 1a on noaa_us h=168 — **B21 replicates on a second sensor network, and harder**

Requested by agent-A after §9 removed bms_air as the Phase-2 cross-check cell. Same target
variable as noaa_uk (**temperature**), same context/horizon geometry, genuinely different
sensor network (40 stations vs 4). Inference only, cuda:2, nothing written.

### 10.1 Registered predictions (see §10.3 for the honesty caveat on A)

1. **A ≥ 20 %** — the target is temperature at h=168, the physics that made noaa_uk
   forecastable. This was the direct test of §9's "bms_air fails on its *target*, not the
   pipeline" claim; a near-zero A here would have falsified it.
2. **B < 0.7·A** — the conditioning gap reproduces. §6's mechanism is *architectural*, so it
   should not care which sensor network produced the data. **B ≥ 0.7·A would have been real
   evidence against §6** — and unlike bms_air this cell can deliver that, because A ≠ 0.
3. Secondary: **C < 10 %**.

### 10.2 Results

val, **19 832 train / 3 289 val** at 40 entities, `cond_norm_mode="sample"`, no checkpoint,
honest blocked+purged alpha on a **healthy 70 % fit set (13 850)** — the §9.10 small-n defect
does not touch this cell.

| probe | dims | → raw target | → latent |
|---|---|---|---|
| **A raw history** | 1 280 | **23.4 %** | 8.3 % |
| B/C `cond_summary`, 8 tok | 2 048 | 0.3 % | 4.4 % |
| B/C `cond_summary`, 32 tok | 8 192 | **2.6 %** | 4.4 % |

**A = 23.4 % · B = 2.6 % · C = 4.4 %** → tool verdict **`FIX THE CONDITIONING (B21)`**.

### 10.3 Prediction outcomes

- **(2) B < 0.7·A — HELD, with room to spare.** Threshold 16.4 %, observed **2.6 %**.
- **(3) C < 10 % — held** (4.4 %).
- **(1) A ≥ 20 % — observed 23.4 %, but DISCOUNT IT.** I formed it before running but recorded
  it in the comms channel only *after* A had come back. agent-A has no way to verify the
  ordering and should not have to take my word for it. Only (2) and (3) were posted while still
  unobserved.

**Within-cell ratio B/A = 11.1 %** — the summarizer delivers about a ninth of the linearly
available signal here. This is the informative case §9 could not provide: A is solidly non-zero,
so a null in B would have been genuine evidence against §6. It is not a null. **§6 gets a
positive replication on an independent network.**

### 10.4 A candidate mechanism — flagged, not concluded

Per agent-A's rule I do not reconcile this with noaa_uk; that joint read is theirs. One
ingredient worth having in front of them:

**noaa_us pools 40 entities into one latent; noaa_uk pools 4.** B20 measured entity pooling as
nearly free on noaa_uk *specifically because* its 4 stations share 85 % of their variance, and
flagged that as cell-specific. noaa_us is the first measured cell where that caveat could bite
hard.

⚠️ Two reasons not to over-read it even if the ratios line up: §6's mechanism (335/336 token
directions unsupervised) is entity-count-**independent**, so pooling would be an *additional*
loss rather than the same one; and two points is a suggestion, not a trend. crypto (113) and
us_equity (221) remain the untested extremes.

### 10.5 What this settles

- **noaa_us h=168 passes 1a's dataset gate** (A = 23.4 %) and can serve as Phase 2's
  cross-check cell.
- **It confirms §9 from a second direction.** temperature 23.4 % vs PM2.5 −0.3 % — same tool,
  metric, horizon, context length and probe. bms_air's failure is its target.
- **Geometry is clean**, unlike bms_air: noaa_us's modal entity count **is** the full 40-station
  panel, and it is the only scoreable count (every other has 0 val windows), so no `--entities`
  override was needed and the modal cell is representative.

**Not settled:** anything nonlinear on this cell; `COND_NORM_MODE="global"` here; a second seed
of the frozen stack; other noaa_us horizons (agent-A asked for the horizon question to be
carried over — untested, and less pressing now that h=168 is healthy).

---

## 11. `"global"` on noaa_us, and the mediator on four cells

Two falsification tests requested by agent-A (comms 2026-08-03 11:10) after their joint read
made entity pooling the dominant loss term on noaa_us (−17.4 of 20.8 pt).

### 11.1 `"global"` on noaa_us — the prediction fails on the measure I would defend

agent-A registered: *`"global"` lifts B far less on noaa_us than on noaa_uk (+4.5 there),
because it addresses only the §6 objective defect and does nothing about pooling; a large lift
would be evidence the decomposition is wrong.*

Setup verified: `[cond-norm] global stats over **8 072 400** rows` (noaa_us's 24 025 × 336),
mean |·| 0.1779 / std 0.3215 — not noaa_uk's 5 472 096 / 0.0896 / 0.5190. **A identical across
arms (23.3968 % both)**, the control that confirms only the conditioning path differs.

| view | sample B | global B | lift | sample C | global C | lift |
|---|---|---|---|---|---|---|
| 2 048 | 0.29 | 3.29 | **+3.00** | 4.41 | 4.95 | +0.54 |
| 8 192 | 2.58 | 4.57 | **+1.99** | 4.43 | 5.49 | +1.06 |

| comparison | noaa_uk | noaa_us | outcome |
|---|---|---|---|
| 2 048-dim row (the row named) | +4.54 | +3.00 | weakly held — less, not *far* less |
| **best-over-views ceiling (§8.4d)** | **−0.54** | **+1.99** | **FAILS — helps noaa_us, hurts noaa_uk** |

**I would defend the ceiling measure**, because §8.4d established it: the 2 048 view is
impoverished, and level-restoration partly compensates for the poor view rather than adding
information. On that measure the prediction is contradicted, not merely weakened.

**A hypothesis for why — untested, recorded as a lead not a finding.** The prediction assumed the
two losses are independent. They may not be, in the direction that matters: **pooling destroys
each entity's idiosyncratic component and preserves the panel-mean component, and the window
level *is* a panel-mean quantity** — so it is exactly what survives pooling, and `"sample"` then
deletes it. On a heavily-pooled cell the level is a larger share of what remains, so restoring it
should help *more*. That also explains the qualitative split: on noaa_uk `"global"` lifts only
the poor view (the rich view can already recover the level from more tokens), while on noaa_us it
lifts **both**, because there it cannot.

### 11.2 The mediator on four cells — with a panel-size correction

🔴 **Reproduction check: noaa_uk matches agent-A, noaa_us does not.** With-self, train split:
noaa_uk **87.9 %** (theirs 88 % ✓), noaa_us **49.0 %** (theirs 42 %, **+7 pt**). Drop rates also
differ (theirs 11.4 %, mine 1.7 %), so the two scripts filter or sample differently. **The
cross-script comparison is void; use one column or the other.** Mine are internally consistent.

| cell | entities | with-self | **LOO** |
|---|---|---|---|
| noaa_uk | 4 | 87.9 % | **76.5 %** |
| noaa_us | 40 | 49.0 % | **46.8 %** |
| crypto | 113 | 43.2 % | **42.4 %** |
| us_equity | 221 | 34.7 % | **34.2 %** |

**Why leave-one-out.** If the panel mean includes the entity itself, a 4-station panel carries a
1/4 self-contribution against a 40-station panel's 1/40 — inflating small panels arithmetically,
independent of real signal sharing. Confirmed and asymmetric as predicted: noaa_uk loses
**11.4 pt** to LOO, noaa_us **2.2**. The noaa_uk–noaa_us gap survives (**+29.6** vs +38.9 pt), so
the mechanism stands — but **~24 % of the apparent gap was panel-size arithmetic**.

**The mediator saturates**: 76.5 → 46.8 → 42.4 → 34.2, with nearly all the fall between 4 and 40
entities. So "more entities is worse" is more precisely "shared variance is what matters, and it
saturates" — and crypto's pooling cost should resemble noaa_us's rather than being far worse.

Per instruction: **no full 1a and no pooling-cost number on crypto/us_equity** (1 225 / 1 099
train, 90 / 72 val — not scoreable). Descriptive statistic only; non-finite dropped and counted.

---

## 7. What I could **not** establish

*(Kept as the closing section, so it covers §4–§5 **and** the §8 follow-up. Numbering left
unchanged because agent-A's comms entries already cite "§7".)* Matching the standard
agent-A's `CMD_BUG_REPORT.md` entries set.

0. **The §8 mechanism.** I registered "additive", the data said "substitutes", and I do
   **not** have a verified account of *why* the two routes cap at the same place. The
   argument I registered — that `"sample"` projects the level out exactly, so the MLP's
   `"sample"` gain cannot be level recovery — survives as stated; what it failed to predict
   is the shared ceiling. Treat "they compensate for the same view deficiency" as the
   description of the data, not as a demonstrated mechanism. In particular I did not test
   the obvious alternative: that the 8-token stride and the level are both proxies for the
   same slow seasonal component, which would make them redundant for a trivial reason.

1. **Nothing about what the *denoiser* can use.** A probe is a **lower bound** on usable
   information, and B21's own top box is the proof: `SUM_FT_MODE="all"` produced the best
   forecast of any run (val CRPS 0.3240 vs 0.3627) while making the conditioning *less*
   linearly decodable (→ latent 1.4 % vs 5.7 %). Probe score and forecast quality provably
   move in opposite directions on this pipeline. My result tightens the lower bound from
   "not *linearly* accessible" to "not accessible to *this probe family*". It does **not**
   license "the information is gone", and it does **not** reinstate B21's retracted
   "discards ~½ the forecastable signal" headline.
2. **One probe family.** A small MLP on a **strided** view (8/32 of 336 tokens). The
   denoiser is far larger and cross-attends over **all 336 tokens**; a sequence model over
   the full token axis is the obvious next probe and was not run. My oracle bound is an
   upper bound *within the searched family only* — 36 configs, ≤50 epochs, fixed
   preprocessing, one optimizer.
3. **One cell, one frozen stack, one conditioning mode.** noaa_uk h=168 only, with the
   single shared VAE/summarizer pair the preset points at. Not replicated on bms_air h=168,
   not on a second stage-1/2 seed.
4. ~~`COND_NORM_MODE="global"` not tested.~~ **Now tested — see §8.** What remains open there:
   the result is one cell and one checkpoint's statistics, the honest MLP rows are still
   depressed by the same selection problem as §4, and I did not test `"global"` at any view
   between 2 048 and 8 192 dims, so "helps the poor view, not the rich one" is established at
   two points rather than as a curve.
5. **The blocked-holdout geometry was not varied.** Blocks 3/8/13 of 15 with a 336-window
   purge; I did not check sensitivity to block count, block positions, or purge width.
6. **The oracle bound's tightness is unknown.** It is leaky by construction (config *and*
   epoch chosen on val) but still bounded by the family in (2), so it is neither an honest
   estimate nor a true ceiling.
7. **§6 is structural, not causal.** I showed the pretraining loss cannot constrain 335 of
   336 token-directions. I did **not** train a summarizer with a modified objective, so
   "supervising more token-directions would close the gap" is untested — it is a hypothesis
   the observation makes available, not a result.
8. **The test split was never touched**, by design, at any point.
