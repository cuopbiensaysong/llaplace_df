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
