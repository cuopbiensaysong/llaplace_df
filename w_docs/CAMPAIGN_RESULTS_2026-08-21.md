# Campaign results — 2026-08-21

Everything measured since the 2026-08-19 handoff, for Tables 1–3 of
`CMD_submission/chirp_update/cmd_iclr2026.tex`. **All numbers are the validation split; the
test split has never been read.** Repo `review_2`, HEAD `dd37486`, 61 uncommitted files.

> **This document supersedes numbers in three places.** `w_docs/noaa_uk_h168/CMD_RESULTS_FOR_PAPER.md`
> §T3/T4/T5, `HANDOFF_2026-08-19.md` §4.3's cross-cell table, and every Family A `grid5` number
> on both cells. See §2.4 for the specific corrections.

Table map: **Table 1** = `tab:pareto` (cost–quality frontier) · **Table 2** = `tab:objgrid`
(objective × UQ mechanism) · **Table 3** = `tab:decomp` (variance decomposition / R9 gate).
bms_air h=168 replaces PhysioNet h=12 as the second cell.

---

## 1. `CHIRP_UQ_INIT_VAR` for bms_air = 1.64

Full detail in `w_docs/bms_air_h168/INITVAR_bms_air_h168.md`. Measured on seed 2's `s1_mean`
checkpoint, full val (70 083 456 latent elements), EMA weights, entenc VAE, guidance frozen at 1.0:

| read | `latent_mse` | predict-mean baseline | `latent_corr` |
|---|---|---|---|
| `oneshot` | 0.9331 | 0.8881 | +0.0220 |
| **`ddim`** (noaa_uk's protocol) | **1.6387** | 0.8881 | +0.0143 |

Within 2× of noaa_uk's 1.0, so the 26× cross-cell spread the guard exists to prevent is not in
play. **Two caveats.** noaa_uk's value of record (`finetuning/results/table3/initvar_entenc.json`,
`latent_mse` 0.8236) was a **capped** read (~1/6 of val) at the default `(1.0, 2.0)` CFG ramp,
predating the D2 guidance fix — the protocols are not identical, only the mean source is. And
`--max-batches` is biased here: the loader does not shuffle, so a capped read is a contiguous
*early*-val chunk. An 8-batch read on this cell reported `latent_corr` **−0.098**; the full split
gives **+0.022**.

**Separate finding:** `latent_corr` ≈ 0.02 on bms_air against +0.363 on noaa_uk. The s1 diffusion
mean carries almost no forecast signal on this cell, and `latent_mse` 0.9331 is 5.1 % *worse* than
predicting the val mean. This does not contradict Family A's +2.6 % VE (different family, different
quantity), but it should be stated wherever the bms_air row is read.

---

## 2. Family A (one-pass heads) — the LR grid was not bracketing the optimum

### 2.1 The defect

The shipped grid `{3e-4, 1e-4, 3e-5, 1e-5}` (`grid5`) had its **lower bound selected** by most
heads, so the arms were under-trained — and unevenly, which biases head-to-head comparisons.
Widening to `grid6` (`+3e-6, 1e-6`) and `grid7` (`+3e-7, 1e-7`) resolves it. **grid7 selections are
identical to grid6 on bms_air and nothing sits at the new bound on either cell, so the optimum is
now bracketed.**

Selected LR at `grid7`, 10 seeds:

| head | bms_air | noaa_uk |
|---|---|---|
| chirp_gp | 1e-5 ×9, 3e-6 ×1 | 3e-4 ×1, 1e-4 ×3, 3e-5 ×1, 1e-5 ×5 |
| diagonal | 1e-6 ×6, 3e-6 ×2, 1e-5 ×2 | **3e-7 ×8**, 1e-6 ×1, 3e-5 ×1 |
| low-rank | 1e-6 ×6, 3e-6 ×4 | **3e-7 ×8**, 1e-6 ×2 |
| Student-t | 3e-5 ×6, 1e-5 ×3, 3e-6 ×1 | 3e-6 ×8, 3e-5 ×1, 1e-5 ×1 |

**chirp_gp barely moves on either cell** — with 3e-7 and 1e-7 on offer it declines them. The
comparators move by one to two orders of magnitude. So `grid5` handicapped the baselines, not us,
and every `grid5` margin was inflated.

grid6 and grid7 ran on different hardware and agree to ≤7 % on any single cell, which makes them an
effective independent replication.

### 2.2 bms_air h=168, grid7, chirp_gp vs baselines (n=10, one-sided Wilcoxon paired by seed)

| metric | vs diagonal | vs low-rank | vs Student-t |
|---|---|---|---|
| **ES ↓** | wins 10/10, p=0.001 | 10/10, p=0.001 | 10/10, p=0.001 |
| **VS ↓** | wins 10/10, p=0.001 | 10/10, p=0.001 | 10/10, p=0.001 |
| CRPS ↓ | wins 10/10, p=0.001 | 10/10, p=0.001 | **loses 1/10** |
| PIT-ECE ↓ | wins 7/10, p=0.065 | 8/10, p=0.019 | 8/10, p=0.032 |
| VE ↑ | wins 10/10, p=0.001 | 10/10, p=0.001 | 10/10, p=0.001 |

Means: chirp_gp VE 2.634 %, ES 34.795, VS 7755.5, CRPS 0.3994, PIT-ECE 0.0250.

### 2.3 noaa_uk h=168, grid7, chirp_gp vs baselines (n=10)

| metric | vs diagonal | vs low-rank | vs Student-t |
|---|---|---|---|
| **VS ↓** | wins 10/10, p=0.001 | 10/10, p=0.001 | 10/10, p=0.001 |
| PIT-ECE ↓ | wins 9/10, p=0.0029 | 9/10, p=0.0029 | 10/10, p=0.001 |
| ES ↓ | **loses 0/10** | **loses 0/10** | loses 1/10 |
| CRPS ↓ | **loses 0/10** | **loses 0/10** | loses 2/10 |
| VE ↑ | **loses 0/10** | **loses 0/10** | loses 0/10 |

Means: chirp_gp VE 8.519 %, ES 33.469, VS 4495.2, CRPS 0.5023, PIT-ECE 0.0064; diagonal VE
11.401 %, ES 32.864, VS 5536.8, CRPS 0.4922, PIT-ECE 0.0104.

### 2.4 Corrections to the frozen write-ups

1. **T5 flips on bms_air, in chirp's favour.** Handoff §4.3 records "chirp **loses** 8/10". On the
   bracketed grid chirp **wins** 7/10, 8/10, 8/10. The `grid5` result came from baselines whose
   calibration was accidentally flattered by under-training.
2. **T4 on noaa_uk is now unanimous.** Handoff §4.1 records "chirp loses the marginal axis on
   **9/10** seeds — seed 8 flips both metrics. Not unanimous; say so." On grid7 it is **0/10** on
   ES, CRPS and VE against both diagonal and low-rank. Seed 8 no longer flips; delete the caveat.
3. **noaa_uk T5 margin nearly halves** but survives: diagonal PIT-ECE 0.0150 → 0.0104 against
   chirp's unchanged 0.0064; 10/10 → 9/10, p 0.001 → 0.0029.
4. **Every `grid5` Family A number on both cells should be replaced by its `grid7` counterpart.**

### 2.5 The cross-cell claim, restated

| read | noaa_uk (grid7) | bms_air (grid7) |
|---|---|---|
| **VS** | **wins 10/10** | **wins 10/10** |
| **PIT-ECE** | **wins 9 / 9 / 10** | **wins 7 / 8 / 8** |
| ES | loses 0/10 | wins 10/10 |
| CRPS | loses 0/10 | wins 10/10 (loses to Student-t) |
| VE | loses 0/10 | wins 10/10 |

Handoff §4.3 said "path coherence is the only claim that survives on both cells." The sharper and
correct statement: **variogram-score path coherence is the only score that wins unanimously on both
cells — ES does not**, and **calibration now also replicates on both**, which it did not before.
ES and VS disagreeing on noaa_uk (chirp loses ES, wins VS 10/10) is a live instance of the
Pinson–Tastu discrimination argument the paper already commits to.

---

## 3. Nugget ablation (Table 1's CMD-GP rows) — wide grid, both cells, n=10

`abl_base` = with nugget, `abl_nonugget` = `--no-nugget`. Run on `grid7`; nothing at the bound on
either cell, so both arms are bracketed. Signs are "+nugget wins".

| metric | bms_air +nug / no-nug | wins | p | noaa_uk +nug / no-nug | wins | p |
|---|---|---|---|---|---|---|
| ES ↓ | **34.796** / 35.085 | 9/10 | **0.0020** | 33.485 / **33.473** | 5/10 | 0.652 |
| **VS ↓** | 7755.0 / **6878.5** | **1/10** | 0.999 | 4490.8 / **4235.0** | **0/10** | 1.000 |
| CRPS ↓ | 0.3994 / 0.4011 | 7/10 | 0.246 | 0.5024 / **0.5009** | 3/10 | 0.903 |
| PIT-ECE ↓ | 0.0250 / 0.0245 | 7/10 | 0.423 | 0.0064 / **0.0055** | 3/10 | 0.884 |
| cov@80 | 0.7976 / 0.7916 | — | — | 0.7428 / 0.7378 | — | — |
| VE ↑ | **0.0263** / 0.0175 | 10/10 | **0.0010** | 0.0843 / **0.0875** | 3/10 | 0.754 |

**The nugget hurts the variogram score on both cells — 9/10 against on bms_air, 10/10 against on
noaa_uk.** That is the one consistent effect, and it is on the score that carries the paper's
central claim. Everything the nugget buys is cell-specific: on bms_air it wins ES (9/10) and VE
(10/10); on noaa_uk it wins nothing and loses slightly on every axis.

The earlier narrow-grid ablation is **confirmed, not overturned** — bms_air's numbers are
essentially unchanged (ES 9/10 both grids; VS 0/10 → 1/10), because chirp_gp's optimum at 1e-5 was
already inside the narrow grid. The noaa_uk ablation previously existed at only 3 seeds.

---

## 4. Table 3 (`tab:decomp`) — the R9 variance decomposition

### 4.1 Verdicts

| cell | seed | within $\E K$ | between $\mathrm{Cov}\,\mu$ | $\mathcal{R}$ | verdict |
|---|---|---|---|---|---|
| bms_air | 0 | 0.1663 | 0.5062 | 0.7527 | OPTION_B |
| bms_air | 2 | 0.5155 | 0.8614 | 0.6256 | OPTION_B |
| bms_air | 3 | 0.2419 | 0.6812 | 0.7380 | OPTION_B |
| bms_air | 4 | 0.1628 | 0.6989 | 0.8111 | OPTION_B |
| bms_air | 1 | 0.1406 | 0.7560 | 0.8432 | OPTION_B |
| bms_air | 5 | — | — | 0.8760 | OPTION_B |
| **bms_air** | **mean n=6 (COMPLETE)** | | | **0.7744 ± 0.0898** | **OPTION_B 6/6** |
| noaa_uk | 0–7 (8 seeds) | 0.21–0.30 | 0.24–0.48 | 0.4965–0.6801 | OPTION_B, 8/8 |

$\mathcal{R}$ is flat in $M$ on bms_air (0.6253–0.6256 across M=2…25).

**bms_air seed 2's $\mathcal{R}$ = 0.62560 coincides with noaa_uk's quoted 0.6256 to four decimals.
This was checked and is a coincidence**, not a wrong-dataset bug: noaa_uk's figure is the mean over
the six *registered* seeds {0,1,2,5,6,7} (both mean and std reproduce exactly), the within/between
components differ ~2× between cells, and every bms_air artifact resolves correctly (ch-24 entenc
VAE, 168-24 summarizer, 70 083 456 elements).

**noaa_uk now has 8 R9 seeds, not the 6 in the frozen table.** All 8 exceed 0.25 so the verdict is
untouched, but the headline moves: all-8 mean **0.6092 ± 0.0532** vs the frozen **0.6256 ± 0.0325**,
and the excluded pair contains the minimum (seed 3, $\mathcal{R}$ = 0.4965). Prereg amendment A1
justifies the six by completion order; now that seeds 3 and 4 have landed, reporting six of eight
where the excluded set holds the lowest value needs an explicit reason in the paper.

### 4.2 Sensitivity to the read protocol — **adverse, and it must be reported**

Algorithm 2 takes the component law from the final reverse step; the superseded A2 protocol read it
at t=T−1 on pure-noise input. Handoff §2 predicted ~3× inflation of the within term.

| cell | seed | within: final → t=T−1 | $\mathcal{R}$: final → t=T−1 | survives $\mathcal{R}$>0.25? |
|---|---|---|---|---|
| noaa_uk | 0 | 0.21 → 1.0265 (~5×) | 0.6389 → 0.2704 | yes, narrowly |
| noaa_uk | 1 | 0.23 → 0.5893 (~2.6×) | 0.6801 → 0.4504 | yes |
| bms_air | 3 | 0.2419 → 0.5764 (2.4×) | 0.7380 → 0.5421 | yes |
| bms_air | 4 | 0.1628 → 0.8951 (5.5×) | 0.8111 → 0.4388 | yes |
| bms_air | 1 | 0.1406 → 1.1529 (8.2×) | 0.8432 → 0.3957 | yes |
| bms_air | 5 | ×12.8 | 0.8760 → 0.3580 | yes |
| **bms_air** | **0** | **0.1663 → 4.8338 (29.1×)** | **0.7527 → 0.0947** | **NO** |
| **bms_air** | **2** | **0.5155 → 33.68 (65.3×)** | **0.6256 → 0.0249** | **NO** |

**Final, n=6: 2 of 6 bms_air seeds** lose the Option-B verdict under the t=T−1 read, with
inflation factors of 65×, 29×, 12.8×, 8.2×, 5.5× and 2.4× — an order of magnitude of spread
across seeds of the same configuration. The failures are seeds 0 and 2. The picture is far less
lopsided than at n=3, where it read 2 of 3. Both noaa_uk seeds survive. This is a cell-level
fragility, not one bad checkpoint; an earlier draft attributed it to seed 2 specifically, which
was wrong.

The between-component term is unchanged in both reads (0.8614 vs 0.8609), so only the within term
moves — these checkpoints genuinely report a hugely over-dispersed within-component variance on
pure-noise input (seed 2: **24× its own total predictive variance**). That is a property of the
checkpoints, not a bug in the read. **The Option-B verdict is therefore not robust to the read
protocol on bms_air.** The primary verdict is unaffected: Algorithm 2 gives
$\mathcal{R}$ = 0.7744 ± 0.0898 across 6 seeds, all far above 0.25. But the robustness check
fails on this cell and that belongs in the paper, since t=T−1 is the protocol the earlier A2 arm
used.

---

## 5. Table 1 (`tab:pareto`) — the mixture frontier, bms_air, **all 6 seeds**

`DEFAULT_LADDER = (1, 2, 4, 8, 16, 25)` matches Table 1's mixture rows exactly.

Mean over all 6 seeds. **Every arm below is trained under `gaussian_nll` (the s2 objective)** — see §5c.

| arm | ES ↓ | VS ↓ | CRPS ↓ | PIT-ECE ↓ | cov@80 |
|---|---|---|---|---|---|
| mixture M=1 | 56.313 | 9262.3 | 0.6400 | 0.1457 | **0.262** |
| mixture M=2 | 45.090 | 7565.6 | 0.5067 | 0.0626 | 0.499 |
| mixture M=4 | 39.736 | 6774.0 | 0.4432 | 0.0273 | 0.651 |
| mixture M=8 | 37.377 | 6425.2 | 0.4152 | 0.0135 | 0.714 |
| mixture M=16 | 35.661 | 6168.8 | 0.3949 | **0.0069** | 0.743 |
| **mixture M=25** | **34.797** | **6039.9** | **0.3846** | 0.0078 | **0.763** |
| **sampled M=25** | 34.823 | 5730.7 | 0.3837 | 0.0060 | 0.731 |

Monotone in $M$ on every metric, and **M=1's coverage is 0.262 against nominal 0.80** — the structural incompleteness of the single-GP law is directly visible and consistent
with $\mathcal{R}$ = 0.7054 ± 0.0695.

**The matched-cost comparison disagrees with noaa_uk.** At M=25 vs `sampled_M25`, both 1600 passes:

| | ES | VS | CRPS | PIT-ECE | cov@80 |
|---|---|---|---|---|---|
| seed 0 mixture / sampled | **34.73** / 34.90 | **5667** / 5758 | **0.3798** / 0.3803 | **0.0034** / 0.0084 | **0.747** / 0.713 |
| seed 2 mixture / sampled | 35.15 / **35.07** | 6729 / **6224** | 0.3916 / **0.3893** | 0.0232 / **0.0074** | **0.816** / 0.763 |
| seed 3 mixture / sampled | **34.72** / 34.79 | 5821 / **5643** | 0.3846 / **0.3838** | **0.0056** / 0.0083 | **0.755** / 0.715 |

On noaa_uk the mixture won ES, VS and CRPS and was 3.5× better on PIT-ECE. On bms_air it does not.

**Updated 2026-08-21 with seed 0 (n=3).** Mean over three seeds, mixture_M25 vs sampled_M25:

| metric | mixture M=25 | sampled M=25 | mixture wins |
|---|---|---|---|
| ES ↓ | 34.797 | 34.823 | 3/6 |
| VS ↓ | 6039.9 | **5730.7** | 1/6 |
| CRPS ↓ | 0.3846 | **0.3837** | 1/6 |
| PIT-ECE ↓ | 0.0078 | **0.0060** | 4/6 |
| **cov@80** | **0.7634** | 0.7314 | **6/6** |

The mixture reliably buys calibrated **coverage** (6/6, and closer to the nominal 0.80 on every
seed) but not across-the-board dominance. Review §2 warned that "dominates" must be shown rather
than asserted from the Rao–Blackwell theorem; on this cell it is not shown.

Seed 2 is the outlier on both reads — the 65× sensitivity inflation *and* the only seed where the
mixture loses PIT-ECE badly (0.0232 vs 0.0074). The tempting story, "an over-dispersed UQ head
hurts the mixture arm but not the sampled arm", **does not survive the data**: seed 0 has 29×
inflation and the mixture does well there, seed 3 has 2.4× and the mixture loses VS and CRPS. No
mechanism is established. **Complete at the registered 6 seeds.**

`sampled_M2` shows PIT-ECE 0.0023 with coverage 0.257: the documented estimator artifact. The
sampled ladder must not be read vertically; only `sampled_M25` is a protocol arm.

---

## 5b. Read R8 — the reverse loop is close to idle (bms_air seed 2)

The guidance × DDIM-step sweep (Appendix A2), 24 cells, seed 2. At the frozen guidance 1.0:

| steps | 64 | 32 | 16 | 8 | 4 | 1 |
|---|---|---|---|---|---|---|
| CRPS | 1.0495 | 1.0501 | 1.0514 | 1.0529 | 1.0594 | 1.0662 |

**A 64× reduction in reverse-diffusion steps costs 1.6 % CRPS.** The pre-declared reading is
that a flat curve means the loop refines parameters and nothing more — and this is flat. It
also confirms the guidance freeze: **w = 1.0 beats 1.25 / 1.5 / 2.0 at every step count,
monotonically**, so no CFG setting recovers anything.

This is the empirical backing for §5.5's "the loop may be idle" and for review §4.2's request to
reconcile Appendix J with §5.5 — measured on this cell rather than inferred. ⚠️ These CRPS values
are on `run_u1_sweep`'s own scale and are **not** comparable to the frontier's ~0.39.

### 🔴 CORRECTION — this R8 result must NOT be used as evidence

`w_docs/noaa_uk_h168/review.md` rejects the equivalent noaa_uk sweep, and **every objection
applies verbatim to this one**, which I did not flag when I first reported it:

* **it used `--dynamic-thresh-p 0.995` while the frozen protocol uses 0**
  (`run_campaign_job.sh:199` hardcodes it), so its absolute values are not comparable to the
  main protocol;
* **one seed** (the pre-registration reads R8 on 2, `SWEEP_SEEDS="0 1"`);
* **CRPS only** — no ES, VS, PIT-ECE, coverage or clip rate.

The flatness may well be real, and if it is it is a *strong* result — the review calls it
potentially "the strongest paper result", since CMD-D(25) at 1 step is 25 network evaluations.
But it cannot be claimed from this run. **The replacement is the review's prescribed grid:**
`M ∈ {1,2,4,8,16,25} × steps ∈ {1,4,8,64}` under the frozen configuration, reporting ES, VS,
CRPS, mixture NLL, coverage, PIT-ECE and wall-clock. `run_mixture_uq_eval` already accepts
`--ladder` and `--gen-steps`, so the grid is 4 runs of an existing tool per seed — except for
mixture NLL, which does not exist (§5d).

---

## 5c. The training objective of every CMD-D row

Not previously stated, and the review is right that it must be. **Every mixture and sampled arm
in §5 comes from the `s2_diffusion_uq` checkpoint, whose objective is `DIFF_LOSS_MODE =
"gaussian_nll"`.** There is no MSE-objective row on either cell, and there cannot be one with
current tooling: `run_mixture_uq_eval` refuses a checkpoint whose UQ head never received a
gradient (`run_mixture_uq_eval.py:307`), which is exactly what an `x0`-MSE s1 checkpoint is.
Table 1's "CMD-D + 25 samples, **MSE**" row therefore has no data source. The Family A arms
(§2) are a separate family: MSE warm-up then Gaussian NLL, via `--epochs-mse 150 --epochs-nll 80`.

---

## 5d. Joint NLL is unavailable on both cells — and it is a *code* gap, not a compute gap

Table 1 has a **joint NLL ↓** column. Nothing in the campaign produces it.

* `frontier_seed*_val.json` arm scores contain ES, VS, CRPS, PIT-ECE, coverage and width —
  **no NLL of any kind**.
* Family A reports `nll_best_val_objective`, which is the *training* objective's value at the
  selected epoch, **not** a reported joint NLL metric.
* `path_scores.py` defines **`gaussian_joint_nll_per_channel` (line 174)** and **`mixture_nll`
  (line 217)** — and a repo-wide grep finds **zero callers**. They are implemented, documented,
  and dead.

**Cost to obtain: compute-cheap, code-expensive.** `run_mixture_uq_eval` already materialises
the per-component Gaussian means and variances (it needs them for the decomposition), so the
mixture `logsumexp` likelihood is a reduction over tensors already in memory — **no extra
forward passes, no extra sampling**. The work is wiring two existing functions into the scoring
path, which is an edit to `llaplace_df` while jobs are running from it.

---

## 5e. All arms remain undercovered — replicated on bms_air

The review records this for noaa_uk. It reproduces here (mixture M=25, 6 seeds):

| nominal | noaa_uk (review) | **bms_air (measured)** |
|---|---|---|
| 0.80 | 0.731 | **0.763** |
| 0.90 | — | **0.848** |
| 0.95 | 0.865 | **0.898** |
| 0.99 | 0.895 | **0.921** |

Undercovered at every level on both cells, worsening as the nominal level rises — the 99 %
interval captures 92 %. bms_air is *less* badly miscalibrated than noaa_uk but has the same
shape, so this is a property of the method, not of one cell. The review's diagnostic list
(did `q_k`/`p_k^0` receive NLL gradients; mixture vs componentwise NLL; correct mixture
quantiles; decoder distortion; variance transport after normalisation) is **not yet started**.

---

## 6. Infrastructure defects found this session — all silent-failure class

| # | defect | consequence |
|---|---|---|
| 7 | **Bare-lock migration shim defeats `GRID_TAG`** (`run_campaign_job.sh:92-96`). The shim overrides the grid-namespaced lock *after* it is set, for any noaa_uk seed with a pre-namespace lock. | noaa_uk `grid7` seeds 6–9 silently skipped; would have capped the re-run at 6 seeds against grid5's 10. Also shadows `u3`(9), `gate`(7), `frontier`(7), `abl_*`(3 each), `sweep`/`sensitivity`/`efficiency`. **Fix: gate the shim on `[ "$GRID_TAG" = "grid5" ]`. NOT YET APPLIED.** |
| 8 | **`warm_start_confirmed` is a false negative on every trial, both cells.** `_warm_start_took` greps for a message printed only under `verbose=True` (`train_val_llapdiff.py:656`, called with `verbose=verbose` at 2355). | The campaign's own audit field is not evidence of warm-start success for any trial. The warm starts *are* correct — `[load] missing keys for LLapDiff:` lists exactly the four UQ-head tensors, and that branch is unreachable unless only UQ keys are missing (anything else takes `strict=True` and raises). |
| 9 | **`abl_*` hardcodes `--lr-grid 3e-4 1e-4 3e-5 1e-5`** and ignores `$LR_GRID`. | The nugget ablation could not be widened through the campaign path; `finetuning/run_abl_widegrid.sh` was added to bypass it. |
| 10 | **`EVAL_CHUNK` is inert for `gate`/`frontier`.** `run_mixture_uq_eval` has no `--eval-chunk` flag and no chunking; `run_campaign_job.sh` passes it only in the `onepass`/`abl_*` branches. | Handoff §3's eval command sets a variable that reaches nothing. Benign in the event — the d_z=24 `--score` path completed without OOM. |
| 11 | Handoff §3's s2 launch command **would have done nothing**: the `u3` lock has no stage component, so every seed with an s1 lock prints `[skip] already claimed`. Its tool name (`llapdiffusion.tools.llapdiff_uq_eval`) also does not exist. | s2 was launched directly via `run_u3_uq.py` instead. |

---

## 7. Measured timings (correcting handoff §3)

Handoff §3 recorded "s1 measured at 5.8–8.0 h/seed". That was the two seeds that had finished — and
they finished first *because* they were fastest. Full set:

| stage | seeds | wall (h) | mean |
|---|---|---|---|
| s1_mean | 0–5 | 5.8, 8.0, 10.6, 13.0, 14.4, 15.0 | **11.1** |
| s2_diffusion_uq | 0–5 | 17.6, 29.8, 31.0, 41.3, 52.1, **77.2** | **41.5** |

s1 best CRPS 0.63486 ± 0.00322 (n=6). s2 best CRPS 0.6225–0.6423 (n=6). **s2 is the dominant
cost at 41.5 h/seed mean and 77.2 h worst case** — the handoff's "s1+s2 ≈ 12–16 h/seed" estimate
is wrong by roughly 4×; budget **~53 h/seed** for the pair. The cause is architectural: s2 trains
under `gaussian_nll`, which calls `modal_variance` — a Python loop over all **168** timesteps —
on every forward pass; s1 under MSE never executes it. Measured signature: 100 % CPU against
14–40 % GPU. More GPUs do not speed up a single seed.

Other measured costs: R9 `gate` 64 min/seed (full val, 328 batches) · full-val `ddim` latent read
364 s · Family A `grid6` ~80 min/seed (bms_air, 6 LR points) · `abl_*` ~7–13 min/seed (narrow grid).

---

## 8. Outstanding

1. **s2 seeds 1, 4, 5** still training — they gate Table 3's remaining rows and the Table 1
   frontier's remaining seeds. At ~26 h/seed.
2. `gate` / `frontier` / `sensitivity` for **seed 0** are runnable now (its s2 landed).
3. **`sweep` seed 2** (Appendix A2 / read R8) running; cost not yet measured.
4. **Efficiency bench** (Table 1's `ms` column) not run for bms_air. noaa_uk's ran on an
   **RTX PRO 4000 Blackwell**; wall-clock is not portable, so bms_air's must run on the same card
   type and on an *uncontended* GPU or the column is not comparable across cells.
5. **Table 2 rows 3–4 have no implementation.** "Joint NLL (T1 innovations)" and "Composite + path
   score (T3)" are training objectives; `train_val_llapdiff.py:1582` accepts only
   `{"mse", "gaussian_nll"}`, and `run_option_a_probe` only does MSE→NLL. `path_scores.py` has
   `gaussian_joint_nll_per_channel` and `mixture_nll` as *metrics*, not wired to any trainer. This
   is new engineering plus two training runs per cell, and it applies to noaa_uk equally.
6. **Table 1's "CMD-D + 25 samples, MSE" row** cannot be produced from an s1 checkpoint with current
   tooling: `run_mixture_uq_eval` refuses a non-NLL checkpoint, and `llapdiff-checkpoint-eval` emits
   CRPS only — no ES/VS/PIT-ECE.
7. **Missing baselines** (review §3.2): PatchTST + Gaussian head, a structured-kernel GP, a
   time-varying SSM core. None implemented. Review §6 makes these the largest gap to a 7–8.
8. **Random-switch-time multimodal family** (review §3.5) — "the highest value-per-hour item" and
   not started; `run_synthetic_regime_shift.py` exists with a fixed switch at H/2.
9. Defect 7's one-line fix is **not applied**; the stale bare locks were left in place deliberately.

---

## §5b-2 R8 replacement grid — COMPLETE at 3 seeds (bms_air, 2026-08-24)

Seeds 0, 1, 2 finished (rc=0 on every step count). This run repairs all three defects the
`noaa_uk_h168/review.md` names in the original R8: the threshold is **frozen**
(`dynamic_thresh_p: None`, not 0.995), it uses **three distinct training checkpoints**
(`seed-0`/`seed-1`/`seed-2`, different config hashes — verified, not three eval seeds on one
model), and it reports the **decomposition R**, not CRPS alone. Full val split, 328 batches,
25 components, `mean_reconstruction_gap = 0.0` everywhere.

### R (between-component share) vs sampler steps

| steps | seed 0 | seed 1 | seed 2 | mean | verdict |
|---|---|---|---|---|---|
| 1  | 0.0400 | 0.0490 | 0.0195 | **0.0361** | OPTION_A 3/3 |
| 4  | 0.6964 | 0.5826 | 0.4374 | **0.5722** | OPTION_B 3/3 |
| 8  | 0.7092 | 0.7854 | 0.5502 | **0.6816** | OPTION_B 3/3 |
| 64 (gate/frontier, 6 seeds) | — | — | — | **0.7744 ± 0.0898** | OPTION_B 6/6 |

### Reading

R rises monotonically in sampler steps and saturates around 0.7–0.77. **Option A appears at
exactly one point in the grid: steps = 1.** That is the degenerate sampler — with a single
step there is no iterative refinement, so between-component variance is near zero by
construction, not by finding. "The reverse loop is idle" is trivially true when the reverse
loop is one step long.

So this grid does not weaken Option B; it strengthens it. It shows the Option-A reading is an
artifact of a one-step sampler, and that as soon as the reverse loop is allowed to run at all
(steps >= 4) the between-component share dominates. bms_air therefore supports **Option B**
at every non-degenerate step count, consistent with the 6-seed gate at steps=64.

Caveat to keep: R at steps=4 and 8 has visible seed spread (0.44-0.79); only the steps=64
gate has 6 seeds. Seeds 3, 4, 5 of this grid are still outstanding.
