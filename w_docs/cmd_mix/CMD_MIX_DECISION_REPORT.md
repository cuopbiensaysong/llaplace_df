# CMD-Mix — decision report

**2026-08-26.** Written to answer one question: *how much more time is this method worth?*

Current-state only. The append-only history is in
[`CMD_MIX_NOAA_UK_G0_G3_HANDOFF.md`](./CMD_MIX_NOAA_UK_G0_G3_HANDOFF.md); superseded
rounds there are labelled as such. Machine-readable truth is
`_campaign_scripts/cmd_mix/frozen/gate_index.json` — if this document disagrees with it,
the file is right.

---

## 1. The short answer

**The machinery is now trustworthy. The science is unproven, and the design is unlikely to
prove it.**

Every gate passes and S4 is authorised. But there is still **no comparative evidence** —
not one arm has been fit under the frozen protocol at full data. And the pre-registered
1% margin is hard to clear at the variability measured so far:

| quantity | value |
|---|---|
| proxy block-level σ (dev_val) | **6.29%** of scale |
| intra-station ρ | 0.28 |
| effective n (4 stations × 11 blocks) | **11.6** |
| observed contrast needed to clear **zero** | **3.33%** |
| observed contrast needed to clear **1%** (Holm) | **5.82%** |
| power at a true 3% effect (major rewrite) | **0.100** |
| development-era effect estimate (different endpoint) | **0.83%** |

If the true effect resembles what development suggested, **no number of seeds clears the
margin** — the point estimate itself does not reach it.

**That is not yet a verdict.** The σ above is a *proxy*, from a persistence-vs-climatology
contrast — two structurally dissimilar predictors with weakly correlated errors. A
CMD-Mix(2)-vs-CMD-Mix(1) contrast pairs two similar models under common random numbers, so
its real σ should be **materially smaller**. The proxy is a conservative upper bound on the
difficulty, not an estimate of it.

**Replacing that proxy costs ~19 GPU-hours and decides everything else.** It is running now.

---

## 2. What is actually established

Five gates, each with machine-readable evidence and per-test maxima:

| gate | what it certifies | status |
|---|---|---|
| G0 | the mixture law matches the shared fixture | PASS — worst 2.842e-13 vs 3.33e-04 tol |
| G1 | split, thresholds, hashes, exposure | PASS |
| G2 | D=1 parity, projectivity, query ordering | PASS — 680 checks, 0 fail |
| G3 | covariance validity, parameterization | PASS — keep `p_exact` |
| G4 | collector + clean upstream stack + latents | PASS |

Plus S4 **smoke**: D∈{1,2,4} at full architecture, **zero** Cholesky failures, adaptive
rescues, skipped steps or non-finite results — meaningful because `strict=True` *raises*
rather than rescuing, so a clean exit is proof.

### The findings that mattered

- **G3 overturned the `p_mono` story.** The kernel is PSD *by construction*
  (`d(rhobar)/dt >= rho_min > 0` in 40/40 audited configurations). The crashes were
  **float32 + TF32**, not a parameterization defect: float64 gives 0 Cholesky failures
  where float32 gives 136, and TF32 turned 8 failing blocks into 128/128. Frozen remedy:
  float64 covariance, TF32 off. **`p_mono` was not adopted.**
- **The original test partition is burned.** CMD-D *selected on it*
  (`selection: {split: test}`, files named `select_test_ema.json`). So the lockbox is a
  fresh out-of-time partition, 2025-01-01 → 2025-08-24, sealed.
- **A normalization "leak" I reported does not exist.** I retracted it: the loader
  recomputes train-only statistics and discards the stored full-series file.
- **The legacy neural stack is unreproducible** — its checkpoints record no architecture.
  G4.1b is *retired by dated waiver*, not passed; the tolerance was never loosened. The
  retrained v2 stack serialises 88 behaviour flags + 5 source hashes so it cannot recur.

---

## 3. Running plan, with stop points

Costs are **measured**, not projected: `cost(D) = 12.3 + 194.0·D` seconds per 3+3 epochs
at full data, float64, batch 32 (D=4 predicted 788 s, measured 809.7 s, +2.7%).

### Stage 1 — σ probe · RUNNING · ~19 GPU-h

`native_D1` and `native_D2`, seeds 0/1/2, full data, frozen LR grid. Started 23:38 on
cuda18 GPU 1. Produces the **real paired-arm σ_block** and the first honest D=1 vs D=2
contrast.

> **STOP POINT 1.** Nothing else should be committed until this lands.

| σ comes back | meaning | action |
|---|---|---|
| **≤ 2%** | need ~1.5% observed to clear zero, ~2.5% to clear 1% | full pilot is worth it |
| **2–4%** | zero reachable; 1% margin marginal | run D=4 + controls, expect a feasibility-only result |
| **≥ 5%** | 1% margin unreachable at n=11 | **stop. Option B (descriptive).** |

### Stage 2 — remaining arms · ~51 GPU-h · other clusters

`native_D4`, `param_matched_{D2,D4}`, `modal_matched_{D2,D4}`,
`full_experts_{D2,D4}` × 3 seeds. Confound controls — they only matter once Stage 1 shows
an effect worth attributing.

> **STOP POINT 2.** If Stage 1 gives no separation, these cannot rescue it.

### Stage 3 — CMD-D comparator · ~108 GPU-h

Must be **retrained** (its checkpoint was test-selected). Measured 35.9 h/seed
(s1 7.5 + s2 16.3 + s3 12.1) × 3 seeds. Needed only for the **major-rewrite** claim
(matched-quality speedup, one-sided 95% LB ≥ 4×).

> **STOP POINT 3.** Skip entirely if Stage 1–2 give no practical superiority.

### Stage 4 — lockbox · once, sealed

Only after one D is selected on dev_test. Not scheduled.

### Totals

| | realistic | cap |
|---|---|---|
| Stage 1 | 19 h | 56 h |
| Stage 2 | 51 h | 152 h |
| Stage 3 | 108 h | 108 h |
| **all** | **282 GPU-h** | **608 GPU-h** |

On 1 GPU: 11.7 days realistic. On 3: 3.9 days. **Stage 1 alone is under a day** and gates
the other 260 hours.

---

## 4. What "good enough to skip Option B" requires

Option B is the descriptive fallback — development evidence only, no confirmatory claim.
To avoid it, CMD-Mix must clear, in order:

1. **Feasibility** — Holm-free 95% lower bound on the paired contrast **> 0**.
2. **Practical superiority** — Holm-adjusted lower bound **> 1%**, against *both* D=1 and
   the matched-capacity control.
3. **Component use** — not a reparameterised D=1: mean responsibility ≤ 0.90,
   second-largest weight ≥ 0.05, normalised component-function distance ≥ 0.10.
4. **Safeguards** — unbiased energy score non-inferiority, then calibration.
5. **Major rewrite** — matched-quality speedup vs CMD-D, one-sided 95% LB ≥ 4×.

**(1) and (2) are separate outcomes and are reported separately.** A result can be
*detectable but not practically superior* — and on current evidence that is the single most
likely outcome.

An honest note on (2): the 2% ES/CRPS non-inferiority margin is **not attainable** at the
proxy σ even when the true difference is exactly zero. It was deliberately left unchanged
rather than widened to fit a proxy; it is revisited once, at Stage 1, against the real σ.

---

## 5. Recommendation

**Spend the ~19 hours on Stage 1. Commit nothing further until it reports.**

The gate work is done and does not need repeating — that cost is sunk and the machinery is
sound. What remains unknown is one number, σ_block, and it is cheap to obtain and decisive.

If σ lands ≥ 5%, Option B is the correct scientific outcome, not a failure: the design
would be shown incapable of resolving a 1% effect at four stations and eleven blocks, and
that is worth stating plainly in the paper rather than spending 260 more GPU-hours to
restate it with wider intervals.

---

## 6. Live status

| item | state |
|---|---|
| gates | G0–G4 PASS, `s4_authorized: true` |
| S4 smoke | clean, zero validity violations |
| Stage 1 | `native_D1` seed 0 running on cuda18 GPU 1 since 23:38 |
| Stage 2 | some arms launched on cuda13; `modal_matched_D4` OOM'd |
| lockbox | **sealed, never opened** |

**cuda13 has 24 GiB cards; cuda18 has 48 GiB.** Peak is ~11.2 GiB (covariance, batch 32,
float64, k-independent) plus a k-dependent term. High-k arms (k=128/131) need
`row_chunk=8`, now automatic in `cmd_mix_arm.sh`. `row_chunk` is **bitwise** value-neutral
(verified 64→1), so it changes memory only; batch stays frozen at 32.

Re-run after the fix:
```bash
WS=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/_campaign_scripts
bash $WS/cmd_mix_arm.sh modal_matched_D4 0 <gpu>
```
