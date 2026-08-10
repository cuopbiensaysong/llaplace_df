# CMD UQ campaign — recovery plan

**Purpose.** The ordered plan that took the U3 ablation from an uninterpretable PhysioNet run to
a defensible Table 3.

> ⚠️ **Rewritten 2026-08-10. Phases 0–2 are DONE and their gate logic is SUPERSEDED** by
> `CMD_UQ_TABLE3_PLAN.md` — the Phase-1 latent gate was measured to be anti-correlated with the
> campaign's objective and is retired. What survives here and is still binding: the **standing
> rules** (§1), the **cell selection** (§2), and the **U3 phase spec** (§3–§6).
>
> The full 796-line original, including the Phase-0/1/2 execution narratives and the
> stop-the-campaign verdicts they produced, is at
> `archive/2026-08-uq-campaign/CMD_UQ_RECOVERY_PLAN.md`.

---

## 1. Standing rules (apply to every run, every phase)

- **`--weights ema`** on `llapdiff-uq-eval` *and* `llapdiff-u1-sweep`. Both default to `raw`,
  and `_load_stack` reads `payload["model"]`, which holds **raw** weights even in
  `*_best_ema.pt` (B16). Measured cost of missing it: val CRPS 0.2673 (raw) vs 0.2593 (EMA) —
  larger than the entire 0.0014 gain a tuning campaign was decided on.
- **Set `CHIRP_UQ_INIT_VAR` explicitly** for every UQ run, at the residual scale of *that*
  dataset × horizon. Never copy a value across cells. See §4.
- **Print train/val/test window counts** before trusting any run on a new cell — and count
  **window starts**, not `sizes[0]`, which runs ~4× larger and green-lights cells the audit
  exists to reject:
  ```python
  w = sum(batch[0][0].shape[0] for batch in loader)   # axis 0 = window starts
  ```
- **Never compare `val_diag_mse_raw` across loss modes.** It is an MSE under
  `DIFF_LOSS_MODE="mse"` and an **NLL** under `"gaussian_nll"`. This produced one phantom
  "340× degradation".
- **Test split is touched once per arm per seed**, and only after a pre-registration for that
  cell is frozen. Gates and diagnostics run on **val**.
- **Record `cond_norm_mode` and `ALLOW_TF32`** for every checkpoint used. The former defaults to
  the legacy `"batch"` when the key is absent, silently reinstating a train/eval mismatch.
- **Gate/decision runs use ≥ 2 seeds.** `cudnn.benchmark` nondeterminism alone produced a
  0.036 CRPS swing between byte-identical configs.

---

## 2. Cell selection — settled

`w_docs/CMD_PHASE0_WINDOW_AUDIT.md` has the full audit. The operative rows:

| cell | train_w | val_w | ent/win | params | verdict |
|---|---|---|---|---|---|
| physionet h12 | **13** | 5 | 445 | 11.75 M | smoke test only |
| crypto h100 | 1 225 | **90** | 118 | 11.75 M | val too thin, 6.0× over |
| us_equity h100 | 1 099 | **72** | 221 | 11.75 M | val too thin, 8.9× over |
| **noaa_uk h168** | 16 286 | 2 183 | 4 | **10.17 M** | **the gate cell** — 4.3× *under*-parameterized |
| bms_air h168 | 95 099 | 17 598 | 12 | 11.75 M | **VOID** — no forecastable signal at any horizon |

**Two axes matter.** Windows set how much the denoiser can learn; entities per window set how
much survives the set-VAE, which mean-pools the entity axis into one latent per window.

**Stop conditions.** < ~500 train windows ⇒ smoke test only. < ~500 val windows ⇒ the gate is
under-powered. Over-parameterized cells cannot support a "modelling defect" verdict.

**Gate 0 does not license a cell.** bms_air has the highest Gate 0 of all five (0.941) and no
forecastable signal whatsoever. Gate 0 asks "can stage 1 reconstruct its own targets?"; the
ridge probe's **A** asks "is the target forecastable from history at all?" Both must be read.

**Phase-2 cross-check cell: `noaa_us` h=168** (A = 23.4 %, same target variable, independent
40-station network). bms_air was ruled out — its target is PM2.5, which is not forecastable at
these horizons; that is physical, not a pipeline defect.

---

## 3. Phase 3 — calibrate `CHIRP_UQ_INIT_VAR` per cell

Once per dataset × horizon, before any UQ arm runs there. The rule that transfers is *initialise
the predicted variance near the residual scale*; the value never does.

```bash
llapdiff-uq-eval --dataset-key <ds> --pred <h> --checkpoint <S1 ckpt> \
  --weights ema --split val --latent-only --mean-source ensemble --vae-entity-encode
```

Read **`latent_mse`**, not `latent_rmse` — this is a *variance*, and confusing it with a std is
the exact failure the phase exists to prevent. Then set the knob to that magnitude and record it
with the residual scale it came from.

Why it matters: on PhysioNet the library default 1e-2 gave initial predicted variance 0.0011
against a squared error of 2.675, leaving the arm at **3 % coverage at nominal 90 %** while
**CRPS moved < 0.001**. CRPS cannot detect this failure.

⚠️ Use a **matched ensemble mean** for this read. At `--mean-source ddim` the residual is
inflated by draw noise — 51 % of val `latent_mse` on this cell — so the derived init would be
~2× too large.

---

## 4. Phase 4 — freeze a pre-registration for the cell

`PREREG_U3.md` is scoped to PhysioNet h=12. **Write a new file**, not an amendment:
`w_docs/PREREG_U3_<dataset>_h<h>.md`. **This has not been done for noaa_uk and blocks the test
split.**

Carry over unchanged: the three-arm design, matched hyperparameters, 5 seeds, test-split-once
discipline, and the declared scope limits (§6 of the old file).

Fix three things while writing it:

1. **Thresholds against measured baselines.** The old G-c assumed unit-scale latents; the actual
   std was 1.46. State thresholds relative to `baseline_rmse_predict_mean` as the tool reports it.
2. **Outcome branches must cover the orderings actually observed.** The PhysioNet run produced
   A2 < A3 < A1, matching none of the four pre-registered branches.
3. **Record the Phase-3 `CHIRP_UQ_INIT_VAR`** and its measured residual scale.

---

## 5. Phase 5 — the guidance audit, before any arm is scored

Guidance `w > 1` sharpens the predictive distribution and is **indistinguishable from
miscalibration**, so it must be frozen before scoring. This is a validity requirement, not tuning.

> **Ordering.** `llapdiff-u1-sweep` scores a *trained* checkpoint, so this runs **after** Phase 6
> training and **before** Phase 6 scoring: **6a training → 5 sweep → freeze → 6b scoring.**

```bash
llapdiff-u1-sweep --dataset-key <ds> --pred <h> --checkpoint <per-arm ckpt> \
  --guidance 1.0 1.25 1.5 2.0 --steps 16 32 64 --split val --weights ema
```

**Executed on noaa_uk h=168: guidance is strictly monotone-harmful and `w = 1.0` is frozen.**
See `CMD_UQ_TABLE3_PLAN.md`. Note this was measured on one stack rather than swept per arm; the
write-up should say what was done rather than cite the per-arm grid.

---

## 6. Phase 6 — run U3

Three arms, identical synthesizer, 5 seeds, driver `finetuning/run_u3_uq.py` (tag `uq_u3`).

**Both design fixes this phase called for are now implemented** — matched means across arms
(Fix 1) and PIT/coverage for A1 (Fix 2) — along with three further scoring defects found in
review. See `CMD_UQ_TABLE3_PLAN.md` for what each was and how it is verified.

**Table 3 columns:** `CRPS ± std | MSE ± std | PIT-ECE | coverage @ 0.8 | mean interval width |
wall (s) | denoiser passes`. Report dispersion on MSE and wall-clock, not only CRPS.

**Report `A1 / A3` as the method-level speedup** and label A1-vs-A2 "calibration at matched
cost" — with means matched, A2 costs what A1 costs (1604 vs 1600 passes) and the A1/A2 ratio is
1.00 by construction.

**Wall-clock caveat.** Fixed harness overhead is ≈0.49 s against ≈8 ms/pass, so A3's timing is
~98 % overhead and `analytic_speedup_x` is nearly meaningless at short horizons. Quote
`denoiser_passes_per_batch`, which is exact and architecture-independent.

---

## 7. What to do with the PhysioNet results

Do not delete them; relabel them.

1. **Report PhysioNet h=12 as a plumbing smoke test.** It is genuinely good at that — a full
   3-stage U3 seed costs ~20 min, and it surfaced B16, the variance-init failure, and B19.
2. **Remove PhysioNet from Table 1 and Table 2.** It is the only filled cell in Table 1 and the
   load-bearing column in Table 2; everything else there is already `[TBD]`.
3. **Add B19 to Appendix F (reproduction traps):**

   > *PhysioNet at short horizons yields only ~13 diffusion training windows under
   > patient-relative splitting (`WINDOW + PRED = 36` against ~48 h records). Data-space CRPS
   > nevertheless appears healthy, because the set-VAE decoder reconstructs each entity largely
   > from its own embedding — CRPS is dominated by a per-entity prior rather than by the
   > forecast. Audit window counts, not row counts.*

4. **Also worth an appendix line:** variance initialisation must be set near the residual scale,
   and CRPS is blind to the resulting miscalibration.

### 🔴 B19 reaches further than this plan does

The five-window estimate invalidates more than Table 3: the **G2/G3 2×2 factorial** (cells at
0.3706–0.3719, i.e. deltas of 0.0001–0.001), the **§8 branch decision** taken on cell (a), the
**22-trial `fixed_bugs` tuning campaign** decided on a 0.0014 gap, and the *"chirp arms show ~6×
lower seed variance"* finding.

This plan recovers the UQ table only. Executed end to end it produces a defensible Table 3
attached to a paper whose **headline claim still rests on five test windows**. Fixing that —
re-running the G3 factorial on noaa_uk h=168 — is a separate track and a prerequisite for the
paper, not for Table 3. It must not be lost because this document did not cover it.
