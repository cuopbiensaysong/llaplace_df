# CMD / LLapDiffusion — Consolidated Bug Report

**Scope.** Every bug found in the CMD (Chirp-Modal Dynamics) work, what it broke, how it was
fixed, and what results it invalidates. Consolidates and supersedes
`fig2_pole_basis_cannot_drift.md`, `pole_rho_init_horizon_bug.md`, and
`CMD_SESSION_NARRATIVE.md`, whose non-bug content — results ledger, retracted claims,
audit recipes — is preserved in §4–§5 below.

Those three originals are kept verbatim in **`archive/`** (see `archive/README.md`) as the
raw record; the phase-by-phase implementation history survives only there. This document is
the one to read and maintain.

**Last updated:** 2026-08-01 · **Branch:** `main` (after the `speed_up` merge) · **Tests:** 411 passing
**Status of the code:** all fixes below are **landed and committed** except where marked OPEN.
`update_method` is deliberately left at `347cd8a` (pre-fix) as the reference point.

---

## 0. Bug index

| # | Bug | Where | Severity | Status |
|---|---|---|---|---|
| **B1** | Pole-coefficient head got exactly zero gradient (zero-init + squared) | `laptrans.py` | 🔴 critical | fixed 2026-07-05 |
| **B2** | Pole recovery ranked modes by coefficient variation → plotted junk modes | `run_synthetic_chirp_benchmark.py`, `plot_llapdiff_poles.py` | 🔴 critical | fixed 2026-07-20 |
| **B3** | Stale-checkpoint trap: runs evaluated 6-day-old checkpoints | `train_val_llapdiff.py` | 🔴 critical | fixed 2026-07-21 |
| **B4** | Stage-1/2 artifacts reused across incompatible caches | `run_synthetic_chirp_benchmark.py` | 🔴 critical | fixed 2026-07-21 |
| **B5** | VAE entity mean-pool cancels per-entity phase (round-trip ceiling 0.62) | `latent_vae.py` (data-side fix) | 🔴 critical | fixed 2026-07-21 |
| **B6** | Horizon-blind ρ init — oscillatory modes dead before the horizon ends | `laptrans.py`, both cores | 🔴 critical | fixed 2026-07-22 |
| **B7** | ρ variation had no upper cap — training re-kills the modes | `laptrans.py` | 🔴 critical | fixed 2026-07-22 |
| **B8** | ρ variation on a mean-1 basis couples variation to mean decay | `laptrans.py` | 🟠 major | fixed 2026-07-22 |
| **B9** | Pole basis cannot drift: ω(0) = ω(L) = sup ω for every mode | `laptrans.py` | 🔴 critical | fixed 2026-07-23 |
| **B10** | ω variation on a mean-1 basis inflates mean frequency | `laptrans.py` | 🔴 critical | fixed 2026-07-24 |
| **B11** | ω-headroom trap: swing budget coupled to mean frequency | `laptrans.py` | 🔴 critical | fixed 2026-07-24 |
| **B12** | `seed_poles` ρ₀ 2× too large under the centred ρ basis | `laptrans.py` | 🟠 major | fixed 2026-07-24 |
| **B13** | `seed_poles` / `_growth_terms` hardcoded φ(0)=2 | `laptrans.py` | 🟠 major | fixed 2026-07-24 |
| **B14** | Recovery metric made the LTI tracking slope identically 0 (a tautology) | `plot_llapdiff_poles.py` | 🟠 major | fixed 2026-07-24 |
| **B15** | Conditioning normalised per **batch** → 41% train/eval mismatch | `llapdiff_utils.py` | 🔴 critical | fixed 2026-07-25 |
| **B16** | `_best.pt` and `_best_ema.pt` hold identical weights; `_load_stack` never applies EMA (`finetuning/` is unaffected) | `train_val_llapdiff.py`, `llapdiff_checkpoint_eval.py` | 🔴 critical | **OPEN** |
| **B17** | `llapdiff-synthetic-regime` crashes at its own default geometry | `run_synthetic_regime_shift.py` | 🟡 minor | **OPEN** (workaround documented) |
| **B18** | `final_eval.py` scored a raw/EMA **hybrid**: the shared `analysis`/`synthesis.encoder` alias kept raw weights | `finetuning/final_eval.py` | 🔴 critical | fixed 2026-07-29 |
| **B19** | PhysioNet h=12 gives the diffusion stage 13 training windows (5 val, 5 test) | dataset geometry | 🔴 critical | **OPEN** |
| **B20** | Stage-1 round-trip is capped at the cross-entity **mean** share: Gate 0 fails on 3 of 5 real cells (B5 on real data) | `latent_vae.py` | 🔴 critical | **OPEN — preliminary** |
| **B21** | The conditioning pipeline discards ~½ the forecastable signal at stage 2 and more at stage 1; no gate detects it | `summarizer.py`, `latent_vae.py` | 🔴 critical | **OPEN — preliminary** |
| **B22** | `SUM_FT_MODE` fine-tunes the summarizer but it is never persisted → silent train/eval mismatch, weights lost | `train_val_llapdiff.py`, `llapdiff_checkpoint_eval.py` | 🔴 critical | fixed 2026-08-01 |

### Knobs added while investigating B21 (2026-08-01)

| knob | default | what it does |
|---|---|---|
| `COND_NORM_MODE = "global"` | not the default (`"sample"` still is) | Normalises the conditioning with **fixed statistics computed once over train**, persisted per checkpoint. Batch-independent like `"sample"` (B15's requirement) but **level-preserving**, and unit-scaled. Measured −0.028 val CRPS vs `"sample"` on noaa_uk h=168, one seed — promising, not yet adopted. Raises if the statistics are missing rather than silently falling back. |

Both `normalize_cond_per_batch` and the resolver reject an unknown mode, and every other
mode's checkpoints resolve their statistics to `None`, so nothing pre-existing moves.

---

## 1. Bugs fixed in this session (2026-07-23 → 2026-07-25)

### B9 — The pole basis cannot drift 🔴

**What was wrong.** `ChirpModalField` built its basis as

```python
freqs = torch.linspace(1.0, float(self.num_basis), self.num_basis)   # f_m = 1..M
phi   = 1.0 + torch.cos(2*pi * f_m * t / L)
a_omega2 = c[:, 1].pow(2)                                            # squared => >= 0
omega(t) = omega_floor + sum_m a_omega2[m] * phi_m(t)
```

Every `f_m` is a **whole number of cycles across the window**, so `phi_m(0) = phi_m(L) = 2`
is each basis function's maximum. Combined with non-negative (squared) coefficients this
forces, for every mode and every conditioning:

- `omega(0) == omega(L)` **exactly** — measured max |difference| = 2e-7 over 512 modes
- `sup_t omega(t)` attained **only at the endpoints** — `argmax` ∈ {0, T−1}, always

The instantaneous frequency could only **dip and return**. A monotone sweep, or a peak
inside the window, was **not in the function class at all**. Amplitude was never the limit
(the class can swing 2.06 rad/step) — it simply could not *end* anywhere but where it started.

**Impact.** The H2 ground truth is a triangle sweep, i.e. exactly monotone-or-one-turning-point
with unequal endpoints. Fitted over 120 stratified test windows using the model's **own**
`_basis()` matrix under its **actual** non-negativity constraint (projected-gradient NNLS):

| basis | R² on the real ground-truth ω(t) |
|---|---|
| `integer` (legacy) — **every Fig-2 run ever** | **0.119 ± 0.266** |
| `half_integer` (fixed) | **0.990 ± 0.008** |

Net endpoint drift is **73% of the within-window range**, and that component is identically
zero in the legacy class. ⇒ The 5-seed "confirmed negative" measured whether the model
*learns* a sweep it could not *represent*.

**Why it hid for so long.** The five "independent" benchmark designs (fm-2, fm-4, sweet-spot,
period-96, period-32) varied sweep *steepness* and *period* but never the *shape class*, so
they shared one confound and their agreement looked like robustness. CRPS ties because a
static superposition of constant-frequency modes approximates a chirp fine over 48 steps.
The signature was visible in every `*_pole_recovery.pdf` the whole time: every recovered ρ
curve starts at its maximum, dips, and returns.

**The fix.** Pair every harmonic with its negative and admit half-cycles:
`phi_m(t) = 1 + s_m·cos(2π f_m t / L)` with `f = {0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0}`
and `s = {+1, −1, +1, −1, …}`.

- A non-negative combination of `(1+cos)` and `(1−cos)` spans a constant plus a **signed**
  multiple of `cos` — so **no signed-coefficient machinery is needed** and every positivity
  guarantee is untouched. At f = 0.5 this gives monotone sweeps in either direction; at
  f = 1, V and Λ shapes.
- **All guarantees preserved:** `phi ∈ [0,2]` and `|phi − 1| ≤ 1`, so the ω Nyquist rescale and
  both ρ headroom rescales are unchanged; `Phi_m = t + s_m·sin(2π f_m t/L)/(2π f_m/L)` keeps
  Theorem A closed-form; ρ ≥ ρ_min keeps Theorem B; ε-init still starts at the LTI poles.
- **Zero-mean over the window survives** because `2·f_m` is an integer ⇒ `sin(2π f_m) = 0` —
  required by the centred ρ basis (`ρ̄(L) = ρ_floor·L`).
- Trade-off: top frequency drops from M to M/4 cycles across the window (8 → 2 at M=8). This
  is the right trade — the observed pathology was capacity spent on high-frequency ripple —
  and M stays sweepable via `--chirp-num-basis`.

**Exposed as** `CHIRP_BASIS` (default `"half_integer"`; `"integer"` reproduces the legacy
class) and `--chirp-basis`, tagging the denoiser dir `_cb-<mode>`. Pre-fix checkpoints
`setdefault` to `"integer"` so they rebuild in the class they were trained in.
`basis_signs` is a **non-persistent** buffer (a pure function of basis+num_basis), so every
pre-fix checkpoint still loads under `strict=True` — verified on real checkpoints.

**No theorem is affected**: the method doc requires only "a nonnegative basis with closed-form
antiderivatives". Integer cycles were an implementation choice, not a theoretical requirement.

**Guarded by** `tests/test_chirp_basis_harmonics.py` (drift achievable in both directions;
legacy mode bit-identical; bounds hold; antiderivative correctness; back-compat).

---

### B10 — The ω variation sat on a mean-1 basis 🔴

**What was wrong.** With B9 fixed, a trained model still read **mean ω = 1.23 against a truth
mean of 0.40**. Cause: the ω variation used `phi = 1 + s·cos`, whose mean over the window is
1, so

```
a²₊(1+cos) + a²₋(1−cos) = (a²₊ + a²₋)  +  (a²₊ − a²₋)·cos
                           └ shifts the MEAN ┘   └ does the useful work ┘
```

Every unit of coefficient needed for *shape* was taxed with an equal *mean* shift, and the
redundant common mode inside each ± pair inflated the mean for free. Measured on the trained
model: **63% of the ω coefficient mass was redundant common mode** (the f=1.5 and f=2.0 pairs
were almost pure common mode).

**This is the identical defect that `CHIRP_RHO_BASIS="centered"` already fixed for ρ (B8) —
it had simply never been applied to ω.**

**The fix.** `CHIRP_OMEGA_BASIS="centered"` (default): `ω(t) = floor + Σ a²·s_m·cos(…)`,
zero-mean over the window, so the time-mean of ω sits exactly on the floor. Verified: the
mean-ω-minus-floor gap went **+1.287 → +0.000**. Positivity/Nyquist by rescale (see B11).
`--chirp-omega-basis` flag, tags `_ob-<mode>`; pre-fix checkpoints `setdefault` to `"nonneg"`.

---

### B11 — The ω-headroom trap 🔴 *(introduced by me, in the B10 fix)*

**What was wrong.** In the B10 fix I bounded the centred ω swing by
`S ≤ min(floor, ω_max − floor)`, reasoning that ω must stay in [0, ω_max]. **But ω ≥ 0 is not
a requirement of this model.** The synthesis is `Σ_k e^{-ρ̄}(c_k cos ω̄_k + b_k sin ω̄_k)` with
free residues, so `ω → −ω` is absorbed *exactly* by `b_k → −b_k` — verified numerically,
difference **0.000e+00**. The sign of ω is pure gauge; only Nyquist binds.

That bound coupled the **swing** budget to the **mean** frequency, creating a
self-reinforcing trap:

> hedge toward DC → floor drops → allowed swing drops with it → *cannot* sweep → hedge further

**Evidence (constraint saturation on trained checkpoints):**

| K | floor | allowed swing | achieved | saturation |
|---|---|---|---|---|
| **1** | 0.0266 | 0.0266 | 0.0264 | **99.2%** ← fully clamped |
| 2 | 0.744 | 0.744 | 0.114 | 23% |
| 8 | 0.781 | 0.473 | 0.168 | 57% |

The K=1 run — which I had called "decisive" — was pinned to a swing of **0.027 against a truth
swing of 0.48**. It could not have tracked whatever it learned. **This retracted my own
"DC hedge is loss-optimal" conclusion**, which was an artefact of my constraint.

**The fix.** Nyquist-only: `S ≤ ω_max − floor`. At a near-DC floor the achievable swing goes
**0.027 → 3.13** (400×), Nyquist still respected. Post-fix, K=1 *chooses* swing **0.544 vs
truth 0.48** (unconstrained, saturation 0.18) — the model does sweep.

Recovery now reports **|ω|** (gauge-invariant; a no-op for the LTI core and for the legacy
non-negative basis, and correct where the centred variation carries ω through zero).

**Note the asymmetry, which is principled:** ρ *must* stay positive (Theorem B contraction),
so the centred ρ bound `S_ρ ≤ floor − ρ_min` is correct and stays. ω is a phase, so only
Nyquist applies.

**Guarded by** `test_centered_omega_swing_budget_is_not_capped_by_the_floor` — pins that a
near-DC floor still permits a swing ≫ floor.

---

### B12 — `seed_poles` ρ₀ was 2× too large under the centred ρ basis 🟠

**What was wrong.** `seed_poles` (which seeds residue extraction for the analysis stage)
computed ρ₀ with the `nonneg` convention (`φ(0)=2`) regardless of `rho_basis`. Under
`rho_basis="centered"` the synthesis path uses `φ−1`, so the **seeded analysis poles were 2×
the poles that actually synthesized the output**.

**Impact.** Arrived with the (uncommitted) centred ρ basis and applied to the **entire final
5-seed campaign**.

**The fix.** `seed_poles` now reads the same centred/non-negative convention as
`instantaneous`/`integrated` for both poles. Guarded by
`test_seed_poles_match_instantaneous_at_zero`, parametrised over both bases.

---

### B13 — `seed_poles` and `_growth_terms` hardcoded φ(0)=2 🟠

**What was wrong.** Both hardcoded `2.0 * coeff.sum(-1)` on the assumption that every basis
function peaks at t=0. Under signed harmonics `φ_m(0) = 1 + s_m` (2 for a positive harmonic,
**0** for its negative partner). `_growth_terms` additionally needed the `s_m` factor in
`phi_prime`.

**The fix.** Both now use the actual `φ(0) = 1 + s_m` vector. `_growth_terms` only matters
when `CHIRP_GROWTH_BUDGET > 0` (the T2 experiment) but had to be correct. Guarded by
`test_growth_excursion_anchored_at_zero` (γ(0)=0, |γ| ≤ c_g, closed-form derivative).

---

### B14 — The recovery metric made the LTI slope a tautology 🟠

**What was wrong.** `modal_contributions` weights modes by `envelope_mass`, which is
**already** `mean(dim=1)` over t. So `omega_eff(t)` is a *fixed convex combination* of the
per-mode `ω_k(t)` and moves only if the modes themselves sweep. Two consequences:

1. For the LTI core (constant ω_k) `omega_eff` is constant and its trend slope is
   **identically 0 by construction** — all 10 LTI rows read `0.0000 ± 0.0000`. The
   "LTI fails structurally" panel contained no evidence either way.
2. A model that produces a swept spectrum by **redistributing energy across
   constant-frequency modes** (differential decay) also scores 0.

Route 2 is not hypothetical. Reconstructed from the 5-seed campaign's per-mode summaries, the
**LTI** arm moves its instantaneous mean frequency by **0.188 ± 0.142** — 59% of the truth's
0.321 swing, up to 94% in seeds 2 and 4 — versus 0.023 for chirp. Its per-mode ρ spread is
11.3× (median 6.8×) against chirp's 1.9×.

⇒ **"LTI is spectrally static" is false and must not be claimed.**

**The fix.** Added `omega_eff_dyn` / `rho_eff_dyn` using instantaneous weights
`E_k(t) = e^{-2ρ̄_k(t)}‖θ_k‖²`, recorded as `omega_trend_slope_dyn` / `omega_eff_dyn_rmse`
alongside the static metrics (kept for continuity). Guarded by
`test_omega_eff_dyn_uses_instantaneous_weights`.

---

### B15 — Conditioning normalised per batch → 41% train/eval mismatch 🔴

**What was wrong.** `normalize_cond_per_batch` z-scored the history-summary conditioning over
`(batch, sequence)`:

```python
m = cs.mean(dim=(0, 1), keepdim=True)      # <- reduces over the BATCH axis
v = cs.var(dim=(0, 1), keepdim=True, unbiased=False)
```

So **a window's conditioning depends on which other windows share its batch.** Training
batches are shuffled (batch statistics ≈ global statistics); eval/test batches are sequential
and strongly correlated, so subtracting the batch mean strips out precisely the shared
structure that distinguishes windows.

**Measured distortion** (same windows, different batch composition):

| context | rel-err vs global | cosine |
|---|---|---|
| sequential batches (**eval**) | 0.388 | 0.933 |
| shuffled batches (**training**) | 0.214 | 0.979 |
| **eval vs training** | **0.411** | **0.920** |

The denoiser learned a mapping on one input distribution and was scored on another — which is
why conditioning was **actively harmful**, and harmful *increasingly* as the model came to
rely on it (see §3).

**The fix.** `COND_NORM_MODE = "sample"` (default): z-score each window over its own sequence
axis only, so the conditioning is a function of that window alone and train/eval agree **by
construction**. `"batch"` preserves legacy behaviour. Flag `--cond-norm-mode`, tags the
denoiser dir `_cn-<mode>`.

**Verified effect** (K=1, p32, 1500 epochs, seed 0):

| | pred_gap | MSE_cond vs uncond @ t=900 | R²_cond @ t=900 |
|---|---|---|---|
| batch (legacy) | 0.215 | 0.395 vs 0.222 — **harmful** | −0.001 |
| **sample (fixed)** | 0.025 | **0.312 vs 0.327 — helpful** | **+0.210** |

The fix does what it claims. It is **not sufficient** on its own — see §3.

**Follow-up (2026-07-27): the mode is now persisted per checkpoint.** As first landed, the
mode was read live from `config`, unlike every other fix — so loading a *pre-fix* checkpoint
evaluated it under `"sample"` while it had been trained under `"batch"`, re-creating the very
mismatch this fix removes, in reverse. It affected the G2/G3 checkpoints, all H2 checkpoints,
`finetuning/results/v1`, and the released checkpoints in `ldt/checkpoints/`.

Closed by persisting `cond_norm_mode` in the checkpoint's `model_config` **alongside**
`cond_adapter` (it is a pipeline setting, not a `LLapDiff` constructor kwarg — it governs how
the summarizer's output is normalised before the denoiser consumes it, so it cannot be passed
to `LLapDiff(**kwargs)`). `build_llapdiff_model` stamps the resolved mode onto the model — from
the checkpoint when rebuilding one, else from the live config — and
`_resolve_cond_norm_mode(diff_model)` prefers that over the global config at every call site.
`_cond_norm_mode_from_checkpoint` returns `"batch"` when the key is absent, so pre-fix
checkpoints keep what they were trained under. Verified on five real pre-fix checkpoints
(G3 chirp cells, `v1` tuning trials, a released x0 checkpoint): all resolve to `"batch"`.
The two remaining `normalize_cond_per_batch` call sites now honour the resolved mode too
(`_build_cond_summary`, used only by the `evaluate_irregular_time_checks` diagnostic, and
`build_context(norm=True)`, which gained an explicit `norm_mode` defaulting to legacy
`"batch"`; every in-tree caller passes `norm=False`).
Pinned by `test_cond_norm_mode_is_persisted_and_defaults_to_legacy`.

---

## 1b. Which stages need retraining (checked 2026-07-27)

**Stage 1 (latent VAE) and stage 2 (history summarizer) do NOT need retraining for any of
these fixes.** Only stage-3 denoiser checkpoints do.

| evidence | result |
|---|---|
| `normalize_cond_per_batch` / `build_context` / `cond_summary` in `trainers/train_val_latent.py` | **no matches** — the VAE never sees conditioning |
| same in `trainers/train_val_summarizer.py` | **no matches** — its objective is history reconstruction (`SUM_LOSS_W_X/V/T`), independent of downstream normalisation |
| where the fixes live | `models/laptrans.py` (pole cores, stage 3), `models/llapdiff_utils.py` (normalisation applied *after* the summarizer), `viz/plot_llapdiff_poles.py` (analysis only) |
| `SUM_FT_MODE` default | `"none"` ⇒ the summarizer is **frozen** in stage 3; denoiser gradients never reach it |
| diffusion precompute cache (`diffusion_cache.py`) | stores `summary_raw` with `norm=False` — normalisation happens at load time ⇒ **existing caches stay valid** |

B15 changes *how the summarizer's output is consumed*, not the summarizer's weights.

⚠️ **The one exception:** with `SUM_FT_MODE` set to `pool`/`top`/`all`, stage 3 fine-tunes the
summarizer *through* the conditioning path — a summarizer fine-tuned under the old
normalisation would then be mismatched and would need redoing. Not active at the default.

*(Separately, B5 — the `--phase-spread` fix — **did** require regenerating the H2 synthetic
caches and their stage-1/2 artifacts, but that is a data-side fix specific to the H2 benchmark
and was completed in 2026-07; it does not apply to physionet or the other real datasets.)*

---

## 2. Bugs fixed earlier in the project (pre-session; from the archived docs)

### B1 — `to_coeffs` zero-init + squared ⇒ exactly zero gradient 🔴 *(2026-07-05)*
`d(a²)/dW = 2a·h = 0` at `a = 0`, so the pole-coefficient head **never trained**. Every chirp
checkpoint before 2026-07-05 had frozen, constant, condition-independent poles — all
chirp-vs-LTI comparisons from those runs are meaningless.
**Fix:** ε-init (std 1e-4) + permanent regression guard
`test_chirp_coeffs_receive_gradient_at_init`.
**Audit:** dead head ⇒ `sd['model.chirp_field.to_coeffs.1.weight'].abs().max()` is exactly 0.

### B2 — Recovery ranked modes by coefficient variation, not output contribution 🔴 *(2026-07-20)*
With K=256 modes and a ~1-D target, zero-residue modes get no gradient, their pole
coefficients drift freely, and the old criterion selected **exactly those** — the published
figures showed ρ ≈ 87–118/step and ω pinned at 0.9π for modes whose output-energy share was
exactly **0.0**.
**Fix (P1–P10):** `modal_capture` hook at the final DDIM step → rank by output contribution
`E_k = mean_t e^{−2ρ̄_k}‖θ_k‖²` → report the E-weighted effective trajectory over **all** modes;
recovery runs for **both** arms; stratified windows; `--recovery-share-threshold` validity gate
with a "SELECTION INVALID" watermark.
**Diagnostic verdict at the time:** the model was recovering fine — only the figure lied.

### B3 — Stale-checkpoint trap 🔴 *(2026-07-21)*
`_best_raw.pt` is written only when `EMA_COMPARE_EVERY > 0` (the benchmark sets 0), the
trainer reported it via a bare `.exists()`, and `_train_or_reuse_stack` preferred
`best_checkpoint_raw` first — so **11 of 30 runs** (all `synthetic_linear_chirp`, 5/6
`synthetic_quadratic_chirp` — exactly the two ω-sweep tasks Fig 2 needs) were **evaluated on a
6-day-old checkpoint** while the run's own model sat unused.
**Fix:** the trainer reports only checkpoints written by the current run (mtime guard +
`written=` filter on `_select_eval_checkpoint_path`); the tool takes the trainer's own
`loaded_checkpoint`.
**Audit:** compare the `checkpoint` column's mtime in the raw CSV against the run time.
⚠️ `llapdiff-checkpoint-eval --checkpoint-kind auto` still searches **raw first** — the same
trap for manual evals.

### B4 — Stage-1/2 artifacts reused across incompatible caches 🔴 *(2026-07-21)*
Artifact dirs are keyed by `(task, seed)` with **no cache identity**, so adding
`--sweep-period` created new data but silently reused the previous cache's VAE + summarizer.
**Fix:** an `artifact_cache.json` fingerprint that fails loudly on mismatch. Pass
`--recompute-artifacts` (or point `--artifact-root` somewhere fresh) when the data changes.

### B5 — VAE entity mean-pool cancels per-entity phase 🔴 *(2026-07-21)*
`latent_vae.py` mean-pools across the entity axis before `mu_head`, while the generator drew
`phase0 ~ U(0, 2π)` per entity — so 64 differently-phased sinusoids cancel and the carrier
never reached the latent. This imposed a **0.62 round-trip ceiling on every H2 result** up to
that date. Ruled out by measurement: input dropout (0.620→0.586), latent width 24→64
(0.586→0.591), fewer entities (8 entities: 0.479, worse).
**Fix:** narrow the phase draw — `--phase-spread 0.393` (π/8) → round-trip **0.857**,
`val_recon` 3.2× better, entity diversity retained.
**Gate 0** (`llapdiff-stage1-roundtrip`) now checks this before any Fig-2 claim: stage-1
round-trip corr must exceed ~0.85 on the run's own cache.
🔴 **The mechanism was never fixed, only side-stepped.** `--phase-spread` narrows the
*generator's* per-entity phases so the panel stops cancelling; the mean-pool is untouched. On
**real** datasets that escape hatch does not exist, and the same bottleneck binds directly —
see **B20**, where the round-trip on every real cell equals that dataset's cross-entity
mean share, and three of five cells fail Gate 0 because of it.

### B6 — Horizon-blind ρ initialisation 🔴 *(2026-07-22, both cores)*
Both cores initialised `ρ ~ U(0.01, 0.2)` with **no horizon term**. The synthesis envelope is
`e^{-ρ̄(t)}`, so survival is decided by **ρ·H**, not ρ:

| horizon | ρ·H at init | consequence |
|---|---|---|
| h = 12 (PhysioNet — where it was plausibly tuned) | 0.12 – 2.4 | healthy |
| h = 48 (H2 benchmark) | 0.5 – 9.6 | most modes dead |
| **h = 168 (NOAA-UK headline)** | **1.7 – 33.6** | **nearly all modes dead** |

Measured on trained h=48 models: every mode at a real signal frequency had ρ ≈ 0.08–0.25,
i.e. **26–76× the ground truth (0.0032)**, envelope mass 0.03–0.12 — extinguished within a few
steps. Only near-DC modes survived, holding up to 92% of output energy. **Both arms affected
equally**, which is why they persistently tied.
**Why training never fixed it:** with a true ρ·H of 0.15 the data contains almost no decay
signal, so ρ is a *null direction* of the loss — recovered ρ values simply span the init range.
A weakly-identified parameter must be initialised at the right **scale**.
**Fix:** anchor to the horizon, `ρ ~ U(0.1, 2.0)/H` ⇒ ρ·H ∈ [0.1, 2]. New `pole_init_horizon`
kwarg threaded `config.PRED → _llapdiff_model_kwargs → LLapDiff → LapFormer →` **both** cores.
**Trap recorded:** the obvious anchor `chirp_time_scale` is *chirp-gated*
(`_resolve_chirp_time_scale` returns `None` for LTI), so using it would have silently fixed
chirp only and biased the headline comparison.
**Guarded by** `tests/test_pole_init_horizon.py`.
**Audit:** modes need **ρ ≲ 2/H** (0.042 at h=48, 0.012 at h=168).

### B7 — ρ variation had no upper cap 🔴 *(2026-07-22)*
Fixing the init was necessary but not sufficient. ω is bounded above by a Nyquist rescale;
**ρ had no upper bound at all**, only positivity. Since `a²` is squared it can only push ρ
*up*, so training inflated ρ freely until the mode's own envelope extinguished it — recreating
the B6 failure by a second route (learned ρ reached 0.987, envelope mass 0.00).
**Fix:** cap ρ with the same smooth rescale used for ω, at `ρ_max = rho_max_scale / H`
(`CHIRP_RHO_MAX_SCALE`, default 4). Closed-form antiderivative preserved; rescale ≈1 at
ε-init; bounding ρ from *above* does not affect Theorem B, which needs only ρ ≥ ρ_min > 0.
**Symmetry worth remembering:** ω is capped by **Nyquist** (what the sampling rate can
resolve); ρ is capped by **envelope survival** (what the horizon can carry). Both are physical
limits of the observation window, and the code had only one of them.

### B8 — ρ variation on a mean-1 basis 🟠 *(2026-07-22)*
The legacy `φ = 1 + cos` expansion has `mean_t φ = 1`, so **any** ρ variation also raises the
mean decay by ~Σa² — damping every oscillatory mode to extinction when the true decay is small.
**Fix:** `CHIRP_RHO_BASIS="centered"` uses `φ − 1 = cos`, whose integral over the window is
exactly zero, so variation contributes **no net decay** (`ρ̄(L) = ρ_floor·L`). Positivity is
then enforced by a headroom rescale (`Σa² ≤ floor − ρ_min` ⇒ ρ ∈ [ρ_min, 2·floor]) instead of
by the sign of the basis. *(B10 is this same fix for ω, applied two days later.)*

---

## 3. Open issues

### B19 — PhysioNet h=12 gives the diffusion stage **13 training windows** 🔴 **OPEN — invalidates every PhysioNet statistical claim**

*Found 2026-07-30 while diagnosing why the U3 latent correlation was ~0 in every arm.*

Measured directly on the loaders the trainer builds (`prepare_eval_stack`, physionet,
h=12, `WINDOW=24`, `PRED=12`, `split_policy="contiguous"`, ratios 0.7/0.1/0.2):

| split | batches | **windows** | entities per window |
|---|---|---|---|
| train | 3 | **13** | 445 |
| val | 1 | **5** | 445 |
| test | 1 | **5** | 445 |

Compare the headline dataset, noaa_uk h=168: **16 286 / 2 183 / 4 702** windows. PhysioNet
h=12 has a **~1250× smaller** training set.

**The 445 entities do not help.** The set-VAE is permutation-invariant over entities and
pools them into one latent per window, so the denoiser's input/output space is
`mu_norm` of shape `[13, 12, 16]` — **2 496 training scalars against 11.5 M parameters
(≈4 600× over-parameterized)**. The loader yields the *identical* 13 windows every epoch
(verified: batch checksums are bit-identical across two passes), so there is no
re-sampling to compensate.

**Root cause, not a code defect.** PhysioNet is split over *patient-relative* time
(`split_scope="physionet_patient_relative_time"`); records span ~48 h, and `WINDOW + PRED
= 36` leaves only ~23 valid start offsets in total. This is a property of the dataset
geometry at this horizon, but it is documented nowhere and the runbook uses PhysioNet
h=12 for load-bearing decisions.

**Observed consequences (seed 0, x0, arm d):**
- train loss **8.23 → 0.075** (memorized) while the val latent x0 MSE never beats the
  trivial baseline: **2.76 vs 2.16 predict-zero**; like-for-like latent x0 MSE is
  **4.3–5.7** across all U2/U3 arms with correlation ≈ **−0.05**.
- Data-space CRPS nevertheless looks healthy (~0.25 val) because the set-VAE decoder
  reconstructs each of the 445 entities largely from its own embedding — CRPS is
  dominated by a per-entity prior, not by the forecast. Disabling conditioning entirely
  costs only 0.2606 → 0.2990.

**What this invalidates.** Every PhysioNet h=12 number is a **5-window** estimate:
- the G2/G3 2×2 factorial (§1 of the runbook) — cells (a)–(d) at 0.3706–0.3719, i.e.
  deltas of 0.0001–0.001, and the "(c) ≈ (d) kill-shot" read;
- the §8 branch decision, which was taken on cell (a) = 0.3710;
- the 22-trial `fixed_bugs` tuning campaign, whose incumbent was chosen on a 0.0014 gap,
  and the `seed8` campaign;
- the "chirp arms show ~6× lower seed variance" unplanned finding;
- the U2/U3 gates and any U3 table computed on PhysioNet.

**Recommendation.** Treat PhysioNet h=12 as a **plumbing smoke test only** — it is
genuinely useful for that (a full 3-stage U3 seed costs ~20 min). Move every statistical
claim, the G3 factorial, and the §8 branch decision to noaa_uk h=168, which the plan
already designates as the headline. If PhysioNet must carry a claim, the window count has
to be raised first (shorter `WINDOW`/`PRED`, or a split that treats patients rather than
relative-time offsets as the sampling unit — the latter is a design change to the
set-VAE's entity pooling, not a config tweak).

**Update 2026-08-01 — B19 is not PhysioNet's only defect.** Gate 0 now runs on real datasets
(**B20**) and PhysioNet h=12 scores a stage-1 round-trip of **0.059** on val against a 0.85
gate. Two separate things are wrong: its cross-entity mean share is only **0.171**, so the
pooling ceiling alone fails the gate before any window count matters; and it is the one cell
that falls *below* its own ceiling on val (0.059) while sitting exactly on it on train
(0.173) — which is this bug, the five-window val split. Every statistical claim on this cell
is therefore blocked twice over, upstream and downstream of the denoiser.

*Relationship to the "denoiser barely uses its conditioning" issue below: on PhysioNet
those observations are fully explained by 13-window memorization. A conditioning-shuffle
probe shows the denoiser **does** respond to its conditioning (39 % output change under a
shuffled summary, 67 % with none), so it is not ignoring the input — it simply cannot
generalize from 13 examples. The conditioning issue should be re-measured on noaa_uk
before being treated as a modelling defect.*

---

### B20 — The stage-1 round-trip ceiling is the cross-entity **mean** share 🔴 **OPEN — preliminary (2026-08-01)**

*Found 2026-08-01, immediately after `llapdiff-stage1-roundtrip` gained a `--dataset-key/--pred`
mode and Gate 0 could be run on a real dataset for the first time. **This is a first-pass
measurement (2 batches per cell) and has not been run at full split coverage** — the mechanism
below is well supported, but treat the exact numbers as provisional.*

**Gate 0 on all five campaign cells** (val split, threshold 0.85, 2 batches, EMA-irrelevant —
stage 1 only):

| cell | round-trip corr | std ratio | n | verdict |
|---|---|---|---|---|
| **noaa_uk h168** | **0.922** | 0.90 | 120 | **PASS** |
| **bms_air h168** | **0.941** | 0.98 | 235 | **PASS** |
| physionet h12 | **0.059** | 0.05 | 110 (266 constant windows skipped) | **FAIL** |
| crypto h100 | 0.657 | 0.75 | 1 124 | **FAIL** |
| us_equity h100 | 0.674 | 0.70 | 2 210 | **FAIL** |

#### The mechanism: the decoder emits the panel mean

`latent_vae.py` mean-pools the entity axis before `mu_head`, so one window becomes **one**
latent with **no entity axis** — this is B5, but here it is measured on real data, where the
data-side escape hatch B5 used (`--phase-spread`, redrawing the generator's per-entity phases)
does not exist. Diagnostic probe (same 2 val batches; window-eligibility filters differ
slightly from the tool's, hence small differences from the table above):

| cell | ent/win | VAE params | train corr | val corr | corr(recon, panel mean) | cross-entity std retained |
|---|---|---|---|---|---|---|
| noaa_uk h168 | 4 | 0.80 M | 0.975 | 0.922 | **1.000** | 0.059 |
| bms_air h168 | 7 | 0.81 M | 0.975 | 0.908 | **0.948** | 0.538 |
| crypto h100 | 113 | 0.82 M | 0.517 | 0.657 | **0.995** | 0.218 |
| us_equity h100 | 221 | 0.83 M | 0.541 | 0.674 | **0.992** | 0.201 |
| physionet h12 | 75 | 0.86 M | 0.173 | 0.059 | 0.306 | 0.907 |

Every entity's reconstruction correlates ≈ 0.99–1.00 with the **cross-entity mean of the
targets**, while retaining only ~20 % of the cross-entity variation (5.9 % on noaa_uk). The
decoder is reproducing the panel average and discarding each entity's deviation from it.

**The consequence is quantitative, and it closes the case.** If the reconstruction *is* the
panel mean, then the round-trip correlation must equal the data's own entity-to-panel-mean
correlation — a property of the **dataset**, with no model in it. Measured directly on the
targets (no VAE involved):

| cell | corr(y_entity, panel mean) | shared variance share | **Gate-0 round-trip** |
|---|---|---|---|
| noaa_uk h168 | **0.922** | 85.1 % | **0.922** |
| bms_air h168 | 0.894 | 80.3 % | 0.908 |
| crypto h100 | **0.657** | 50.6 % | **0.657** |
| us_equity h100 | 0.678 | 48.0 % | 0.674 |
| physionet h12 | 0.171 | 37.0 % | 0.059 (train: 0.173) |

⇒ **Gate 0 is not measuring VAE quality on these cells. It is measuring how much of each
entity's target is shared across the panel**, and the pooling architecture pins the VAE to
exactly that ceiling. noaa_uk (4 co-located UK weather stations, 85 % shared) and bms_air
(7 sensors, 80 %) pass because the panel mean nearly *is* each entity's signal. Crypto and
us_equity fail because financial targets are ~50 % idiosyncratic and the panel mean — a market
factor — cannot carry the other half.

#### Capacity is NOT the cause — explicitly ruled out

The obvious hypothesis is that the failing cells are over-parameterized. Three independent
disproofs:

1. **The denoiser is not in the loop.** Gate 0 instantiates `LatentVAE` and nothing else
   (`tools/run_stage1_roundtrip.py`; the file contains no reference to `LLapDiff`,
   `LapFormer` or `laptrans`). The 10.2–11.8 M-parameter backbone of the B19 discussion is
   stage 3 and never runs here.
2. **The VAEs are the same size everywhere** — 0.80–0.86 M parameters across all five cells,
   an 8 % spread. There is no larger model on the failing cells.
3. **The failing cells UNDERFIT, they do not overfit.** Over-parameterization predicts
   train ≫ val. Measured train ≤ val, stably across sample sizes:

   | cell | batches | train | val |
   |---|---|---|---|
   | crypto | 2 / 8 / 20 | 0.517 / 0.522 / 0.518 | 0.657 / 0.632 / 0.617 |
   | us_equity | 2 / 8 / 20 | 0.541 / 0.591 / 0.587 | 0.674 / 0.672 / 0.643 |

   The VAE cannot reconstruct its **own training windows** on those cells (0.52 vs the 0.975
   it reaches on noaa_uk/bms_air). That is a structural bottleneck, not excess capacity.
   *(Splits are chronological, so a small train sample is regime-specific; the 20-batch rows
   are there to show the ordering is not a sampling artefact.)*

#### PhysioNet is a second, distinct failure on top of the ceiling

PhysioNet is the only cell that falls **below** its own pooling ceiling, and only on val:

- Its panel-mean ceiling is **0.171**. On **train** it achieves **0.173** — i.e. exactly the
  ceiling, like every other cell. On **val** it collapses to **0.059**.
- Its `corr(recon, panel mean)` is only **0.306** and its cross-entity std ratio is **0.907** —
  the opposite signature to crypto/us_equity. Its decoder produces output that varies across
  entities but matches neither the entity nor the panel: it is not reproducing anything.
- The train/val gap is the natural reading of **B19**: five val windows. This is the one place
  in the table where an over-parameterization argument does apply — but to the *val estimate*,
  not to the reconstruction mechanism.

So PhysioNet carries **three** compounding defects: the B5 pooling ceiling (0.171 — already
below the 0.85 gate before anything else happens), the B19 five-window val split, and the
degeneracy below.

#### ⚠️ PhysioNet's number needed a tool fix to be readable at all

The first run reported `corr = nan`, `std_ratio = 206413`. Not a numerical blow-up — `mu` and
the reconstruction are finite throughout. Cause: **71 % of PhysioNet's val entity-windows are
constant** (median true target std exactly **0**; sparse clinical series carried forward).
`np.corrcoef` on a constant series is undefined, `roundtrip_correlation` aggregated with a
plain `np.mean`, and a single `nan` destroyed the aggregate while near-constant windows
(true std ~1.5e-08, above the 1e-8 floor) inflated the std ratio to ~1e6. Synthetic
benchmark targets are never constant, which is why this never surfaced before real datasets
were admitted. Constant and undefined windows are now dropped and counted (`n_degenerate`),
and the filter is a verified no-op where nothing is degenerate (noaa_uk and bms_air score
identically before and after).

**Audit rule:** read `n` and the skipped count, never `corr` alone. A cell scored on a
minority of its split is telling you about that minority.

#### What this affects

- **Gate 0 as a gate still works and its verdicts stand**: noaa_uk and bms_air — the Phase-1
  and Phase-2 datasets of `CMD_UQ_RECOVERY_PLAN.md` — pass, so the campaign's chosen cells are
  sound and the plan does not need re-routing.
- **It independently corroborates Phase 0** from a different direction: the three cells the
  window audit rejected on supervision volume are the same three that fail here on stage-1
  representational capacity.
- **It raises the bar for any future finance cell.** Any crypto/us_equity result is bounded by
  a ~0.66 stage-1 round-trip, i.e. the denoiser is being asked to forecast a latent that cannot
  represent half the target. A "ties baseline" outcome there is not attributable to the model.
- **It does not by itself invalidate published numbers**: nothing in the paper currently rests
  on crypto or us_equity, and PhysioNet is already retired by B19.

#### Not yet established (why this entry is marked preliminary)

- Full-split coverage; everything above is 2 batches per cell (20 for the train/val check).
- Whether an entity-conditioned decoder path recovers the idiosyncratic component.
  `VAE_ENTITY_CONDITION` is **True** and the decoder does receive an entity embedding, yet the
  cross-entity std collapses anyway — that tension is unexplained and is the first thing to
  look at.
- Whether the 0.85 threshold is the right gate for a panel whose shared share is genuinely
  ~50 %. Arguably the honest gate is "round-trip ≈ panel-mean ceiling" (is the VAE at its
  structural limit?) *plus* a separate check that the ceiling is high enough to be worth
  forecasting. Those are two different questions and the current single number conflates them.

---

### B21 — The conditioning pipeline discards most of the forecastable signal 🔴 **OPEN — preliminary (2026-08-01)**

*Found 2026-08-01 while running Phase 1a (the ridge positive control) of
`CMD_UQ_RECOVERY_PLAN.md` on noaa_uk h=168. **Preliminary**: one cell, one seed, val split,
linear probes only.*

> ## 🔴 Read this first — the central inference below is weaker than originally written
>
> Every number in this entry comes from a **linear** probe, and a later experiment in the
> same session shows linear decodability is **not** a valid proxy for information the
> denoiser can use. `SUM_FT_MODE="all"` produced the **best forecast of any run**
> (val CRPS 0.3240 vs 0.3627 control) while making the conditioning **less** linearly
> decodable (`cond_summary` → raw target 6.8 % vs 7.7 % frozen; → latent **1.4 % vs 5.7 %**).
>
> So a representation can carry *more* usable signal and *less* linear signal at the same
> time. A ridge probe measures a **lower bound** on usable information, not its quantity.
>
> **What survives:** the measurements themselves (raw history is far more linearly decodable
> than the summary; the level component that z-scoring discards is more linearly decodable
> than the part it keeps). Those are facts about linear decodability.
>
> **What does not survive as stated:** "the summarizer discards ~½ the forecastable signal".
> The summarizer is a deterministic function of the history, so it can only lose or *reformat*
> — and the `SUM_FT_MODE="all"` result is direct evidence that reformatting is happening. The
> gap is consistent with loss, with nonlinear re-encoding, or with both, and this entry cannot
> separate them. Treat the headline as **"the conditioning is much less linearly decodable
> than the raw history"**, which is what was actually measured.

#### The measurement

Ridge from a frozen representation of the history to the h=168 target, fit on 14 446 train
windows, evaluated on 2 183 val windows, alpha selected on a chronological holdout **inside
train** (val never touched), baseline = `baseline_rmse_predict_mean` exactly as
`llapdiff-uq-eval` computes it. **Temporal resolution matched at 32 of 336 positions for
every feature set**; the target is the per-entity raw temperature trajectory:

| input to ridge | dim | relative RMSE reduction |
|---|---|---|
| **raw history, temperature channel only** | **128** | **27.3 %** |
| `cond_summary` (the denoiser's conditioning) | 8 192 | 14.2 % |
| `cond_summary_raw` | 8 192 | 13.5 % |

**128 raw numbers beat 8 192 dimensions of summarizer output by ~2×.** Giving the summarizer
64× more dimensions at the same temporal resolution recovers only about half the signal that
is sitting in the raw history.

Signal remaining after each stage of the pipeline:

| stage | linear headroom |
|---|---|
| raw data (history → target) | **27.3 %** |
| after the summarizer (stage 2) | ~14 % |
| after the VAE, i.e. the latent the denoiser must predict (stage 1) | **5.6 %** |

⚠️ Those three are measured in different spaces and at different feature dimensions, so this
is an ordering, **not** a clean decomposition. At matched 2 048 features the stage-1 step is
raw 7.7 % → latent 5.6 %.

#### Stage 2 — two mechanisms, both by design

1. **The training objective is not the downstream task.** The summarizer optimises *history
   reconstruction* (`SUM_LOSS_W_X/V/T`, `trainers/train_val_summarizer.py`). Nothing pushes it
   to preserve the component that predicts the *future*. It is not broken; it is solving a
   different problem, and the pipeline then treats its output as if it were a sufficient
   statistic for forecasting.
2. **Per-window z-scoring erases the level — and the erased part is the more informative
   half.** `normalize_cond_per_batch(mode="sample")` reduces over `dims=(1,)`, so per window
   and per feature dim it subtracts the mean over tokens and divides by the std over tokens.
   Measured on noaa_uk: the across-window std of the *per-window mean* of `cond_summary` is
   **exactly 0.00000** (`cond_summary_raw`: 0.04092) — the level is projected out, and no
   linear combination of the normalised tokens can recover it, ever.

   Restoring **exactly** what the normalisation removes (the per-window mean and std over the
   token axis, 2 × 256 numbers):

   | features | dim | reduction |
   |---|---|---|
   | `cond_summary` (level erased) | 2 048 | 7.7 % |
   | + erased MEAN restored | 2 304 | 9.3 % |
   | + erased MEAN and STD restored | 2 560 | 9.3 % |
   | `cond_summary_raw` (never normalised) | 2 048 | 10.8 % |
   | **the erased MEAN alone** | **256** | **12.4 %** |

   **The 256 numbers thrown away are more predictive on their own than the entire 2 048-dim
   normalised summary that is kept.** (The combined rows sit *below* the mean alone because
   ridge has one shared alpha, and the optimal regularisation for 256 dense dims differs from
   that for 2 048 sparse ones — a limitation of the probe, not of the information.)

   🔴 **Correction to an earlier draft of this entry.** It said `cond_summary_raw` is also
   handed to the denoiser, "so the model is not fully blind". **That is wrong at the default
   configuration.** `LapFormer._select_summary_tokens` (`lapformer.py:461`) returns
   `cond_summary_raw` only when `use_raw=True`, and both gates — `COND_POOL_USE_RAW` and
   `ANALYSIS_QK_USE_RAW` — default to **False** and are not set in `config.py`. So the
   denoiser never sees the raw summary, and the level component reaches it by **no path at
   all**.

   This does **not** retract B15, which fixed a real batch-dependence bug. The problem is the
   remedy chosen: `"sample"` removes the batch dependence *and* the level. A third option
   achieves both — **global** normalisation with fixed statistics computed once over train:
   batch-independent by construction (B15's requirement) and level-preserving. That mode does
   not exist; the code offers only `"batch"` and `"sample"`.

   ⚠️ **`SUM_FT_MODE` cannot address this.** The z-scoring is a projection applied *after* the
   summarizer, so whatever the fine-tuned encoder emits, the normalised output still has a
   per-token-axis mean of exactly zero. The stage-2 fine-tuning experiment tests mechanism (1)
   only.

   **`COND_POOL_USE_RAW=True` tested 2026-08-01 — it makes things worse, and the failure is
   informative.** Matched pair, noaa_uk h=168 arm d seed 0, 60 epochs, campaign eval profile,
   differing in exactly this one flag (verified in the checkpoint at
   `model_config["llapdiff"]["pole_pool_use_raw_summary"]`):

   | | e5 | e30 | e60 | best val CRPS |
   |---|---|---|---|---|
   | control | 0.4483 | 0.4096 | **0.3627** | 0.3627 (e60) |
   | `COND_POOL_USE_RAW=True` | **0.4128** | 0.4400 | 0.4459 | 0.4128 (**e5**) |

   It **stalls rather than diverges**: train loss plateaus at ~0.88 from epoch 20 while the
   control keeps descending to 0.626 (1.01 → 0.83 → 0.75 → 0.626). So the flag impairs
   *optimisation itself*, not just generalisation.

   ⚠️ **This does not falsify mechanism (2) — it falsifies this delivery mechanism.** Two
   reasons. First, the treated run's epoch-5 CRPS (0.4128) is the *best* early value of any
   run including the control (0.4483), which is what you would expect if the level genuinely
   helps before something else dominates. Second, `cond_summary_raw` is **unnormalised**:
   measured std 0.574 overall, with per-window per-dim stds ranging 0.05–0.43. Feeding that
   straight into the pole conditioning vector is a scale problem, and the pole field is
   exactly where a badly-scaled input does most damage (it sets ρ and ω).

   Note what the flag does and does not reach: it gates `make_pole_cond` (→ ρ(t), ω(t)) and
   `make_block_cond`, but the latter is inert (`block_summary_adaln=False`) and the trunk's
   main cross-attention over the summary tokens is **not gated at all**. So this tested the
   pole path only.

   **`COND_NORM_MODE="global"` implemented and tested 2026-08-01 — it works, and it confirms
   the scale diagnosis.** Fixed statistics computed once over the train split (per hidden
   dim, reduced over windows and tokens: 5 472 096 rows, mean |·| 0.0896, std 0.5190),
   persisted in the checkpoint and reloaded at eval. Batch-independent (so B15's requirement
   still holds), level-preserving, and unit-scaled.

   | run (60 ep, seed 0, matched) | best val CRPS | vs control |
   |---|---|---|
   | `ctrl_e60` — `"sample"`, the current default | 0.3627 | — |
   | `rawpole_e60` — level restored, **bad scale** | 0.4128 | **+0.050 (worse, stalls)** |
   | **`globalnorm_e60`** — level restored, **unit scale** | **0.3344** | **−0.028** |

   It is better than the control at **11 of 12 evals**, and dramatically so early
   (e5: 0.3665 vs 0.4483). The three rows together are the cleanest result in this entry:
   *the same information* helps or hurts depending only on the scale it arrives at.

   ⚠️ **One seed.** −0.028 is below the 0.036 nondeterminism band of §4 retraction 1, so the
   endpoint alone would not carry the claim; the 11/12 trajectory and the early-epoch gap are
   what make it credible. **Not yet a recommendation to change the default** — that needs
   multi-seed confirmation and a second dataset.

   Unlike `SUM_FT_MODE="all"` (below), this intervention is **campaign-compatible**: it leaves
   stage 1/2 frozen and shared across arms, so it does not violate the parity requirement that
   every cross-arm comparison depends on.

   **The probe under `"global"` (2026-08-02) reproduces the dissociation at the top of this
   entry, in the opposite direction to the CRPS.** Exact A/B — raw summaries collected once,
   both normalisations applied offline, same summarizer, same windows, same honest alpha:

   | conditioning | dims | → latent | → raw target |
   |---|---|---|---|
   | `"sample"` | 2 048 / 8 192 | 5.6 % / 4.2 % | 7.7 % / 14.2 % |
   | `"global"` | 2 048 / 8 192 | 4.9 % / **6.8 %** | **12.3 % / 16.4 %** |
   | raw history | 128 | **11.6 %** | **27.3 %** |

   `"global"` clearly improves the **raw-target** readout (+4.6 points at 2 048) and improves
   val CRPS, yet the **latent** readout barely moves. Two consequences: the latent is a bad
   yardstick for conditioning quality (reinforcing the top box), and the raw history still
   beats every summarizer variant by ~1.7× on the raw target with 64× fewer features — so
   `"global"` narrows the B21 gap without closing it.

   This is what blocks Phase 1a of `CMD_UQ_RECOVERY_PLAN.md`: the raw history clears that
   gate's 10 % threshold (11.6 %) while no conditioning variant does (best 6.8 %).

#### Stage 1 — a second loss that Gate 0 cannot see

The VAE's *latent coordinates* are harder to predict than the signal they encode:

| target | reduction |
|---|---|
| VAE round-trip reconstruction of the raw targets | **9.6 %** |
| the latent `mu_norm` itself | **5.6 %** |

The VAE does **not** destroy forecastability in data space — it slightly *improves* it (9.6 %
vs 7.7 % for the raw targets), consistent with **B20**: it discards the idiosyncratic,
high-frequency component, which is also the least predictable one. But what the denoiser is
asked to regress is the latent code, and that is the least predictable quantity in the table.
Reconstruction quality and latent predictability are different properties, and the same
reconstruction can be reached through a smooth or a scrambled code.

🔴 **Gate 0 is blind to this.** noaa_uk **passes Gate 0 at 0.922** while handing the denoiser a
latent on which a linear readout of its own conditioning gets 5.6 %. Gate 0 measures
*reconstruction*; nothing in the plan measures *latent predictability*. A stage-1 health check
that a pipeline can pass while being unfit for its downstream purpose is not a sufficient gate.

#### What this changes

- **The Phase-1a verdict must be read carefully.** By the letter of the pre-registered rule
  (ridge from the frozen summarizer tokens, < 10 % ⇒ uninformative ⇒ move datasets), noaa_uk
  h=168 **fails**: 6.2 % honest / 9.8 % under a leaky oracle-alpha upper bound. But 1a's stated
  *purpose* is "establish whether the target is forecastable **at all** on this dataset", and
  by that test the answer is clearly **yes** — 27.3 % from the raw history. **1a as written
  conflates dataset quality with summarizer quality.**
- **Do NOT move the gate to bms_air on the strength of 1a.** The failure branch assumes the
  cause is a dataset lacking temporal structure. Here it is not: the same summarizer would
  travel to the new cell and most likely reproduce the result. This supersedes the earlier
  recommendation in this session.
- **It sits upstream of the open "denoiser barely uses its conditioning" issue** (§3). That
  entry compares a ridge probe on `cond_summary` (R² +0.747) against the trained denoiser
  (weak/negative) and concludes the denoiser cannot exploit its conditioning. B21 shows
  `cond_summary` is *itself* impoverished relative to the raw history, so there are two losses
  in series, not one.

#### Not yet established

- ~~Whether `SUM_FT_MODE` recovers any of the gap.~~ **Tested 2026-08-01 — essentially null.**
  noaa_uk h=168, arm d, seed 0, `SUM_FT_MODE="pool"`, 60 epochs, campaign eval profile.
  Frozen and fine-tuned summarizers probed in the same process, same windows, same ridge
  procedure, so the encoder is the only difference:

  | probe (2 048 summary dims) | frozen | fine-tuned |
  |---|---|---|
  | `cond_summary` → RAW target | 7.7 % | **8.6 %** |
  | `cond_summary` → LATENT target | 5.7 % | **5.2 %** |
  | raw history → RAW target (reference) | 27.3 % | 27.3 % |

  The weights genuinely moved (`max |fine-tuned − frozen| = 5.9e-02`, so this is not a no-op),
  but the representation gained ~1 point on the raw target, lost ~0.5 on the latent, and got
  nowhere near the 27.3 % reference. **It does not close the B21 gap.**

  Read this as weak evidence, not a refutation of mechanism (1). Four reasons: `"pool"`
  unlocks only **131 776 of 5 234 592** summarizer parameters (**2.5 %**) — `"top"` (5.2 %) and
  `"all"` (100 %) are untested; one seed, 60 epochs; the gradient reaching the summarizer is
  the diffusion loss routed through the conditioning path, which is exactly the path the open
  §3 issue says is weak, so a null result may reflect a weak training signal rather than an
  unimportant mechanism; and it **cannot** touch mechanism (2) by construction.

  **Update: a matched control now exists** (the `COND_POOL_USE_RAW` pair supplied one — same
  60 epochs, same seed, same profile, defaults elsewhere), so the CRPS *is* readable after
  all: **0.3467 (`pool`) vs 0.3627 (control) = −0.016**, and the treated run sits below the
  control at every eval from epoch 30 onward (0.3745/0.3714/0.3648/0.3817/0.3666/0.3602/0.3467
  vs 0.4096/0.4065/0.4009/0.3936/0.3827/0.3694/0.3627).

  ⚠️ **Do not promote this to a finding on one seed.** §4 retraction 1 records two
  *byte-identical* configs on this codebase producing CRPS 0.2243 vs 0.1886 — a 0.036 swing
  from `cudnn.benchmark` nondeterminism alone, i.e. **more than twice this effect**. The
  consistent 7-eval offset is worth more than the endpoint, but it is still n=1. It also
  points the opposite way to the representation probe above (which moved −0.5 points on the
  latent), so mechanism (1) remains genuinely unresolved.

  ⚠️ Running this at all required fixing **B22** first: `SUM_FT_MODE` was silently discarding
  the fine-tuned summarizer, so the experiment would have been unevaluable.

  **`SUM_FT_MODE="all"` tested next — and it changes the conclusion. Mechanism (1) is
  supported, by CRPS, with a dose-response.** One control, three treatments, all 60 epochs,
  seed 0, arm d, same eval profile, differing in one flag:

  | run | fine-tuned params | best val CRPS | vs control | non-finite grad skips |
  |---|---|---|---|---|
  | `ctrl_e60` | 0 | 0.3627 | — | 0 |
  | `sumft_pool_e60` | 131 776 (2.5 %) | 0.3467 | −0.016 | 0 |
  | **`sumft_all_e60`** | **5 234 592 (100 %)** | **0.3240** | **−0.039** | 0 |

  **Monotone in the fine-tuning surface** — 0 → 2.5 % → 100 % of the summarizer gives
  0.3627 → 0.3467 → 0.3240. A three-point ordered dose-response is materially stronger than
  any single pairwise delta, and the `all` effect (−0.039) exceeds the 0.036
  nondeterminism band of §4 retraction 1. Stability concerns did not materialise: zero
  non-finite gradient skips in all three runs, including full fine-tuning at
  `SUM_FT_LR_MULT=0.1`.

  🔴 **But the representation probe moves the OPPOSITE way.** On the `all` summarizer,
  `cond_summary` → raw target falls to **6.8 %** (frozen: 7.7 %) and → latent to **1.4 %**
  (frozen: 5.7 %). The best-forecasting encoder is the least linearly decodable one. So
  fine-tuning is **not** adding linearly-readable signal; it is reshaping the representation
  into something this denoiser can consume. That is why the box at the top of this entry
  retracts the "discards ~½ the signal" framing.

  Still one seed per cell, 60 epochs, one dataset. The dose-response is what carries the
  claim, not any single run. `SUM_FT_MODE="all"` is **not** thereby recommended for the
  campaign: it makes the summarizer arm-specific, which breaks the shared-frozen-stage-1/2
  parity requirement that every cross-arm comparison in this project depends on (§1b).
- Whether the loss is in the summarizer's **architecture** or its **objective**. **Two
  architectural explanations are now eliminated (2026-08-02, CPU-only, no model touched):**

  1. **Not a dimensional bottleneck.** Input is `[N=4, K=336, F=5]` = **6 720** numbers per
     window; `cond_summary` is `[S=336, H=256]` = **86 016**. The summary is **12.8× larger
     than its own input**, so nothing is lost to compression.
  2. **Not entity pooling** — on this cell. `LaplaceAE.forward` step 5 aggregates across
     entities exactly as the VAE does (`encoded_mean = (encoded_bn * entity_weight).sum(dim=1)
     / entity_denom`), so the pipeline mean-pools the entity axis **twice**, at stage 2 and
     again at stage 1. But it costs almost nothing here:

     | history representation | dims | reduction |
     |---|---|---|
     | per-entity — what the summarizer is given | 128 | **27.3 %** |
     | entity-mean — what it keeps after pooling | 32 | **25.3 %** |
     | `cond_summary` — what comes out | 2 048 | **7.7 %** |

     Entity pooling costs **2 points**; the remaining ~17 points are lost downstream of it.
     Consistent with B20: noaa_uk's 4 stations share 85 % of their variance, so averaging
     them discards little. ⚠️ **This is cell-specific** — crypto (113 entities, 50 % shared)
     and us_equity (221, 48 %) would pay far more, and the double pooling is worth
     re-measuring there before it is dismissed.

  ⇒ What remains is the **encoder + query-pooling + objective**, and both surviving strands of
  evidence point at re-encoding rather than destruction: the `SUM_FT` dose-response (better
  CRPS as more of the summarizer is fine-tuned) and the `SUM_FT="all"` inversion (best CRPS,
  *worst* linear probe). So "1.7× is lost" is very likely the wrong frame — see the box at the
  top of this entry — and the defensible statement remains "not linearly accessible".
- Whether feeding raw history features to the denoiser alongside the summary closes the gap.
- Replication on a second cell, a second seed, and with a nonlinear probe. The denoiser is
  nonlinear with cross-attention over all 336 tokens and may extract more from `cond_summary`
  than ridge does — though the comparison is fair in that both feature sets received identical
  linear treatment, and the winner needed 128 dimensions against 8 192.

---

### B22 — `SUM_FT_MODE` silently discards the fine-tuned summarizer 🔴 *(found & fixed 2026-08-01)*

**Found** while setting up the B21 follow-up experiment — `SUM_FT_MODE` is the one existing
knob that directly tests B21's mechanism, and it turned out not to be evaluable.

**What was wrong.** With `SUM_FT_MODE` ∈ {`pool`, `top`, `all`} the stage-3 trainer unfreezes
part of the summarizer and trains it through the conditioning path
(`_select_summarizer_finetune_named_params`, `train_val_llapdiff.py:1948`). But:

- `_checkpoint_payload()` stored `model`, `ema`, `optimizer`, `mu_mean/mu_std` and two
  `summ_ft_*` counters — **no summarizer state**. Verified on a real checkpoint: the only
  matching keys are `summ_ft_skipped_nonfinite_grad_steps` and `summ_max_nonfinite_grad_steps`.
- The trainer **reads** `config.SUM_CKPT` (line 2058) and never writes it. `grep torch.save`
  over the whole trainer returns exactly one hit, and it is not the summarizer.
- `_load_stack` unconditionally reloaded the summarizer from `cfg.SUM_CKPT`
  (`llapdiff_checkpoint_eval.py:381`) — the **original shared frozen artifact**.

So the denoiser was trained against a fine-tuned conditioning encoder and scored against a
different one, with no warning; and the fine-tuned weights were destroyed at process exit.
This is the same family as **B15** (train/eval conditioning mismatch) and **B18** (raw/EMA
mismatch): the training and evaluation paths disagree about which artifact is in force.

**Scope of the damage: none, so far.** `SUM_FT_MODE` defaults to `"none"` and no campaign in
this repo has enabled it, so no existing result is affected. It was a trap waiting for the
first person to turn the knob on — which B21 makes likely.

**The fix.** Persist and prefer, mirroring what B15's follow-up did for `cond_norm_mode`:

- `_checkpoint_payload` adds `payload["summarizer"]` and `payload["sum_ft_mode"]`, **guarded
  by `if sum_ft_named_params`** — these state dicts run 21 MB (noaa_uk) to 1.1 GB
  (us_equity), so the default path must not carry one.
- `_load_stack` uses `payload["summarizer"]` when present and otherwise falls back to
  `cfg.SUM_CKPT`. Absent the key — every pre-fix checkpoint and every `SUM_FT_MODE="none"`
  run — behaviour is byte-identical to before. Verified on a real checkpoint.

**Guarded by** `tests/test_eval_protocol_split.py::test_finetuned_summarizer_is_persisted_and_preferred_at_eval`
(a source-level pin on both call sites, including the ordering; it does not exercise a real
fine-tuning run).

⚠️ **Still open around this knob**, per §1b: a summarizer trained under the old
normalisation and then fine-tuned *through* the conditioning path is an untested combination.
And `finetuning/`'s trial pruning keeps only `_best_ema.pt`, which now carries the summarizer —
worth watching for disk growth on large-summarizer datasets.

---

### B16 — EMA checkpoints are identical to raw; EMA is never applied at eval 🔴 **OPEN**

`llapdiff_pred-*_best.pt` and `_best_ema.pt` contain **identical model weights** — verified:
sum of per-tensor `max|raw − ema|` over all parameters = **exactly 0.0**, both stamped
`epoch=635`.

`_checkpoint_payload()` writes the *raw* model into both files; they differ only in **which
validation metric selected the epoch**, not in which weights are stored. The actual EMA
weights live under a separate `ema` key that `_load_stack` never applies.

**Impact — scoped (corrected 2026-07-27).** This bites only the consumers that go through
`_load_stack`, which reads `payload["model"]` and never applies `payload["ema"]`:
`llapdiff-checkpoint-eval`, the H2 chirp benchmark, and the T1/T4/UQ tools. For those,
**the reported CRPS is a raw-weight number labelled EMA** despite `USE_EMA_EVAL = True` —
including the G2/G3 factorial and the 5-seed Fig-2 campaign.

**Opt-out coverage (2026-08-01).** Three consumers now carry `--weights {raw,ema}`, which
applies the shadow via the alias-proof `_apply_ema_weights`: `llapdiff-checkpoint-eval`
(B18), `llapdiff-uq-eval`, and — added 2026-08-01 — **`llapdiff-u1-sweep`**, whose absence
would have frozen the campaign's guidance choice on raw weights. Measured on the
`fixed_bugs` physionet incumbent at one fixed sampling cell: **val CRPS 0.2673 (raw) vs
0.2593 (EMA)**, i.e. 0.008 — larger than that campaign's entire 0.0014 tuning gain. All
three still **default to `raw`** so previously published cells stay reproducible, so the
underlying defect is unchanged and this entry stays OPEN. `llapdiff-t1-poles`,
`llapdiff-t4-timing` and the H2 benchmark remain raw-only.

**The `finetuning/` harness is only partly affected.** `eval_sampling.py` — the selection
path — applies the EMA shadow correctly: it copies `payload["ema"]` onto the live model
parameters before scoring (and marks a cell `skipped_no_ema` when the shadow is missing), so
selection numbers tagged `weights=ema` really are EMA. ⚠️ **`final_eval.py` did NOT** — see
**B18**; an earlier version of this paragraph claimed it did, and that was wrong.

**Suggested fix:** either write the EMA weights into `model` when saving `_best_ema.pt`, or
have `_load_stack` apply `payload["ema"]` when `USE_EMA_EVAL` is set — the second matches what
`finetuning/eval_sampling.py` already does and would make the whole codebase consistent. Add a
test asserting the two files differ when EMA is enabled.

### B18 — `final_eval.py` scored a raw/EMA hybrid 🔴 *(found & fixed 2026-07-29)*

**What was wrong.** `LapFormer.__init__` passes `self.analysis` into
`LaplacePseudoInverse(self.analysis, …)`, which keeps it as `self.encoder`. So
`model.analysis.*` and `model.synthesis.encoder.*` are the **same tensors**:
`state_dict()` lists both names (202 keys), `named_parameters()` de-duplicates (168).
`EMA` shadows `named_parameters()`, so the shadow carries only the `model.analysis.*`
name. `final_eval.py` materialised its EMA checkpoint key-wise:

```python
for name, tensor in shadow.items():
    model_state[name] = tensor      # updates model.analysis.*, not the alias
```

leaving `model.synthesis.encoder.*` at its **raw** value — and since the alias is applied
later by `load_state_dict` (index 29 vs 2), raw won for all 23 aliased tensors. The
evaluated model was EMA everywhere except the **entire Laplace analysis encoder**
(`comp_emb`, `pole_embedding`, `time_key_proj`, `attention`, `out_proj`, `_rho_raw`,
`_omega_raw`), which was un-averaged.

**Impact.** Measured on physionet h=12 arm d, seed 0, same checkpoint and sampling cell:

| | forecast CRPS |
|---|---|
| hybrid (what `--phase final` scored) | 0.3577 |
| true EMA | **0.3472** |

**0.010 CRPS, biased high — larger than the entire tuning gain over defaults (0.0014).**
Every `--phase final` number with `weights=ema` produced before this fix is void. Selection
scores are unaffected (`eval_sampling.py` copies onto live parameters, where aliasing is a
non-issue because it is one object).

**The fix.** Stop materialising a file. `llapdiff-checkpoint-eval` gained
`--weights {raw,ema}`; `evaluate_checkpoint` applies the shadow via
`_apply_ema_weights(diff_model, payload)` over `named_parameters()` after `_load_stack`,
which is alias-proof by construction. `final_eval.py` passes `--weights ema`. The CLI
default stays `raw`, so the published reference numbers are untouched. `weights` and
`generator_seed` are now recorded in every result payload. Verified: the fixed path
reproduces the independent `eval_sampling.py` score to **2.9e-06**.

**Audit recipe.** Any code that transplants an EMA shadow into a `state_dict` by key is
suspect whenever a module is reachable under two names. Copy onto `named_parameters()`
instead, or compare `len(state_dict())` against `len(list(named_parameters()))` — here 202
vs 168 was the tell.

**Two smaller estimator artefacts found alongside** (not bugs, but they bound what a CRPS
delta can mean at this scale): `evaluate_regression` draws its 200 CRPS comparison pairs from
the **global** RNG, so a cell's score depends on its position in a multi-cell sweep (~0.0015);
and `evaluate_checkpoint` left the DDIM generator unseeded (~0.0003), now controllable with
`--generator-seed`.

---

### B17 — `llapdiff-synthetic-regime` crashes at its own default geometry 🟡 **OPEN**

With `boundary_crossing` defaults (series 288, window 96, horizon 48) the
`global_purged_horizon` split leaves a val band of ~15 window-starts, but a val window needs
its full 48-step target interval inside the band ⇒ val is **structurally empty** and the run
dies with `ValueError: target-interval purged split produced an empty train/val/test split`.
The `strict_unseen_regime` defaults (432/373) have the same problem.

**Two constraints for a working geometry:** (i) `val_ratio · (series_length − window −
horizon + 1) > horizon`; (ii) the change point must sit a full `--lookback-steps` **inside**
the test region.
**Verified workaround:** `--series-length 768 --change-point 606` (12 crossing windows/asset).
Sanity-check any other geometry with `--validate-split-only` first.

### The denoiser barely uses its conditioning ⚠️ **OPEN — not a localised bug**

Measured on the H2 benchmark (K=1, p32, converged):

| quantity | R² on the true latent trajectory |
|---|---|
| ridge probe on the model's **own** `cond_summary` | **+0.747** |
| the trained denoiser's 25-draw ensemble mean | **−0.30** (before B15) / weak (after) |

A *linear readout* of the model's own inputs massively beats the trained model. At `steps=1`
(the one-shot conditional mean, where `x_T` carries no information) the output had **std 0.09
vs the target's 0.62** and corr ≈ 0 — i.e. it predicted the *marginal* mean.

**The target is sound:** the true latent carrier sits at dominant ω = **0.360** against a
ground truth of **0.362** (single-sinusoid R² 0.54, the expected value for a chirp). Stage 1
and the benchmark data are fine.

Fixing B15 moved R²_cond from −0.001 to **+0.210**, so conditioning stopped being harmful —
but 0.21 is still far below the 0.747 ceiling.

⇒ **Poles cannot track because the FORECAST does not track.** This single upstream failure
explains every H2 observation: flat pole slopes, CRPS pinned at 0.19–0.20 across *all* arms /
K / basis / ρ settings, and insensitivity to every intervention.

**Hypothesis for the remainder (untested):** conditioning reaches the trunk only via
cross-attention into K modal tokens whose residues come from `x_t`; at high noise those tokens
are noise-dominated, so the conditioning must overwrite them through residual updates — a weak
path by construction. `block_summary_adaln` is currently **False**; enabling AdaLN injection
is the natural next experiment.

---

## 4. What these bugs invalidate — results ledger

**Every H2 / Fig-2 conclusion in this project's history is suspect**, because the denoiser
barely conditions (§3). In particular the "confirmed negative" below never actually tested the
CMD tracking claim.

| result | status |
|---|---|
| Any chirp checkpoint trained **before 2026-07-05** | **void** (B1: frozen poles) |
| Any H2 recovery figure **before 2026-07-20** | **void** (B2: junk-mode selection) |
| Any H2 result on a **2π phase-spread** cache | **void** (B5: 0.62 round-trip ceiling) |
| Any H2 result **before 2026-07-22** | **void** (B6/B7: oscillatory modes dead) |
| The **5-seed "confirmed negative"** (slope −0.0032 ± 0.0113) | **retracted** (B9: target unrepresentable; B12: ρ₀ 2×) |
| **CRPS** from `_load_stack` tools (checkpoint-eval, H2 benchmark, T1/T4/UQ) | labelled EMA, actually raw weights (B16); `finetuning/` results are genuinely EMA |
| **NOAA-UK h=168 headline** | never run — `ldt/output/noaa_uk/` does not exist |
| **G2/G3 PhysioNet h=12 factorial** | *probably safe* from B6 (ρ·H ∈ [0.12, 2.4]); still affected by B16 (it used `llapdiff-checkpoint-eval`) |

**Retracted claims (kept so they are not re-derived):**
1. *"chirp beats LTI by 18% CRPS"* — **retracted**. Single-run CRPS is untrustworthy: the
   benchmark trains with `cudnn.benchmark=True`, and two byte-identical LTI configs gave
   CRPS 0.2243 vs 0.1886 — a **0.036 swing from pure nondeterminism, larger than every
   chirp-vs-LTI margin ever quoted**. Any CRPS claim needs multi-seed.
2. *"chirp has ~12× lower variance"* — **retracted**, a comparison error: those numbers came
   from runs differing in *config*, not seed. At fixed config across seeds the variance ratio
   is **0.98×** (equal). The earlier G3 "~6× lower seed variance" was PhysioNet h=12 and gets
   no support at h=48 — do not cite it for H2.
3. *"the model hedges to DC because future phase is uncertain"* — **retracted twice**: first
   superseded by B6 (decay was killing the modes), then again by B11 (my own headroom trap
   made hedging and inability-to-sweep the same event).
4. *"LTI is spectrally static"* — **false** (B14): the LTI arm moves its instantaneous mean
   frequency by 59% of the truth's swing via differential decay.

**Preserved positive results (still believed):**
- **G2/G3 PhysioNet h=12, 5 seeds** (2026-07-05): (a) lti+head 0.3710 ± 0.0117,
  (b) lti−head 0.3719 ± 0.0125, (c) chirp+head 0.3706 ± 0.0024, (d) chirp−head **0.3709 ±
  0.0018**. Reads: (d) ≈ (a) exactly ("delete the hack, keep the accuracy"); (c) ≈ (d) (the
  head is redundant once poles vary); (d) > (b) *not* observed at h=12 (expected — the
  headline regime is h=168). This decided the §8 branch: **≈0.36 reproduction gap ⇒
  "match + certify + strictly generalize"** (the 0.320 anchor is retired; the released
  checkpoint reference 0.367 *is* reproduced).
- **Gate 0** (stage-1 round-trip) passes at **0.857** on `--phase-spread 0.393` caches.
- The VAE latent carrier **matches** the data-space truth ω (0.360 vs 0.362 measured
  2026-07-25; 0.48 vs 0.467 measured earlier), and truth poles explain 83–92% of latent
  variance ⇒ the Fig-2 target is fair and well-posed.
- **Basis representability after B9/B10/B11**: R² 0.03 → **0.99** on the real ground truth.
- **Prop. A.1**: companion-form integration error ~4.6 vs normal-form ~2e-11.

**A measurement discipline note.** The Fig-2 pole tracking slope is **noise-dominated at
single seed**. Observed aggregates across configurations: **+0.170, −0.035, +0.069, +0.054**,
with per-window values from **−0.44 to +0.79**. With 4 windows × 3 draws these are not
distinguishable from each other or from zero. **Do not read any single-run slope as signal.**

---

## 5. Methodological traps and audit recipes

**Before believing any H2 result, check in this order:**

1. **Checkpoint freshness** (B3) — compare the `checkpoint` column's mtime in
   `chirp_benchmark_raw.csv` against the run time.
2. **Artifact identity** (B4) — confirm `artifact_cache.json` matches the cache in use.
3. **Gate 0** (B5, B20) — `llapdiff-stage1-roundtrip` must exceed ~0.85 on the run's *own*
   cache. It also runs on a real cell (`--dataset-key <ds> --pred <H>`, scored on val).
   ⚠️ Compare the result against that dataset's **cross-entity mean share**, not against 0.85
   alone: the mean-pooled VAE is pinned to that ceiling (B20), so a "failure" may be the panel
   being idiosyncratic rather than the VAE being bad. Read `n` and the skipped-window count
   too — PhysioNet scores on 29 % of its val windows, the rest being constant.
4. **Modes alive** (B6/B7) — oscillating modes with `envelope_mass < 0.1` are dead; learned ρ
   more than ~10× the expected decay is a red flag; recovered ρ spanning exactly the init
   range means ρ is unidentified and the init is doing all the work.
5. **Poles can drift** (B9) —
   ```python
   rho, omega = field.instantaneous(cond, t_rel)   # t_rel spanning [0, L]
   net = omega[:, -1] - omega[:, 0]
   # legacy basis: net == 0 to float precision, and omega.argmax(dim=1) is 0 or T-1
   ```
   Red flags in a recovery figure: every ρ (or ω) curve starting at its maximum; curves
   returning to their initial value at the window edge; a recovered trajectory that ripples at
   the basis frequencies but has no trend against a truth that clearly trends.
6. **The model actually conditions** (§3) — denoise the same `x_t` with and without
   `cond_summary`; if `MSE_cond ≈ MSE_uncond` at high t, the denoiser learned nothing
   conditional and *no* pole conclusion is meaningful.

**Estimator caveats** (do not re-derive these the hard way):
- A single-sinusoid periodogram **cannot** fit a chirp (R² = 0.33 even on clean truth); a
  chirp-aware fit (ω₀ + dω/dt) reaches R² = 0.68.
- A *linear* chirp fit mis-reads a window straddling a sweep turning point (it reports
  dω/dt ≈ 0 for a V-shaped ω). Match the fit to the sweep geometry.
- Always use **stratified** windows; consecutive windows have almost no target variation, so
  any cross-window tracking slope computed on them is meaningless.
- `forecast_freq_*` is computed on the raw 24-dim latent and is **noise-fooled** — it reads
  RMSE ≈ 1.4 against a truth whose entire range is 0.49. Calibrate against the true latent
  before drawing any conclusion from it.

**Design rule from §11 of the ρ report (still standing).** The pole head is conditioned on the
history summary and the diffusion timestep **alone**:
```
cond_vec   = make_pole_cond(t_vec, cond_summary, cond_summary_raw)   # history only
rho, omega = chirp_field.integrated(cond_vec, t_rel)
```
Observations enter through `x_t`, **never** through `cond_vec`. So recovered pole trajectories
are a pure *extrapolation from the conditioning window*, and sampling-time inpainting cannot
help (measured: free −0.002 vs imputed −0.009). Pole recovery can only succeed when the
history determines the pole trajectory over the horizon — which is why `--sweep-period 32`
(3 sweep cycles in the history) exists, versus `96` (exactly one cycle, phase not inferable).

---

## 6. Configuration knobs introduced by these fixes

| knob | default | CLI flag | dir tag | fixes |
|---|---|---|---|---|
| `CHIRP_BASIS` | `"half_integer"` | `--chirp-basis` | `_cb-` | B9 |
| `CHIRP_OMEGA_BASIS` | `"centered"` | `--chirp-omega-basis` | `_ob-` | B10 |
| `CHIRP_RHO_BASIS` | `"centered"` | `--chirp-rho-basis` | `_rb-` | B8 |
| `CHIRP_RHO_MAX_SCALE` | `4.0` | `--chirp-rho-max-scale` | `_rmax-` | B7 |
| `COND_NORM_MODE` | `"sample"` | `--cond-norm-mode` | `_cn-` | B15 |
| `pole_init_horizon` | `config.PRED` | — (automatic) | — | B6 |
| `--phase-spread` | `2π` (use `0.393`) | `--phase-spread` | cache `_phase-` | B5 |

**Back-compat rule applied throughout:** every new model kwarg `setdefault`s to the *legacy*
value in `_llapdiff_config_from_checkpoint`, so pre-fix checkpoints rebuild in the function
class they were trained in and still load under `strict=True`.

## 7. Test map

| test file | pins |
|---|---|
| `tests/test_chirp_basis_harmonics.py` | B9, B10, B11, B12, B13 |
| `tests/test_pole_init_horizon.py` | B6 |
| `tests/test_pole_recovery.py` | B2, B14 |
| `tests/test_chirp_modal.py` | B1 (`test_chirp_coeffs_receive_gradient_at_init`), bounds, LTI-equivalence |
| `tests/test_stage1_roundtrip.py` | B5 / Gate 0; B20 (constant-window filter, dataset-mode guards) |
| `tests/test_synthetic_chirp_benchmark.py` | B3, B4 |
| `tests/test_chirp_uq.py` | B16 (`--weights` on uq-eval and u1-sweep), mean-only reads without a UQ head |

Run: `python -m pytest tests/ -q` → **411 passing** (see `CLAUDE.md` for the venv +
`PYTHONPATH` preamble; the `conda activate llapdiff` line is from the original environment).

---

*Diagnostic scripts used for the 2026-07-23…25 findings (probe, conditioning gap, target
inspection, representability) are under
`~/brain/.copilot2/jobs/aff6d405/tmp/`. Figures: `ldt/results/fig2_diagnosis.pdf`,
`ldt/results/fig_representability.pdf`.*
