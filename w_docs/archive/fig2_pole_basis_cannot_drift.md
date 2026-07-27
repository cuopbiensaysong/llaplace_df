# The Pole Basis Cannot Drift — Root-Cause Report

*Found 2026-07-23 while re-examining the Fig-2 "confirmed negative". The 5-seed campaign
measured whether the model **learns** a frequency sweep that its pole basis could not
**represent**. Affects the chirp core only, at every horizon.*

---

## TL;DR

`ChirpModalField` builds its basis as

```python
freqs = torch.linspace(1.0, float(self.num_basis), self.num_basis)   # f_m = 1..M
phi   = 1.0 + torch.cos(2 * pi * f_m * t / L)
a_omega2 = c[:, 1].pow(2)                                            # >= 0, squared
omega(t) = omega_floor + sum_m a_omega2[m] * phi_m(t)
```

Every `f_m` is a **whole number of cycles across the window**, so `phi_m(0) = phi_m(L) = 2`
is each basis function's maximum. With nonnegative coefficients this forces, for every mode
and every conditioning:

- **`omega(0) == omega(L)`** exactly — measured max |difference| = 2e-7 over 512 modes
- **`sup_t omega(t)` is attained only at the endpoints** — `argmax` in {0, T-1}, always

The instantaneous frequency can only **dip and return**. A monotone sweep, or a peak inside
the window, is not in the function class at all. Amplitude is not the limit — the class can
swing 2.06 rad/step; it simply cannot *end* anywhere but where it started.

The H2 ground truth is a triangle sweep, which within a window is exactly that: monotone,
or a single interior turning point, with unequal endpoints.

## 1. How much of the target this excludes

Fitted over 120 stratified test windows of the sweet-spot cache, using the model's **own**
`_basis()` matrix and its **actual** constraint (nonnegative coefficients on `phi` plus a
free floor, by projected-gradient NNLS — the headroom rescale in `_coeffs` only shrinks the
reachable set further):

| basis | R² on the real ground-truth omega(t) |
|---|---|
| `integer` (f = 1..M, all signs +1) — **every Fig-2 run** | **0.119 ± 0.266** |
| `half_integer` (f = 0.5, 0.5, 1.0, 1.0, … with ± signs) | **0.990 ± 0.008** |

Net endpoint drift `|omega(end) − omega(start)|` = 0.271 rad/step = **73% of the
within-window range** — and that component is identically zero in the legacy class.

(An earlier estimate of 0.351 for the legacy basis allowed *signed* coefficients. The real
model has squared ones, so 0.119 is the honest figure.)

## 2. Why it stayed hidden through five benchmark designs

- The five designs — fm-2, fm-4, sweet-spot, period-96, period-32 — varied sweep
  **steepness** and **period**, never the **shape class**. They shared one confound, so
  their agreement looked like robustness.
- The LTI arm's `omega_trend_slope` is 0 **by construction** (constant `omega_k` ⇒ constant
  `omega_eff`), so the contrast arm never looked wrong. All 10 LTI rows read `0.0000 ±
  0.0000` — a tautology, not a measurement. See §4.
- CRPS ties, because a **static superposition of constant-frequency modes** approximates a
  chirp perfectly well over 48 steps. Nothing downstream looked broken.
- The signature was in the figures the whole time: in
  `*_pole_recovery.pdf`, every recovered rho curve starts at its maximum, dips, and returns
  — the fingerprint of nonnegative coefficients on a basis whose members all peak at t=0.

## 3. The fix

Pair every harmonic with its negative and admit half-cycles:
`phi_m(t) = 1 + s_m cos(2 pi f_m t / L)`, with `f = {0.5, 0.5, 1.0, 1.0, 1.5, 1.5, ...}`
and `s = {+1, -1, +1, -1, ...}`.

- A nonnegative combination of `(1 + cos)` and `(1 - cos)` spans a constant plus a **signed**
  multiple of `cos`, so **no signed-coefficient machinery is needed**. At f = 0.5 that gives
  monotone sweeps in either direction; at f = 1, V and Λ shapes.
- **Every guarantee is preserved.** `phi` is still in [0, 2] and `|phi - 1| <= 1`, so the
  omega Nyquist rescale and both rho headroom rescales in `_coeffs` are unchanged.
  `Phi_m = t + s_m sin(2 pi f_m t/L)/(2 pi f_m/L)` keeps Theorem A closed-form; rho >= rho_min
  keeps Theorem B; eps-init still starts at the LTI poles.
- **Zero-mean over the window survives**, because `2*f_m` is an integer ⇒ `sin(2 pi f_m) = 0`.
  This is what the `centered` rho basis needs (`rho_bar(L) = rho_floor * L`, no net decay).
- Trade-off: the top frequency drops from M to M/4 cycles across the window (8 → 2 at M=8).
  That is the right trade — the observed pathology was capacity spent on high-frequency
  ripples — and M stays sweepable via `--chirp-num-basis`.

Exposed as `CHIRP_BASIS` (default `"half_integer"`; `"integer"` reproduces the legacy class)
and `--chirp-basis` on the benchmark tool, tagging the denoiser dir `_cb-<mode>`.
Pre-fix checkpoints `setdefault` to `"integer"`, so they rebuild in the class they were
trained in. `ChirpModalField(basis=...)` also defaults to `"integer"` for the same reason.

Two hardcoded assumptions had to go with it, both from `phi_m(0) = 2`:
`seed_poles` (the seeded analysis poles) and `_growth_terms` (`g_0`, and the sign in
`phi_prime`). Pinned by `tests/test_chirp_basis_harmonics.py`.

**Also fixed in passing:** `seed_poles` computed rho0 with the `nonneg` convention even
under `rho_basis="centered"`, so the seeded analysis poles were **2x** the poles that
actually synthesized the output. That inconsistency arrived with the (uncommitted) centered
basis and applied to the entire final 5-seed campaign.

## 4. Companion metric defect (separate, smaller)

`modal_contributions` weights modes by `envelope_mass`, which is **already** `mean(dim=1)`
over t. So `omega_eff(t)` is a *fixed convex combination* of the per-mode `omega_k(t)` and
moves only if the modes themselves sweep. Consequences:

1. For the LTI core `omega_eff` is constant and its slope is identically 0 — the "LTI fails
   structurally" panel contains no evidence either way.
2. A model that produces a swept spectrum by **redistributing energy across
   constant-frequency modes** (differential decay) scores 0 as well.

Route 2 is not hypothetical. Reconstructed from the saved per-mode summaries of the 5-seed
campaign, the **LTI** arm moves its instantaneous mean frequency by **0.188 ± 0.142** — 59%
of the truth's 0.321 swing, and up to 94% in seeds 2 and 4 — versus 0.023 for chirp. Its
per-mode rho spread is 11.3x (median 6.8x) against chirp's 1.9x.

⇒ **"LTI is spectrally static" is false and must not be claimed.** `omega_eff_dyn` /
`omega_trend_slope_dyn` (instantaneous weights `E_k(t) = e^{-2 rho_bar_k(t)} ||theta_k||^2`)
are now recorded alongside the static ones so both arms are measurable.

## 5. Scope beyond Fig 2

The constraint is horizon-independent: at h=168 the chirp core likewise cannot express any
net frequency drift across the forecast. For a method whose claim is time-varying poles,
that is substantive. Like the rho-init fix, it lands **before** the H1 headline campaign is
spent (no `ldt/output/noaa_uk/` exists — H1 has never been run).

No theorem is affected: the method doc requires only "a nonnegative basis with closed-form
antiderivatives" (§4, "e.g. squared random-Fourier features or RBFs"). Integer cycles were
an implementation choice, not a theoretical requirement.

## 6. What this does NOT fix

Representability is **necessary, not sufficient**. The §11 obstacle in
`pole_rho_init_horizon_bug.md` is separate and unaffected: the pole head is conditioned on
the history summary and the diffusion timestep alone, so the pole trajectory over the
horizon is a pure extrapolation and the sweep's turning point must be **inferable from the
context window**. The pilot runs both `--sweep-period 96` (the baseline config, a
one-variable comparison) and `--sweep-period 32` (3 sweep cycles in the history) precisely
to separate "cannot represent" from "cannot infer".

A flat slope after this fix is still a real negative — a much stronger one.

## 7. Audit recipe

For any chirp checkpoint, ask whether its poles can drift at all:

```python
rho, omega = field.instantaneous(cond, t_rel)      # t_rel spanning [0, L]
net = omega[:, -1] - omega[:, 0]
# legacy basis: net == 0 to float precision, and omega.argmax(dim=1) is 0 or T-1
```

Red flags in a recovery figure: every rho (or omega) curve starting at its maximum; curves
that return to their initial value at the window edge; a recovered trajectory that ripples
at the basis frequencies but has no trend against a truth that clearly trends.

---

*Related: `w_docs/pole_rho_init_horizon_bug.md` (the rho init/cap findings and the §11
history-only conditioning result), `w_docs/fig2_signature_figure_strategy.md` (campaign plan
and gates), `w_docs/CMD_SESSION_NARRATIVE.md` (running log).*
