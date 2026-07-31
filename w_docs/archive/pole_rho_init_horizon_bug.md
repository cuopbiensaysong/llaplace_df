# Horizon-Blind ρ Initialisation — Root-Cause Report

*Found 2026-07-22 while investigating why the H2 signature figure (Fig. 2) would not
separate the chirp and LTI arms. Hypothesis raised by the user ("ρ(t) is very large
compared to the ground truth, it leads to flat trajectories"), then confirmed against
trained checkpoints and the source. Affects **both** the chirp and LTI cores.*

---

## TL;DR

Both modal cores initialise the pole decay rate from a **hardcoded, horizon-independent**
range `ρ ~ U(0.01, 0.2)`. The synthesis envelope is `e^{-ρ̄(t)}`, so the quantity that
decides whether a mode survives the forecast is **ρ·H**, not ρ. At long horizons the
legacy range puts most modes far past the point of extinction:

| horizon | ρ·H at init | consequence |
|---|---|---|
| h = 12 (PhysioNet — where this was plausibly tuned) | 0.12 – 2.4 | healthy |
| h = 48 (H2 chirp benchmark) | 0.5 – 9.6 | most modes dead |
| **h = 168 (NOAA-UK headline)** | **1.7 – 33.6** | **nearly all modes dead** |

Measured on trained h=48 models: every mode sitting at a real signal frequency had
**ρ ≈ 0.08–0.25, i.e. 26–76× the ground truth (0.0032)**, with energy-envelope mass
0.03–0.12 — extinguished within a few steps of a 48-step horizon. The only survivors were
near-DC modes. The model output therefore collapses toward a flat/DC trajectory, which is
exactly the "flat recovered pole trajectories" and "forecast hedges to DC" symptoms.

**Both arms are affected equally**, so every chirp-vs-LTI comparison to date compared two
equally crippled models — the most likely explanation for why they persistently tied.

---

## 1. Symptom

Across every H2 configuration tried (fm-2 and fm-4 difficulty, LR 1.5e-4 and 5e-4,
under- and fully-converged, K ∈ {8, 32, 256}), the same picture recurred:

- recovered pole trajectories `ω_eff(t)` were **flat** — trend slope ≈ 0 against a truth
  that genuinely swept;
- the decoded forecast correlated only **~0.25** with the target and looked damped;
- **chirp and LTI tied** on CRPS, and LTI's constant-pole recovery was often *better*;
- the dominant output mode was consistently **near-DC** (ω ≈ 0.02–0.04) holding up to
  **92%** of the modal output energy.

The natural reading — which I initially adopted — was "the model hedges to DC because
future phase is uncertain, so an MSE-optimal forecast is damped." That reading is wrong,
or at best a second-order effect. The real cause is upstream.

## 2. Mechanism: ρ·H is the quantity that matters

The modal synthesis is

```
z(t) = Σ_k e^{-ρ̄_k(t)} [ c_k cos ω̄_k(t) + b_k sin ω̄_k(t) ],     ρ̄_k(t) = ∫₀ᵗ ρ_k(s) ds
```

For a roughly constant ρ, the **amplitude** envelope across a horizon H is `e^{-ρH}` and
the **energy** envelope (what the E_k contribution ranking uses) is `e^{-2ρH}`. Define

```
envelope_mass = mean_t e^{-2ρ̄(t)}  =  (1 - e^{-2ρH}) / (2ρH)      (constant ρ)
```

| ρ (H = 48) | ρ·H | envelope_mass | mode status |
|---|---|---|---|
| 0.0032 (truth) | 0.15 | 0.861 | alive |
| 0.01 | 0.48 | 0.643 | alive |
| 0.05 | 2.4 | 0.207 | dying |
| 0.10 | 4.8 | 0.104 | dying |
| 0.20 | 9.6 | 0.052 | dead |
| 0.50 | 24 | 0.021 | dead |

A mode with ρ·H ≫ 1 contributes only at the very start of the horizon and is numerically
gone thereafter. It cannot carry an oscillation across the window, no matter how correct
its frequency ω is. **Getting ω right is useless if ρ has already killed the mode.**

## 3. Evidence from trained models

Ground truth for the chirp tasks: `ρ = 0.0032` (constant), so `ρ·H = 0.15` — the true
signal barely decays over the horizon (envelope keeps 0.86).

Measured from the recovery JSONs (`selected_modes`, first stratified window), splitting
modes into those at real signal frequencies (ω > 0.15) and near-DC ones:

| run | arm | oscillating modes: ρ | × truth | envelope_mass | status |
|---|---|---|---|---|---|
| fm-2 converged | **lti** | 0.083, 0.113 | 26–35× | 0.100, 0.071 | dead |
| fm-2 converged | **chirp** | 0.130, 0.175 | 41–55× | 0.061, 0.044 | dead |
| fm-4 LR 5e-4 | **lti** | 0.130, 0.116 | 36–41× | 0.061, 0.069 | dead |
| fm-4 LR 5e-4 | **chirp** | 0.104, 0.133, 0.119 | 33–42× | 0.072, 0.051, 0.056 | dead |

And the survivors, in every case, are near-DC:

| run | arm | surviving mode | ρ | envelope_mass | energy share |
|---|---|---|---|---|---|
| fm-4 LR 5e-4 | chirp | ω = 0.019 | 0.019 | 0.41 | **0.53** |
| fm-4 LR 1.5e-4 | chirp | ω = 0.039 | — | — | **0.92** |
| fm-4 LR 5e-4 | lti | ω = 0.032 | 0.014 | 0.50 | 0.25 |
| fm-2 converged | lti | ω = 0.027 | 0.009 | 0.61 | 0.34 |

The pattern is unambiguous and symmetric across arms: **modes at the signal's frequency
are killed by their own decay; only low-ρ, near-DC modes survive to carry output energy.**
The observed "DC hedging" is a *consequence* of this, not an independent phenomenon.

## 4. Root cause in code

`llapdiffusion/models/laptrans.py`, both cores, pre-fix:

```python
# LaplaceTransformEncoder.reset_parameters   (the LTI core)
target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)

# ChirpModalField.reset_parameters           (the chirp core)
target_rho = torch.empty_like(self._rho_base).uniform_(0.01, 0.2)
```

Neither expression references the forecast horizon. The range `(0.01, 0.2)` is a sensible
*absolute* decay band for a short horizon and a poor one for a long horizon, because the
meaningful quantity ρ·H scales with H (see the TL;DR table).

## 5. Why ρ is never learned away

One might expect training to correct a bad init. It does not, for an identifiability
reason: with a true ρ·H of 0.15, the data contains **almost no decay signal**. The
observed amplitude decays by only ~14% across the whole horizon, which is comfortably
inside the noise. ρ is therefore close to a *null direction* of the loss — the likelihood
is nearly flat in it — so the parameter stays wherever the prior/initialisation put it.

This is directly visible in the measurements: recovered ρ values (0.009 – 0.245) simply
span the init range (0.01 – 0.2). An earlier note in the project log recorded exactly this
without diagnosing it — *"recovered ρ ∈ [0.06, 0.25] ≈ the init range; the prior floor
fills the null direction."* That was the bug announcing itself.

**Consequence:** a weakly-identified parameter must be initialised at the right *scale*,
because training will not rescue it. Here the right scale is set by the horizon.

## 6. Scope — which runs are affected

Any run whose horizon makes `ρ·H ≫ 1` for a large part of the init range:

- **H2 chirp benchmark (h = 48)** — confirmed by direct measurement (this report).
- **NOAA-UK headline (h = 168)** — *predicted from the init range* (ρ·H ∈ [1.7, 33.6]);
  not yet measured on a trained h=168 checkpoint, but the arithmetic is the same and the
  effect is strictly worse. This should be re-run after the fix regardless of the Fig-2
  outcome.
- **PhysioNet (h = 12)** — largely unaffected (ρ·H ∈ [0.12, 2.4]); the G2/G3 factorial
  results are probably safe, which is consistent with them behaving sensibly.

Both the chirp **and** LTI arms are affected, so this does not simply handicap one method:
it compresses the difference between them, which is why they kept tying.

## 7. The fix

Anchor the ρ init to the horizon so modes survive the window:

```python
# rho*H in [0.1, 2]  ->  envelope_mass 0.9 ... 0.25
lo, hi = 0.1 / horizon, 2.0 / horizon
target_rho = ... .uniform_(lo, hi)
```

Threading (`pole_init_horizon`):

```
config.PRED  ->  _llapdiff_model_kwargs  ->  LLapDiff  ->  LapFormer
                                                  ├─> LaplaceTransformEncoder (LTI)
                                                  └─> ChirpModalField          (chirp)
```

Design points:

1. **Applied to both cores.** A one-sided fix would have handed the chirp arm an
   artificial advantage in precisely the comparison the paper rests on.
2. **A trap worth recording:** the obvious anchor, `chirp_time_scale`, is *chirp-gated* —
   `_resolve_chirp_time_scale` returns `None` when `DENOISER_MODAL_TYPE != "chirp"`. Using
   it would have silently fixed chirp only. Hence a dedicated `pole_init_horizon`, set
   from `config.PRED` for **both** arms.
3. **Backward compatible.** `setdefault("pole_init_horizon", None)` keeps pre-fix
   checkpoints on the legacy init, and `None` (e.g. the adaptive per-sample time scale,
   which has no fixed horizon) also retains the legacy range.
4. **Pinned by tests** — `tests/test_pole_init_horizon.py`: scaling at h=48 and h=168,
   the LTI core receiving the same anchor, legacy behaviour preserved without a horizon,
   and end-to-end threading through `LLapDiff` for both modal types. Suite: 335 passing.

## 8. Why this was hard to see

It never looked like a decay problem. It presented as, in order:

1. *"The recovery tool is broken"* — plausible, and a separate real bug (the earlier
   variation-energy mode ranking) was found and fixed first, which masked this.
2. *"The benchmark is too easy"* — also true and independently fixed (`--freq-multiplier`);
   LTI genuinely did not fail at the original sweep.
3. *"The model hedges to DC because phase is unpredictable"* — a coherent story that fit
   the forecast evidence, and wrong.

Each of those was a real finding, which is what made the chain so long. The decisive
question — *"are the modes at the right frequency even alive?"* — was never asked until
the ρ magnitudes were compared against ground truth directly.

The general lesson: when a model's output is persistently damped, check whether its
basis functions can survive the prediction window **before** attributing the damping to
the loss, the data, or the optimiser.

## 9. Audit recipe (how to detect this in any run)

From a trained checkpoint's recovery JSON:

```python
# For each selected mode: is it alive across the horizon?
#   envelope_mass = mean_t exp(-2*rho_bar(t))   (already recorded per mode)
# Compare against the ground-truth / expected decay scale.
alive = [m for m in window["selected_modes"] if m["envelope_mass"] > 0.2]
osc   = [m for m in window["selected_modes"] if m["omega_mean"] > 0.15]
```

Red flags:

- oscillating modes with `envelope_mass < 0.1` (dead before the horizon ends);
- learned ρ more than ~10× the expected/true decay rate;
- the highest-energy mode having ω ≈ 0 (near-DC) while the signal clearly oscillates;
- recovered ρ values spanning exactly the init range → ρ is unidentified, and the init is
  doing all the work.

Quick sanity number for any horizon H: **modes need ρ ≲ 2/H.** At h=48 that is ρ ≲ 0.042;
at h=168, ρ ≲ 0.012.

## 9b. Second instance: ρ has no upper cap (found 2026-07-22, same day)

Fixing the *init* was necessary but not sufficient. The A/B showed the horizon-anchored
init works for the **LTI** core — its oscillating modes came back to ρ ≈ 0.017–0.031 with
envelope mass 0.27–0.44 (alive) — but the **chirp** core still had ρ = 0.103 / 0.359 /
0.987 with envelope mass 0.08 / 0.01 / **0.00**.

The arithmetic identifies the culprit. With the fix the chirp ρ *floor* is bounded to
[0.002, 0.042] by construction, yet learned ρ reached 0.987, so the excess is entirely the
**variation term**:

```
ρ_k(t) = ρ_floor_k + Σ_m a²_ρ,km · φ_m(t)      a² is squared ⇒ can only push ρ UP
```

ω is bounded above by a Nyquist rescale (`sup_t ω ≤ ω_max = π`); **ρ had no upper bound at
all**, only positivity. Training therefore inflates ρ freely until the mode's own envelope
extinguishes it — re-creating the exact failure the init fix was meant to remove, by a
second route.

**Fix:** cap ρ with the same smooth rescale used for ω, at `ρ_max = rho_max_scale / H`
(default `rho_max_scale = 4`, i.e. ρ·H ≤ 4, envelope mass ≥ ~0.12):

```python
rho_head  = (rho_max - rho_floor).clamp_min(1e-8)
rho_total = 2.0 * a_rho2.sum(-1, keepdim=True)      # phi_m <= 2
a_rho2    = a_rho2 * (rho_head / (rho_total + rho_head))
```

Properties: the coefficients remain a plain linear combination of the basis, so the
closed-form antiderivative (Theorem A) is preserved; the rescale is ≈1 at ε-init so
chirp still starts at the LTI poles; and bounding ρ from *above* does not affect the
Theorem-B contraction, which requires only `ρ ≥ ρ_min > 0` (guaranteed by the floor).
Verified under deliberately large coefficients: ρ·H pins at 4.00 and envelope mass holds
at 0.166 instead of collapsing to 0.00.

**Symmetry worth remembering:** ω is capped by *Nyquist* (what the sampling rate can
resolve); ρ is capped by *envelope survival* (what the horizon can carry). Both are
physical limits of the observation window, and the code had only one of them.

## 10. Open items

- **A/B in flight:** `fig2_sweetspot` (pre-fix ρ) vs `fig2_sweetspot_rhofix` (fixed ρ), at
  identical difficulty, identical LR, and a *copied identical stage 1*, so the ρ init is
  the only difference. This isolates the effect.
- **Removing a shared handicap does not guarantee separation.** The fix restores a fair
  test; it does not by itself make chirp beat LTI. Three outcomes are all informative:
  chirp separates (Fig. 2 is real), they still tie (a credible failure branch at last), or
  LTI gains more (also worth knowing).
- **Re-run h=168** after the fix — the predicted impact there is much larger than at h=48,
  and it affects the paper's headline rather than a synthetic figure.
- All results so far are **single-seed**; multi-seed confirmation is still owed.

---

*Related: `w_docs/fig2_signature_figure_strategy.md` (Fig-2 campaign plan and gates),
`h2_pole_recovery_problems_fixes.md` (the earlier, separate recovery-tool bug),
`w_docs/CMD_RUNBOOK.md` (operational knobs: `--freq-multiplier`, `--base-frequency`,
`--epochs`/`--early-stop`, `--base-lr`, Gate-0 `llapdiff-stage1-roundtrip`).*

---

## §11. Follow-on finding: recovered poles are a function of the HISTORY ONLY

Established 2026-07-22 while trying to rescue Fig. 2. After the rho fixes the chirp modes
are alive (envelope 0.38-0.65, rho within 3-6x truth) yet the recovered `omega_eff(t)`
still does not track the swept truth (slope +0.002). Two experiments narrowed why:

1. **Sampling-time inpainting does not help.** Re-running the pole capture with the true
   latent revealed at every 3rd step (phase pinned by observations) left the slope
   unchanged: free -0.002 vs imputed -0.009.
2. **The architecture explains it.** The pole head is conditioned on the history summary
   and the diffusion timestep alone:

   ```
   cond_vec  = make_pole_cond(t_vec, cond_summary, cond_summary_raw)   # history only
   rho, omega = chirp_field.integrated(cond_vec, t_rel)
   ```

   Observations enter through `x_t`, never through `cond_vec`. So the poles cannot be
   informed by the forecast-window data under any inpainting scheme.

**Consequence (design rule):** recovered pole trajectories are an *extrapolation from the
conditioning window*. Pole recovery can only succeed when the history determines the pole
trajectory over the forecast horizon.

**Where the benchmark violated this:** `window = 96` with `sweep_period = 96` puts exactly
**one** sweep cycle in the history — the model must infer both the period and the phase of
the frequency modulation from a single cycle and then extrapolate 48 steps. That is a
benchmark design flaw, not a model failure.

**Fix under test:** `--sweep-period 32` gives **3 sweep cycles in the history** while also
*increasing* the within-horizon excursion (dOmega 0.354 -> 0.480) and the LTI desync
(1.41 -> 1.92 rad), at an unchanged, reconstruction-safe omega range [0.10, 0.59].
Run: `ldt/results/fig2_inferable_p32`.

**Measurement caveats recorded so the next person does not repeat them:**
- A single-sinusoid periodogram cannot fit a chirp (R^2 = 0.33 even on clean truth); a
  chirp-aware fit (omega0 + domega/dt) reaches R^2 = 0.68.
- A *linear* chirp fit still mis-reads a window that straddles a sweep turning point (it
  reports domega/dt ~ 0 for a V-shaped omega). Match the fit to the sweep geometry.
- Always use stratified windows; consecutive windows have almost no target variation and
  make any cross-window tracking slope meaningless.
