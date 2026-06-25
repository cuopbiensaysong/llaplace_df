# Chirp-Modal Fix — Finding 2: Basis frequencies not scaled to the time axis

**Effect of the bug:** the chirp degenerates to ~LTI at native horizons, so the
time-varying-pole expressiveness the method advertises does not materialize.

**File touched:** `models/laptrans.py` (`ChirpModalField` basis / integration only).

---

## 1. The problem (precise)

`ChirpModalField` uses fixed integer frequencies *per unit relative-time*
(`models/laptrans.py : 378–380`):

```python
freqs = torch.linspace(1.0, float(self.num_basis), self.num_basis)  # 1..M cycles per unit t̃
self.register_buffer("basis_freqs", freqs, persistent=True)
```

but the relative time fed in is in **raw native units** (the trainer builds query offsets
relative to the first/last context timestamp with no normalization; horizons run to
100–168). In `_basis` (`models/laptrans.py : 423–431`):

```python
two_pi_f = (2.0 * math.pi) * self.basis_freqs
wt  = t_rel * two_pi_f
phi = 1.0 + torch.cos(wt)                 # instantaneous
Phi = t_rel + torch.sin(wt) / two_pi_f    # antiderivative
```

With `f_m` up to 8 and `t̃` up to ~168:

- the **instantaneous** poles `ρ_k(t̃), ω_k(t̃)` ripple at `f_m·H` cycles (hundreds across the
  window) — not a smooth drift;
- in the **integrated** poles that drive synthesis, the oscillatory term of `Phi` has
  amplitude `1/(2π f_m) ≤ 0.16`, negligible next to the linear `t̃` term (~168). So
  `ρ̄_k, ω̄_k` collapse to a constant-slope ramp ⇒ **LTI with a learned slope**.

**Verified numerically**: at `H=168` the genuinely time-varying ("wiggle") part of `ω̄_k` is
≈`2e-4` of the linear ramp; at normalized `H≈1` it is ≈`3e-2` (healthy). The advertised chirp
expressiveness does not materialize at training scale.

## 2. Design principle

The basis must resolve variation on the scale of the **window** — a few cycles across
`[0, L]`, not per unit time. Equivalently: normalize time to `τ = t̃ / L ∈ ~[0,1]` and keep
small integer frequencies in `τ`. The closed-form antiderivative is preserved; only a scale
factor changes.

Reparameterization (with `Ψ_m(τ) = τ + sin(2π f_m τ)/(2π f_m)` the antiderivative in `τ`):

```
ρ_k(t̃)   = ρ_floor_k + Σ_m a²_km (1 + cos(2π f_m · t̃/L))
ρ̄_k(t̃)  = ρ_floor_k·t̃ + Σ_m a²_km · L · Ψ_m(t̃/L)        # note the ·L
```

Check `d/dt̃[L·Ψ_m(t̃/L)] = (L)(1/L)φ_m(t̃/L) = φ_m(τ)` ✓ and `ρ̄_k(0)=0` ✓.

## 3. Fix (minimal patch — frequency rescaling = time normalization)

Introduce a per-sample time scale `L` (data-adaptive, robust to units and to negative
imputation times via `abs`) and divide the frequencies by it. This is mathematically the
`τ`-normalization above.

```python
# models/laptrans.py  ChirpModalField._basis
def _basis(self, t_rel, time_scale):
    # time_scale: [B,1,1] (per-sample window length L), L > 0
    f = self.basis_freqs.to(t_rel)                      # [M], cycles ACROSS the window
    two_pi_f = (2.0 * math.pi) * f / time_scale         # [B,1,M]  (= 2π f / L)
    wt  = t_rel * two_pi_f                              # [B,T,M]  (= 2π f · τ)
    phi = 1.0 + torch.cos(wt)                          # instantaneous (in τ)
    Phi = t_rel + torch.sin(wt) / two_pi_f             # antiderivative; sin term ~ (L/2πf)·sin(2πfτ)
    return phi, Phi
```

```python
# models/laptrans.py  ChirpModalField.integrated
def integrated(self, cond, t_rel):
    rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
    a_rho2, a_omega2 = self._coeffs(cond)
    L = t_rel.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)   # [B,1,1] per-sample
    _, Phi = self._basis(t_rel, L)
    rho_var   = torch.einsum("bkm,btm->btk", a_rho2,   Phi)
    omega_var = torch.einsum("bkm,btm->btk", a_omega2, Phi)
    rho_bar   = rho_floor.view(1, 1, self.k) * t_rel + rho_var
    omega_bar = omega_floor.view(1, 1, self.k) * t_rel + omega_var
    return rho_bar, omega_bar
```

`seed_poles` is **unchanged**: it uses `φ_m(0)=2`, which is scale-invariant. (If you keep a
separate `_basis` call anywhere for instantaneous poles, thread the same `L` through it.)

Notes:
- `L = max|t̃|` per sample is robust and adapts to mixed time units; for strict
  reproducibility/checkpoint comparability, prefer a fixed config constant
  `CHIRP_TIME_SCALE` (= the horizon length in native units) and pass it through instead of
  recomputing. Expose both; default to data-adaptive.
- The oscillatory amplitude of `Phi` becomes `~L/(2π f_m)`, comparable to the linear `t̃` term
  (~`L`), so the wiggle/linear ratio is `~1/(2π f_m) ∈ [0.02, 0.16]` — healthy, scale-free.
- Optional: make `basis_freqs` learnable. The antiderivative stays closed-form for any fixed-
  per-step `f` (it is a parameter, differentiable), so this is safe; keep `f ≥ f_min > 0`.

## 4. Tests to add / update

New non-degeneracy test:

```python
def test_chirp_is_nondegenerate_at_native_horizon():
    """With time normalization, the time-varying part is non-negligible at a long horizon."""
    torch.manual_seed(0)
    B, K, C, M, H = 1, 3, 16, 6, 168.0
    field = ChirpModalField(k=K, cond_dim=C, num_basis=M)
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=0.5)   # activate time-variation
    cond = torch.randn(B, C)
    t_rel = torch.linspace(0.0, H, 400).view(1, 400, 1)
    rho_bar, _ = field.integrated(cond, t_rel)                   # [B,T,K]
    # remove the best-fit linear-in-t ramp per mode; the residual is the genuine "wiggle".
    t = t_rel.squeeze(-1).squeeze(0)
    for k in range(K):
        y = rho_bar[0, :, k]
        slope = (t * y).sum() / (t * t).sum()
        wiggle = (y - slope * t).abs().max()
        assert wiggle / (slope * H).abs().clamp_min(1e-9) > 1e-2   # was ~2e-4 before the fix
```

Keep the existing invariants under the new scaling (these must still pass):
- `test_chirp_field_integral_correctness` — `ρ̄(0)=0`, monotone increasing, and
  `d/dt̃ ρ̄ = instantaneous ρ`. **Update its finite-difference closed-form** `inst` to divide
  the frequency by `L` (i.e., `two_pi_f = 2π · basis_freqs / L`) so it matches the new basis.
- `test_chirp_basis_recovers_lti_with_constant_poles` and the LTI-equivalence-at-init tests —
  unaffected (zero coeffs ⇒ `rho_var=0` regardless of `L`), but re-run to confirm.

## 5. Validation (standalone)

1. **Unit check:** run the new + updated tests above.
2. **Numerical sanity:** confirm the integrated-pole wiggle/linear ratio is `O(0.01–0.1)` at
   native horizons (it was ~`2e-4` before the fix).
3. **Experiment:** train/eval chirp with the fix and compare against the pre-fix chirp and the
   LTI baseline. This is the first point at which the "chirp vs LTI" comparison is meaningful,
   because the chirp now actually varies in time.

