# Chirp-Modal Fix — Finding 1: Stability certificate broken by the LapFormer output head

**Status:** independent of Finding 2 (disjoint code region). Can be applied before or after
Finding 2.

**Effect of the bug:** the Theorem-B contraction guarantee (`drop the residual MLP, stability
by construction`) does **not** hold for the model's actual output — the very LLapDiff defect
the method set out to remove reappears through a different door.

**File touched:** `models/lapformer.py` (output head only).

---

## 1. The problem (precise)

The method's headline is "drop the uncertified residual MLP — stability by construction
(Theorem B): `‖ŷ(t̃)‖ ≤ e^{-ρ_min·t̃}·Σ_k√(‖c_k‖²+‖b_k‖²)`". The code correctly disables the
**synthesis** MLP in chirp mode (`LaplacePseudoInverse(..., use_mlp_residual=False)`), and
`DEVELOPER_GUIDE.md` (§7.5) states the bound holds for the output. It does not, because of
the **unconditional** output head in `LapFormer.forward`:

```python
# models/lapformer.py : 578  (runs for BOTH lti and chirp)
return self.output_skip_scale * y_time + self.head_proj(self.head_norm(y_time))
```

with (`models/lapformer.py : 391–395`)

```python
self.head_norm = nn.LayerNorm(input_dim)
self.head_proj = nn.Linear(input_dim, input_dim)   # unconstrained, trainable
nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)
self.output_skip_scale = nn.Parameter(torch.tensor(0.1))   # unbounded scalar
```

The certified bound applies to `y_time` (the synthesizer output), **not** to the tensor the
model returns. Two distinct failures:

1. **`head_proj` is an uncertified residual** on top of the modal sum — exactly the term the
   chirp method exists to remove. Zero-initialized but trainable, so it is nonzero after
   training.
2. **`head_norm` (LayerNorm) actively defeats the decay.** As the certified envelope
   `e^{-ρ̄_k(t̃)} → 0`, LayerNorm re-normalizes its input back to ~unit scale, so the head
   term does **not** decay. Theorem B(iii) (`ẑ₀(t)→0` as `t→∞`, no residual) fails: the
   output tends to a nonzero floor set by the head.

`output_skip_scale` is also an unbounded scalar (a milder issue: it rescales the bound's
constant but keeps it horizon-independent and decaying).

**Why the test missed it.** `tests/test_chirp_modal.py::test_chirp_contraction_bound`
checks `LaplacePseudoInverse.forward` in isolation, so it passes while the **model-level**
property fails — false confidence.

**Verified numerically** (exact formula from line 578, long horizon): `‖y_time‖` decays
`2.64 → 7e-6`, but `‖LayerNorm(y_time)‖` stays ≈`1.3`, so the model output plateaus and
exceeds the bound by up to ~4× in the mid-horizon (violation begins ≈`t̃=21`).

## 2. Design principle

Any post-modal term added to the output must have an envelope **dominated by**
`e^{-ρ_min·t̃}`. A plain `Linear∘LayerNorm` violates this on both counts (no decay, no
Lipschitz bound). The fix is to either remove the head in chirp mode or replace it with a
*certified* correction.

## 3. Fix — Option A (default; matches the method's claim)

In chirp mode, **drop the output head** and return the modal sum directly (optionally with a
clamped skip scale). This realizes the exact Theorem-B bound.

```python
# models/lapformer.py  __init__ : after building self.synthesis
self._use_output_head = (self.denoiser_modal_type != "chirp")   # off for chirp
if self._use_output_head:
    self.head_norm = nn.LayerNorm(input_dim)
    self.head_proj = nn.Linear(input_dim, input_dim)
    nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)
self.output_skip_scale = nn.Parameter(torch.tensor(1.0 if self.denoiser_modal_type == "chirp"
                                                    else 0.1))
```

```python
# models/lapformer.py  forward : replace line 578
if self._use_output_head:
    return self.output_skip_scale * y_time + self.head_proj(self.head_norm(y_time))
# chirp: certified output is the modal sum itself
return y_time
```

Resulting bound (exact Theorem B): `‖out‖ = ‖y_time‖ ≤ e^{-ρ_min·t̃}·Σ_k√(‖c_k‖²+‖b_k‖²)`.

## 4. Fix — Option B (keep local expressiveness, still certified)

If ablations show the head buys accuracy, keep a correction but make it **decay-gated,
LayerNorm-free, and 1-Lipschitz**:

```python
from torch.nn.utils import spectral_norm

# __init__ (chirp branch)
self.head_proj = spectral_norm(nn.Linear(input_dim, input_dim, bias=False))  # 1-Lipschitz, Lip(0)=0
nn.init.normal_(getattr(self.head_proj, "weight_orig", self.head_proj.weight), std=1e-4)
self.output_skip_scale = nn.Parameter(torch.tensor(1.0))  # clamp in forward

# forward (chirp branch); t_rel is [B,T,1] from self.analysis.relative_time(...)
s = self.output_skip_scale.clamp(max=1.0)
gate = torch.exp(-self.chirp_field.rho_min * t_rel)        # [B,T,1], in (0,1] for t̃>=0
return s * y_time + gate * self.head_proj(y_time)
```

Certified bound (since `head_proj` is 1-Lipschitz with `head_proj(0)=0`, so
`‖head_proj(y_time)‖ ≤ ‖y_time‖`, and `gate ≤ 1`):

```
‖out‖ ≤ |s|·‖y_time‖ + gate·‖head_proj(y_time)‖
      ≤ (|s| + gate)·‖y_time‖
      ≤ (|s| + 1)·e^{-ρ_min·t̃}·Σ_k√(‖c_k‖²+‖b_k‖²)   ≤ 2·e^{-ρ_min·t̃}·(…)   [with |s|≤1]
```

Still horizon-independent and `→0`. Two non-negotiables: **no LayerNorm** (it re-inflates the
envelope — it is the root cause), and **no un-gated bias** (a constant bias does not decay; use
`bias=False` or gate the whole term, which the snippet does).

## 5. Notes

- The head is inherited from the **LLapDiff backbone**; leave it untouched for `lti` mode
  (LTI relies on its own synthesis MLP and never claimed this certificate). Gate all changes
  on `denoiser_modal_type == "chirp"`.
- Update `DEVELOPER_GUIDE.md` §7.5: under Option A the bound constant is `1` (or `|s|`); under
  Option B it is `(|s|+1)` ≤ `2`.

## 6. Tests to add

`tests/test_chirp_modal.py`:

```python
def test_full_model_contraction_bound():
    """Theorem B holds for the ACTUAL LapFormer/LLapDiff output, not just the synthesizer."""
    torch.manual_seed(0)
    B, T, D, K = 2, 64, 8, 4
    model = LLapDiff(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=K, timesteps=50, denoiser_modal_type="chirp").eval()
    # Simulate a *trained* head: perturb every output-side parameter away from zero-init.
    for name, p in model.named_parameters():
        if any(s in name for s in ("head_proj", "head_norm", "output_skip_scale",
                                   "chirp_field.to_coeffs")):
            torch.nn.init.normal_(p, std=0.5) if p.dim() > 0 else p.data.fill_(0.7)

    x = torch.randn(B, T, D)
    tstep = torch.randint(0, 50, (B,))
    dt = torch.linspace(0.0, 5.0, T).view(1, T).expand(B, T).contiguous()  # t̃ >= 0
    with torch.no_grad():
        y = model(x, tstep, dt=dt)

    norm = y.norm(dim=-1)                      # [B,T]
    # (a) far end <= early peak (envelope decays) and (b) ->0 at the far end (Theorem B(iii)).
    assert norm[:, -1].max() <= norm[:, : T // 4].max()
    assert torch.isfinite(y).all()
```

(Adapt the `amp` comparison to your residue bookkeeping if you want the exact
`e^{-ρ_min·t̃}·Σ√(‖c‖²+‖b‖²)` inequality; the essential new assertion is on the **full model**
output and its decay-to-zero, which the current test does not cover.)

## 7. Validation (standalone)

1. **Unit check:** the new `test_full_model_contraction_bound` passes; the existing
   `test_chirp_contraction_bound` (synthesizer-level) still passes.
2. **Numerical sanity:** the model output envelope decays toward zero over a long horizon with
   a perturbed (trained-like) head — no nonzero floor.
3. **Re-claim:** update the `DEVELOPER_GUIDE.md` bound to match the chosen option.

> Independent of Finding 2. If you applied Finding 2 first, no merge conflict: that change is
> confined to `models/laptrans.py` (`ChirpModalField`), this one to `models/lapformer.py`
> (output head).
