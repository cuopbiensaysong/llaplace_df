# Chirp-Modal Fix — Finding 1 (CORRECTED): Stability certificate broken by the LapFormer output head

**Supersedes** the previous Finding-1 note. The earlier "Option A" returned the raw modal sum
`y_time` and set `output_skip_scale = 1.0`. That was wrong: it removed the **output scaling**
together with the uncertified residual, which blows up the prediction magnitude and the loss
scale. This document keeps the (certified) scaling and removes **only** the uncertified term.

**File touched:** `models/lapformer.py` (output head only). Independent of Finding 2.

---

## 1. What went wrong with the first patch (failure mode to avoid)

The output head was doing **two separate jobs**:

```python
# models/lapformer.py : 578
return self.output_skip_scale * y_time + self.head_proj(self.head_norm(y_time))
#      └──── certified output SCALING ────┘   └──── the uncertified RESIDUAL ────┘
```

- `output_skip_scale * y_time` — a learnable **magnitude** on the modal sum. **Certified and
  legitimate.** It exists because the synthesizer sums `K` modes (PhysioNet config: `K=256`),
  so the raw sum `ẑ₀(0)=Σₖ cₖ` is ~an order of magnitude larger than the unit-scale VAE latent
  `z₀`. The init value `≈0.1` shrinks the 256-mode sum back to `z₀`'s scale.
- `head_proj(head_norm(...))` — the **non-decaying, uncertified residual**. This is the *only*
  Finding-1 defect.

The first patch deleted both. Removing the scaling left the prediction ~`10×` too large, so
`‖z₀ − ẑ₀‖² ≈ ‖−9·z₀‖² ≈ 80·‖z₀‖²`. Observed: train loss `1.5→0.9` became `70→27.5` (~50×
scale jump), destabilized optimization (huge initial gradients the LR schedule wasn't tuned
for), and CRPS `0.367 → 0.469`. **The regression was a scale artifact of the patch, not a
property of removing the residual.**

> Rule: never drop `output_skip_scale * y_time`. Drop **only** `head_proj(head_norm(...))`.

## 2. The actual defect (unchanged)

The certified bound `‖ŷ(t̃)‖ ≤ e^{-ρ_min·t̃}·Σₖ√(‖cₖ‖²+‖bₖ‖²)` applies to `y_time`, not to the
returned tensor, because `head_proj(head_norm(y_time))` is (a) an unconstrained trainable
`Linear` and (b) preceded by `LayerNorm`, which re-normalizes its input back to ~unit scale as
the modal envelope decays — so the term does **not** decay and the output tends to a nonzero
floor (violates Theorem B(iii)). `tests/test_chirp_modal.py::test_chirp_contraction_bound`
only checks the synthesizer in isolation, so it passes while the model-level property fails.

## 3. Design principle

Keep the certified scaling; remove or *certify* the residual. Any post-modal additive term
must have an envelope **dominated by** `e^{-ρ_min·t̃}`. `LayerNorm` is the root cause and must
be removed; an unconstrained `Linear` must be made 1-Lipschitz; any bias must be decay-gated.

## 4. Fix — Option A (default; keep scaling, drop residual)

```python
# models/lapformer.py  __init__ : after building self.synthesis
# Keep output_skip_scale for chirp at the SAME init as LTI (0.1) — this is the scaling, not the bug.
self.output_skip_scale = nn.Parameter(torch.tensor(0.1))
self._use_output_head = (self.denoiser_modal_type != "chirp")   # uncertified head: off for chirp
if self._use_output_head:
    self.head_norm = nn.LayerNorm(input_dim)
    self.head_proj = nn.Linear(input_dim, input_dim)
    nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)
```

```python
# models/lapformer.py  forward : replace line 578
if self._use_output_head:
    return self.output_skip_scale * y_time + self.head_proj(self.head_norm(y_time))
# chirp: scaled modal sum, no uncertified residual
return self.output_skip_scale * y_time
```

**Certified bound** (carries the scaling constant):

```
‖out‖ = |s|·‖y_time‖ ≤ |s|·e^{-ρ_min·t̃}·Σₖ√(‖cₖ‖²+‖bₖ‖²)        (s = output_skip_scale)
```

still horizon-independent and `→0` for any finite `|s|`. For the *clean* certificate
(constant `≤1`), clamp/parameterize the scale:

```python
s = self.output_skip_scale.clamp(max=1.0)     # or: s = F.softplus(self.output_skip_scale_raw)
return s * y_time
```

This restores the loss scale to ~1 and should bring CRPS back near baseline.

### Cleaner root-cause alternative (optional)

To eliminate the magic `0.1`, normalize the `K`-mode sum so it is naturally `O(1)`: scale the
synthesizer output by `1/√K` (or unit-normalize the residues) inside `LaplacePseudoInverse`,
then `output_skip_scale` can init at `1.0`. The certified bound is unchanged (the `1/√K` folds
into the residue norms).

## 5. Fix — Option B (recommended if corrected Option A is still worse than 0.367)

If corrected Option A underperforms the pre-fix CRPS (`0.367`), the head's residual was
contributing **real predictive value** (likely on short/sparse data like PhysioNet, where the
modal basis alone under-fits). Then *certify* the head instead of deleting it: keep the
scaling, add a **decay-gated, LayerNorm-free, 1-Lipschitz** correction with a **gated affine**
to recover the per-channel scale/shift the `LayerNorm` affine used to provide.

```python
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

# __init__ (chirp branch)
self.output_skip_scale = nn.Parameter(torch.tensor(0.1))             # certified scaling (keep!)
self.head_proj = spectral_norm(nn.Linear(input_dim, input_dim, bias=False))  # 1-Lipschitz, Lip(0)=0
nn.init.normal_(getattr(self.head_proj, "weight_orig", self.head_proj.weight), std=1e-4)
self.head_bias = nn.Parameter(torch.zeros(input_dim))               # per-channel shift (gated below)

# forward (chirp branch); t_rel is [B,T,1] from self.analysis.relative_time(...)
s    = self.output_skip_scale.clamp(max=1.0)
gate = torch.exp(-self.chirp_field.rho_min * t_rel)                  # [B,T,1], in (0,1] for t̃>=0
corr = self.head_proj(y_time) + self.head_bias                      # affine, NO LayerNorm
return s * y_time + gate * corr
```

**Certified bound** (1-Lipschitz `head_proj` with `head_proj(0)=0`; `gate ≤ 1`; bias is gated
so it decays):

```
‖out‖ ≤ |s|·‖y_time‖ + gate·(‖head_proj(y_time)‖ + ‖head_bias‖)
      ≤ |s|·‖y_time‖ + gate·(‖y_time‖ + ‖head_bias‖)
      ≤ e^{-ρ_min·t̃}·[ (|s| + 1)·Σₖ√(‖cₖ‖²+‖bₖ‖²) + ‖head_bias‖ ]
```

still horizon-independent and `→0` (the `‖head_bias‖` term is gated by `e^{-ρ_min·t̃}`).

Non-negotiables: **no LayerNorm** (it re-inflates the envelope — the root cause); `head_proj`
**spectral-normed** (1-Lipschitz); the bias **multiplied by `gate`** (an un-gated constant bias
would not decay and would re-break the certificate).

## 6. Diagnostics (confirm the scale story before long runs)

```python
# at init, with a fresh chirp model:
print("‖y_time‖ / ‖z0‖ at init:", y_time.norm(dim=-1).mean().item() / z0.norm(dim=-1).mean().item())
#   expect ~K^0.5-ish (≈ several×, ~10 for K=256): confirms why raw-drop blew up the scale.

# from the ORIGINAL (pre-fix) checkpoint:
print("converged output_skip_scale:", state_dict["model.output_skip_scale"].item())
#   expect ≈0.1: confirms it was doing the down-scaling, not the bug.
```

If both hold, the loss explosion is fully explained by the dropped scaling.

## 7. Notes

- The head is inherited from the **LLapDiff backbone**; leave it untouched for `lti` mode
  (LTI relies on its own synthesis MLP). Gate all changes on `denoiser_modal_type == "chirp"`.
- Update `DEVELOPER_GUIDE.md` §7.5: Option A bound constant is `|s| ≤ 1`; Option B is
  `(|s|+1) ≤ 2` plus a gated `‖head_bias‖`.

## 8. Tests

`tests/test_chirp_modal.py`:

```python
def test_full_model_contraction_bound():
    """Theorem B holds for the ACTUAL model output (with the scaling constant)."""
    torch.manual_seed(0)
    B, T, D, K = 2, 64, 8, 4
    model = LLapDiff(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=K, timesteps=50, denoiser_modal_type="chirp").eval()
    # Simulate a *trained* model: perturb output-side params away from zero-init.
    for name, p in model.named_parameters():
        if any(s in name for s in ("head_proj", "head_bias", "output_skip_scale",
                                   "chirp_field.to_coeffs")):
            (torch.nn.init.normal_(p, std=0.5) if p.dim() > 0 else p.data.fill_(0.7))
    x  = torch.randn(B, T, D)
    ts = torch.randint(0, 50, (B,))
    dt = torch.linspace(0.0, 5.0, T).view(1, T).expand(B, T).contiguous()  # t̃ >= 0
    with torch.no_grad():
        y = model(x, ts, dt=dt)
    norm = y.norm(dim=-1)                       # [B,T]
    assert norm[:, -1].max() <= norm[:, : T // 4].max()   # envelope decays (no nonzero floor)
    assert torch.isfinite(y).all()

def test_output_scale_preserved_for_chirp():
    """Regression guard: chirp must NOT drop output_skip_scale (the loss-scale bug)."""
    model = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=4, timesteps=50, denoiser_modal_type="chirp")
    assert hasattr(model.model, "output_skip_scale")
    assert abs(float(model.model.output_skip_scale)) <= 1.0 + 1e-6   # init 0.1, clamped
```

## 9. Validation (standalone)

1. Run the tests above; the existing synthesizer-level `test_chirp_contraction_bound` still
   passes.
2. Re-train on PhysioNet with corrected Option A. Expect train loss back to ~`1.x→0.9` scale
   and CRPS back near `0.367`.
3. If CRPS is still above `0.367`, switch to Option B (certified head + gated affine) — the
   residual was carrying signal, so certify it rather than delete it.

> Independent of Finding 2 (this touches `models/lapformer.py`; Finding 2 touches
> `models/laptrans.py`). No merge conflict.
