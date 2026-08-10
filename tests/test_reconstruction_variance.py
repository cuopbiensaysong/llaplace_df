"""The stage-1 round-trip residual as a data-space variance component.

Eq. (7) is a law for `z_0`, so its pushforward describes `decode(z)` while the target is
`y = decode(encode(y)) + r`. `r` is a variance component no data-space arm accounts for, and it
is measurable: the latent metrics score against `encode(y)` and the data metrics against `y`.

These pin the machinery. **On noaa_uk h=168 the term turns out to be small** — pooled variance
0.00265 against a data-space residual of ~0.40, recovering +0.003 of a +0.106 coverage gap — so
it is NOT the explanation for the latent→data calibration loss. The tests exist so the estimate
stays correct and so the correction can be applied wherever stage 1 is weaker.
"""

import pytest
import torch

from llapdiffusion.tools.run_analytic_uq_eval import ReconstructionNoiseVAE


class _FlatVAE:
    """Decodes to a constant, so any spread in the output is the added term alone."""
    latent_channel = 8

    def __init__(self, shape):
        self._shape = shape

    def decode_mu(self, mu_est, entity_pad):
        return torch.full(self._shape, 3.0)


def test_the_wrapper_adds_exactly_the_requested_marginal_std():
    B, H, N, C = 2, 5, 3, 1
    std = torch.full((H, C), 0.25)
    g = torch.Generator().manual_seed(0)
    w = ReconstructionNoiseVAE(_FlatVAE((B, H, N, C)), std, generator=g)

    draws = torch.stack([w.decode_mu(None, None) for _ in range(6000)])
    assert draws.std().item() == pytest.approx(0.25, abs=0.01)
    assert draws.mean().item() == pytest.approx(3.0, abs=0.01)


def test_per_horizon_variance_is_applied_per_horizon():
    """Reconstruction error is not flat over the horizon; a scalar would smear it."""
    B, H, N, C = 2, 4, 3, 1
    std = torch.tensor([[0.0], [0.1], [0.5], [2.0]])
    g = torch.Generator().manual_seed(1)
    w = ReconstructionNoiseVAE(_FlatVAE((B, H, N, C)), std, generator=g)

    draws = torch.stack([w.decode_mu(None, None) for _ in range(4000)])
    # [S,B,H,N,C] -> std over draws -> average over B and N, KEEPING the horizon.
    per_h = draws.std(dim=0).mean(dim=(0, 2)).squeeze(-1)      # [H]
    for got, want in zip(per_h.tolist(), [0.0, 0.1, 0.5, 2.0]):
        assert got == pytest.approx(want, abs=0.05)


def test_the_wrapper_is_transparent_to_the_rest_of_the_stack():
    """decode_latents_with_vae dispatches on vae.decode_mu and checks vae.latent_channel, so
    the wrapper must delegate everything it does not override."""
    w = ReconstructionNoiseVAE(_FlatVAE((1, 1, 1, 1)), torch.zeros(1, 1))
    assert w.latent_channel == 8
    assert hasattr(w, "decode_mu")


def test_zero_variance_is_a_no_op():
    """So the correction can be switched off without changing a single number."""
    B, H, N, C = 2, 3, 2, 1
    w = ReconstructionNoiseVAE(_FlatVAE((B, H, N, C)), torch.zeros(H, C),
                               generator=torch.Generator().manual_seed(2))
    assert torch.equal(w.decode_mu(None, None), torch.full((B, H, N, C), 3.0))
