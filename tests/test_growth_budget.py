"""Tests for the Theorem-B' bounded-growth head (T2): the budgeted excursion
gamma_k(t) on the integrated decay, its closed-form derivative, the B' bounds,
and signed-increment support in the variance quadrature."""

import math

import pytest
import torch

from llapdiffusion.models.laptrans import (
    ChirpModalField,
    LaplacePseudoInverse,
    LaplaceTransformEncoder,
)
from llapdiffusion.models.llapdiff import LLapDiff

C_G = math.log(2.0)


def _growth_field(seed=0, k=3, cond_dim=16, num_basis=6, L=4.0, rho_min=1e-2, activate=True):
    torch.manual_seed(seed)
    field = ChirpModalField(
        k=k, cond_dim=cond_dim, num_basis=num_basis, time_scale=L,
        rho_min=rho_min, growth_budget=C_G,
    )
    if activate:
        torch.nn.init.normal_(field.to_growth[-1].weight, std=2.0)
        torch.nn.init.normal_(field.to_growth[-1].bias, std=2.0)
    return field


def test_growth_head_inert_at_init_and_disabled_at_zero_budget():
    field = _growth_field(activate=False)  # zero-init head
    cond = torch.randn(2, 16)
    t = torch.linspace(0.0, 4.0, 30).view(1, 30, 1).expand(2, 30, 1).contiguous()
    gamma, gamma_prime = field._growth_terms(cond, t)
    torch.testing.assert_close(gamma, torch.zeros_like(gamma))
    torch.testing.assert_close(gamma_prime, torch.zeros_like(gamma_prime))
    # c_g = 0 builds no head at all (Theorem B verbatim).
    plain = ChirpModalField(k=3, cond_dim=16, num_basis=6, growth_budget=0.0)
    assert not hasattr(plain, "to_growth")
    with pytest.raises(ValueError):
        ChirpModalField(k=3, cond_dim=16, num_basis=6, growth_budget=-0.1)


def test_growth_excursion_anchored_and_budgeted():
    field = _growth_field()
    cond = torch.randn(4, 16)
    t = torch.linspace(0.0, 4.0, 50).view(1, 50, 1).expand(4, 50, 1).contiguous()
    gamma, _ = field._growth_terms(cond, t)
    torch.testing.assert_close(gamma[:, 0, :], torch.zeros_like(gamma[:, 0, :]))  # gamma(0)=0
    assert (gamma <= C_G + 1e-6).all()  # capped by the budget


def test_instantaneous_matches_integrated_derivative_with_growth():
    """The closed-form gamma' is the true derivative of the excursion (B' pole path)."""
    field = _growth_field()
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=0.5)  # time-varying rho too
    cond = torch.randn(2, 16)
    h = 1e-3
    tc = torch.full((2, 1, 1), 1.234)
    rb_plus, _ = field.integrated(cond, tc + h)
    rb_minus, _ = field.integrated(cond, tc - h)
    rho_fd = (rb_plus - rb_minus) / (2 * h)
    rho_inst, _ = field.instantaneous(cond, tc)
    torch.testing.assert_close(rho_fd, rho_inst, atol=1e-3, rtol=1e-3)


def test_theorem_b_prime_envelope_bounds():
    """rho_bar >= rho_min*t - c_g everywhere (i.e. ||Phi|| <= e^{c_g} e^{-rho_min t}),
    and the envelope can genuinely GROW somewhere (negative instantaneous rho)."""
    field = _growth_field(rho_min=1e-2)
    cond = torch.randn(8, 16)
    t = torch.linspace(0.0, 4.0, 200).view(1, 200, 1).expand(8, 200, 1).contiguous()
    rho_bar, _ = field.integrated(cond, t)
    assert (rho_bar >= field.rho_min * t - C_G - 1e-5).all()  # B'(i) at s=0
    rho_inst, _ = field.instantaneous(cond, t)
    assert (rho_inst < 0).any()  # growth is actually expressible, not just permitted


def test_theorem_b_prime_synthesis_bound():
    """||y(t)|| <= e^{c_g} e^{-rho_min t} sum_k sqrt(||c||^2+||b||^2)  (Eq. 4')."""
    torch.manual_seed(0)
    B, T, K, D = 2, 24, 3, 6
    field = _growth_field(k=K)
    enc = LaplaceTransformEncoder(k=K, feat_dim=D, hidden_dim=16, num_heads=4)
    synth = LaplacePseudoInverse(enc, use_mlp_residual=False)
    cond = torch.randn(B, 16)
    theta = torch.randn(B, 2 * K, D)
    t = torch.linspace(0.0, 4.0, T).view(1, T, 1).expand(B, T, 1).contiguous()
    rho_bar, omega_bar = field.integrated(cond, t)
    y = synth(theta, rho_bar=rho_bar, omega_bar=omega_bar)
    amp = torch.sqrt(
        theta[:, :K, :].pow(2).sum(-1) + theta[:, K:, :].pow(2).sum(-1)
    ).sum(-1)
    bound = math.exp(C_G) * torch.exp(-field.rho_min * t.squeeze(-1)) * amp.unsqueeze(1)
    assert (y.norm(dim=-1) <= bound + 1e-4).all()


def test_modal_variance_supports_signed_increments():
    """Piecewise rho: growth then decay — the exponential-integrator recurrence must
    match the exact piecewise closed form with a NEGATIVE first-segment rho."""
    B, K = 1, 1
    rho1, rho2, q, t1, t2 = -0.3, 0.5, 0.7, 2.0, 5.0
    n1, n2 = 200, 300
    ta = torch.linspace(t1 / n1, t1, n1)
    tb = torch.linspace(t1 + (t2 - t1) / n2, t2, n2)
    t = torch.cat([ta, tb]).view(1, -1, 1)
    rho_bar = torch.where(
        t <= t1, rho1 * t, rho1 * t1 + rho2 * (t - t1)
    ).expand(B, -1, K).clone()
    s = ChirpModalField.modal_variance(
        rho_bar, t, torch.full((B, K), q), torch.zeros(B, K)
    )
    # Exact: v(t1) = q(1-e^{-2 rho1 t1})/(2 rho1) (valid for negative rho1);
    # then v(t) = e^{-2 rho2 (t-t1)} v(t1) + q(1-e^{-2 rho2 (t-t1)})/(2 rho2).
    v_t1 = q * (1.0 - math.exp(-2 * rho1 * t1)) / (2 * rho1)
    dt2 = t2 - t1
    v_t2 = math.exp(-2 * rho2 * dt2) * v_t1 + q * (1.0 - math.exp(-2 * rho2 * dt2)) / (2 * rho2)
    assert abs(float(s[0, n1 - 1, 0]) - v_t1) < 1e-4
    assert abs(float(s[0, -1, 0]) - v_t2) < 1e-4
    assert torch.isfinite(s).all() and (s >= 0).all()


def test_growth_budget_threading_and_checkpoint_default():
    model = LLapDiff(
        data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
        timesteps=50, denoiser_modal_type="chirp", chirp_growth_budget=C_G,
    )
    assert model.model.chirp_field.growth_budget == pytest.approx(C_G)
    assert any("to_growth" in k for k in model.state_dict())

    from llapdiffusion.trainers.train_val_llapdiff import _llapdiff_config_from_checkpoint

    payload = {"model_config": {"llapdiff": {"data_dim": 8, "hidden_dim": 32}}}
    assert _llapdiff_config_from_checkpoint(payload)["chirp_growth_budget"] == 0.0
