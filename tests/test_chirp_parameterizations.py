"""Tests for the Phase-4 pole-function parameterizations (P-mono, P-grid) behind
CHIRP_PARAMETERIZATION, alongside the default P-exact."""

import math

import pytest
import torch

from llapdiffusion.models.laptrans import (
    CHIRP_PARAMETERIZATIONS,
    ChirpModalField,
    normalize_chirp_parameterization,
)
from llapdiffusion.models.llapdiff import LLapDiff

VARIANTS = ("p_mono", "p_grid")


def _field(parameterization, seed=0, k=3, cond_dim=16, num_basis=6, L=4.0, **kwargs):
    torch.manual_seed(seed)
    return ChirpModalField(
        k=k, cond_dim=cond_dim, num_basis=num_basis, time_scale=L,
        parameterization=parameterization, **kwargs,
    )


def _activate(field):
    head = {"p_mono": "to_mono", "p_grid": "to_grid", "p_exact": "to_coeffs"}[
        field.parameterization
    ]
    layer = getattr(field, head)[-1]
    torch.nn.init.normal_(layer.weight, std=1.0)
    torch.nn.init.normal_(layer.bias, std=1.0)
    return field


def test_normalize_chirp_parameterization():
    assert normalize_chirp_parameterization(" P_MONO ") == "p_mono"
    assert set(CHIRP_PARAMETERIZATIONS) == {"p_exact", "p_mono", "p_grid"}
    with pytest.raises(ValueError):
        normalize_chirp_parameterization("p_magic")


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_invariants(variant):
    """integrated(0)=0; rho_bar strictly increasing; instantaneous poles positive
    with omega below the Nyquist cap — for trained-like (activated) heads."""
    field = _activate(_field(variant))
    cond = torch.randn(4, 16)
    zero = torch.zeros(4, 1, 1)
    rb0, ob0 = field.integrated(cond, zero)
    torch.testing.assert_close(rb0, torch.zeros_like(rb0))
    torch.testing.assert_close(ob0, torch.zeros_like(ob0))

    t = torch.linspace(0.05, 4.0, 80).view(1, 80, 1).expand(4, 80, 1).contiguous()
    rho_bar, omega_bar = field.integrated(cond, t)
    # Nondecreasing everywhere (an increment of a saturated ~1e-10 pole can round to
    # exactly 0 in float32 cumsum) and strictly increasing overall.
    assert (rho_bar[:, 1:] - rho_bar[:, :-1] >= 0).all()
    assert (omega_bar[:, 1:] - omega_bar[:, :-1] >= 0).all()
    assert (rho_bar[:, -1] > rho_bar[:, 0]).all()
    assert (omega_bar[:, -1] > omega_bar[:, 0]).all()

    rho, omega = field.instantaneous(cond, t)
    assert (rho > 0).all()
    assert (omega > 0).all() and (omega <= math.pi + 1e-5).all()

    rho0, omega0 = field.seed_poles(cond)
    assert (rho0 > 0).all() and (omega0 > 0).all() and (omega0 <= math.pi + 1e-5).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_near_lti_at_init(variant):
    """At init the variants sit at (p_grid: exactly) or eps-near (p_mono) the LTI floors."""
    field = _field(variant)
    cond = torch.randn(3, 16)
    t = torch.linspace(0.0, 4.0, 40).view(1, 40, 1).expand(3, 40, 1).contiguous()
    rho, omega = field.instantaneous(cond, t)
    rho_floor, omega_floor = field._floor_poles(t.dtype, t.device)
    assert (rho - rho_floor.view(1, 1, -1)).abs().max() < 1e-2
    assert (omega - omega_floor.view(1, 1, -1)).abs().max() < 1e-2


def test_pmono_derivative_matches_integrated():
    """P-mono's instantaneous poles are the closed-form derivative of the monotone
    integrated poles (finite-difference check)."""
    field = _activate(_field("p_mono"))
    cond = torch.randn(2, 16)
    h = 1e-3
    tc = torch.full((2, 1, 1), 1.234)
    rb_p, ob_p = field.integrated(cond, tc + h)
    rb_m, ob_m = field.integrated(cond, tc - h)
    rho_inst, omega_inst = field.instantaneous(cond, tc)
    torch.testing.assert_close((rb_p - rb_m) / (2 * h), rho_inst, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close((ob_p - ob_m) / (2 * h), omega_inst, atol=1e-3, rtol=1e-3)


def test_pgrid_integration_is_trapezoid_of_instantaneous():
    """P-grid's integral is by construction the cumulative trapezoid of the pointwise
    poles over [0, t_1, ..., t_T]."""
    field = _activate(_field("p_grid"))
    cond = torch.randn(2, 16)
    t = torch.linspace(0.5, 4.0, 8).view(1, 8, 1).expand(2, 8, 1).contiguous()
    rho_bar, _ = field.integrated(cond, t)
    zero = torch.zeros(2, 1, 1)
    rho0, _ = field._pgrid_inst(cond, zero)
    rho_t, _ = field._pgrid_inst(cond, t)
    rho_all = torch.cat([rho0, rho_t], dim=1)
    tt = torch.cat([torch.zeros(2, 1), t.squeeze(-1)], dim=1)
    seg = (tt[:, 1:] - tt[:, :-1]).unsqueeze(-1)
    expected = (0.5 * seg * (rho_all[:, 1:] + rho_all[:, :-1])).cumsum(dim=1)
    torch.testing.assert_close(rho_bar, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_heads_receive_gradient_at_init(variant):
    """No stationary-trap regressions: both new heads must have nonzero gradients
    from a fresh init under a loss on the integrated poles."""
    field = _field(variant)
    cond = torch.randn(2, 16)
    t = torch.linspace(0.1, 4.0, 20).view(1, 20, 1).expand(2, 20, 1).contiguous()
    rho_bar, omega_bar = field.integrated(cond, t)
    (rho_bar.sum() + omega_bar.sum()).backward()
    head = {"p_mono": "to_mono", "p_grid": "to_grid"}[variant]
    grad = getattr(field, head)[-1].weight.grad
    assert grad is not None and grad.abs().max().item() > 0.0


def test_growth_budget_composes_with_pgrid():
    field = _field("p_grid", growth_budget=math.log(2.0), rho_min=1e-2)
    torch.nn.init.normal_(field.to_growth[-1].weight, std=2.0)
    torch.nn.init.normal_(field.to_growth[-1].bias, std=2.0)
    cond = torch.randn(4, 16)
    t = torch.linspace(0.0, 4.0, 100).view(1, 100, 1).expand(4, 100, 1).contiguous()
    rho_bar, _ = field.integrated(cond, t)
    assert (rho_bar >= field.rho_min * t - math.log(2.0) - 1e-5).all()  # B'(i)


@pytest.mark.parametrize("variant", VARIANTS)
def test_llapdiff_forward_and_threading(variant):
    torch.manual_seed(0)
    model = LLapDiff(
        data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
        timesteps=50, denoiser_modal_type="chirp", chirp_parameterization=variant,
    ).eval()
    assert model.model.chirp_field.parameterization == variant
    x = torch.randn(2, 5, 8)
    ts = torch.randint(0, 50, (2,))
    dt = torch.sort(torch.rand(2, 5), dim=1).values
    with torch.no_grad():
        y = model(x, ts, dt=dt)
    assert y.shape == (2, 5, 8) and torch.isfinite(y).all()


def test_uq_head_works_with_pmono():
    """modal_variance consumes rho_bar and is parameterization-agnostic."""
    torch.manual_seed(0)
    model = LLapDiff(
        data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
        timesteps=50, predict_type="x0", denoiser_modal_type="chirp",
        chirp_uq_head=True, chirp_parameterization="p_mono",
    ).eval()
    x = torch.randn(2, 5, 8)
    ts = torch.randint(0, 50, (2,))
    dt = torch.sort(torch.rand(2, 5), dim=1).values
    with torch.no_grad():
        mean, var = model(x, ts, dt=dt, return_variance=True)
    assert torch.isfinite(mean).all() and (var > 0).all()


@pytest.mark.parametrize("variant", ("p_exact",) + VARIANTS)
def test_coefficient_penalty_positive_differentiable_and_shrinks(variant):
    """CHIRP_COEFF_L2: the penalty is positive for activated heads, differentiable,
    and its gradient points toward the LTI special case (smaller coefficients)."""
    field = _activate(_field(variant))
    cond = torch.randn(4, 16)
    penalty = field.coefficient_penalty(cond)
    assert penalty.dim() == 0 and float(penalty) > 0
    penalty.backward()
    head = {"p_exact": "to_coeffs", "p_mono": "to_mono", "p_grid": "to_grid"}[variant]
    grad = getattr(field, head)[-1].weight.grad
    assert grad is not None and grad.abs().max().item() > 0


def test_coeff_l2_in_diffusion_loss_and_lti_guard():
    from llapdiffusion.models.llapdiff_utils import diffusion_loss

    torch.manual_seed(0)
    B, T, D = 2, 5, 8
    model = LLapDiff(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=4, timesteps=50, denoiser_modal_type="chirp")
    x0 = torch.randn(B, T, D)
    t = torch.randint(1, 50, (B,))
    dt = torch.sort(torch.rand(B, T), dim=1).values

    torch.manual_seed(1)
    base, base_stats = diffusion_loss(model, model.scheduler, x0, t, cond_summary=None,
                                      predict_type="v", dt=dt, return_stats=True)
    torch.manual_seed(1)  # same noise draw -> identical reconstruction term
    lam = 10.0
    reg, reg_stats = diffusion_loss(model, model.scheduler, x0, t, cond_summary=None,
                                    predict_type="v", dt=dt, return_stats=True,
                                    coeff_l2=lam)
    assert "coeff_penalty" in reg_stats and "coeff_penalty" not in base_stats
    torch.testing.assert_close(reg - base, lam * reg_stats["coeff_penalty"],
                               atol=1e-6, rtol=1e-5)

    # lti model: requesting the penalty fails loudly.
    lti = LLapDiff(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4,
                   laplace_k=4, timesteps=50)
    with pytest.raises(RuntimeError, match="chirp"):
        diffusion_loss(lti, lti.scheduler, x0, t, cond_summary=None,
                       predict_type="v", dt=dt, coeff_l2=lam)


def test_checkpoint_missing_parameterization_defaults_to_p_exact():
    from llapdiffusion.trainers.train_val_llapdiff import _llapdiff_config_from_checkpoint

    payload = {"model_config": {"llapdiff": {"data_dim": 8, "hidden_dim": 32}}}
    assert _llapdiff_config_from_checkpoint(payload)["chirp_parameterization"] == "p_exact"
