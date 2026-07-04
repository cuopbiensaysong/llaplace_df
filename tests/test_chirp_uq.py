"""Tests for the Theorem-C analytic UQ stack (U2) and the one-shot arm pieces (U3):
UQ head + variance quadrature, return_variance plumbing, Gaussian-NLL loss mode,
PIT/reliability metrics, and the max_only timestep sampler."""

import math
from types import SimpleNamespace

import pytest
import torch

from llapdiffusion.models.laptrans import ChirpModalField
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.models.llapdiff_utils import (
    NoiseScheduler,
    diffusion_loss,
    sample_training_timesteps,
)
from llapdiffusion.models.uq_metrics import (
    gaussian_nll,
    gaussian_pit,
    pit_calibration_error,
    reliability_curve,
)


def _uq_model(predict_type="x0", **kwargs):
    return LLapDiff(
        data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
        timesteps=50, predict_type=predict_type, denoiser_modal_type="chirp",
        chirp_uq_head=True, **kwargs,
    )


def test_modal_variance_matches_constant_pole_closed_form():
    """Constant rho: v(t) = q (1 - e^{-2 rho t}) / (2 rho); s = e^{-2 rho t} p0 + v."""
    B, T, K = 2, 400, 3
    rho = torch.tensor([0.05, 0.2, 0.6]).view(1, 1, K)
    t = torch.linspace(0.0, 5.0, T).view(1, T, 1).expand(B, T, 1).contiguous()
    rho_bar = rho * t  # constant instantaneous rho
    q = torch.rand(B, K) + 0.5
    p0 = torch.rand(B, K) + 0.1

    s = ChirpModalField.modal_variance(rho_bar, t, q, p0)
    v_exact = q.unsqueeze(1) * (1.0 - torch.exp(-2.0 * rho_bar)) / (2.0 * rho)
    s_exact = torch.exp(-2.0 * rho_bar) * p0.unsqueeze(1) + v_exact
    torch.testing.assert_close(s, s_exact, atol=1e-4, rtol=1e-3)  # trapezoid on fine grid


def test_modal_variance_positive_and_stable_under_large_decay():
    """Huge integrated decay must not overflow (all exponents are <= 0)."""
    B, T, K = 1, 16, 2
    t = torch.linspace(0.0, 168.0, T).view(1, T, 1)
    rho_bar = 0.5 * t.expand(B, T, K).clone()  # rho_bar up to 84 -> e^{2*84} would overflow
    q = torch.full((B, K), 0.3)
    p0 = torch.full((B, K), 0.2)
    s = ChirpModalField.modal_variance(rho_bar, t, q, p0)
    assert torch.isfinite(s).all() and (s > 0).all()
    # The exponential-integrator segment rule is exact for constant rho, even on this
    # very coarse grid: steady state of the Lyapunov equation is q / (2 rho).
    torch.testing.assert_close(
        s[:, -1, :], torch.full((B, K), 0.3 / (2 * 0.5)), atol=1e-5, rtol=1e-5
    )


def test_uq_params_init_uniform():
    torch.manual_seed(0)
    field = ChirpModalField(k=4, cond_dim=16, num_basis=6, uq_head=True)
    cond = torch.randn(3, 16)
    p0, q = field.uq_params(cond)
    assert p0.shape == (3, 4) and q.shape == (3, 4)
    torch.testing.assert_close(p0, torch.full_like(p0, 1e-2))  # zero-init deltas
    torch.testing.assert_close(q, torch.full_like(q, 1e-2))
    # Without the head the accessor refuses.
    bare = ChirpModalField(k=4, cond_dim=16, num_basis=6)
    with pytest.raises(RuntimeError):
        bare.uq_params(cond)


def test_forward_return_variance_shapes_and_scaling():
    torch.manual_seed(0)
    B, T, D = 2, 6, 8
    model = _uq_model().eval()
    x = torch.randn(B, T, D)
    ts = torch.randint(0, 50, (B,))
    dt = torch.sort(torch.rand(B, T), dim=1).values
    with torch.no_grad():
        mean, var = model(x, ts, dt=dt, return_variance=True)
        assert mean.shape == (B, T, D) and var.shape == (B, T, D)
        assert torch.isfinite(var).all() and (var > 0).all()
        # Variance scales with the squared (clamped) output scale.
        model.model.output_skip_scale.fill_(1.0)
        _, var_full = model(x, ts, dt=dt, return_variance=True)
        model.model.output_skip_scale.fill_(0.5)
        _, var_half = model(x, ts, dt=dt, return_variance=True)
    torch.testing.assert_close(var_half, 0.25 * var_full)


def test_uq_head_guards():
    # Theorem C is a law for z0 -> x0 parameterization only.
    with pytest.raises(ValueError, match="x0"):
        _uq_model(predict_type="v")
    # UQ requires the chirp core.
    with pytest.raises(ValueError, match="chirp"):
        LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
                 timesteps=50, predict_type="x0", chirp_uq_head=True)
    # ... and the certified output path (no LayerNorm head).
    with pytest.raises(ValueError, match="certified"):
        _uq_model(output_head="on")
    # return_variance without the head is a usage error.
    plain = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
                     timesteps=50, denoiser_modal_type="chirp").eval()
    with pytest.raises(RuntimeError):
        plain(torch.randn(1, 4, 8), torch.zeros(1, dtype=torch.long),
              dt=torch.sort(torch.rand(1, 4), 1).values, return_variance=True)


def test_gaussian_nll_loss_mode():
    torch.manual_seed(0)
    B, T, D = 2, 5, 8
    model = _uq_model()
    scheduler = model.scheduler
    x0 = torch.randn(B, T, D)
    t = torch.randint(1, 50, (B,))
    dt = torch.sort(torch.rand(B, T), dim=1).values

    loss = diffusion_loss(
        model, scheduler, x0, t, cond_summary=None, predict_type="x0",
        dt=dt, loss_mode="gaussian_nll",
    )
    assert torch.isfinite(loss)
    loss.backward()  # variance path is differentiable
    grads = [p.grad for n, p in model.named_parameters() if "to_uq" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

    with pytest.raises(ValueError, match="x0"):
        diffusion_loss(model, scheduler, x0, t, cond_summary=None,
                       predict_type="v", dt=dt, loss_mode="gaussian_nll")
    with pytest.raises(ValueError, match="loss_mode"):
        diffusion_loss(model, scheduler, x0, t, cond_summary=None,
                       predict_type="x0", dt=dt, loss_mode="nope")


def test_diff_loss_mode_config_validation():
    from llapdiffusion.trainers.train_val_llapdiff import _diff_loss_mode

    assert _diff_loss_mode(SimpleNamespace()) == "mse"
    assert _diff_loss_mode(
        SimpleNamespace(DIFF_LOSS_MODE="gaussian_nll", CHIRP_UQ_HEAD=True)
    ) == "gaussian_nll"
    with pytest.raises(ValueError, match="CHIRP_UQ_HEAD"):
        _diff_loss_mode(SimpleNamespace(DIFF_LOSS_MODE="gaussian_nll"))


def test_checkpoint_missing_uq_head_defaults_to_false():
    from llapdiffusion.trainers.train_val_llapdiff import _llapdiff_config_from_checkpoint

    payload = {"model_config": {"llapdiff": {"data_dim": 8, "hidden_dim": 32}}}
    assert _llapdiff_config_from_checkpoint(payload)["chirp_uq_head"] is False


def test_max_only_timestep_sampler():
    scheduler = NoiseScheduler(timesteps=50, schedule="cosine")
    t = sample_training_timesteps(scheduler, 7, torch.device("cpu"), sampler="max_only")
    assert t.shape == (7,) and (t == 49).all()


def test_warm_start_uq_model_from_mse_checkpoint():
    """The recommended NLL warm-start: a CHIRP_UQ_HEAD model loads an MSE-trained
    (no-UQ) chirp checkpoint, keeping fresh UQ-head params; other mismatches raise."""
    from llapdiffusion.trainers.train_val_llapdiff import _load_diff_init_state

    torch.manual_seed(0)
    common = dict(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4,
                  laplace_k=4, timesteps=50, denoiser_modal_type="chirp")
    mse_model = LLapDiff(**common)  # no UQ head
    torch.manual_seed(1)
    uq_model = _uq_model()

    ref = mse_model.state_dict()["model.analysis._rho_raw"].clone()
    _load_diff_init_state(uq_model, mse_model.state_dict())
    torch.testing.assert_close(uq_model.state_dict()["model.analysis._rho_raw"], ref)
    assert any("to_uq" in k for k in uq_model.state_dict())  # UQ head kept (fresh)

    # A genuinely incompatible checkpoint (lti: different keys) still fails loudly.
    lti_model = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4,
                         laplace_k=4, timesteps=50)
    with pytest.raises(RuntimeError):
        _load_diff_init_state(uq_model, lti_model.state_dict())


def test_generate_clip_stats_populated():
    torch.manual_seed(0)
    model = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=4, timesteps=50, denoiser_modal_type="chirp").eval()
    clip_stats = {}
    with torch.no_grad():
        model.generate(
            shape=(2, 5, 8), steps=4, guidance_strength=1.0, eta=0.0,
            dt=torch.sort(torch.rand(2, 5), 1).values,
            dynamic_thresh_p=0.9, clip_stats=clip_stats,
        )
    assert clip_stats["steps"] == 4  # one threshold application per DDIM step
    assert 0.0 <= clip_stats["clipped_fraction_sum"] / clip_stats["steps"] <= 1.0
    # Thresholding off -> nothing recorded.
    off_stats = {}
    with torch.no_grad():
        model.generate(
            shape=(2, 5, 8), steps=4, guidance_strength=1.0, eta=0.0,
            dt=torch.sort(torch.rand(2, 5), 1).values,
            dynamic_thresh_p=0.0, clip_stats=off_stats,
        )
    assert off_stats == {}


def test_pit_metrics_calibrated_vs_overconfident():
    torch.manual_seed(0)
    n = 20000
    mean = torch.randn(n)
    var = torch.rand(n) * 2.0 + 0.5
    y = mean + var.sqrt() * torch.randn(n)

    u = gaussian_pit(y, mean, var)
    assert u.shape == (n,) and (u >= 0).all() and (u <= 1).all()
    assert pit_calibration_error(u) < 0.02  # calibrated -> ~uniform PIT

    coverage = reliability_curve(u, levels=(0.5, 0.9))
    assert abs(coverage[0.5] - 0.5) < 0.03
    assert abs(coverage[0.9] - 0.9) < 0.03

    # Overconfident (variance 10x too small): coverage collapses, error grows.
    u_over = gaussian_pit(y, mean, var / 10.0)
    assert pit_calibration_error(u_over) > 0.1
    assert reliability_curve(u_over, levels=(0.9,))[0.9] < 0.7

    # NLL prefers the calibrated variance.
    assert gaussian_nll(y, mean, var) < gaussian_nll(y, mean, var / 10.0)

    # Mask selects elements.
    mask = torch.zeros(n, dtype=torch.bool)
    mask[: n // 2] = True
    assert gaussian_pit(y, mean, var, mask=mask).shape == (n // 2,)
