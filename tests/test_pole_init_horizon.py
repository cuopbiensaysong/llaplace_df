"""Horizon-anchored rho initialisation.

The synthesis envelope is e^{-rho*t}, so what matters is rho*H. A horizon-independent
init (the legacy uniform(0.01, 0.2)) puts rho*H in [0.5, 9.6] at H=48 and [1.7, 33.6]
at H=168 -- nearly every oscillatory mode decays to nothing before the horizon ends,
leaving only near-DC modes alive. Measured on a trained H=48 chirp model: modes at the
signal frequency had rho ~0.10-0.13 (21-76x the truth 0.0032) with envelope mass
0.03-0.12, while the surviving 53%-energy mode sat at omega~0.02. Both cores are
anchored so chirp-vs-LTI stays a fair comparison.
"""

import math

import torch

from llapdiffusion.models.laptrans import ChirpModalField, LaplaceTransformEncoder
from llapdiffusion.models.llapdiff import LLapDiff


def _envelope_mass(rho, H):
    a = 2.0 * rho * H
    return (1.0 - math.exp(-a)) / a if a > 0 else 1.0


def test_chirp_rho_init_scales_with_horizon():
    torch.manual_seed(0)
    for H in (48.0, 168.0):
        f = ChirpModalField(k=64, cond_dim=16, num_basis=8, time_scale=H)
        rho, _ = f._floor_poles(torch.float32, torch.device("cpu"))
        rho_h = (rho * H).detach()
        # rho*H in [0.1, 2] => modes survive the horizon.
        assert float(rho_h.min()) >= 0.05, f"H={H}: {float(rho_h.min())}"
        assert float(rho_h.max()) <= 2.5, f"H={H}: {float(rho_h.max())}"
        # Every mode retains real envelope mass (legacy init gave ~0.03 at H=48).
        assert min(_envelope_mass(float(r), H) for r in rho.detach()) > 0.2


def test_lti_rho_init_scales_with_horizon_too():
    """The LTI arm gets the same anchor, or the comparison is biased."""
    torch.manual_seed(0)
    H = 48.0
    enc = LaplaceTransformEncoder(k=64, feat_dim=8, hidden_dim=32, num_heads=4,
                                  rho_init_horizon=H)
    rho = torch.nn.functional.softplus(enc._rho_raw) + enc.alpha_min
    assert float((rho * H).max()) <= 2.5
    assert min(_envelope_mass(float(r), H) for r in rho.detach()) > 0.2


def test_legacy_init_preserved_without_horizon():
    """No horizon (adaptive time scale / pre-fix checkpoints) keeps the old range."""
    torch.manual_seed(0)
    f = ChirpModalField(k=64, cond_dim=16, num_basis=8, time_scale=None)
    rho, _ = f._floor_poles(torch.float32, torch.device("cpu"))
    assert float(rho.max()) > 0.05          # legacy uniform(0.01, 0.2)
    enc = LaplaceTransformEncoder(k=64, feat_dim=8, hidden_dim=32, num_heads=4)
    rho_l = torch.nn.functional.softplus(enc._rho_raw) + enc.alpha_min
    assert float(rho_l.max()) > 0.05


def test_llapdiff_threads_pole_init_horizon_to_both_cores():
    for modal in ("chirp", "lti"):
        torch.manual_seed(0)
        m = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=32,
                     timesteps=50, denoiser_modal_type=modal, pole_init_horizon=48.0)
        if modal == "chirp":
            rho, _ = m.model.chirp_field._floor_poles(torch.float32, torch.device("cpu"))
        else:
            rho = torch.nn.functional.softplus(m.model.analysis._rho_raw) + m.model.analysis.alpha_min
        assert float((rho * 48.0).max()) <= 2.5, f"{modal} not anchored"


def test_chirp_rho_is_capped_so_modes_survive_the_horizon():
    """The rho VARIATION term is squared, so it can only push rho up from the floor.
    Without a cap, training inflated rho to ~0.99 at H=48 (envelope mass 0.00) and the
    modes at the signal frequency contributed nothing. omega is capped by Nyquist; rho
    is capped by envelope survival."""
    torch.manual_seed(0)
    H = 48.0
    f = ChirpModalField(k=8, cond_dim=16, num_basis=8, time_scale=H)
    assert f.rho_max == 4.0 / H

    # Drive the coefficient head hard, as training does.
    torch.nn.init.normal_(f.to_coeffs[-1].weight, std=2.0)
    cond = torch.randn(4, 16)
    t = torch.arange(H).view(1, -1, 1).expand(4, -1, 1).contiguous()
    rho, _ = f.instantaneous(cond, t)
    rho_bar, _ = f.integrated(cond, t)

    assert float(rho.max()) <= 4.0 / H + 1e-6            # cap holds pointwise
    assert float(rho.min()) > 0.0                        # Theorem B still needs rho > 0
    envelope = torch.exp(-2.0 * rho_bar).mean(dim=1)
    assert float(envelope.min()) > 0.1                   # modes survive (was 0.00)


def test_rho_cap_disabled_without_horizon():
    """No horizon (adaptive time scale) => no cap, legacy behaviour."""
    torch.manual_seed(0)
    f = ChirpModalField(k=8, cond_dim=16, num_basis=8, time_scale=None)
    assert f.rho_max is None


def test_rho_cap_preserves_lti_equivalence_at_init():
    """At eps-init the rescale must be ~1 so chirp still starts at the LTI poles."""
    torch.manual_seed(0)
    H = 48.0
    f = ChirpModalField(k=8, cond_dim=16, num_basis=8, time_scale=H)
    cond = torch.randn(4, 16)
    a_rho2, _ = f._coeffs(cond)
    floor, _ = f._floor_poles(torch.float32, torch.device("cpu"))
    t = torch.zeros(4, 1, 1)
    rho, _ = f.instantaneous(cond, t)
    # phi_m(0)=2 so rho(0) = floor + 2*sum a^2; at eps-init the variation is negligible.
    torch.testing.assert_close(rho[:, 0, :], floor.expand(4, -1), atol=1e-3, rtol=1e-2)


# --------------------------------------------------------------------------------------
# Centred rho basis: decouple rho variation from mean decay.
# phi = 1+cos has mean 1, so ANY rho variation raised mean rho and extinguished the
# oscillatory modes (measured rho up to 7.2 = 346x the truth). phi-1 = cos integrates to
# exactly zero over the window (basis freqs are whole cycles), so variation adds NO decay.
# --------------------------------------------------------------------------------------

def _field(basis, H=48.0, k=8, M=8, scale=4.0):
    return ChirpModalField(k=k, cond_dim=16, num_basis=M, time_scale=H,
                           rho_basis=basis, rho_max_scale=scale)


def test_centered_basis_contributes_zero_net_decay_over_the_window():
    """rho_bar(L) == rho_floor * L exactly: the variation shapes rho(t) inside the
    window but adds no net decay, so the envelope is set purely by the floor."""
    torch.manual_seed(0)
    H = 48.0
    f = _field("centered", H=H)
    torch.nn.init.normal_(f.to_coeffs[-1].weight, std=2.0)   # drive it hard
    cond = torch.randn(4, 16)
    t_end = torch.full((4, 1, 1), H)
    rho_bar, _ = f.integrated(cond, t_end)
    floor, _ = f._floor_poles(torch.float32, torch.device("cpu"))
    expected = floor.view(1, 1, -1) * H
    torch.testing.assert_close(rho_bar, expected.expand_as(rho_bar), atol=1e-4, rtol=1e-3)


def test_centered_basis_keeps_rho_positive_and_bounded():
    """Theorem B needs rho >= rho_min; the headroom rescale guarantees it, and the
    bounded floor gives rho <= 2*rho_max as a side effect."""
    torch.manual_seed(0)
    H = 48.0
    f = _field("centered", H=H)
    torch.nn.init.normal_(f.to_coeffs[-1].weight, std=5.0)   # adversarial
    cond = torch.randn(8, 16)
    t = torch.arange(H).view(1, -1, 1).expand(8, -1, 1).contiguous()
    rho, _ = f.instantaneous(cond, t)
    assert float(rho.min()) >= f.rho_min - 1e-9, float(rho.min())
    assert float(rho.max()) <= 2.0 * f.rho_max + 1e-6, float(rho.max())
    rho_bar, _ = f.integrated(cond, t)
    assert float(torch.exp(-2.0 * rho_bar).mean(dim=1).min()) > 0.2   # modes alive


def test_centered_instantaneous_is_derivative_of_integrated():
    """Theorem A: the closed form must still be the antiderivative after centring."""
    torch.manual_seed(0)
    f = _field("centered", H=48.0)
    torch.nn.init.normal_(f.to_coeffs[-1].weight, std=1.0)
    cond = torch.randn(2, 16)
    h, tc = 1e-3, torch.full((2, 1, 1), 11.7)
    rb_p, _ = f.integrated(cond, tc + h)
    rb_m, _ = f.integrated(cond, tc - h)
    rho_fd = (rb_p - rb_m) / (2 * h)
    rho_inst, _ = f.instantaneous(cond, tc)
    torch.testing.assert_close(rho_fd, rho_inst, atol=1e-3, rtol=1e-3)


def test_centered_basis_still_lti_equivalent_at_eps_init():
    """At eps-init the variation is negligible, so chirp still starts at the LTI poles."""
    torch.manual_seed(0)
    f = _field("centered", H=48.0)
    cond = torch.randn(4, 16)
    floor, _ = f._floor_poles(torch.float32, torch.device("cpu"))
    t = torch.arange(48.0).view(1, -1, 1).expand(4, -1, 1).contiguous()
    rho, _ = f.instantaneous(cond, t)
    torch.testing.assert_close(rho, floor.view(1, 1, -1).expand_as(rho), atol=1e-3, rtol=1e-2)


def test_nonneg_basis_reproduces_legacy_behaviour():
    """Back-compat: pre-fix checkpoints keep the 1+cos basis, where the variation DOES
    raise the mean (that is the bug, preserved for reproducibility)."""
    torch.manual_seed(0)
    H = 48.0
    f = _field("nonneg", H=H)
    torch.nn.init.normal_(f.to_coeffs[-1].weight, std=2.0)
    cond = torch.randn(4, 16)
    t_end = torch.full((4, 1, 1), H)
    rho_bar, _ = f.integrated(cond, t_end)
    floor, _ = f._floor_poles(torch.float32, torch.device("cpu"))
    assert float(rho_bar.mean()) > float((floor.view(1, 1, -1) * H).mean()) * 1.05


def test_rho_basis_normalizer_rejects_garbage():
    from llapdiffusion.models.laptrans import normalize_chirp_rho_basis
    import pytest as _pytest
    assert normalize_chirp_rho_basis(" Centered ") == "centered"
    assert normalize_chirp_rho_basis("nonneg") == "nonneg"
    with _pytest.raises(ValueError):
        normalize_chirp_rho_basis("bogus")


def test_llapdiff_threads_rho_basis():
    torch.manual_seed(0)
    m = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=8,
                 timesteps=50, denoiser_modal_type="chirp", pole_init_horizon=48.0,
                 chirp_rho_basis="centered", chirp_rho_max_scale=2.0)
    assert m.model.chirp_field.rho_basis == "centered"
    assert m.model.chirp_field.rho_max == 2.0 / 48.0
