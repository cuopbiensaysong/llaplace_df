"""Tests for the chirp pole-basis harmonic set (``CHIRP_BASIS``).

The legacy ``integer`` basis uses f_m = 1..M with every sign +1, so every basis function
completes a whole number of cycles across the window and phi_m(0) = phi_m(L) = 2 is its
maximum. With nonnegative (squared) coefficients that pins

    omega(0) == omega(L) == sup_t omega(t)

for EVERY mode: the instantaneous frequency can only dip and return, so a monotone sweep --
which is what the H2 chirp truth does across a window -- is outside the function class
entirely. The ``half_integer`` basis pairs each harmonic with its negative so a nonnegative
coefficient reaches either sign, restoring drift while keeping phi in [0, 2] (so every bound
in ``_coeffs`` is untouched) and keeping each element zero-mean over [0, L] (so the
``centered`` rho basis still adds no net decay).
"""

import math

import torch

from llapdiffusion.models.laptrans import (
    ChirpModalField,
    chirp_basis_freqs_signs,
    normalize_chirp_basis,
)
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.trainers.train_val_llapdiff import _llapdiff_config_from_checkpoint

import pytest


def _driven(basis: str, *, M: int = 8, L: float = 48.0, K: int = 8, C: int = 16,
            rho_basis: str = "centered", omega_basis: str = "nonneg",
            std: float = 2.0, seed: int = 0) -> ChirpModalField:
    """A field with trained-like (large) coefficients, so tests probe the FUNCTION CLASS
    rather than the eps-init neighbourhood of LTI."""
    torch.manual_seed(seed)
    field = ChirpModalField(k=K, cond_dim=C, num_basis=M, time_scale=L,
                            rho_init_horizon=L, rho_basis=rho_basis,
                            omega_basis=omega_basis, basis=basis)
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=std)
    torch.nn.init.normal_(field.to_coeffs[-1].bias, std=std)
    return field


def _grid(B: int, L: float, T: int = 97) -> torch.Tensor:
    return torch.linspace(0.0, L, T).view(1, T, 1).expand(B, T, 1).contiguous()


def test_normalize_chirp_basis():
    assert normalize_chirp_basis(" Half_Integer ") == "half_integer"
    assert normalize_chirp_basis("INTEGER") == "integer"
    with pytest.raises(ValueError):
        normalize_chirp_basis("quarter")


def test_legacy_basis_is_bit_identical():
    """``integer`` must reproduce the pre-change basis exactly: f = 1..M, all signs +1,
    phi = 1 + cos, Phi = t + sin/(2 pi f). Guards every existing checkpoint."""
    M, L = 8, 48.0
    freqs, signs = chirp_basis_freqs_signs("integer", M)
    torch.testing.assert_close(freqs, torch.linspace(1.0, float(M), M))
    torch.testing.assert_close(signs, torch.ones(M))

    field = _driven("integer", M=M, L=L)
    t_rel = _grid(2, L)
    phi, Phi = field._basis(t_rel, field._time_scale(t_rel))
    two_pi_f = (2.0 * math.pi) * field.basis_freqs / L
    torch.testing.assert_close(phi, 1.0 + torch.cos(t_rel * two_pi_f))
    torch.testing.assert_close(Phi, t_rel + torch.sin(t_rel * two_pi_f) / two_pi_f)


def test_half_integer_pairs_every_harmonic_with_both_signs():
    freqs, signs = chirp_basis_freqs_signs("half_integer", 8)
    torch.testing.assert_close(freqs, torch.tensor([0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]))
    torch.testing.assert_close(signs, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))
    # An odd M drops the trailing negative partner rather than changing the frequencies.
    freqs_odd, signs_odd = chirp_basis_freqs_signs("half_integer", 5)
    torch.testing.assert_close(freqs_odd, torch.tensor([0.5, 0.5, 1.0, 1.0, 1.5]))
    torch.testing.assert_close(signs_odd, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]))


def test_legacy_basis_cannot_drift_across_the_window():
    """The defect this module exists for: with integer cycles omega(L) == omega(0) exactly,
    and the maximum sits at the endpoints, for every mode and every conditioning."""
    L = 48.0
    field = _driven("integer", L=L)
    cond = torch.randn(32, 16)
    t_rel = _grid(32, L)
    with torch.no_grad():
        _, omega = field.instantaneous(cond, t_rel)

    torch.testing.assert_close(omega[:, -1], omega[:, 0], atol=1e-5, rtol=0)
    # sup_t omega is attained at an endpoint (t=0 and t=L tie, so argmax is one of them).
    argmax = omega.argmax(dim=1)
    assert bool(((argmax == 0) | (argmax == omega.shape[1] - 1)).all())


def test_half_integer_basis_drifts_in_both_directions():
    """The fix: net omega(L) - omega(0) is reachable, and reachable with EITHER sign --
    a one-sided basis would bias every recovered trajectory the same way."""
    L = 48.0
    field = _driven("half_integer", L=L)
    cond = torch.randn(64, 16)
    t_rel = _grid(64, L)
    with torch.no_grad():
        _, omega = field.instantaneous(cond, t_rel)

    net = (omega[:, -1] - omega[:, 0]).reshape(-1)
    varying = (omega.max(dim=1).values - omega.min(dim=1).values).reshape(-1) > 1e-4
    net = net[varying]
    assert net.numel() > 0
    assert float(net.abs().max()) > 0.1          # drift is available at a useful scale
    assert float((net > 1e-3).float().mean()) > 0.2   # rising modes exist
    assert float((net < -1e-3).float().mean()) > 0.2  # ...and falling ones


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
@pytest.mark.parametrize("omega_basis", ["nonneg", "centered"])
def test_omega_cap_holds_under_driven_coefficients(basis, omega_basis):
    """Both omega bases keep instantaneous omega inside [0, omega_max] pointwise under
    trained-like coefficients: nonneg via floor + 2*sum a^2 <= omega_max, centered via a
    two-sided rescale around the floor."""
    L = 48.0
    field = _driven(basis, L=L, omega_basis=omega_basis, std=3.0)
    cond = torch.randn(16, 16)
    t_rel = _grid(16, L, T=257)
    with torch.no_grad():
        _, omega = field.instantaneous(cond, t_rel)
        _, omega0 = field.seed_poles(cond)
    # Nyquist is the real constraint, on |omega|: the SIGN of omega is a pure gauge
    # (omega -> -omega is absorbed by the free sin residues b -> -b), so the centred
    # basis is allowed to carry omega through zero.
    assert float(omega.abs().max()) <= math.pi + 1e-5
    assert float(omega0.abs().max()) < math.pi
    if omega_basis == "nonneg":
        assert float(omega.min()) >= -1e-5  # legacy basis is nonnegative by construction


def test_centered_omega_swing_budget_is_not_capped_by_the_floor():
    """Regression for a trap of my own making: bounding the centred swing by
    min(floor, omega_max - floor) "to keep omega >= 0" couples the SWING budget to the
    MEAN frequency, so a model hedging toward DC also loses the ability to sweep. Since
    the sign of omega is a gauge, only Nyquist binds. A near-DC floor must still permit a
    swing far larger than the floor itself.
    """
    L = 48.0
    field = _driven("half_integer", L=L, K=4, omega_basis="centered", std=4.0)
    with torch.no_grad():
        field._omega_base.fill_(-6.0)  # push the floor to ~DC: omega_max*sigmoid(-6)~0.008
        cond = torch.randn(32, 16)
        _, omega_floor = field._floor_poles(cond.dtype, cond.device)
        _, omega = field.instantaneous(cond, _grid(32, L, T=193))
    floor = float(omega_floor.mean())
    swing = float((omega.max(dim=1).values - omega.min(dim=1).values).mean())
    assert floor < 0.05, f"test setup: floor should be near DC, got {floor}"
    # Under the buggy bound this was pinned at ~2*floor (<0.1); it must now be O(1).
    assert swing > 1.0, f"swing {swing} collapsed with the floor (the trap is back)"
    assert float(omega.abs().max()) <= math.pi + 1e-5


def test_centered_omega_does_not_inflate_the_mean_frequency():
    """The reason the p96 pilot failed: the mean-1 (nonneg) basis couples omega variation
    to the mean, so a swept frequency drags the mean up. The centered basis keeps the
    time-mean of omega(t) at the floor, so a redundant +/- common mode cancels instead of
    accumulating."""
    L = 48.0
    cond = torch.randn(64, 16)
    t_end = torch.full((64, 1, 1), L)
    means = {}
    for omega_basis in ("nonneg", "centered"):
        field = _driven("half_integer", L=L, omega_basis=omega_basis, std=2.0, seed=1)
        with torch.no_grad():
            # omega_bar(L) / L IS the window-mean of omega(t), in closed form -- free of
            # the O(1/T) quadrature error a sampled mean would carry.
            _, omega_bar = field.integrated(cond, t_end)
            _, omega_floor = field._floor_poles(cond.dtype, cond.device)
        mean_gap = float((omega_bar.squeeze(1) / L - omega_floor.view(1, -1)).abs().mean())
        means[omega_basis] = mean_gap
    # centered: the window-mean of omega sits exactly on the floor (basis integrates to 0).
    assert means["centered"] < 1e-5
    # nonneg: variation drags the mean well above the floor.
    assert means["nonneg"] > 0.1


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
@pytest.mark.parametrize("rho_basis", ["centered", "nonneg"])
def test_rho_stays_within_theorem_b_bounds(basis, rho_basis):
    """Theorem B needs rho >= rho_min > 0; the centered basis additionally caps rho at
    2*floor. Both come from |phi - 1| <= 1, which the signs preserve."""
    L = 48.0
    field = _driven(basis, L=L, rho_basis=rho_basis, std=3.0)
    cond = torch.randn(16, 16)
    t_rel = _grid(16, L, T=257)
    with torch.no_grad():
        rho, _ = field.instantaneous(cond, t_rel)
        rho_floor, _ = field._floor_poles(cond.dtype, cond.device)
    assert float(rho.min()) >= field.rho_min - 1e-6
    if rho_basis == "centered":
        assert float(rho.max()) <= 2.0 * float(rho_floor.max()) + 1e-6


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
def test_centered_rho_adds_no_net_decay(basis):
    """The centered basis's defining property: every element integrates to zero over
    [0, L], so rho_bar(L) = rho_floor * L regardless of the variation. Holds for
    half-integer frequencies because 2*f_m is an integer, hence sin(2 pi f_m) = 0."""
    L = 48.0
    field = _driven(basis, L=L, rho_basis="centered", std=3.0)
    cond = torch.randn(8, 16)
    t_end = torch.full((8, 1, 1), L)
    with torch.no_grad():
        rho_bar, _ = field.integrated(cond, t_end)
        rho_floor, _ = field._floor_poles(cond.dtype, cond.device)
    torch.testing.assert_close(
        rho_bar.squeeze(1), rho_floor.view(1, -1).expand_as(rho_bar.squeeze(1)) * L,
        atol=1e-4, rtol=1e-3,
    )


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
@pytest.mark.parametrize("omega_basis", ["nonneg", "centered"])
def test_integrated_is_the_antiderivative_of_instantaneous(basis, omega_basis):
    """Theorem A: Phi must stay the exact antiderivative of phi under the signs and under
    the centred (Phi - t) omega variation."""
    L = 48.0
    field = _driven(basis, L=L, omega_basis=omega_basis, std=2.0)
    cond = torch.randn(4, 16)
    h = 1e-3
    tc = torch.full((4, 1, 1), 11.7)
    with torch.no_grad():
        rb_plus, ob_plus = field.integrated(cond, tc + h)
        rb_minus, ob_minus = field.integrated(cond, tc - h)
        rho_inst, omega_inst = field.instantaneous(cond, tc)
    # Tolerance is set by the central-difference truncation error at these driven
    # coefficients, not by the antiderivative: a wrong Phi is off by O(1), not 0.1%.
    torch.testing.assert_close((rb_plus - rb_minus) / (2 * h), rho_inst, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close((ob_plus - ob_minus) / (2 * h), omega_inst, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
@pytest.mark.parametrize("omega_basis", ["nonneg", "centered"])
def test_seed_poles_match_instantaneous_at_zero(basis, omega_basis):
    """seed_poles shortcuts phi_m(0); under signs that is 1 + s_m, not a flat 2, and it
    must read the same centred/nonneg convention as instantaneous for BOTH poles."""
    L = 48.0
    field = _driven(basis, L=L, omega_basis=omega_basis, std=2.0)
    cond = torch.randn(6, 16)
    zero = torch.zeros(6, 1, 1)
    with torch.no_grad():
        rho0, omega0 = field.seed_poles(cond)
        rho_inst, omega_inst = field.instantaneous(cond, zero)
    torch.testing.assert_close(rho0, rho_inst.squeeze(1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(omega0, omega_inst.squeeze(1), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
def test_growth_excursion_anchored_at_zero(basis):
    """_growth_terms anchors gamma at t=0 via phi_m(0) and differentiates phi; both used
    to hardcode the all-positive basis. gamma(0) = 0 and |gamma| <= c_g must survive."""
    L, c_g = 48.0, math.log(2.0)
    torch.manual_seed(0)
    field = ChirpModalField(k=4, cond_dim=16, num_basis=8, time_scale=L, rho_init_horizon=L,
                            rho_basis="centered", basis=basis, growth_budget=c_g)
    torch.nn.init.normal_(field.to_growth[-1].weight, std=2.0)
    torch.nn.init.normal_(field.to_growth[-1].bias, std=2.0)
    cond = torch.randn(8, 16)
    t_rel = _grid(8, L)
    with torch.no_grad():
        gamma, _ = field._growth_terms(cond, t_rel)
    torch.testing.assert_close(gamma[:, 0], torch.zeros_like(gamma[:, 0]), atol=1e-6, rtol=0)
    assert float(gamma.abs().max()) <= c_g + 1e-6

    # gamma' is the closed-form derivative of gamma (small-h central difference at a
    # point, as in tests/test_growth_budget.py -- a coarse grid cannot resolve the
    # fastest harmonic).
    h = 1e-3
    tc = torch.full((8, 1, 1), 11.7)
    with torch.no_grad():
        g_plus, _ = field._growth_terms(cond, tc + h)
        g_minus, _ = field._growth_terms(cond, tc - h)
        _, gamma_prime = field._growth_terms(cond, tc)
    torch.testing.assert_close((g_plus - g_minus) / (2 * h), gamma_prime, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("basis", ["integer", "half_integer"])
def test_chirp_is_near_lti_at_eps_init(basis):
    """Strict generalization (Theorem A) must not depend on the harmonic set: at eps-init
    the coefficients are ~0, so the poles are constant whichever basis is used."""
    L = 48.0
    torch.manual_seed(0)
    field = ChirpModalField(k=4, cond_dim=16, num_basis=8, time_scale=L,
                            rho_init_horizon=L, rho_basis="centered", basis=basis)
    cond = torch.randn(4, 16)
    t_rel = _grid(4, L)
    with torch.no_grad():
        rho, omega = field.instantaneous(cond, t_rel)
    # eps-init puts the coefficients at ~1e-8, so the poles are constant to float32 noise.
    assert float((rho.max(dim=1).values - rho.min(dim=1).values).max()) < 1e-5
    assert float((omega.max(dim=1).values - omega.min(dim=1).values).max()) < 1e-5


def test_llapdiff_threads_chirp_basis():
    torch.manual_seed(0)
    model = LLapDiff(data_dim=8, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=4,
                     timesteps=50, denoiser_modal_type="chirp", chirp_num_basis=8,
                     chirp_basis="half_integer")
    field = model.model.chirp_field
    assert field.basis == "half_integer"
    torch.testing.assert_close(
        field.basis_signs, torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    )


def test_basis_signs_stay_out_of_the_state_dict():
    """basis_signs must NOT be persistent. It is a pure function of (basis, num_basis),
    both already in the checkpoint's model_config, and two load paths are strict
    (viz/plot_llapdiff_poles, train_val_llapdiff._load_diff_model) -- a new persistent
    buffer would make every pre-fix chirp checkpoint fail to load.
    """
    field = ChirpModalField(k=2, cond_dim=8, num_basis=8, time_scale=48.0,
                            basis="half_integer")
    keys = field.state_dict().keys()
    assert "basis_signs" not in keys
    assert "basis_freqs" in keys  # pre-existing and persistent; leave it alone

    # A legacy state_dict (integer basis, no signs entry) loads strictly.
    legacy = ChirpModalField(k=2, cond_dim=8, num_basis=8, time_scale=48.0, basis="integer")
    rebuilt = ChirpModalField(k=2, cond_dim=8, num_basis=8, time_scale=48.0, basis="integer")
    rebuilt.load_state_dict(legacy.state_dict())  # strict=True by default
    torch.testing.assert_close(rebuilt.basis_signs, torch.ones(8))


def test_pre_fix_checkpoints_keep_the_legacy_basis():
    """A checkpoint written before this change carries no chirp_basis key and must rebuild
    on the integer basis -- its weights were trained in that function class."""
    payload = {"model_config": {"data_dim": 8, "hidden_dim": 32, "num_layers": 2,
                                "num_heads": 4, "laplace_k": 4, "timesteps": 50,
                                "denoiser_modal_type": "chirp"}}
    cfg = _llapdiff_config_from_checkpoint(payload)
    assert cfg["chirp_basis"] == "integer"
    assert ChirpModalField(k=2, cond_dim=8, num_basis=4, time_scale=48.0).basis == "integer"
