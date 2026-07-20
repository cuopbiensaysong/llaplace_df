"""Tests for pole-recovery diagnostics: the modal_capture plumbing and the
output-contribution ranking (h2_pole_recovery_problems_fixes.md P3/P7/P9).

The old recovery figure ranked modes by coefficient variation energy without
consulting the residues, so it plotted unconstrained junk modes. These tests pin
the corrected path: capture (theta, poles) from the actual generation's final
denoising step and rank by per-mode output energy E_k.
"""

import math

import pytest
import torch

from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.viz.plot_llapdiff_poles import modal_contributions


def _model(modal_type: str, timesteps: int = 20) -> LLapDiff:
    torch.manual_seed(0)
    return LLapDiff(
        data_dim=8, hidden_dim=32, num_layers=2, num_heads=4,
        laplace_k=4, timesteps=timesteps, denoiser_modal_type=modal_type,
    ).eval()


def _dt(B: int, T: int) -> torch.Tensor:
    return torch.cumsum(torch.rand(B, T) + 0.1, dim=1)


def test_forward_fills_chirp_capture():
    model = _model("chirp")
    B, T = 2, 6
    capture: dict = {}
    with torch.no_grad():
        model(torch.randn(B, T, 8), torch.randint(1, 20, (B,)), dt=_dt(B, T), modal_capture=capture)
    assert capture["modal_type"] == "chirp"
    assert capture["theta"].shape == (B, 8, 8)  # [B, 2K, D]
    for key in ("rho_bar", "omega_bar", "rho_inst", "omega_inst"):
        assert capture[key].shape == (B, T, 4)


def test_forward_fills_lti_capture():
    model = _model("lti")
    B, T = 2, 6
    capture: dict = {}
    with torch.no_grad():
        model(torch.randn(B, T, 8), torch.randint(1, 20, (B,)), dt=_dt(B, T), modal_capture=capture)
    assert capture["modal_type"] == "lti"
    assert capture["theta"].shape == (B, 8, 8)
    assert capture["rho_const"].shape[-1] == 4
    assert capture["omega_const"].shape[-1] == 4


def test_forward_without_capture_unchanged():
    model = _model("chirp")
    B, T = 2, 6
    x, t, dt = torch.randn(B, T, 8), torch.randint(1, 20, (B,)), _dt(B, T)
    with torch.no_grad():
        base = model(x, t, dt=dt)
        again = model(x, t, dt=dt, modal_capture={})
    torch.testing.assert_close(base, again)


@pytest.mark.parametrize("modal_type", ["chirp", "lti"])
def test_generate_captures_final_step(modal_type):
    """modal_capture records the FINAL denoising step (t_idx == 0 when the full
    schedule is walked), i.e. the poles/residues that produced the forecast."""
    model = _model(modal_type)
    B, T = 2, 5
    capture: dict = {}
    out = model.generate(
        shape=(B, T, 8), steps=20, guidance_strength=1.0,
        dt=_dt(B, T), generator=torch.Generator().manual_seed(0),
        modal_capture=capture,
    )
    assert out.shape == (B, T, 8)
    assert capture["t_idx"] == 0
    assert capture["theta"].shape == (B, 8, 8)
    assert capture["modal_type"] == modal_type


def test_modal_contributions_analytic():
    """E_k = mean_t exp(-2 rho_bar_k(t)) * (||c_k||^2 + ||b_k||^2), and the
    effective trajectories are the E_k-weighted means over ALL modes."""
    B, T, K, D = 1, 4, 3, 2
    theta = torch.zeros(B, 2 * K, D)
    theta[0, 0] = torch.tensor([3.0, 4.0])   # c_0: norm^2 = 25
    theta[0, K + 1] = torch.tensor([1.0, 0.0])  # b_1: norm^2 = 1
    t_rel = torch.arange(1.0, T + 1).view(1, T, 1)
    rho_bar = torch.stack(
        [0.1 * t_rel[..., 0], 0.5 * t_rel[..., 0], 50.0 * t_rel[..., 0]], dim=-1
    )  # [B,T,K]
    omega = torch.stack(
        [0.3 * torch.ones(B, T), 0.7 * torch.ones(B, T), 3.0 * torch.ones(B, T)], dim=-1
    )
    capture = {
        "modal_type": "chirp", "theta": theta, "t_rel": t_rel,
        "rho_bar": rho_bar, "omega_bar": omega, "rho_inst": 2.0 * omega, "omega_inst": omega,
    }
    out = modal_contributions(capture)

    env = torch.exp(-2.0 * rho_bar).mean(dim=1)  # [B,K]
    expected = env * torch.tensor([[25.0, 1.0, 0.0]])
    torch.testing.assert_close(out["energy"], expected)
    torch.testing.assert_close(out["energy_share"].sum(-1), torch.ones(B))
    w = expected.unsqueeze(1)
    torch.testing.assert_close(
        out["omega_eff"], (omega * w).sum(-1) / w.sum(-1).clamp_min(1e-30)
    )
    assert out["t_rel"].shape == (B, T)


def test_modal_contributions_lti_expands_constant_poles():
    B, T, K, D = 2, 5, 3, 4
    theta = torch.randn(B, 2 * K, D)
    rho_c = torch.rand(B, K) * 0.2 + 0.01
    omega_c = torch.rand(B, K) * math.pi
    t_rel = torch.cumsum(torch.rand(B, T) + 0.1, dim=1).unsqueeze(-1)
    capture = {
        "modal_type": "lti", "theta": theta, "t_rel": t_rel,
        "rho_const": rho_c, "omega_const": omega_c,
    }
    out = modal_contributions(capture)
    assert out["rho"].shape == (B, T, K)
    # Constant in time, matching the captured poles.
    torch.testing.assert_close(out["omega"][:, 0, :], omega_c)
    torch.testing.assert_close(out["omega"][:, -1, :], omega_c)
    env = torch.exp(-2.0 * rho_c.unsqueeze(1) * t_rel).mean(dim=1)
    res2 = theta[:, :K].pow(2).sum(-1) + theta[:, K:].pow(2).sum(-1)
    torch.testing.assert_close(out["energy"], env * res2)


def test_contribution_ranking_ignores_zero_residue_junk_modes():
    """The P3 regression: a mode with huge pole variation but zero residues must
    rank BELOW a small quiet mode that carries all the output."""
    B, T, K, D = 1, 6, 4, 2
    theta = torch.zeros(B, 2 * K, D)
    theta[0, 2] = torch.tensor([1.0, 1.0])  # only mode 2 synthesizes anything
    t_rel = torch.arange(1.0, T + 1).view(1, T, 1)
    # Mode 0: junk — enormous decay (the old criterion's favorite). Mode 2: usable.
    rho_bar = torch.ones(B, T, K) * t_rel
    rho_bar[..., 0] *= 100.0
    rho_bar[..., 2] *= 0.01
    capture = {
        "modal_type": "chirp", "theta": theta, "t_rel": t_rel,
        "rho_bar": rho_bar, "omega_bar": rho_bar,
        "rho_inst": torch.ones(B, T, K), "omega_inst": torch.ones(B, T, K),
    }
    out = modal_contributions(capture)
    assert int(out["energy"].argmax(dim=-1)) == 2
    assert float(out["energy_share"][0, 2]) > 0.999
