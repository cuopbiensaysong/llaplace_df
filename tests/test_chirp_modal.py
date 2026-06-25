"""Tests for the chirp-modal (time-varying pole) denoiser variant.

Covers the Direction-1 method (chirp_modal_method.md): the chirp synthesizer is a strict
generalization of the LTI core (Theorem A), is stable by construction (Theorem B), and is
flag-gated so existing LTI checkpoints stay loadable.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from llapdiffusion.models.laptrans import (
    ChirpModalField,
    LaplacePseudoInverse,
    LaplaceTransformEncoder,
    normalize_modal_type,
)
from llapdiffusion.models.llapdiff import LLapDiff


def _t_rel(B: int, T: int) -> torch.Tensor:
    return torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, 1).contiguous()


def test_normalize_modal_type():
    assert normalize_modal_type("LTI") == "lti"
    assert normalize_modal_type(" Chirp ") == "chirp"
    with pytest.raises(ValueError):
        normalize_modal_type("nope")


def test_chirp_basis_recovers_lti_with_constant_poles():
    """Constant poles (rho_bar=rho*t, omega_bar=omega*t) ==> chirp basis == LTI basis."""
    torch.manual_seed(0)
    B, T, K = 3, 7, 5
    t_rel = _t_rel(B, T)
    rho = torch.rand(B, K) * 0.2 + 0.01
    omega = torch.rand(B, K) * math.pi

    lti = LaplaceTransformEncoder.basis_matrix(t_rel, rho, omega)
    rho_bar = t_rel * rho.unsqueeze(1)  # [B,T,K]
    omega_bar = t_rel * omega.unsqueeze(1)
    chirp = LaplaceTransformEncoder.chirp_basis_matrix(rho_bar, omega_bar)
    torch.testing.assert_close(lti, chirp)


def test_synthesis_equivalence_lti_vs_chirp():
    """Same residues + constant poles ==> identical synthesized trajectory (strict gen.)."""
    torch.manual_seed(0)
    B, T, K, D = 2, 6, 4, 8
    enc = LaplaceTransformEncoder(k=K, feat_dim=D, hidden_dim=16, num_heads=4)
    synth = LaplacePseudoInverse(enc, use_mlp_residual=False)

    theta = torch.randn(B, 2 * K, D)
    rho = torch.rand(B, K) * 0.2 + 0.01
    omega = torch.rand(B, K) * math.pi
    t_rel = _t_rel(B, T)

    y_lti = synth(theta, rho=rho, omega=omega, dt=t_rel.squeeze(-1))
    # Reconstruct the same t_rel the LTI path used, then drive the chirp path with it.
    t_rel_used = enc.relative_time(B, T, theta.dtype, theta.device, dt=t_rel.squeeze(-1))
    rho_bar = t_rel_used * rho.unsqueeze(1)
    omega_bar = t_rel_used * omega.unsqueeze(1)
    y_chirp = synth(theta, rho_bar=rho_bar, omega_bar=omega_bar)
    torch.testing.assert_close(y_lti, y_chirp)


def test_chirp_field_zero_coeffs_is_constant_pole():
    """At init the coeff head is zero, so integrated poles are exactly rho_floor * t."""
    torch.manual_seed(0)
    B, T, K, C = 2, 6, 5, 16
    field = ChirpModalField(k=K, cond_dim=C, num_basis=8)
    cond = torch.randn(B, C)
    t_rel = _t_rel(B, T)

    rho_bar, omega_bar = field.integrated(cond, t_rel)
    rho_floor, omega_floor = field._floor_poles(t_rel.dtype, t_rel.device)
    torch.testing.assert_close(rho_bar, t_rel * rho_floor.view(1, 1, K))
    torch.testing.assert_close(omega_bar, t_rel * omega_floor.view(1, 1, K))

    # Seed poles at t=0 equal the floor when coeffs vanish.
    rho0, omega0 = field.seed_poles(cond)
    torch.testing.assert_close(rho0, rho_floor.view(1, K).expand(B, K).contiguous())
    torch.testing.assert_close(omega0, omega_floor.view(1, K).expand(B, K).contiguous())


def test_chirp_field_integral_correctness():
    """rho_bar(0)=0, monotone increasing, and d/dt rho_bar matches instantaneous rho.

    Uses a fixed time_scale L so the field is a pure pointwise function of t (the default
    data-adaptive L = max|t_rel| would change between the +-h finite-difference calls).
    """
    torch.manual_seed(0)
    B, K, C, M = 1, 3, 16, 6
    L = 4.0
    field = ChirpModalField(k=K, cond_dim=C, num_basis=M, time_scale=L)
    # Activate the time-varying part.
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=0.5)
    cond = torch.randn(B, C)

    # Boundary: integrated poles vanish at t=0.
    z = torch.zeros(B, 1, 1)
    rho_bar0, omega_bar0 = field.integrated(cond, z)
    torch.testing.assert_close(rho_bar0, torch.zeros_like(rho_bar0))
    torch.testing.assert_close(omega_bar0, torch.zeros_like(omega_bar0))

    # Monotonicity (instantaneous decay > 0 => integrated decay strictly increasing).
    t_rel = torch.linspace(0.0, 3.0, 50).view(1, 50, 1)
    rho_bar, _ = field.integrated(cond, t_rel)
    assert (rho_bar[:, 1:, :] - rho_bar[:, :-1, :] > 0).all()

    # Finite-difference derivative vs the closed-form instantaneous rho (freq normalized by L).
    h = 1e-3
    tc = torch.full((B, 1, 1), 1.234)
    rb_plus, _ = field.integrated(cond, tc + h)
    rb_minus, _ = field.integrated(cond, tc - h)
    deriv = (rb_plus - rb_minus) / (2 * h)  # [B,1,K]

    rho_floor, _ = field._floor_poles(tc.dtype, tc.device)
    a_rho2, _ = field._coeffs(cond)  # [B,K,M]
    two_pi_f = (2.0 * math.pi) * field.basis_freqs / L
    phi = 1.0 + torch.cos(tc * two_pi_f)  # [B,1,M]
    inst = rho_floor.view(1, 1, K) + torch.einsum("bkm,btm->btk", a_rho2, phi)
    torch.testing.assert_close(deriv, inst, atol=1e-3, rtol=1e-3)


def test_chirp_is_nondegenerate_at_native_horizon():
    """With time normalization, the time-varying part is non-negligible at a long horizon."""
    torch.manual_seed(0)
    B, K, C, M, H = 1, 3, 16, 6, 168.0
    field = ChirpModalField(k=K, cond_dim=C, num_basis=M)
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=0.5)  # activate time-variation
    cond = torch.randn(B, C)
    t_rel = torch.linspace(0.0, H, 400).view(1, 400, 1)
    rho_bar, _ = field.integrated(cond, t_rel)  # [B,T,K]
    # remove the best-fit linear-in-t ramp per mode; the residual is the genuine "wiggle".
    t = t_rel.squeeze(-1).squeeze(0)
    for k in range(K):
        y = rho_bar[0, :, k]
        slope = (t * y).sum() / (t * t).sum()
        wiggle = (y - slope * t).abs().max()
        assert wiggle / (slope * H).abs().clamp_min(1e-9) > 1e-2  # was ~2e-4 before the fix


def test_chirp_contraction_bound():
    """||y(t)|| <= e^{-rho_min t} * sum_k sqrt(||c_k||^2 + ||b_k||^2)  (Theorem B)."""
    torch.manual_seed(0)
    B, T, K, D, C = 2, 12, 4, 6, 16
    rho_min = 1e-2
    field = ChirpModalField(k=K, cond_dim=C, num_basis=8, rho_min=rho_min)
    torch.nn.init.normal_(field.to_coeffs[-1].weight, std=0.5)
    enc = LaplaceTransformEncoder(k=K, feat_dim=D, hidden_dim=16, num_heads=4)
    synth = LaplacePseudoInverse(enc, use_mlp_residual=False)

    cond = torch.randn(B, C)
    theta = torch.randn(B, 2 * K, D)
    t_rel = torch.linspace(0.0, 5.0, T).view(1, T, 1).expand(B, T, 1).contiguous()

    rho_bar, omega_bar = field.integrated(cond, t_rel)
    y = synth(theta, rho_bar=rho_bar, omega_bar=omega_bar)  # [B,T,D]

    c = theta[:, :K, :]
    b = theta[:, K:, :]
    amp = torch.sqrt(c.pow(2).sum(-1) + b.pow(2).sum(-1)).sum(-1)  # [B]
    bound = torch.exp(-rho_min * t_rel.squeeze(-1)) * amp.unsqueeze(1)  # [B,T]
    norm = y.norm(dim=-1)  # [B,T]
    assert (norm <= bound + 1e-5).all()


def test_full_model_contraction_bound():
    """Theorem B holds for the ACTUAL LapFormer/LLapDiff output, not just the synthesizer.

    The LTI output head (LayerNorm + Linear) is dropped in chirp mode; without that fix a
    trained-like head re-inflates the decaying envelope and the output plateaus.
    """
    torch.manual_seed(0)
    B, T, D, K = 2, 64, 8, 4
    model = LLapDiff(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4,
                     laplace_k=K, timesteps=50, denoiser_modal_type="chirp").eval()
    assert not model.model._use_output_head  # head removed in chirp mode
    # Simulate a *trained* model: perturb output-side params away from zero-init.
    for name, p in model.named_parameters():
        if any(s in name for s in ("head_proj", "head_norm", "output_skip_scale",
                                   "chirp_field.to_coeffs")):
            torch.nn.init.normal_(p, std=0.5) if p.dim() > 0 else p.data.fill_(0.7)

    x = torch.randn(B, T, D)
    tstep = torch.randint(0, 50, (B,))
    dt = torch.linspace(0.0, 5.0, T).view(1, T).expand(B, T).contiguous()  # t_rel >= 0
    with torch.no_grad():
        y = model(x, tstep, dt=dt)

    norm = y.norm(dim=-1)  # [B,T]
    # (a) far end <= early peak (envelope decays) and (b) finite everywhere (Theorem B(iii)).
    assert norm[:, -1].max() <= norm[:, : T // 4].max()
    assert torch.isfinite(y).all()


def test_lapformer_chirp_forward_shapes_and_finite():
    torch.manual_seed(0)
    B, T, D, K = 2, 5, 8, 6
    common = dict(data_dim=D, hidden_dim=32, num_layers=2, num_heads=4, laplace_k=K, timesteps=50)
    x = torch.randn(B, T, D)
    tstep = torch.randint(0, 50, (B,))
    # Irregular query offsets.
    dt = torch.sort(torch.rand(B, T), dim=1).values

    model = LLapDiff(**common, denoiser_modal_type="chirp").eval()
    assert model.model.chirp_field is not None
    assert model.model.synthesis.use_mlp_residual is False
    with torch.no_grad():
        y = model(x, tstep, dt=dt)
    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()


def test_checkpoint_missing_modal_type_defaults_to_lti():
    from llapdiffusion.trainers.train_val_llapdiff import _llapdiff_config_from_checkpoint

    payload = {"model_config": {"llapdiff": {"data_dim": 8, "hidden_dim": 32}}}
    cfg = _llapdiff_config_from_checkpoint(payload)
    assert cfg["denoiser_modal_type"] == "lti"


def test_run_preds_routes_chirp_into_its_own_dirs(monkeypatch, tmp_path):
    """chirp runs nest under modal-chirp/ so they don't overwrite lti checkpoints."""
    from llapdiffusion import pipeline

    calls = []
    cfg = SimpleNamespace(
        PREDICT_TYPE="v",
        DENOISER_MODAL_TYPE="chirp",
        OUT_DIR=str(tmp_path / "out"),
        CKPT_DIR=str(tmp_path / "ckpt"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "pole_plots"),
    )

    def fake_run_single_pred(pred, **kwargs):
        calls.append((kwargs["base_out_dir"], kwargs["base_ckpt_dir"]))
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)

    assert pipeline.run_preds([5], config=cfg) == {5: {"eval_stats": {}}}
    assert calls == [(tmp_path / "out" / "modal-chirp", tmp_path / "ckpt" / "modal-chirp")]
    assert cfg.OUT_DIR == str(tmp_path / "out" / "modal-chirp")
    assert cfg.POLE_PLOT_DIR == str(tmp_path / "out" / "modal-chirp" / "pole_plots")


def test_run_preds_composes_predict_then_modal_routing(monkeypatch, tmp_path):
    """predict-type and modal-type segments compose: predict-x0/modal-chirp."""
    from llapdiffusion import pipeline

    calls = []
    cfg = SimpleNamespace(
        PREDICT_TYPE="x0",
        DENOISER_MODAL_TYPE="chirp",
        OUT_DIR=str(tmp_path / "out"),
        CKPT_DIR=str(tmp_path / "ckpt"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "pole_plots"),
    )

    def fake_run_single_pred(pred, **kwargs):
        calls.append((kwargs["base_out_dir"], kwargs["base_ckpt_dir"]))
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)

    assert pipeline.run_preds([5], config=cfg) == {5: {"eval_stats": {}}}
    assert calls == [
        (tmp_path / "out" / "predict-x0" / "modal-chirp",
         tmp_path / "ckpt" / "predict-x0" / "modal-chirp")
    ]


def test_run_preds_default_lti_keeps_base_dirs(monkeypatch, tmp_path):
    """The default lti core adds no segment (paths unchanged)."""
    from llapdiffusion import pipeline

    calls = []
    cfg = SimpleNamespace(
        PREDICT_TYPE="v",
        DENOISER_MODAL_TYPE="lti",
        OUT_DIR=str(tmp_path / "out"),
        CKPT_DIR=str(tmp_path / "ckpt"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "pole_plots"),
    )

    def fake_run_single_pred(pred, **kwargs):
        calls.append((kwargs["base_out_dir"], kwargs["base_ckpt_dir"]))
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)

    assert pipeline.run_preds([5], config=cfg) == {5: {"eval_stats": {}}}
    assert calls == [(tmp_path / "out", tmp_path / "ckpt")]
    assert cfg.OUT_DIR == str(tmp_path / "out")
