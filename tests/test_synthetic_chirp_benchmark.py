"""Tests for the H2 ground-truth chirp benchmark pieces: the synthetic pole-profile
generator (+ persisted ground truth), the pole-trajectory extraction, and the
Prop.-A.1 companion-vs-normal-form numerics."""

import numpy as np
import pytest
import torch

from llapdiffusion.datasets.synthetic_regime_dataset import (
    CHIRP_TASKS,
    SyntheticRegimeCacheConfig,
    load_ground_truth_poles,
    prepare_synthetic_regime_cache,
)
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.viz.plot_llapdiff_poles import extract_chirp_pole_trajectories


def _prep(tmp_path, task, **overrides):
    cfg = SyntheticRegimeCacheConfig(
        task=task,
        window=32,
        horizon=16,
        data_dir=str(tmp_path / task),
        num_entities=3,
        series_length=160,
        change_point=120,
        seed=7,
        overwrite=True,
        **overrides,
    )
    prepare_synthetic_regime_cache(cfg)
    return cfg


@pytest.mark.parametrize("task", CHIRP_TASKS)
def test_chirp_tasks_generate_cache_with_pole_truth(tmp_path, task):
    cfg = _prep(tmp_path, task)
    truth = load_ground_truth_poles(cfg.data_dir)
    assert sorted(truth) == [0, 1, 2]
    for payload in truth.values():
        assert payload["rho"].shape == (cfg.series_length,)
        assert payload["omega"].shape == (cfg.series_length,)
        assert np.isfinite(payload["rho"]).all() and np.isfinite(payload["omega"]).all()


def test_linear_chirp_truth_matches_formula(tmp_path):
    cfg = _prep(tmp_path, "synthetic_linear_chirp", shared_poles=True)
    truth = load_ground_truth_poles(cfg.data_dir)
    omega = truth[0]["omega"]
    # Linear ramp from 2*pi*f0 to 2*pi*f0*freq_multiplier.
    assert omega[0] < omega[-1]
    np.testing.assert_allclose(omega[-1] / omega[0], cfg.freq_multiplier, rtol=1e-5)
    diffs = np.diff(omega.astype(np.float64))
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-3, atol=1e-7)  # constant slope
    # rho stays at the constant base decay.
    rho = truth[0]["rho"]
    np.testing.assert_allclose(rho, rho[0], rtol=1e-6)


def test_quadratic_chirp_is_convex_increasing(tmp_path):
    cfg = _prep(tmp_path, "synthetic_quadratic_chirp")
    omega = load_ground_truth_poles(cfg.data_dir)[0]["omega"].astype(np.float64)
    assert (np.diff(omega) >= -1e-9).all()
    assert (np.diff(omega, 2) >= -1e-7).all()  # accelerating (quadratic) chirp


def test_ramp_damping_directions(tmp_path):
    up = load_ground_truth_poles(_prep(tmp_path, "synthetic_ramp_damping_up").data_dir)[0]["rho"]
    down = load_ground_truth_poles(_prep(tmp_path, "synthetic_ramp_damping_down").data_dir)[0]["rho"]
    assert (np.diff(up.astype(np.float64)) > 0).all()
    assert (np.diff(down.astype(np.float64)) < 0).all()
    assert (up > 0).all() and (down > 0).all()


def test_growth_decay_envelope_peaks_at_change_point(tmp_path):
    cfg = _prep(tmp_path, "synthetic_growth_decay")
    rho = load_ground_truth_poles(cfg.data_dir)[0]["rho"].astype(np.float64)
    assert (rho[: cfg.change_point] < 0).all()  # envelope grows before the change point
    assert (rho[cfg.change_point :] > 0).all()
    # Total growth equals the configured log-amplitude budget.
    np.testing.assert_allclose(-rho[: cfg.change_point].sum(), cfg.growth_log_amplitude, rtol=1e-4)
    envelope_log = -np.cumsum(rho)
    assert int(envelope_log.argmax()) == cfg.change_point - 1


def test_shared_poles_gives_identical_truth_across_entities(tmp_path):
    cfg = _prep(tmp_path, "synthetic_linear_chirp", shared_poles=True)
    truth = load_ground_truth_poles(cfg.data_dir)
    for aid in (1, 2):
        np.testing.assert_array_equal(truth[aid]["omega"], truth[0]["omega"])
        np.testing.assert_array_equal(truth[aid]["rho"], truth[0]["rho"])


def test_extract_chirp_pole_trajectories_shapes_and_lti_rejection():
    torch.manual_seed(0)
    B, S, HID, K, H = 2, 5, 32, 6, 12
    model = LLapDiff(data_dim=8, hidden_dim=HID, num_layers=2, num_heads=4,
                     laplace_k=K, timesteps=50, denoiser_modal_type="chirp").eval()
    cond = torch.randn(B, S, HID)
    t_grid = torch.arange(1, H + 1, dtype=torch.float32)

    traj = extract_chirp_pole_trajectories(
        model, t_idx=1, cond_summary=cond, cond_summary_raw=cond, t_grid=t_grid, top_modes=3
    )
    assert traj["rho"].shape == (B, H, 3)
    assert traj["omega"].shape == (B, H, 3)
    assert traj["mode_indices"].shape == (B, 3)
    assert traj["variation_energy"].shape == (B, K)
    assert torch.isfinite(traj["rho"]).all() and torch.isfinite(traj["omega"]).all()
    assert (traj["omega"] > 0).all() and (traj["rho"] > 0).all()

    lti = LLapDiff(data_dim=8, hidden_dim=HID, num_layers=2, num_heads=4,
                   laplace_k=K, timesteps=50).eval()
    with pytest.raises(ValueError):
        extract_chirp_pole_trajectories(
            lti, t_idx=1, cond_summary=cond, cond_summary_raw=cond, t_grid=t_grid
        )


def test_companion_form_error_dwarfs_normal_form_error():
    """Prop A.1 numerics: naive exp(int A) is exact for the normal form, wrong for
    the companion realization of the same chirped oscillator."""
    from llapdiffusion.tools.plot_companion_vs_normal_form import (
        _closed_form_normal,
        _companion_A,
        _naive_exp_companion,
        _normal_A,
        _rk4_transition,
    )

    t_grid = np.linspace(0.0, 10.0, 21)
    omega0, alpha, rho = 1.0, 0.15, 0.05

    comp_ref = _rk4_transition(lambda t: _companion_A(t, omega0, alpha), t_grid, 2e-3)
    comp_err = np.linalg.norm(_naive_exp_companion(t_grid, omega0, alpha) - comp_ref, axis=(1, 2))
    norm_ref = _rk4_transition(lambda t: _normal_A(t, omega0, alpha, rho), t_grid, 2e-3)
    norm_err = np.linalg.norm(_closed_form_normal(t_grid, omega0, alpha, rho) - norm_ref, axis=(1, 2))

    assert norm_err.max() < 1e-6  # Theorem A: closed form == reference
    assert comp_err.max() > 1e-1  # Prop A.1: naive exponential is not the propagator
    assert comp_err.max() > 1e3 * norm_err.max()


def test_regular_sampling_bitcompatible_grid(tmp_path):
    """Default gap_distribution='regular' keeps the historical dense hourly grid."""
    cfg = _prep(tmp_path, "synthetic_linear_chirp")
    truth = load_ground_truth_poles(cfg.data_dir)
    times = truth[0]["times"].astype(np.float64)
    np.testing.assert_allclose(np.diff(times), 1.0, atol=1e-9)  # unit gaps
    assert times[0] == 0.0
    import json
    meta = json.loads((tmp_path / "synthetic_linear_chirp" / "cache_ratio_index" / "meta.json").read_text())
    gen = meta["generation_config"]
    assert gen["gap_distribution"] == "regular"
    assert gen["gap_var_realized"] == 0.0


def test_gamma_gaps_statistics_shared_grid_and_meta(tmp_path):
    """Gamma renewal gaps: irregular strictly-increasing shared grid with the
    advertised tunable moments (Var = mean^2 / shape), recorded in the meta."""
    cfg = _prep(
        tmp_path, "synthetic_linear_chirp",
        gap_distribution="gamma", gap_mean=1.0, gap_shape=2.0,
    )
    truth = load_ground_truth_poles(cfg.data_dir)
    times = truth[0]["times"].astype(np.float64)
    gaps = np.diff(times)
    assert (gaps > 0).all()
    assert gaps.std() > 0.1  # genuinely irregular
    assert abs(gaps.mean() - 1.0) < 0.25
    # All entities share the SAME grid (joint-panel batching requirement).
    for aid in (1, 2):
        np.testing.assert_array_equal(truth[aid]["times"], truth[0]["times"])
    import json
    meta = json.loads((tmp_path / "synthetic_linear_chirp" / "cache_ratio_index" / "meta.json").read_text())
    gen = meta["generation_config"]
    assert gen["gap_distribution"] == "gamma" and gen["gap_shape"] == 2.0
    # Var(Delta) = mean^2/shape = 0.5 — realized within statistical slack (n=160).
    assert 0.2 < gen["gap_var_realized"] < 1.0


def test_gap_aware_discretization_matches_closed_form():
    """With constant poles, unit amplitude, zero baseline/phase/noise, the generated
    signal equals sin(2*pi*f*t) * exp(-d*t) at the cumulative renewal times — the
    gap-aware discretization is exact for constant poles."""
    from llapdiffusion.datasets.synthetic_regime_dataset import _generate_signal

    cfg = SyntheticRegimeCacheConfig(
        task="synthetic_freq_shift",
        series_length=120, change_point=119, window=32, horizon=16,
        amplitude_min=1.0, amplitude_max=1.0,
        baseline_min=0.0, baseline_max=0.0,
        phase_min=0.0, phase_max=0.0,
        noise_std=0.0,
        gap_distribution="gamma", gap_mean=1.0, gap_shape=3.0,
    )
    rng = np.random.default_rng(5)
    gaps = np.maximum(rng.gamma(3.0, 1.0 / 3.0, size=120), 1e-3)
    f0, d0 = 0.03, 0.008
    signal, frequency, decay = _generate_signal(
        cfg, np.random.default_rng(9), gaps, base_frequency=f0, base_decay=d0
    )
    t_cum = np.cumsum(gaps)  # historical convention: the first gap is counted
    expected = np.exp(-d0 * t_cum) * np.sin(2 * np.pi * f0 * t_cum)
    np.testing.assert_allclose(
        signal[: cfg.change_point].astype(np.float64),
        expected[: cfg.change_point],
        atol=1e-5,
    )
    np.testing.assert_allclose(frequency[: cfg.change_point], f0, rtol=1e-6)
    np.testing.assert_allclose(decay, d0, rtol=1e-6)


def test_benchmark_gap_tag_in_cache_dir(tmp_path):
    from types import SimpleNamespace

    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _cache_dir, _gap_tag

    args = SimpleNamespace(
        data_root=str(tmp_path), series_length=768, change_point=None,
        num_entities=8, gap_distribution="gamma", gap_mean=1.0, gap_shape=4.0,
    )
    assert _gap_tag(args) == "gaps-gamma-m1-k4"
    assert "gaps-gamma-m1-k4" in str(_cache_dir("synthetic_linear_chirp", args))
    args.gap_distribution = "regular"
    assert "gaps-regular" in str(_cache_dir("synthetic_linear_chirp", args))


def test_benchmark_geometry_validation():
    """The purged split needs the val band to exceed one horizon; short series must
    be rejected with a helpful message (288/96/48 is the classic structural trap)."""
    from types import SimpleNamespace

    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _validate_geometry

    ok = SimpleNamespace(series_length=768, window=96, horizon=48)
    _validate_geometry(ok)  # should not raise
    bad = SimpleNamespace(series_length=288, window=96, horizon=48)
    with pytest.raises(ValueError, match="series_length"):
        _validate_geometry(bad)


def test_benchmark_configure_sets_arm_fields(tmp_path):
    from types import SimpleNamespace

    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _configure, _summary_rows

    args = SimpleNamespace(
        window=32, horizon=16, series_length=160, change_point=None, num_entities=3,
        data_root=str(tmp_path / "data"), artifact_root=str(tmp_path / "art"),
        gap_distribution="gamma", gap_mean=1.0, gap_shape=4.0,
        smoke=True, verbose=False, debug=False,
    )
    cfg = _configure("synthetic_linear_chirp", "chirp", 3, args)
    assert cfg.DENOISER_MODAL_TYPE == "chirp"
    assert cfg.DENOISER_OUTPUT_HEAD == "auto"
    assert cfg.SEED == 3
    assert cfg.EPOCHS == 1  # smoke schedule
    cfg_lti = _configure("synthetic_linear_chirp", "lti", 3, args)
    assert cfg_lti.DENOISER_MODAL_TYPE == "lti"
    # Both arms share the same frozen upstream artifacts (parity requirement).
    assert cfg_lti.VAE_CKPT == cfg.VAE_CKPT and cfg_lti.SUM_CKPT == cfg.SUM_CKPT
    assert cfg_lti.CKPT_DIR != cfg.CKPT_DIR  # but not the diffusion checkpoints

    rows = [
        {"task": "t", "arm": "lti", "crps": 1.0, "mae": 1.0, "mse": 1.0},
        {"task": "t", "arm": "lti", "crps": 3.0, "mae": 1.0, "mse": 1.0},
    ]
    summary = _summary_rows(rows)
    assert summary[0]["runs"] == 2 and summary[0]["crps_mean"] == 2.0
