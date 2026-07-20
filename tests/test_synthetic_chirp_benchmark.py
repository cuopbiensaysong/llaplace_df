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


def test_select_top_modes_escalates_until_share_threshold():
    """P3/P9: selection is by output-energy share, escalating top-N (x2, cap 16)
    until the selected modes explain the threshold share of the output."""
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _select_top_modes

    # One dominant mode: base N suffices and stays minimal.
    share = np.array([0.9, 0.05, 0.03, 0.02] + [0.0] * 12)
    sel, s, valid, n = _select_top_modes(share, top_n=2, threshold=0.5)
    assert list(sel) == [0, 1] and valid and n == 2 and s == pytest.approx(0.95)

    # Spread energy: 2 -> 4 -> 8 modes needed to pass 50%.
    share = np.full(16, 1.0 / 16.0)
    sel, s, valid, n = _select_top_modes(share, top_n=2, threshold=0.5)
    assert n == 8 and valid and s == pytest.approx(0.5)

    # Unreachable threshold: caps at 16 and flags invalid.
    share = np.full(64, 1.0 / 64.0)
    sel, s, valid, n = _select_top_modes(share, top_n=4, threshold=0.5)
    assert n == 16 and not valid and s == pytest.approx(0.25)


def test_select_top_modes_ranks_by_contribution_not_variation():
    """The junk-drawer regression: a zero-share mode must never be selected ahead
    of the modes that actually synthesize the forecast."""
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _select_top_modes

    share = np.array([0.0, 0.6, 0.0, 0.4])
    sel, s, valid, n = _select_top_modes(share, top_n=2, threshold=0.5)
    assert set(sel.tolist()) == {1, 3} and valid


def test_stratified_pick_spreads_across_window_starts():
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _stratified_pick

    cands = [(i // 4, i % 4, 0, 500 + i) for i in range(40)]  # starts 500..539
    picked = _stratified_pick(cands, 4)
    starts = [c[3] for c in picked]
    assert len(picked) == 4
    assert starts[0] == 500 and starts[-1] == 539  # covers both ends of the span
    assert min(np.diff(starts)) >= 10  # roughly even spacing, not the first batch
    # Fewer candidates than requested: keep them all.
    assert len(_stratified_pick(cands[:2], 4)) == 2


def test_native_step_grid_assertion():
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _assert_native_step_grid

    _assert_native_step_grid(np.cumsum(np.full(48, 1.0)))  # regular native steps
    _assert_native_step_grid(np.cumsum(np.random.default_rng(0).gamma(4.0, 0.25, 48)))
    with pytest.raises(AssertionError, match="native steps"):
        _assert_native_step_grid(np.cumsum(np.full(48, 3600.0)))  # seconds, not steps
    with pytest.raises(AssertionError, match="native steps"):
        _assert_native_step_grid(np.zeros(48))  # non-increasing


def _fig_window(aid=0, start=500, valid=True, n_modes=3, H=12, arm="chirp", seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full(H, 1.0))
    shares = rng.dirichlet(np.ones(n_modes)) * (0.9 if valid else 0.2)
    return {
        "arm": arm, "asset_id": aid, "window_start": start,
        "t_norm_span": (0.7, 0.75),
        "t_grid": t,
        "mode_ids": np.arange(n_modes),
        "mode_shares": shares,
        "rho_modes": rng.uniform(0.01, 0.3, (H, n_modes)),
        "omega_modes": rng.uniform(0.2, 0.6, (H, n_modes)),
        "rho_eff": rng.uniform(0.01, 0.3, H),
        "omega_eff": rng.uniform(0.2, 0.6, H),
        "rho_true": np.full(H, 0.02),
        "omega_true": np.linspace(0.4, 0.5, H),
        "selected_share": float(shares.sum()),
        "selection_valid": bool(valid),
    }


def test_plot_recovery_small_multiples_with_lti_overlay(tmp_path):
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _plot_recovery

    chirp = [_fig_window(start=s, seed=s) for s in (500, 550, 600)]
    lti = [dict(_fig_window(start=s, n_modes=2, arm="lti", seed=s + 100),
                omega_eff=np.full(12, 0.45), rho_eff=np.full(12, 0.1)) for s in (500, 550, 600)]
    out = tmp_path / "recovery.pdf"
    _plot_recovery(chirp, lti, out, title="test")
    assert out.exists() and out.stat().st_size > 0
    # Missing lti payload must not break the chirp-only figure.
    out2 = tmp_path / "recovery_no_lti.pdf"
    _plot_recovery(chirp, [], out2, title="test")
    assert out2.exists()
    # Invalid selection still renders (watermarked), never crashes.
    out3 = tmp_path / "recovery_invalid.pdf"
    _plot_recovery([_fig_window(valid=False)], [], out3, title="test")
    assert out3.exists()


def test_sweep_period_triangle_puts_full_excursion_in_every_window():
    """P5: with sweep_period ~ (window+horizon), the pole excursion recurs inside
    every window — including the tail test windows — instead of once per series."""
    from llapdiffusion.datasets.synthetic_regime_dataset import (
        SyntheticRegimeCacheConfig,
        _pole_profiles,
    )

    L, period, H = 768, 144.0, 48
    cfg = SyntheticRegimeCacheConfig(
        task="synthetic_linear_chirp", series_length=L, change_point=576,
        sweep_period=period,
    )
    times = np.arange(L, dtype=np.float64)
    base_f = 1.0 / 32.0
    freq, _ = _pole_profiles(
        cfg, base_frequency=base_f, base_decay=0.005,
        t_norm=times / times[-1], times_h=times,
    )
    # Bounded by the multiplier range and periodic in absolute time.
    assert freq.min() == pytest.approx(base_f) and freq.max() == pytest.approx(2.0 * base_f)
    np.testing.assert_allclose(freq[: L - int(period)], freq[int(period):], rtol=1e-6)
    # Every horizon-length slice in the tail test region sees real variation,
    # and a typical one sees >= 30% (the doc's acceptance).
    ratios = [
        (freq[s : s + H].max() - freq[s : s + H].min()) / freq[s : s + H].min()
        for s in range(600, L - H)
    ]
    assert min(ratios) >= 0.15 and float(np.mean(ratios)) >= 0.30

    # Default None keeps the legacy series-long monotone ramp.
    cfg_legacy = SyntheticRegimeCacheConfig(
        task="synthetic_linear_chirp", series_length=L, change_point=576,
    )
    freq_legacy, _ = _pole_profiles(
        cfg_legacy, base_frequency=base_f, base_decay=0.005,
        t_norm=times / times[-1], times_h=times,
    )
    np.testing.assert_allclose(
        freq_legacy, base_f * (1.0 + (2.0 - 1.0) * times / times[-1]), rtol=1e-6
    )


def test_sweep_period_rejected_for_piecewise_tasks(tmp_path):
    from llapdiffusion.datasets.synthetic_regime_dataset import (
        SyntheticRegimeCacheConfig,
        prepare_synthetic_regime_cache,
    )

    for task in ("synthetic_freq_shift", "synthetic_growth_decay"):
        cfg = SyntheticRegimeCacheConfig(
            task=task, window=32, horizon=16, series_length=160, change_point=120,
            num_entities=2, data_dir=str(tmp_path / task), sweep_period=64.0,
        )
        with pytest.raises(ValueError, match="sweep_period"):
            prepare_synthetic_regime_cache(cfg)


def test_benchmark_sweep_tag_and_task_gating(tmp_path):
    from types import SimpleNamespace

    from llapdiffusion.tools.run_synthetic_chirp_benchmark import (
        _cache_dir,
        _task_sweep_period,
    )

    args = SimpleNamespace(
        data_root=str(tmp_path), series_length=768, change_point=None,
        num_entities=8, gap_distribution="gamma", gap_mean=1.0, gap_shape=4.0,
        sweep_period=144.0,
    )
    assert _task_sweep_period("synthetic_linear_chirp", args) == 144.0
    assert _task_sweep_period("synthetic_freq_shift", args) is None  # piecewise
    assert "_sweep-144" in _cache_dir("synthetic_linear_chirp", args).name
    assert "_sweep" not in _cache_dir("synthetic_freq_shift", args).name
    args.sweep_period = None
    assert "_sweep" not in _cache_dir("synthetic_linear_chirp", args).name


def test_plot_cross_window_stitched_figure(tmp_path):
    from llapdiffusion.datasets.synthetic_regime_dataset import (
        SyntheticRegimeCacheConfig,
        prepare_synthetic_regime_cache,
    )
    from llapdiffusion.tools.run_synthetic_chirp_benchmark import _plot_cross_window

    cache = tmp_path / "cache"
    prepare_synthetic_regime_cache(
        SyntheticRegimeCacheConfig(
            task="synthetic_linear_chirp", window=32, horizon=16, series_length=160,
            change_point=120, num_entities=2, data_dir=str(cache), shared_poles=True,
        )
    )
    chirp = [_fig_window(start=s, H=16, seed=s) for s in (60, 90)]
    lti = [dict(_fig_window(start=s, H=16, arm="lti", seed=s + 7),
                omega_eff=np.full(16, 0.4), rho_eff=np.full(16, 0.05)) for s in (60, 90)]
    out = tmp_path / "stitched.pdf"
    _plot_cross_window(chirp, lti, str(cache), out, title="test", window=32)
    assert out.exists() and out.stat().st_size > 0
