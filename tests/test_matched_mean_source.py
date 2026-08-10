"""Fix 1d — the analytic arm's mean must be the ensemble mean, not one draw.

`generate(eta=0)` is deterministic given `x_T`, but `x_T` is random, so a single call returns
ONE DRAW from the predictive distribution rather than its mean. The sampled arm's point forecast
is the mean of `num_samples` such draws. At `--mean-source ddim` the two arms therefore differ
in LOCATION before any UQ question is asked, and the A1-vs-A2 row reports that difference as if
it were a calibration result: measured on a live S2 checkpoint, MSE 0.394 against 0.093 on the
same weights.

These pin the three things that make the fix real rather than nominal:

1. `ensemble` actually averages N draws, and `ddim` is exactly its N = 1 case — one code path,
   so the arms cannot drift apart for any reason other than the draw count.
2. The averaged mean converges on the true mean as N grows (a single draw does not).
3. The cost is reported honestly: matching the means COSTS the A1-vs-A2 speedup, and the
   report says so rather than leaving a stale ratio in the table.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))

from llapdiffusion.tools import run_analytic_uq_eval as uq  # noqa: E402


class _FakeScheduler:
    timesteps = 100

    def q_sample(self, x0, t, noise):
        return x0 + 0.1 * noise, noise


class _FakeModel:
    """`generate` returns TRUE_MEAN + noise, so the ensemble mean has a known target."""

    TRUE_MEAN = 3.0

    def __init__(self, noise_scale=1.0):
        self.scheduler = _FakeScheduler()
        self.generate_calls = 0
        self.noise_scale = noise_scale

    def generate(self, *, shape, generator=None, **kwargs):
        self.generate_calls += 1
        noise = torch.randn(shape, generator=generator)
        return self.TRUE_MEAN + self.noise_scale * noise

    def __call__(self, x_t, t, *, return_variance=False, **kwargs):
        out = torch.full_like(x_t, 7.0)
        return (out, torch.full_like(x_t, 0.25)) if return_variance else out


_CFG = SimpleNamespace(TEST_STEPS=8, TEST_GUIDANCE=1.0, GUIDANCE_POWER=0.3,
                       NUM_EVAL_SAMPLES=16, SEED=42)
_SHAPE = (4, 6, 3)


def _mean(model, source, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    mean, _ = uq._predict_mean_var(
        model, _CFG, mu_shape=_SHAPE, cond_summary=None, cond_summary_raw=None,
        dt_model=None, mean_source=source, device=torch.device("cpu"),
        ensemble_size=n, generator=g,
    )
    return mean


def test_ddim_is_exactly_the_n_equals_one_case():
    """One code path, so the only difference between the arms is the number of draws."""
    a = _mean(_FakeModel(), "ddim", n=25, seed=11)      # ensemble_size ignored for ddim
    b = _mean(_FakeModel(), "ensemble", n=1, seed=11)
    assert torch.equal(a, b)


def test_ensemble_averages_the_requested_number_of_draws():
    model = _FakeModel()
    _mean(model, "ensemble", n=12)
    assert model.generate_calls == 12

    single = _FakeModel()
    _mean(single, "ddim", n=12)
    assert single.generate_calls == 1, "ddim must stay a single trajectory"


def test_the_averaged_mean_converges_where_a_single_draw_does_not():
    """The property the fix exists for: N=1 carries a full draw's worth of location error."""
    err_1 = (_mean(_FakeModel(), "ddim", n=1, seed=3) - _FakeModel.TRUE_MEAN).abs().mean()
    err_256 = (_mean(_FakeModel(), "ensemble", n=256, seed=3) - _FakeModel.TRUE_MEAN).abs().mean()

    assert float(err_256) < 0.2 * float(err_1)
    assert float(err_256) < 0.1


def test_the_mean_is_reproducible_for_a_given_seed():
    """A '± std over seeds' must not carry estimator noise no seed controls."""
    assert torch.equal(
        _mean(_FakeModel(), "ensemble", n=8, seed=5),
        _mean(_FakeModel(), "ensemble", n=8, seed=5),
    )
    assert not torch.equal(
        _mean(_FakeModel(), "ensemble", n=8, seed=5),
        _mean(_FakeModel(), "ensemble", n=8, seed=6),
    )


def test_oneshot_ignores_the_ensemble_size():
    """A3 IS a single forward — averaging would make it a different arm."""
    model = _FakeModel()
    _mean(model, "oneshot", n=25)
    assert model.generate_calls == 0


# ------------------------------------------------------------------- the cost accounting

def test_matching_the_means_costs_the_a1_vs_a2_speedup_and_the_report_says_so():
    steps, n = 64, 25
    sampled = n * steps
    analytic_matched = uq._denoiser_passes("ensemble", steps=steps, ensemble_size=n)
    analytic_single = uq._denoiser_passes("ddim", steps=steps, ensemble_size=n)
    oneshot = uq._denoiser_passes("oneshot", steps=steps, ensemble_size=n)

    # Matched means: the analytic arm now costs MORE than the sampled one (the +1 is the
    # variance read). There is no A1/A2 speedup left to report.
    assert analytic_matched == sampled + 1
    assert analytic_matched > sampled
    # The speedup survives only against the one-shot arm.
    assert oneshot == 1 and sampled / oneshot == 1600
    # And the unmatched arm is where the old 23.5x came from.
    assert analytic_single == steps + 1


def test_the_label_carries_the_ensemble_size():
    assert uq._mean_source_label("ensemble", 25) == "ensemble25"
    assert uq._mean_source_label("ensemble", 4) == "ensemble4"
    assert uq._mean_source_label("ddim", 25) == "ddim"
    assert uq._mean_source_label("oneshot", 25) == "oneshot"


# ------------------------------------------------------------------- the driver's protocol

def test_the_driver_asks_for_matched_means_on_the_diffusion_arms():
    import run_u3_uq as u3

    assert u3.ARM_EVAL["A1_diffusion_sampled"][1] == "ensemble"
    assert u3.ARM_EVAL["A2_diffusion_analytic"][1] == "ensemble"
    assert u3.ARM_EVAL["A3_oneshot_analytic"][1] == "oneshot", "A3 must stay a single forward"


@pytest.mark.parametrize(
    "a1, a2, expect_matched",
    [("ensemble25", "ensemble25", True), ("ensemble25", "ddim", False)],
)
def test_the_table_refuses_to_present_an_unmatched_row_as_a_result(a1, a2, expect_matched):
    import run_u3_uq as u3

    payload = {
        "dataset": "noaa_uk", "pred": 168, "arm": "d", "split": "val", "weights": "ema",
        "seeds": [0], "coverage_level": 0.8,
        "arms": {
            "A1_diffusion_sampled": {"mean_source": a1},
            "A2_diffusion_analytic": {"mean_source": a2},
            "A3_oneshot_analytic": {"mean_source": "oneshot"},
        },
    }
    md = u3._render_markdown(payload)

    if expect_matched:
        assert "calibration at MATCHED COST" in md
        assert "do NOT share a mean" not in md
    else:
        assert "do NOT share a mean" in md
        assert "must not be read as a UQ result" in md


# ------------------------------------------------- the timestep the law is read at (parity)

class _EnergyModel(_FakeModel):
    """Variance ∝ the modal energy of the x_t it is handed — the real head's structure.

    `LapFormer` computes Var = einsum(s_modal(cond), theta(x_t)²), so the spread scales with
    how noisy the input is. That is why the read timestep changes the law.
    """

    def __call__(self, x_t, t, *, return_variance=False, **kwargs):
        out = torch.full_like(x_t, 7.0)
        if not return_variance:
            return out
        return out, x_t.pow(2).mean() * torch.ones_like(x_t)


class _ScalingScheduler(_FakeScheduler):
    """q_sample with a real noise ramp, so t actually changes |x_t|."""

    def q_sample(self, x0, t, noise):
        frac = (t.float().view(-1, *([1] * (x0.dim() - 1))) / float(self.timesteps))
        return (1.0 - frac) * x0 + frac * 8.0 * noise, noise


def _variance_at(t):
    model = _EnergyModel()
    model.scheduler = _ScalingScheduler()
    g = torch.Generator().manual_seed(0)
    _, var = uq._predict_mean_var(
        model, _CFG, mu_shape=_SHAPE, cond_summary=None, cond_summary_raw=None,
        dt_model=None, mean_source="ensemble", device=torch.device("cpu"),
        ensemble_size=4, variance_timestep=t, generator=g,
    )
    return float(var.mean())


def test_the_read_timestep_changes_the_law():
    """If it did not, A2 reading at 1 and A3 at T-1 would be harmless. It is not."""
    assert _variance_at(1) < 0.25 * _variance_at(99)


def test_the_default_read_matches_where_the_oneshot_arm_reads():
    """Parity: A2 and A3 must read the Theorem-C law in the same noise regime, or the row
    between them reports the read protocol as if it were the arms' calibration."""
    assert _variance_at(None) == pytest.approx(_variance_at(_ScalingScheduler.timesteps - 1))


def test_an_out_of_range_timestep_is_clamped_not_crashed():
    assert _variance_at(10_000) == pytest.approx(_variance_at(_ScalingScheduler.timesteps - 1))
    assert _variance_at(-5) == _variance_at(0)


# --------------------------------------------- averaging the variance read (--variance-draws)

class _NoisyEnergyModel(_FakeModel):
    """Variance = the modal energy of x_t, which at T-1 is dominated by the noise draw."""

    def __call__(self, x_t, t, *, return_variance=False, **kwargs):
        out = torch.full_like(x_t, 7.0)
        if not return_variance:
            return out
        self.variance_reads = getattr(self, "variance_reads", 0) + 1
        return out, x_t.pow(2)


def _read(n, seed):
    model = _NoisyEnergyModel()
    model.scheduler = _ScalingScheduler()
    g = torch.Generator().manual_seed(seed)
    _, var = uq._predict_mean_var(
        model, _CFG, mu_shape=(8, 16, 4), cond_summary=None, cond_summary_raw=None,
        dt_model=None, mean_source="ensemble", device=torch.device("cpu"),
        ensemble_size=2, variance_draws=n, generator=g,
    )
    return var, model


def test_variance_draws_costs_exactly_n_forwards():
    _, model = _read(4, seed=0)
    assert model.variance_reads == 4
    _, model = _read(1, seed=0)
    assert model.variance_reads == 1


def test_averaging_shrinks_the_per_element_draw_noise():
    """The estimator noise falls like 1/sqrt(n); the point of the flag."""
    def spread(n):
        a, _ = _read(n, seed=1)
        b, _ = _read(n, seed=2)
        return float(((a - b).abs() / a.clamp_min(1e-9)).median())

    s1, s16 = spread(1), spread(16)
    assert s16 < 0.45 * s1, f"expected ~1/4 of the n=1 spread, got {s16:.3f} vs {s1:.3f}"


def test_the_cost_accounting_counts_every_variance_read():
    """A2's passes must not understate what it actually spends."""
    base = uq._denoiser_passes("ensemble", steps=64, ensemble_size=25, variance_draws=1)
    four = uq._denoiser_passes("ensemble", steps=64, ensemble_size=25, variance_draws=4)
    assert base == 25 * 64 + 1 and four == 25 * 64 + 4

    # A3 is a single forward that yields mean AND variance together: averaging would change
    # the arm, not just its estimator, so the flag must not touch it.
    assert uq._denoiser_passes("oneshot", steps=64, ensemble_size=25, variance_draws=8) == 1


def test_oneshot_ignores_variance_draws():
    model = _NoisyEnergyModel()
    g = torch.Generator().manual_seed(0)
    uq._predict_mean_var(
        model, _CFG, mu_shape=_SHAPE, cond_summary=None, cond_summary_raw=None,
        dt_model=None, mean_source="oneshot", device=torch.device("cpu"),
        variance_draws=8, generator=g,
    )
    assert getattr(model, "variance_reads", 0) == 1
