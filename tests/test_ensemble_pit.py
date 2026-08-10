"""`ensemble_pit` — the sampled-arm calibration estimator Table 3's A1 was missing.

Without it, A1 has no PIT/coverage and the A1-vs-A2 comparison rests on CRPS alone — the
metric the recovery plan's Phase 3 proves blind to a 3 % -> 87 % coverage swing. These pin
that it agrees with the Gaussian estimator on a Gaussian, so the two arms are scored by
estimators that actually match.
"""

import pytest
import torch

from llapdiffusion.models.uq_metrics import (
    ensemble_pit,
    gaussian_pit,
    pit_calibration_error,
    reliability_curve,
)


def test_calibrated_ensemble_is_uniform():
    """Draws from the true predictive law give PIT ~ U(0,1), i.e. ~0 calibration error."""
    torch.manual_seed(0)
    y = torch.randn(4000)
    draws = y.unsqueeze(0) + torch.randn(200, 4000)      # y is one more draw from N(y, 1)
    u = ensemble_pit(draws, y + torch.randn(4000))       # observation from the same law

    assert u.min() >= 0.0 and u.max() <= 1.0
    assert pit_calibration_error(u) < 0.02


@pytest.mark.parametrize("scale, direction", [(0.25, "over"), (4.0, "under")])
def test_miscalibration_is_detected_in_the_right_direction(scale, direction):
    """An over-confident ensemble pushes PIT mass to the tails; an under-confident one
    piles it at the centre. This is exactly the failure CRPS cannot see."""
    torch.manual_seed(1)
    y = torch.randn(4000)
    draws = torch.randn(200, 4000) * scale

    u = ensemble_pit(draws, y)
    tails = ((u < 0.05) | (u > 0.95)).float().mean().item()

    assert pit_calibration_error(u) > 0.05
    if direction == "over":
        assert tails > 0.30, "an over-confident ensemble must put mass in the tails"
    else:
        assert tails < 0.02, "an under-confident ensemble must avoid the tails"


def test_agrees_with_the_gaussian_estimator_on_a_gaussian():
    """A1 and A2/A3 must be scored by the SAME estimator, not two lookalikes: on a Gaussian
    law the ensemble form has to reproduce the closed form."""
    torch.manual_seed(2)
    y = torch.randn(500)
    mean, var = torch.zeros(500), torch.ones(500)
    draws = torch.randn(20000, 500)

    u_ens = ensemble_pit(draws, y)
    u_gauss = gaussian_pit(y, mean, var)

    assert (u_ens - u_gauss).abs().mean() < 0.01
    assert abs(pit_calibration_error(u_ens) - pit_calibration_error(u_gauss)) < 0.01


@pytest.mark.parametrize("S", [5, 25])
def test_exact_at_the_ensemble_size_the_pipeline_ACTUALLY_runs(S):
    """🔴 The regression test the first implementation lacked, and would have failed.

    Every other test here sits at S = 100-20 000; U3 runs `NUM_EVAL_SAMPLES = 25` and the
    campaign val profile runs 5. The naive `(below + 0.5*ties)/S` form reads PIT-ECE 0.0132 and
    coverage 0.846 at nominal 0.9 on a PERFECTLY calibrated ensemble at S = 25 -- i.e. it
    manufactures ~5 pt of A1 under-coverage out of nothing. Tolerances here are 15x tighter
    than the rest of the file precisely because this is the operating point.
    """
    g = torch.Generator().manual_seed(7)
    y = torch.randn(400_000, generator=g)
    draws = torch.randn(S, 400_000, generator=g)

    u = ensemble_pit(draws, y, generator=g)
    lo, hi = 0.05, 0.95
    coverage = float(((u >= lo) & (u <= hi)).float().mean())

    assert pit_calibration_error(u) < 0.002
    assert coverage == pytest.approx(0.9, abs=0.01)


def test_randomization_is_reproducible_with_a_generator():
    """A calibration number must not move between runs of the same command."""
    draws, y = torch.randn(25, 1000), torch.randn(1000)
    a = ensemble_pit(draws, y, generator=torch.Generator().manual_seed(0))
    b = ensemble_pit(draws, y, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(a, b)


def test_reliability_curve_consumes_it_unchanged():
    torch.manual_seed(3)
    y = torch.randn(2000)
    u = ensemble_pit(y.unsqueeze(0) + torch.randn(100, 2000), y + torch.randn(2000))
    curve = reliability_curve(u)
    assert curve and all(0.0 <= float(v) <= 1.0 for v in curve.values())


def test_rejects_a_degenerate_ensemble():
    with pytest.raises(ValueError, match="at least 2 draws"):
        ensemble_pit(torch.zeros(1, 5), torch.zeros(5))
