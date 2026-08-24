"""Path-level scores, checked against their definitions and against known-good limits.

The point of these scores is to detect a method that gets marginals right and dependence
wrong, so the tests that matter most are the ones that separate those two cases: a sampler
with correct marginals but destroyed cross-time structure must score *worse* on the
variogram and energy scores while scoring identically on CRPS.
"""

import math

import pytest
import torch

from llapdiffusion.models.path_scores import (
    coverage_and_width,
    crps,
    energy_score,
    gaussian_joint_nll_per_channel,
    mixture_nll,
    randomized_pit,
    simultaneous_band_coverage,
    variogram_score,
    variogram_weights,
)


# --------------------------------------------------------------------------------------
# estimator parity
# --------------------------------------------------------------------------------------

def test_energy_score_estimators_differ_in_the_documented_direction_and_size():
    """naive - unbiased = +spread/(2S). The naive estimator is biased UPWARD, which is the
    direction that flatters an arm scored in closed form against a sampled one."""
    torch.manual_seed(0)
    S, B, T, D = 25, 4, 6, 2
    samples = torch.randn(S, B, T, D)
    y = torch.randn(B, T, D)
    es = energy_score(samples, y)
    gap = es["naive"] - es["unbiased"]
    assert (gap > 0).all()

    # Recover the spread term and check the gap is exactly spread/(2S).
    iu, ju = torch.triu_indices(S, S, offset=1)
    spread = (samples[iu] - samples[ju]).pow(2).flatten(start_dim=-2).sum(-1).sqrt().mean(0)
    torch.testing.assert_close(gap, spread / (2 * S), rtol=1e-5, atol=1e-6)


def test_crps_estimators_differ_and_converge_as_S_grows():
    torch.manual_seed(0)
    y = torch.zeros(1, 1, 1)
    gaps = []
    for S in (10, 100, 1000):
        s = torch.randn(S, 1, 1, 1)
        c = crps(s, y)
        gaps.append(float(c["naive"] - c["unbiased"]))
    assert all(g > 0 for g in gaps)
    assert gaps[0] > gaps[1] > gaps[2]


def test_energy_score_reduces_to_crps_in_one_dimension():
    """CRPS is the 1-D energy score; if the two disagree, one of them is wrong."""
    torch.manual_seed(1)
    samples = torch.randn(40, 3, 1, 1)
    y = torch.randn(3, 1, 1)
    es = energy_score(samples, y)
    cr = crps(samples, y)
    torch.testing.assert_close(es["unbiased"], cr["unbiased"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(es["naive"], cr["naive"], rtol=1e-5, atol=1e-6)


def test_crps_of_a_point_mass_is_the_absolute_error():
    y = torch.tensor([[[2.0]]])
    samples = torch.full((16, 1, 1, 1), 0.5)
    assert float(crps(samples, y)["unbiased"]) == pytest.approx(1.5, abs=1e-6)


def test_energy_score_is_minimised_by_the_true_distribution():
    """Propriety, empirically: the correct sampler must beat a shifted and an
    over-dispersed one."""
    torch.manual_seed(0)
    truth = torch.randn(4000, 1, 5, 1)
    y = torch.randn(1, 5, 1)
    good = float(energy_score(truth, y)["unbiased"].mean())
    shifted = float(energy_score(truth + 1.0, y)["unbiased"].mean())
    wide = float(energy_score(truth * 3.0, y)["unbiased"].mean())
    assert good < shifted and good < wide


# --------------------------------------------------------------------------------------
# the property CRPS cannot see
# --------------------------------------------------------------------------------------

def _ar1(S, T, phi, generator):
    """Stationary AR(1): every marginal is N(0,1) and Corr(X_r, X_r') = phi^|r-r'|.

    Stationarity is what makes the shuffle test meaningful -- permuting values across time
    then preserves each marginal *exactly* and changes only the lag structure. A process
    with a common level plus i.i.d. noise would not work: its pairwise differences are
    exchangeable, so a within-path permutation leaves the variogram score untouched. That
    is a real blind spot of the score, not of this implementation.
    """
    x = torch.randn(S, 1, generator=generator)
    out = [x]
    for _ in range(T - 1):
        x = phi * x + math.sqrt(1 - phi ** 2) * torch.randn(S, 1, generator=generator)
        out.append(x)
    return torch.cat(out, dim=1).view(S, 1, T, 1)


def test_scores_rank_dependence_the_way_the_paper_claims():
    """Two forecasts with *identical* marginals and opposite dependence: the true AR(1)
    against i.i.d. noise. This is the discrimination the paper's path-level argument rests
    on, so it is measured rather than assumed.

    Averaged over 120 truths at S=1000 the observed ratios (i.i.d. / correct) are:

        CRPS       1.0000   -- blind by construction: it is computed per query time
        energy     1.046    -- strictly proper, but discriminates dependence only weakly,
                                which is why certifying coherence with it alone would be
                                choosing the instrument least able to detect the problem
        variogram  2.785    -- the score that actually sees it

    Thresholds below are set well inside those measurements.
    """
    g = torch.Generator().manual_seed(0)
    S, T, phi, n_truth = 1000, 10, 0.9, 120
    t = torch.arange(float(T)).view(1, T)
    fc_true = _ar1(S, T, phi, g)       # correct dependence
    fc_iid = _ar1(S, T, 0.0, g)        # same marginals, no dependence

    acc = {k: [0.0, 0.0] for k in ("crps", "energy", "variogram")}
    for _ in range(n_truth):
        y = _ar1(1, T, phi, g)[0]
        for i, fc in enumerate((fc_true, fc_iid)):
            acc["crps"][i] += float(crps(fc, y)["unbiased"].mean())
            acc["energy"][i] += float(energy_score(fc, y)["unbiased"].mean())
            acc["variogram"][i] += float(variogram_score(fc, y, t).mean())
    ratio = {k: v[1] / v[0] for k, v in acc.items()}

    assert abs(ratio["crps"] - 1.0) < 0.01, f"CRPS should be ~blind here: {ratio['crps']:.4f}"
    assert 1.01 < ratio["energy"] < 1.20, f"energy ratio out of range: {ratio['energy']:.4f}"
    assert ratio["variogram"] > 2.0, f"variogram failed to discriminate: {ratio['variogram']:.4f}"
    # And the ordering the paper relies on: VS is the sharper instrument.
    assert ratio["variogram"] > ratio["energy"] > ratio["crps"]


def test_variogram_weights_are_predeclared_and_truncated():
    t = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
    w = variogram_weights(t)
    assert torch.diagonal(w[0]).abs().max() == 0.0            # no self-pairs
    torch.testing.assert_close(w[0, 0, 1], torch.tensor(1.0))
    torch.testing.assert_close(w[0, 0, 3], torch.tensor(0.1))
    assert w[0, 0, 1] > w[0, 0, 3]                            # nearer pairs weigh more

    # Truncation caps the weight of a near-coincident pair at 1/median gap.
    t2 = torch.tensor([[0.0, 1e-9, 1.0, 2.0]])
    w2 = variogram_weights(t2)
    median_gap = torch.tensor([1e-9, 1.0, 1.0]).median()
    assert float(w2[0, 0, 1]) == pytest.approx(1.0 / float(median_gap), rel=1e-5)


def test_variogram_score_is_zero_for_a_deterministic_perfect_forecast():
    y = torch.randn(2, 5, 1)
    samples = y.unsqueeze(0).expand(8, 2, 5, 1).contiguous()
    t = torch.arange(5.0).view(1, 5).expand(2, 5)
    assert float(variogram_score(samples, y, t).abs().max()) < 1e-8


# --------------------------------------------------------------------------------------
# likelihoods
# --------------------------------------------------------------------------------------

def test_joint_nll_matches_a_hand_computed_gaussian():
    torch.manual_seed(0)
    T = 4
    A = torch.randn(T, T)
    cov = A @ A.T + torch.eye(T)
    mean = torch.randn(T)
    y = torch.randn(T)

    got = gaussian_joint_nll_per_channel(
        y.view(1, T, 1), mean.view(1, T, 1), cov.view(1, T, T, 1), jitter=0.0
    )
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    torch.testing.assert_close(got[0], -dist.log_prob(y), rtol=1e-4, atol=1e-5)


def test_joint_nll_sums_independent_channels():
    torch.manual_seed(0)
    T, D = 3, 2
    cov = torch.stack([torch.eye(T) * (d + 1.0) for d in range(D)], dim=-1).unsqueeze(0)
    y, mean = torch.randn(1, T, D), torch.zeros(1, T, D)
    total = gaussian_joint_nll_per_channel(y, mean, cov, jitter=0.0)
    parts = sum(
        float(gaussian_joint_nll_per_channel(
            y[..., d : d + 1], mean[..., d : d + 1], cov[..., d : d + 1], jitter=0.0
        ))
        for d in range(D)
    )
    assert float(total) == pytest.approx(parts, rel=1e-5)


def test_joint_nll_respects_the_mask():
    torch.manual_seed(0)
    T = 4
    cov = torch.eye(T).view(1, T, T, 1)
    y, mean = torch.randn(1, T, 1), torch.zeros(1, T, 1)
    mask = torch.tensor([[[True], [False], [True], [False]]])
    got = float(gaussian_joint_nll_per_channel(y, mean, cov, mask, jitter=0.0))
    kept = y[0, [0, 2], 0]
    want = float(0.5 * (kept.pow(2).sum() + 2 * math.log(2 * math.pi)))
    assert got == pytest.approx(want, rel=1e-5)


def test_mixture_nll_is_logsumexp_not_the_average():
    nlls = torch.tensor([[1.0], [5.0]])
    got = float(mixture_nll(nlls))
    want = -(math.log((math.exp(-1.0) + math.exp(-5.0)) / 2))
    assert got == pytest.approx(want, rel=1e-6)
    # Strictly below the average of the components -- the quantity the paper declines to
    # report, because it is easier.
    assert got < float(nlls.mean())


def test_mixture_of_identical_components_equals_a_single_component():
    n = torch.full((7, 3), 2.5)
    torch.testing.assert_close(mixture_nll(n), n[0])


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------

def test_randomized_pit_is_uniform_for_a_calibrated_ensemble():
    """The naive (below + 0.5 ties)/S rank is biased at finite S; the randomized rank is
    not. Checked at S=25, the protocol's ensemble size."""
    torch.manual_seed(0)
    S, N = 25, 40_000
    samples = torch.randn(S, N, 1, 1)
    y = torch.randn(N, 1, 1)
    u = randomized_pit(samples, y, generator=torch.Generator().manual_seed(0))
    assert float(u.mean()) == pytest.approx(0.5, abs=0.01)
    hist = torch.histc(u, bins=20, min=0.0, max=1.0) / u.numel()
    assert float((hist - 0.05).abs().max()) < 0.006


def test_coverage_is_nominal_for_a_calibrated_ensemble_and_width_grows_with_level():
    torch.manual_seed(0)
    samples = torch.randn(2000, 200, 1, 1)
    y = torch.randn(200, 1, 1)
    out = coverage_and_width(samples, y)
    for lv in (0.5, 0.8, 0.9):
        assert out[f"coverage_{lv}"] == pytest.approx(lv, abs=0.05)
    assert out["width_0.5"] < out["width_0.9"] < out["width_0.99"]


def test_simultaneous_band_coverage_reports_the_path_maximum():
    y = torch.tensor([[[0.0], [3.0], [1.0]]])
    mean = torch.zeros(1, 3, 1)
    sd = torch.ones(1, 3, 1)
    assert float(simultaneous_band_coverage(y, mean, sd)) == pytest.approx(3.0)

    # A pointwise band at 2 sigma covers 2 of 3 times but not the trajectory.
    lam = simultaneous_band_coverage(y, mean, sd)
    assert float(lam) > 2.0


def test_simultaneous_band_is_wider_than_the_pointwise_band_for_a_real_process():
    """Path coverage costs more than marginal coverage -- the gap is what the statistic is
    for, and a band that ignored it would over-promise."""
    torch.manual_seed(0)
    N, T = 5000, 12
    draws = torch.randn(N, T, 1)
    lam = simultaneous_band_coverage(draws, torch.zeros(N, T, 1), torch.ones(N, T, 1))
    lam95 = float(torch.quantile(lam, 0.95))
    assert lam95 > 1.96, "simultaneous 95% band must exceed the pointwise 95% band"
