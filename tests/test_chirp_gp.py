"""Theorem-C machinery, checked against the theory rather than against itself.

These are the correctness gates the paper reports as appendix Table A9: the variance
recurrence against dense quadrature, the constant-pole closed form, the small-increment
series branch, and Markov joint sampling against a Cholesky of the materialized kernel.
A closed-form predictive law is only worth its name if it agrees with the integral it
claims to evaluate, so each test compares against an independently computed quantity.
"""

import math

import pytest
import torch

from llapdiffusion.models.chirp_gp import (
    cross_time_kernel,
    cross_time_kernel_full,
    marginal_law,
    modal_energies,
    modal_increments,
    modal_lambda_projective,
    sample_joint_markov,
)
from llapdiffusion.models.laptrans import ChirpModalField


def _constant_pole_setup(B=2, T=25, K=3, D=4, rho=None, omega=None, q=None, p0=None, tmax=4.0):
    """Constant instantaneous poles, where Theorem C degenerates to a closed form."""
    torch.manual_seed(0)
    rho = torch.tensor([0.05, 0.3, 0.9])[:K] if rho is None else rho
    omega = torch.tensor([0.7, 1.9, 3.1])[:K] if omega is None else omega
    t = torch.linspace(tmax / T, tmax, T).view(1, T, 1).expand(B, T, 1).contiguous()
    rho_bar = rho.view(1, 1, K) * t
    omega_bar = omega.view(1, 1, K) * t
    q = torch.rand(B, K) * 0.5 + 0.2 if q is None else q
    p0 = torch.rand(B, K) * 0.5 + 0.1 if p0 is None else p0
    theta = torch.randn(B, 2 * K, D)
    return dict(theta=theta, rho_bar=rho_bar, omega_bar=omega_bar, t=t, q=q, p0=p0,
                rho=rho, omega=omega, B=B, T=T, K=K, D=D)


# --------------------------------------------------------------------------------------
# The quadrature
# --------------------------------------------------------------------------------------

def test_modal_increments_is_at_least_as_accurate_as_the_sequential_loop():
    """One quadrature, two consumers -- and the vectorised one must not be a downgrade.

    ``modal_increments`` evaluates the recurrence as a log-domain cumulative sum rather than a
    Python loop, so it is not bit-identical to ``ChirpModalField.modal_variance``: the
    summation order differs. Bit-equality is therefore the wrong assertion. What must hold is
    that the two agree to float32 precision *and* that the vectorised form is no less accurate
    against a float64 reference -- otherwise the speedup would have been bought with error.
    """
    s = _constant_pole_setup()
    got = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    want = ChirpModalField.modal_variance(s["rho_bar"], s["t"], s["q"], s["p0"])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    ref = ChirpModalField.modal_variance(
        s["rho_bar"].double(), s["t"].double(), s["q"].double(), s["p0"].double()
    )
    err = lambda x: float(((x.double() - ref).abs() / ref.abs()).max())
    assert err(got) <= err(want), (
        f"vectorised quadrature is less accurate than the loop: {err(got):.3e} vs {err(want):.3e}"
    )


def _dense_v(rho_fn, q_val, times, *, n_per_unit=200_000):
    """v(t) = int_0^t e^{-2[rhobar(t) - rhobar(s)]} q ds by dense float64 trapezoid.

    The grid is rebuilt per query time so it lands exactly on t: truncating a shared grid
    at the largest node <= t leaves an O(grid spacing) bias, which at 2e4 nodes is ~1e-4
    relative -- big enough to masquerade as a defect in the recurrence under test.
    """
    out = []
    for ti in times:
        n = max(int(n_per_unit * float(ti)), 1000)
        s = torch.linspace(0.0, float(ti), n + 1, dtype=torch.float64)
        integrand = torch.exp(-2.0 * (rho_fn(torch.tensor(float(ti))) - rho_fn(s))) * q_val
        out.append(torch.trapz(integrand, s))
    return torch.stack(out)


def test_variance_recurrence_is_exact_for_piecewise_constant_decay():
    """Gate A9 row 1. The exponential integrator is *exact* when the instantaneous decay is
    piecewise constant on the query grid, which is the regime it is derived for."""
    torch.manual_seed(0)
    T, q_val = 24, 0.6
    dt = torch.rand(T, dtype=torch.float64) * 0.2 + 0.05        # irregular gaps
    times = torch.cumsum(dt, 0)
    rho_seg = torch.rand(T, dtype=torch.float64) * 0.8 + 0.1    # constant within each segment
    rho_bar = torch.cumsum(rho_seg * dt, 0)

    lam = modal_increments(
        rho_bar.view(1, T, 1), times.view(1, T, 1),
        torch.tensor([[q_val]], dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64),
    )["lam"][0, :, 0]

    # Independent reference: integrate segment by segment in closed form, propagating the
    # carried-over mass through the exact transition e^{-2 d_rhobar}.
    ref, acc = [], torch.tensor(0.0, dtype=torch.float64)
    for r in range(T):
        d_rho = rho_seg[r] * dt[r]
        acc = torch.exp(-2 * d_rho) * acc + q_val * (1 - torch.exp(-2 * d_rho)) / (2 * rho_seg[r])
        ref.append(acc)
    ref = torch.stack(ref)

    rel = ((lam - ref).abs() / ref.abs()).max()
    assert rel < 1e-6, f"piecewise-constant exactness: rel err {rel:.3e}"


def test_variance_recurrence_converges_to_the_true_integral_for_a_drifting_pole():
    """The honest quadrature claim: under a genuinely time-varying rho the integrand has no
    elementary antiderivative, so the recurrence is a quadrature and carries discretization
    error. Pin that it (a) is small at a realistic grid and (b) actually converges."""
    q_val = 0.5
    a, b, L = 0.4, 0.3, 2.0

    def rho_bar_fn(t):          # rho(s) = a + b cos(2 pi s / L)  =>  closed-form integral
        return a * t + (b * L / (2 * math.pi)) * torch.sin(2 * math.pi * t / L)

    errs = {}
    for T in (16, 32, 64, 128):
        times = torch.linspace(4.0 / T, 4.0, T, dtype=torch.float64)
        lam = modal_increments(
            rho_bar_fn(times).view(1, T, 1), times.view(1, T, 1),
            torch.tensor([[q_val]], dtype=torch.float64),
            torch.zeros(1, 1, dtype=torch.float64),
        )["lam"][0, :, 0]
        ref = _dense_v(rho_bar_fn, q_val, times, n_per_unit=20_000)
        errs[T] = float(((lam - ref).abs() / ref.abs()).max())

    assert errs[128] < 2e-3, f"quadrature error at a fine grid: {errs[128]:.3e}"
    # Refining the grid must actually help -- a flat curve would mean a bias, not a
    # discretization error.
    assert errs[128] < errs[16] / 4, f"no convergence with grid refinement: {errs}"


def test_constant_pole_closed_form_to_float_precision():
    """Constant rho: v = q (1 - e^{-2 rho t})/(2 rho), the algebraic-Lyapunov steady state."""
    s = _constant_pole_setup()
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    rho, t = s["rho"].view(1, 1, -1), s["t"]
    closed = torch.exp(-2 * rho * t) * s["p0"].unsqueeze(1) + s["q"].unsqueeze(1) * (
        -torch.expm1(-2 * rho * t)
    ) / (2 * rho)
    torch.testing.assert_close(lam, closed, rtol=1e-6, atol=1e-7)

    # And the t -> infinity limit is the algebraic-Lyapunov steady state q / (2 rho).
    t_far = torch.full((1, 1, 1), 400.0)                       # [B,T,1], one query
    lam_far = modal_increments(
        rho * t_far, t_far, s["q"][:1], s["p0"][:1]
    )["lam"]
    torch.testing.assert_close(
        lam_far[0, 0], (s["q"][0] / (2 * s["rho"])), rtol=1e-5, atol=1e-6
    )


def test_small_increment_branch_is_accurate_and_finite():
    """Gate A9 row 4: the (1-e^{-2x})/(2x) series branch, rel err <= 1e-8 against float64."""
    B, T, K = 1, 6, 1
    t = torch.linspace(1e-9, 6e-9, T).view(1, T, 1).double()
    rho_bar = torch.full((B, T, K), 0.0, dtype=torch.float64)
    q = torch.ones(B, K, dtype=torch.float64)
    p0 = torch.zeros(B, K, dtype=torch.float64)
    lam = modal_increments(rho_bar, t, q, p0)["lam"]
    # rho == 0 exactly: v(t) = q * t.
    torch.testing.assert_close(lam[0, :, 0], t[0, :, 0], rtol=1e-8, atol=0)
    assert torch.isfinite(lam).all()


def test_time_varying_q_is_accepted_and_reduces_to_the_constant_case():
    s = _constant_pole_setup()
    q_t = s["q"].unsqueeze(1).expand(s["B"], s["T"], s["K"]).contiguous()
    a = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    b = modal_increments(s["rho_bar"], s["t"], q_t, s["p0"])["lam"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------------------

def test_kernel_diagonal_equals_the_marginal_variance():
    """Cov(z(t), z(t)) must be Var(z(t)) -- the marginal is a special case, not a parallel
    formula."""
    s = _constant_pole_setup()
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    _, var = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    torch.testing.assert_close(torch.diagonal(ker, dim1=1, dim2=2).permute(0, 2, 1), var)


def test_per_channel_kernel_matches_the_constant_pole_closed_form():
    """For t >= t':  K_d = Lambda(t') e^{-rho (t-t')} cos(omega (t-t')) (c_d^2 + b_d^2)."""
    s = _constant_pole_setup()
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    energy = modal_energies(s["theta"], s["K"])                       # [B,K,D]

    # Built with an explicit O(T^2) loop rather than vectorised tricks: Lambda is taken at
    # the EARLIER QUERY, which is an index, not the smaller Lambda value. Those coincide only
    # while Lambda is increasing, i.e. p0 < q/(2 rho) -- false for the fast-decaying modes
    # in this setup, so an elementwise `minimum` silently tests the wrong formula.
    t = s["t"][:, :, 0]
    want = torch.zeros_like(ker)
    for bi in range(s["B"]):
        for r in range(s["T"]):
            for c in range(s["T"]):
                lo = min(r, c)
                dt_ = abs(float(t[bi, r] - t[bi, c]))
                w = lam[bi, lo] * torch.exp(-s["rho"] * dt_) * torch.cos(s["omega"] * dt_)
                want[bi, r, c] = (w.unsqueeze(-1) * energy[bi]).sum(0)
    torch.testing.assert_close(ker, want, rtol=1e-5, atol=1e-6)


def test_per_channel_kernel_is_symmetric_but_the_full_one_is_not():
    """The antisymmetric part of C_k Rot C_k^T has a zero diagonal, so it survives only in
    the cross-channel blocks. Both facts are load-bearing and easy to get backwards."""
    s = _constant_pole_setup(D=3)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]

    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    torch.testing.assert_close(ker, ker.transpose(1, 2), rtol=1e-6, atol=1e-7)

    full = cross_time_kernel_full(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    # K(t',t) = K(t,t')^T, and it is genuinely not equal to K(t,t').
    torch.testing.assert_close(
        full, full.transpose(1, 2).transpose(-1, -2), rtol=1e-5, atol=1e-6
    )
    assert (full - full.transpose(1, 2)).abs().max() > 1e-3


def test_per_channel_kernel_is_the_channel_diagonal_of_the_full_one():
    s = _constant_pole_setup(T=12, D=3)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    fast = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    full = cross_time_kernel_full(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    torch.testing.assert_close(
        fast, torch.diagonal(full, dim1=-2, dim2=-1), rtol=1e-5, atol=1e-6
    )


def test_kernel_is_positive_semidefinite():
    """It is the covariance of an actual process, so this must hold for every channel."""
    s = _constant_pole_setup(T=30, D=3)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    for bi in range(s["B"]):
        for d in range(s["D"]):
            ev = torch.linalg.eigvalsh(ker[bi, :, :, d].double())
            assert ev.min() > -1e-8, f"kernel not PSD: min eigenvalue {ev.min():.3e}"


def test_row_chunking_does_not_change_the_kernel():
    s = _constant_pole_setup(T=17)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    a = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam, row_chunk=17)
    b = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam, row_chunk=5)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_nugget_lands_on_the_diagonal_only():
    """White across query times: a variance floor that buys no path coherence."""
    s = _constant_pole_setup(T=8, D=3)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    sig2 = torch.rand(s["B"], s["D"]) + 0.5
    bare = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    with_n = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam, sigma2=sig2)
    delta = with_n - bare
    eye = torch.eye(s["T"]).view(1, s["T"], s["T"], 1)
    torch.testing.assert_close(delta, eye * sig2.unsqueeze(1).unsqueeze(1), rtol=1e-6, atol=1e-7)


def test_alpha_scales_the_law_as_alpha_squared():
    """The no-head arm returns alpha * modal sum, so the covariance carries alpha^2."""
    s = _constant_pole_setup(T=6)
    lam = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])["lam"]
    a = 0.37
    k1 = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    k2 = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], lam, alpha=a)
    torch.testing.assert_close(k2, (a ** 2) * k1, rtol=1e-6, atol=1e-7)

    m1, v1 = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], lam)
    m2, v2 = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], lam, alpha=a)
    torch.testing.assert_close(m2, a * m1, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(v2, (a ** 2) * v1, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def test_markov_sampling_reproduces_the_analytic_mean_and_kernel():
    """Gate A9 row 5. The O(Kh) recursion must sample the law the O(Kh^2) kernel describes;
    if it does not, every path-level score is measuring a different object than the one the
    paper derives."""
    s = _constant_pole_setup(B=1, T=8, K=2, D=2, tmax=3.0)
    inc = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])
    mean, _ = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])

    g = torch.Generator().manual_seed(1234)
    draws = sample_joint_markov(
        s["theta"], s["rho_bar"], s["omega_bar"], inc["incr"], inc["d_rho"], s["p0"],
        num_samples=200_000, generator=g,
    )                                                                 # [S,1,T,D]

    emp_mean = draws.mean(dim=0)
    torch.testing.assert_close(emp_mean, mean, rtol=0, atol=0.02)

    centred = draws - emp_mean.unsqueeze(0)
    emp_cov = torch.einsum("sbtd,sbud->btud", centred, centred) / draws.shape[0]
    scale = ker.abs().max()
    err = (emp_cov - ker).abs().max() / scale
    assert err < 0.03, f"Markov draws vs analytic kernel: max rel err {err:.3f}"


def test_markov_sampling_agrees_with_cholesky_of_the_materialized_kernel():
    """The two sampling routes must be equal in distribution -- one is O(Kh), the other
    O(h^3). We keep the cheap one and check it here rather than trusting it."""
    s = _constant_pole_setup(B=1, T=6, K=2, D=1, tmax=2.0)
    inc = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])
    mean, _ = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])

    S = 60_000
    g = torch.Generator().manual_seed(7)
    markov = sample_joint_markov(
        s["theta"], s["rho_bar"], s["omega_bar"], inc["incr"], inc["d_rho"], s["p0"],
        num_samples=S, generator=g,
    )[:, 0, :, 0]                                                     # [S,T]

    cov = ker[0, :, :, 0].double()
    L = torch.linalg.cholesky(cov + 1e-10 * torch.eye(s["T"], dtype=torch.float64))
    z = torch.randn(S, s["T"], generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    chol = (z @ L.T).float() + mean[0, :, 0]

    # Compare moments rather than samples: equality is in distribution.
    torch.testing.assert_close(markov.mean(0), chol.mean(0), rtol=0, atol=0.02)
    cm = torch.cov(markov.T)
    cc = torch.cov(chol.T)
    assert (cm - cc).abs().max() / cov.abs().max().float() < 0.04


def test_vectorised_and_sequential_samplers_agree_in_distribution():
    """The closed-form scan replaced a step-by-step recursion. They consume randomness in a
    different order, so draws cannot match pairwise -- the moments must."""
    from llapdiffusion.models.chirp_gp import _sample_joint_markov_sequential

    s = _constant_pole_setup(B=1, T=8, K=2, D=2, tmax=3.0)
    inc = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])
    args = (s["theta"], s["rho_bar"], s["omega_bar"], inc["incr"], inc["d_rho"], s["p0"])
    mean, _ = marginal_law(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])
    ker = cross_time_kernel(s["theta"], s["rho_bar"], s["omega_bar"], inc["lam"])

    # Compared against the ANALYTIC law, not against each other: two Monte Carlo estimates
    # differ by the sum of their own errors, so a direct comparison is twice as noisy as the
    # thing it is trying to detect and would need 4x the draws for the same sensitivity.
    S = 120_000
    for name, draws in (
        ("vectorised", sample_joint_markov(*args, num_samples=S,
                                           generator=torch.Generator().manual_seed(11))),
        ("sequential", _sample_joint_markov_sequential(
            *args, num_samples=S, generator=torch.Generator().manual_seed(12))),
    ):
        d = draws[:, 0, :, 0]
        assert (d.mean(0) - mean[0, :, 0]).abs().max() < 0.03, f"{name}: mean off"
        emp = torch.cov(d.T)
        rel = (emp - ker[0, :, :, 0]).abs().max() / ker[0, :, :, 0].abs().max()
        assert rel < 0.05, f"{name}: covariance off by {rel:.3f}"


def test_sampler_falls_back_when_the_closed_form_would_overflow():
    """`1/A_k` has magnitude e^{rhobar}, so the scan is guarded. The guard must actually fire
    and the fallback must still produce finite, correctly shaped draws."""
    from llapdiffusion.models.chirp_gp import _sample_joint_markov_sequential

    B, T, K, D = 1, 6, 2, 2
    t = torch.linspace(1.0, 6.0, T).view(1, T, 1)
    rho_bar = torch.full((B, T, K), 0.0)
    rho_bar[:] = torch.linspace(10.0, 60.0, T).view(1, T, 1)      # rhobar up to 60 >> 30
    omega_bar = 0.3 * t.expand(B, T, K)
    q, p0 = torch.ones(B, K), torch.ones(B, K)
    inc = modal_increments(rho_bar, t, q, p0)
    theta = torch.randn(B, 2 * K, D)
    out = sample_joint_markov(theta, rho_bar, omega_bar, inc["incr"], inc["d_rho"], p0,
                              num_samples=16, generator=torch.Generator().manual_seed(0))
    assert out.shape == (16, B, T, D)
    assert torch.isfinite(out).all(), "fallback produced non-finite draws"


def test_sampling_is_differentiable():
    """(T3) needs a path score with gradients flowing back to the residues."""
    s = _constant_pole_setup(B=1, T=5, K=2, D=2)
    theta = s["theta"].clone().requires_grad_(True)
    inc = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])
    draws = sample_joint_markov(
        theta, s["rho_bar"], s["omega_bar"], inc["incr"], inc["d_rho"], s["p0"],
        num_samples=4, generator=torch.Generator().manual_seed(0),
    )
    draws.pow(2).mean().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert theta.grad.abs().max() > 0


def test_nugget_draws_are_white_across_query_times():
    s = _constant_pole_setup(B=1, T=6, K=1, D=1, tmax=1.0)
    inc = modal_increments(s["rho_bar"], s["t"], s["q"], s["p0"])
    zero_theta = torch.zeros_like(s["theta"])          # kill the modal part: only the nugget
    sig2 = torch.full((1, 1), 0.75)
    draws = sample_joint_markov(
        zero_theta, s["rho_bar"], s["omega_bar"], inc["incr"], inc["d_rho"], s["p0"],
        num_samples=80_000, sigma2=sig2, generator=torch.Generator().manual_seed(3),
    )[:, 0, :, 0]
    cov = torch.cov(draws.T)
    torch.testing.assert_close(torch.diag(cov), torch.full((6,), 0.75), rtol=0, atol=0.02)
    off = cov - torch.diag(torch.diag(cov))
    assert off.abs().max() < 0.02


# --------------------------------------------------------------------------------------
# projective consistency of the variance (the Prop. C.1 repair)
# --------------------------------------------------------------------------------------

def _drifting(times, k=2):
    """rho(s) = a + b cos(2 pi s / L) with a closed-form integral, so rhobar is exactly a
    function of time and any residual grid dependence must come from the v quadrature."""
    a = torch.tensor([0.30, 0.60])[:k].view(1, 1, k)
    b = torch.tensor([0.20, 0.35])[:k].view(1, 1, k)
    L = 8.0
    t = times.view(1, -1, 1)
    rho_bar = a * t + (b * L / (2 * math.pi)) * torch.sin(2 * math.pi * t / L)
    omega_bar = 0.8 * t
    return rho_bar, omega_bar


def test_query_grid_quadrature_is_NOT_projective():
    """The defect this repair exists for, pinned so it cannot silently come back as the
    default. Lambda from `modal_increments` depends on which query times were requested."""
    torch.manual_seed(0)
    full_t = torch.linspace(0.5, 8.0, 24)
    keep = torch.arange(0, 24, 4)                 # coarsen the grid 4x
    q = torch.tensor([[0.5, 0.3]])
    p0 = torch.tensor([[0.2, 0.1]])

    rb_full, _ = _drifting(full_t)
    lam_full = modal_increments(rb_full, full_t.view(1, -1, 1), q, p0)["lam"]
    rb_sub, _ = _drifting(full_t[keep])
    lam_sub = modal_increments(rb_sub, full_t[keep].view(1, -1, 1), q, p0)["lam"]

    rel = float((lam_sub - lam_full[:, keep]).abs().max() / lam_full[:, keep].abs().max())
    assert rel > 1e-3, (
        "the query-grid quadrature is expected to be grid-DEPENDENT; if this now passes at "
        "1e-6 the default changed and the repair below may be redundant"
    )


def test_fixed_anchor_grid_restores_projective_consistency():
    """The repair: Lambda computed against a query-independent anchor grid is a function of
    time alone, so a subset and a superset agree to float precision.

    Measured on the full R11 harness: worst relative kernel discrepancy goes from 2.975e-02
    (shipped) to 5.6e-07 under (P-exact) and to exactly 0.0 under (P-mono), and does not
    improve further with more anchor nodes -- i.e. what remains is float32 noise, not
    discretization.
    """
    full_t = torch.linspace(0.5, 8.0, 24)
    keep = torch.arange(0, 24, 4)
    q = torch.tensor([[0.5, 0.3]])
    p0 = torch.tensor([[0.2, 0.1]])

    horizon = 10.0                                 # a WINDOW property, not max(query)
    t_anchor = torch.linspace(0.0, horizon, 129).view(1, -1, 1)
    rb_anchor, _ = _drifting(t_anchor.view(-1))

    def lam_on(times):
        rb, _ = _drifting(times)
        return modal_lambda_projective(rb, times.view(1, -1, 1), rb_anchor, t_anchor, q, p0)

    lam_full = lam_on(full_t)
    lam_sub = lam_on(full_t[keep])
    rel = float((lam_sub - lam_full[:, keep]).abs().max() / lam_full[:, keep].abs().max())
    assert rel < 1e-6, f"anchored Lambda is still query-dependent: rel {rel:.3e}"


def test_anchored_lambda_still_matches_the_true_integral():
    """Consistency is worthless if it is consistently wrong: the anchored value must still
    be the variance integral, not merely a reproducible number."""
    times = torch.linspace(0.5, 8.0, 16)
    q_val, p0_val = 0.5, 0.0
    q = torch.full((1, 2), q_val)
    p0 = torch.zeros(1, 2)

    t_anchor = torch.linspace(0.0, 10.0, 2049).view(1, -1, 1)
    rb_anchor, _ = _drifting(t_anchor.view(-1))
    rb, _ = _drifting(times)
    anchored = modal_lambda_projective(rb, times.view(1, -1, 1), rb_anchor, t_anchor, q, p0)

    # Dense reference for mode 0.
    a, b, L = 0.30, 0.20, 8.0
    rbf = lambda t: a * t + (b * L / (2 * math.pi)) * torch.sin(2 * math.pi * t / L)
    ref = _dense_v(rbf, q_val, times.double(), n_per_unit=20_000)
    rel = float(((anchored[0, :, 0].double() - ref).abs() / ref.abs()).max())
    assert rel < 5e-3, f"anchored Lambda drifted from the true integral: rel {rel:.3e}"


def test_variance_quadrature_has_finite_gradients_through_a_zero_width_segment():
    """Regression: a zero-width segment must not poison the BACKWARD pass.

    `incr` is exactly zero whenever a segment has zero width, which the fixed anchor grid
    guarantees at its first node (it starts at t=0). `log(0) = -inf` is harmless in the
    forward -- it contributes nothing to the sum, and every forward-only check passes -- but
    `logcumsumexp` backpropagates **NaN** through a -inf input, and that NaN reaches every
    parameter. The observable symptom is not an error: training silently does nothing while
    the predictive law still looks correct. It cost a full set of ablation runs.
    """
    T, K = 12, 2
    t = torch.cat([torch.zeros(1), torch.linspace(1.0, 8.0, T - 1)]).view(1, T, 1)
    assert float(t[0, 0, 0]) == 0.0 and float(t[0, 1, 0] - t[0, 0, 0]) > 0

    q = torch.full((1, K), 0.5, requires_grad=True)
    p0 = torch.full((1, K), 0.2, requires_grad=True)
    rho_bar = (0.3 * t).expand(1, T, K).clone().requires_grad_(True)

    out = modal_increments(rho_bar, t, q, p0)
    assert float(out["incr"][0, 0, 0]) == 0.0, "the zero-width segment should give incr == 0"
    out["lam"].sum().backward()
    for name, g in (("rho_bar", rho_bar.grad), ("q", q.grad), ("p0", p0.grad)):
        assert g is not None and torch.isfinite(g).all(), f"non-finite gradient w.r.t. {name}"


def test_projective_lambda_has_finite_gradients():
    """The same check through the anchored path, which is where the zero-width node lives."""
    T, K, N = 10, 2, 32
    t = torch.linspace(1.0, 8.0, T).view(1, T, 1)
    t_anchor = torch.linspace(0.0, 10.0, N + 1).view(1, -1, 1)      # starts at exactly 0
    q = torch.full((1, K), 0.5, requires_grad=True)
    p0 = torch.full((1, K), 0.2, requires_grad=True)
    rb = (0.3 * t).expand(1, T, K).clone().requires_grad_(True)
    rb_a = (0.3 * t_anchor).expand(1, N + 1, K).clone().requires_grad_(True)

    modal_lambda_projective(rb, t, rb_a, t_anchor, q, p0).sum().backward()
    for name, g in (("rho_bar", rb.grad), ("rho_bar_anchor", rb_a.grad),
                    ("q", q.grad), ("p0", p0.grad)):
        assert g is not None and torch.isfinite(g).all(), f"non-finite gradient w.r.t. {name}"


def test_anchor_grid_must_start_at_zero():
    t_anchor = torch.linspace(1.0, 10.0, 33).view(1, -1, 1)
    rb, _ = _drifting(t_anchor.view(-1))
    with pytest.raises(ValueError, match="must start at t=0"):
        modal_lambda_projective(
            rb, t_anchor, rb, t_anchor, torch.ones(1, 2), torch.zeros(1, 2)
        )


def test_modal_energies_rejects_a_mismatched_theta():
    with pytest.raises(ValueError, match=r"theta must be \[B,2K,D\]"):
        modal_energies(torch.randn(2, 7, 3), 4)
