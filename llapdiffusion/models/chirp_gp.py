"""The chirp-modal Gaussian process: mean, cross-time kernel, exact joint sampling.

Theorem C turns the predicted parameters into a *process* on physical time, not a bag of
per-time marginals. The denoiser already evaluates the marginal (``lapformer.py``,
``return_variance``); everything a joint claim needs -- the cross-time kernel, exact joint
draws, the innovations likelihood -- lives here, so that one implementation of the algebra
serves the diffusion arm and the one-pass arm alike.

Notation follows the paper. Per mode ``k``, with the gauge ``mu_k^0 = e_1``,
``C_k = [c_k b_k] in R^{d_z x 2}``, ``P_k^0 = p0_k I_2``, ``Q_k = q_k I_2``:

    Phi_k(t,s) = e^{-[rhobar_k(t) - rhobar_k(s)]} Rot(omegabar_k(t) - omegabar_k(s))
    m(t)       = sum_k e^{-rhobar_k(t)} [c_k cos omegabar_k(t) + b_k sin omegabar_k(t)]
    Lambda_k(t)= e^{-2 rhobar_k(t)} p0_k + v_k(t)
    Cov(z(t), z(t')) = sum_k Lambda_k(t') e^{-[rhobar_k(t)-rhobar_k(t')]}
                       C_k Rot(omegabar_k(t)-omegabar_k(t')) C_k^T   (t >= t')

with ``C_k Rot(a) C_k^T = cos a (c_k c_k^T + b_k b_k^T) + sin a (b_k c_k^T - c_k b_k^T)``.

Two consequences of that last identity are worth stating because they are easy to get
wrong. The full ``d_z x d_z`` kernel is **not** symmetric under swapping ``(t, t')`` -- the
antisymmetric part is exactly the rotation. But ``(b_k c_k^T - c_k b_k^T)`` has a **zero
diagonal**, so the *per-channel* kernel, which is what path-level scores and lag-covariance
diagnostics consume, reduces to

    K_d(t,t') = sum_k Lambda_k(t_min) e^{-|drhobar_k|} cos(domegabar_k) (c_kd^2 + b_kd^2)

and *is* symmetric. Both are provided; the per-channel form is the default because the full
one costs ``O(K h^2 d_z^2)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


# Exponent floor shared with ChirpModalField.modal_variance: guards float range under the
# signed increments Theorem B' allows, while leaving the Assumption-R case untouched.
_RHO_FLOOR = -20.0
# |x| below this uses the Taylor branch of (1 - e^{-2x})/(2x); matches modal_variance.
_NEAR_ZERO = 1e-6


def _one_minus_exp_ratio(d_rho: torch.Tensor) -> torch.Tensor:
    """(1 - e^{-2x}) / (2x), with the x -> 0 limit taken as 1 - x.

    Identical to the branch inside ``ChirpModalField.modal_variance`` -- deliberately, so
    the increments below reproduce that quadrature exactly rather than approximating it.
    """
    near_zero = d_rho.abs() <= _NEAR_ZERO
    safe = torch.where(near_zero, torch.ones_like(d_rho), d_rho)
    return torch.where(near_zero, 1.0 - d_rho, (-torch.expm1(-2.0 * d_rho)) / (2.0 * safe))


def modal_increments(
    rho_bar: torch.Tensor,   # [B,T,K]
    t_rel: torch.Tensor,     # [B,T,1] or [B,T]
    q: torch.Tensor,         # [B,K] (constant q_k) or [B,T,K] (q_k(.))
    p0: torch.Tensor,        # [B,K]
) -> Dict[str, torch.Tensor]:
    """Run the Theorem-C quadrature once and keep everything the process needs.

    ``ChirpModalField.modal_variance`` returns only ``Lambda``; a Markov sampler and an
    innovations likelihood also need the per-segment transition and process-noise terms.
    Recomputing them elsewhere would be a second copy of the quadrature free to drift from
    the first, so they are produced together here and ``Lambda`` is asserted equal to
    ``modal_variance`` in the tests.

    Returns ``lam`` (= Lambda_k(t_r)), ``incr`` (the per-segment process variance
    ``Delta_r``), and ``d_rho`` (``rhobar_r - rhobar_{r-1}``, with a virtual node at
    ``t=0, rhobar=0``), each ``[B,T,K]``.

    The Lambda recurrence follows from the v recurrence:
        Lambda_r = e^{-2 rhobar_r} p0 + v_r
                 = e^{-2 d_rho_r} (e^{-2 rhobar_{r-1}} p0 + v_{r-1}) + Delta_r
                 = e^{-2 d_rho_r} Lambda_{r-1} + Delta_r,     Lambda_0 = p0 at t=0.
    """
    B, T, K = rho_bar.shape
    t = t_rel.reshape(B, T)
    if q.dim() == 2:
        q = q.unsqueeze(1).expand(B, T, K)
    elif q.shape != (B, T, K):
        raise ValueError(f"q must be [B,K] or [B,T,K]; got {tuple(q.shape)}")

    zt = torch.zeros(B, 1, dtype=t.dtype, device=t.device)
    zk = torch.zeros(B, 1, K, dtype=rho_bar.dtype, device=rho_bar.device)
    d_t = (t - torch.cat([zt, t[:, :-1]], dim=1)).clamp_min(0.0)          # [B,T]
    d_rho = (rho_bar - torch.cat([zk, rho_bar[:, :-1]], dim=1)).clamp_min(_RHO_FLOOR)
    incr = d_t.unsqueeze(-1) * q * _one_minus_exp_ratio(d_rho)            # [B,T,K], >= 0

    # The recurrence v_r = e^{-2 d_rho_r} v_{r-1} + incr_r unrolls exactly to
    #     v_r = sum_{k<=r} e^{-2 (rhobar_r - rhobar_k)} incr_k,
    # a first-order linear recurrence, so it needs no sequential loop at all. Evaluating it
    # as a cumulative sum naively would form e^{+2 rhobar_k}, which overflows for a decaying
    # mode; `logcumsumexp` does the same sum with the running max factored out, so the
    # intermediate never leaves float range while the result is unchanged.
    #
    # `rhobar_eff` is the cumulative sum of the CLAMPED increments rather than `rho_bar`
    # itself, so this reproduces the loop bit-for-bit in the Theorem-B' regime where an
    # increment can hit the -20 floor; under Assumption R the clamp never fires and the two
    # are identical anyway.
    #
    # The log-space accumulation is done in float64. The exponent carries `2 rhobar`, which
    # reaches ~720 at a long horizon with a fast mode; adding log(incr) ~ O(1) to 720 and
    # subtracting 720 again costs most of float32's mantissa. Measured against a float64
    # reference this was the scan's only regression (rel 2e-5 at rhobar=360) and it is the
    # classic cancellation of a log-domain sum. The tensor is [B,T,K] and small, so the
    # doubled precision is free; in float64 the scan is *more* accurate than the sequential
    # loop it replaces (max rel 2.2e-07 against the loop's 3.6e-07).
    rho_eff = torch.cumsum(d_rho, dim=1).double()
    # 🔴 The floor is load-bearing and must NOT be 0. An increment is exactly zero whenever a
    # segment has zero width -- which the fixed anchor grid of `modal_lambda_projective`
    # guarantees at its first node, since it starts at t=0. `log(0) = -inf` is harmless in the
    # forward (it contributes nothing to the sum) but `logcumsumexp` **backpropagates NaN**
    # through a -inf input, and that NaN reaches every parameter. It is invisible in a
    # forward-only test: the law is correct and training is silently dead. `1e-300` is
    # representable in float64, gives log ~= -690, and contributes e^{-690} to the sum, i.e.
    # nothing -- while keeping the gradient finite.
    log_incr = torch.log(incr.double().clamp_min(1e-300))
    log_v = torch.logcumsumexp(2.0 * rho_eff + log_incr, dim=1) - 2.0 * rho_eff
    v = torch.exp(log_v).to(rho_bar.dtype)
    v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))           # all-zero prefix

    lam = torch.exp(-2.0 * rho_bar.clamp_min(_RHO_FLOOR)) * p0.unsqueeze(1) + v
    return {"lam": lam, "incr": incr, "d_rho": d_rho}


def modal_lambda_projective(
    rho_bar_query: torch.Tensor,    # [B,Tq,K]  integrated decay at the requested times
    t_query: torch.Tensor,          # [B,Tq,1] or [B,Tq]
    rho_bar_anchor: torch.Tensor,   # [B,Ta,K]  integrated decay on the FIXED grid
    t_anchor: torch.Tensor,         # [B,Ta,1] or [B,Ta]  (must start at 0 and be sorted)
    q: torch.Tensor,                # [B,K] or [B,Ta,K]
    p0: torch.Tensor,               # [B,K]
) -> torch.Tensor:
    """``Lambda_k`` evaluated as a function of **time alone**, not of the requested query set.

    Why this exists. Under (P-exact)/(P-mono) the integrated poles ``rhobar, omegabar`` have
    closed forms, so the predictive *mean* is already a function on ``[0,H]`` and is exactly
    projective -- measured discrepancy 0.0 when query points are added or removed. The
    *variance* is not: ``modal_increments`` evaluates

        v_k(t) = int_0^t e^{-2[rhobar_k(t) - rhobar_k(s)]} q_k(s) ds

    by a recurrence **over the requested query nodes**, so refining or coarsening the query
    set changes the quadrature and hence the kernel. Measured on a 24-point irregular grid
    under (P-exact): dropping to 25% of the points moves the kernel by 2.9e-2 relative, while
    the mean moves by 1.0e-7; inserting a point *after* all others moves it by exactly 0,
    which is the signature of a grid-dependent quadrature and of nothing else.

    That makes Proposition C.1 false as stated for the shipped implementation -- the object is
    a Gaussian process in its mean and only a grid law in its kernel. The repair is to run the
    same overflow-free recurrence on a **fixed grid that does not depend on the query set**,
    then propagate exactly from the enclosing anchor node to each query time:

        v(t) = e^{-2[rhobar(t) - rhobar(t_n)]} v(t_n) + q (t - t_n) (1 - e^{-2 drhobar})/(2 drhobar)

    with ``t_n`` the last anchor node at or before ``t``. Both terms are functions of ``t``
    alone, so the result is projective by construction, and the residual quadrature error is
    the anchor grid's -- deterministic, query-independent, and controllable by its density.
    """
    B, Tq, K = rho_bar_query.shape
    tq = t_query.reshape(B, Tq)
    ta = t_anchor.reshape(B, -1)
    if not bool((ta[:, :1] == 0).all()):
        raise ValueError("the anchor grid must start at t=0, where the initial condition sits")

    # v on the fixed grid: query-independent by construction.
    anchor = modal_increments(rho_bar_anchor, t_anchor, q, p0)
    v_anchor = anchor["lam"] - torch.exp(
        -2.0 * rho_bar_anchor.clamp_min(_RHO_FLOOR)
    ) * p0.unsqueeze(1)                                                  # [B,Ta,K]

    # Enclosing anchor node for each query time. `right=True` then -1 gives the last node
    # with t_n <= t; clamped at 0 so a query at exactly 0 anchors on the initial condition.
    idx = torch.searchsorted(ta.contiguous(), tq.contiguous(), right=True) - 1
    idx = idx.clamp_min(0)                                               # [B,Tq]

    gather = idx.unsqueeze(-1).expand(B, Tq, K)
    v_n = torch.gather(v_anchor, 1, gather)                              # [B,Tq,K]
    rho_n = torch.gather(rho_bar_anchor, 1, gather)
    t_n = torch.gather(ta, 1, idx)                                       # [B,Tq]

    if q.dim() == 2:
        q_seg = q.unsqueeze(1).expand(B, Tq, K)
    else:
        q_seg = torch.gather(q, 1, gather)

    d_rho = (rho_bar_query - rho_n).clamp_min(_RHO_FLOOR)
    d_t = (tq - t_n).clamp_min(0.0).unsqueeze(-1)
    v_q = torch.exp(-2.0 * d_rho) * v_n + d_t * q_seg * _one_minus_exp_ratio(d_rho)
    return torch.exp(-2.0 * rho_bar_query.clamp_min(_RHO_FLOOR)) * p0.unsqueeze(1) + v_q


def modal_energies(theta: torch.Tensor, k: int) -> torch.Tensor:
    """Per-mode, per-channel readout energy ``c_kd^2 + b_kd^2``, i.e. ``diag(C_k C_k^T)``.

    ``theta`` is ``[B, 2K, D]`` with the cosine residues ``c_k`` first and the sine
    residues ``b_k`` second -- the layout ``chirp_basis_matrix`` synthesises against.
    """
    if theta.shape[1] != 2 * k:
        raise ValueError(f"theta must be [B,2K,D] with K={k}; got {tuple(theta.shape)}")
    return theta[:, :k, :].pow(2) + theta[:, k:, :].pow(2)          # [B,K,D]


def marginal_law(
    theta: torch.Tensor,        # [B,2K,D]
    rho_bar: torch.Tensor,      # [B,T,K]
    omega_bar: torch.Tensor,    # [B,T,K]
    lam: torch.Tensor,          # [B,T,K]
    *,
    alpha: float | torch.Tensor = 1.0,
    sigma2: Optional[torch.Tensor] = None,   # [B,D] or [D] nugget variance
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paper eq. (predictive): the per-time mean and variance of the returned tensor.

    Reproduces the block inlined in ``LapFormer.forward``; kept here as the single
    definition so the diffusion arm and the one-pass arm cannot disagree.
    """
    k = rho_bar.shape[-1]
    decay = torch.exp(-rho_bar)
    basis = torch.cat([decay * torch.cos(omega_bar), decay * torch.sin(omega_bar)], dim=-1)
    mean = torch.bmm(basis, theta)                                   # [B,T,D]
    var = torch.einsum("btk,bkd->btd", lam, modal_energies(theta, k))
    a2 = (alpha ** 2) if not torch.is_tensor(alpha) else alpha.pow(2)
    mean, var = alpha * mean, a2 * var
    if sigma2 is not None:
        var = var + (sigma2.unsqueeze(1) if sigma2.dim() == 2 else sigma2.view(1, 1, -1))
    return mean, var


def cross_time_kernel(
    theta: torch.Tensor,        # [B,2K,D]
    rho_bar: torch.Tensor,      # [B,T,K]
    omega_bar: torch.Tensor,    # [B,T,K]
    lam: torch.Tensor,          # [B,T,K]
    *,
    alpha: float | torch.Tensor = 1.0,
    sigma2: Optional[torch.Tensor] = None,
    row_chunk: int = 64,
) -> torch.Tensor:
    """Paper eq. (jointpredictive), per channel: ``[B,T,T,D]``.

    The channel-diagonal of ``b_k c_k^T - c_k b_k^T`` vanishes, so the rotation's
    antisymmetric part drops and what remains is symmetric in ``(t, t')``. The nugget is
    white across query times and therefore enters only the diagonal -- which is exactly why
    it buys a variance floor but no path coherence.

    Chunked over rows because the ``[B,T,T,K]`` intermediate, not the result, is what
    dominates memory (``K`` is typically 128-256 against ``D`` of 16-24).
    """
    B, T, K = rho_bar.shape
    energy = modal_energies(theta, K)                                # [B,K,D]
    D = energy.shape[-1]
    out = torch.empty(B, T, T, D, dtype=theta.dtype, device=theta.device)

    for lo in range(0, T, row_chunk):
        hi = min(lo + row_chunk, T)
        rb_r = rho_bar[:, lo:hi].unsqueeze(2)                        # [B,t,1,K]
        ob_r = omega_bar[:, lo:hi].unsqueeze(2)
        lam_r = lam[:, lo:hi].unsqueeze(2)
        rb_c = rho_bar.unsqueeze(1)                                  # [B,1,T,K]
        ob_c = omega_bar.unsqueeze(1)
        lam_c = lam.unsqueeze(1)

        # Query times are sorted, so "earlier" is the smaller index. Selecting Lambda by
        # INDEX rather than by the value of rhobar matters under Theorem B', where the
        # excursion makes rhobar non-monotone while time of course is not.
        rows = torch.arange(lo, hi, device=theta.device).view(1, -1, 1, 1)
        cols = torch.arange(T, device=theta.device).view(1, 1, -1, 1)
        row_is_later = rows >= cols
        lam_lo = torch.where(row_is_later, lam_c, lam_r)             # Lambda at min(t,t')
        d_rho = torch.where(row_is_later, rb_r - rb_c, rb_c - rb_r)  # >= 0 under Assumption R
        d_omega = ob_r - ob_c                                        # cos is even: sign free

        w = lam_lo * torch.exp(-d_rho.clamp_min(_RHO_FLOOR)) * torch.cos(d_omega)
        out[:, lo:hi] = torch.einsum("brck,bkd->brcd", w, energy)

    a2 = (alpha ** 2) if not torch.is_tensor(alpha) else alpha.pow(2)
    out = a2 * out
    if sigma2 is not None:
        s = sigma2.unsqueeze(1) if sigma2.dim() == 2 else sigma2.view(1, -1)   # [B,1,D]/[1,D]
        eye = torch.eye(T, dtype=torch.bool, device=theta.device).view(1, T, T, 1)
        out = out + eye * s.unsqueeze(1)
    return out


def cross_time_kernel_full(
    theta: torch.Tensor,
    rho_bar: torch.Tensor,
    omega_bar: torch.Tensor,
    lam: torch.Tensor,
    *,
    alpha: float | torch.Tensor = 1.0,
    sigma2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The full channel-block kernel ``[B,T,T,D,D]`` -- ``O(K h^2 d_z^2)``, so opt-in.

    This is the object that is genuinely **not** symmetric under ``(t,t') -> (t',t)``:
    ``K(t',t) = K(t,t')^T``. Provided mainly so the per-channel fast path can be checked
    against the definition rather than trusted.
    """
    B, T, K = rho_bar.shape
    c, b = theta[:, :K, :], theta[:, K:, :]                          # [B,K,D]
    sym = torch.einsum("bkd,bke->bkde", c, c) + torch.einsum("bkd,bke->bkde", b, b)
    asym = torch.einsum("bkd,bke->bkde", b, c) - torch.einsum("bkd,bke->bkde", c, b)

    rows = torch.arange(T, device=theta.device).view(1, -1, 1, 1)
    cols = torch.arange(T, device=theta.device).view(1, 1, -1, 1)
    row_is_later = rows >= cols
    rb_r, rb_c = rho_bar.unsqueeze(2), rho_bar.unsqueeze(1)
    ob_r, ob_c = omega_bar.unsqueeze(2), omega_bar.unsqueeze(1)
    lam_lo = torch.where(row_is_later, lam.unsqueeze(1), lam.unsqueeze(2))
    d_rho = torch.where(row_is_later, rb_r - rb_c, rb_c - rb_r)
    d_omega = torch.where(row_is_later, ob_r - ob_c, ob_c - ob_r)

    env = lam_lo * torch.exp(-d_rho.clamp_min(_RHO_FLOOR))           # [B,T,T,K]
    out = torch.einsum("brck,bkde->brcde", env * torch.cos(d_omega), sym) \
        + torch.einsum("brck,bkde->brcde", env * torch.sin(d_omega), asym)
    # Below the diagonal the pair is transposed, per the ordering convention.
    out = torch.where(row_is_later.unsqueeze(-1), out, out.transpose(-1, -2))

    a2 = (alpha ** 2) if not torch.is_tensor(alpha) else alpha.pow(2)
    out = a2 * out
    if sigma2 is not None:
        s = sigma2.unsqueeze(1) if sigma2.dim() == 2 else sigma2.view(1, -1)
        nug = torch.diag_embed(s)                                    # [B,1,D,D]/[1,D,D]
        eye = torch.eye(T, dtype=torch.bool, device=theta.device).view(1, T, T, 1, 1)
        out = out + eye * nug.unsqueeze(1)
    return out


def _sample_joint_markov_sequential(
    theta, rho_bar, omega_bar, incr, d_rho, p0, *,
    num_samples=25, alpha=1.0, sigma2=None, generator=None,
) -> torch.Tensor:
    """The original step-by-step recursion, kept as the numerically unconditional path.

    The vectorised form needs ``1/A_k``, whose magnitude is ``e^{rhobar_k}``; this one never
    forms it and so is safe for any integrated decay. It is used only when the closed form
    would leave float range, which no shipped configuration reaches.
    """
    B, T, K = rho_bar.shape
    S, dev, dt_ = int(num_samples), theta.device, theta.dtype
    c, b = theta[:, :K, :], theta[:, K:, :]

    def _randn(*shape):
        return torch.randn(*shape, generator=generator, device=dev, dtype=dt_)

    sd0 = p0.clamp_min(0.0).sqrt().view(1, B, K, 1)
    e1 = torch.zeros(1, 1, 1, 2, dtype=dt_, device=dev)
    e1[..., 0] = 1.0
    xi = _randn(S, B, K, 2) * sd0 + e1
    prev_ob = torch.zeros(B, K, dtype=dt_, device=dev)
    steps = []
    for r in range(T):
        d_ob = omega_bar[:, r] - prev_ob
        env = torch.exp(-d_rho[:, r].clamp_min(_RHO_FLOOR))
        cos_, sin_ = torch.cos(d_ob), torch.sin(d_ob)
        x, y = xi[..., 0], xi[..., 1]
        xr, yr = env * (cos_ * x - sin_ * y), env * (sin_ * x + cos_ * y)
        nsd = incr[:, r].clamp_min(0.0).sqrt()
        xi = torch.stack((xr + nsd * _randn(S, B, K), yr + nsd * _randn(S, B, K)), dim=-1)
        steps.append(torch.einsum("sbk,bkd->sbd", xi[..., 0], c)
                     + torch.einsum("sbk,bkd->sbd", xi[..., 1], b))
        prev_ob = omega_bar[:, r]
    out = alpha * torch.stack(steps, dim=2)
    if sigma2 is not None:
        s = sigma2.unsqueeze(1) if sigma2.dim() == 2 else sigma2.view(1, 1, -1)
        out = out + _randn(S, B, T, theta.shape[-1]) * s.clamp_min(0.0).sqrt().unsqueeze(0)
    return out


def sample_joint_markov(
    theta: torch.Tensor,
    rho_bar: torch.Tensor,
    omega_bar: torch.Tensor,
    incr: torch.Tensor,          # [B,T,K] per-segment process variance from modal_increments
    d_rho: torch.Tensor,         # [B,T,K] rhobar_r - rhobar_{r-1}
    p0: torch.Tensor,            # [B,K]
    *,
    num_samples: int = 25,
    alpha: float | torch.Tensor = 1.0,
    sigma2: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Exact joint draws in ``O(K h d_z)`` by the Markov recursion on ``xi_k`` -- no Cholesky.

    ``xi_k(0) ~ N(e_1, p0_k I_2)`` at the virtual node ``t=0``, then

        xi_r = e^{-d_rho_r} Rot(d_omega_r) xi_{r-1} + w_r,   w_r ~ N(0, Delta_r I_2),

    and ``z_r = alpha sum_k C_k xi_k(t_r) + eps_r``. These are draws from the *same* law
    ``cross_time_kernel`` describes, which is the point: estimator parity requires scoring
    the analytic arm with the identical sample-based estimator used for the sampled arms,
    and drawing through a Cholesky of the materialized kernel would cost ``O(h^3)`` for no
    statistical gain. The tests check the two agree in distribution.

    Returns ``[S,B,T,D]``.
    """
    B, T, K = rho_bar.shape
    S = int(num_samples)
    dev, dt_ = theta.device, theta.dtype
    c, b = theta[:, :K, :], theta[:, K:, :]                          # [B,K,D]

    def _randn(*shape):
        return torch.randn(*shape, generator=generator, device=dev, dtype=dt_)

    # Initial state at t=0: mean e_1, isotropic covariance p0 I_2.
    sd0 = p0.clamp_min(0.0).sqrt().view(1, B, K)
    xi0 = torch.complex(_randn(S, B, K) * sd0 + 1.0, _randn(S, B, K) * sd0)

    # In the rotation-scaling normal form the 2-D state is one complex number: Rot(a) acts as
    # multiplication by e^{ia}, so the recursion
    #     xi_r = e^{-drho_r} Rot(domega_r) xi_{r-1} + w_r
    # is the scalar first-order linear recurrence  z_r = a_r z_{r-1} + w_r  with
    # a_r = exp(-drho_r + i domega_r). Its cumulative product telescopes in closed form,
    #     A_r = prod_{j<=r} a_j = exp(-rho_c_r + i omega_c_r),
    # so no sequential loop is needed:
    #     z_r = A_r ( z_0 + sum_{k<=r} w_k / A_k ).
    # `rho_c`/`omega_c` are cumulative sums of the (clamped) increments, so this matches the
    # loop it replaces including in the Theorem-B' regime.
    zk = torch.zeros(B, 1, K, dtype=dt_, device=dev)
    d_omega = omega_bar - torch.cat([zk, omega_bar[:, :-1]], dim=1)
    rho_c = torch.cumsum(d_rho.clamp_min(_RHO_FLOOR), dim=1)         # [B,T,K]
    omega_c = torch.cumsum(d_omega, dim=1)

    # `1/A_k` has magnitude e^{rho_c}, so the closed form is only safe while that stays in
    # float range. The pole cap (rho_max = CHIRP_RHO_MAX_SCALE / horizon) keeps rho_c at a few
    # units in every shipped configuration, but a hand-built field can exceed it, so the bound
    # is checked rather than assumed and the exact sequential path is kept for that case.
    if float(rho_c.max()) > 30.0:
        return _sample_joint_markov_sequential(
            theta, rho_bar, omega_bar, incr, d_rho, p0,
            num_samples=S, alpha=alpha, sigma2=sigma2, generator=generator)

    A = torch.polar(torch.exp(-rho_c), omega_c).unsqueeze(0)         # [1,B,T,K] complex
    inv_A = torch.polar(torch.exp(rho_c), -omega_c).unsqueeze(0)
    noise_sd = incr.clamp_min(0.0).sqrt().unsqueeze(0)               # [1,B,T,K]
    w = torch.complex(_randn(S, B, T, K), _randn(S, B, T, K)) * noise_sd
    z = A * (xi0.unsqueeze(2) + torch.cumsum(inv_A * w, dim=2))      # [S,B,T,K]

    # z_r = sum_k (c_k Re xi + b_k Im xi)
    out = alpha * (torch.einsum("sbtk,bkd->sbtd", z.real, c)
                   + torch.einsum("sbtk,bkd->sbtd", z.imag, b))
    if sigma2 is not None:
        s = sigma2.unsqueeze(1) if sigma2.dim() == 2 else sigma2.view(1, 1, -1)
        out = out + _randn(S, B, T, theta.shape[-1]) * s.clamp_min(0.0).sqrt().unsqueeze(0)
    return out
