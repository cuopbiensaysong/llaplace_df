import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from llapdiffusion.models.time_utils import relative_time_offsets

RHO_CONDITIONING_MODES = ("legacy_effective", "raw")
MODAL_TYPES = ("lti", "chirp")
CHIRP_PARAMETERIZATIONS = ("p_exact", "p_mono", "p_grid")


def normalize_chirp_parameterization(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in CHIRP_PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown chirp parameterization '{value}'. Use one of {CHIRP_PARAMETERIZATIONS}."
        )
    return mode

__all__ = [
    "LaplaceTransformEncoder",
    "LaplacePseudoInverse",
    "ChirpModalField",
    "CHIRP_PARAMETERIZATIONS",
    "CHIRP_BASES",
    "RHO_CONDITIONING_MODES",
    "MODAL_TYPES",
    "chirp_basis_freqs_signs",
    "normalize_chirp_basis",
    "normalize_chirp_parameterization",
    "normalize_rho_conditioning_mode",
    "normalize_modal_type",
]


CHIRP_BASES = ("half_integer", "integer")


def normalize_chirp_basis(value: object) -> str:
    """Harmonic set for the pole basis phi_m(t) = 1 + s_m cos(2 pi f_m t / L).

    ``integer`` is the legacy set f_m = 1..M with every sign +1. Every basis function
    then completes a WHOLE number of cycles across the window, so phi_m(0) = phi_m(L) = 2
    is its maximum -- and with nonnegative (squared) coefficients that forces
    omega(0) = omega(L) = sup_t omega(t) for EVERY mode. The instantaneous frequency can
    only dip and return: a monotone sweep, or a peak inside the window, is not in the
    function class at all. Measured against the H2 chirp truth, whose net drift across a
    window is ~70% of its within-window range, the whole legacy span reaches only
    R^2 = 0.35.

    ``half_integer`` uses f_m in {0.5, 0.5, 1.0, 1.0, ...} with alternating signs, i.e.
    each frequency present with BOTH signs. A nonnegative combination of the pair
    (1 + cos) and (1 - cos) spans a constant plus a SIGNED multiple of cos, so f = 0.5
    gives monotone sweeps in either direction and f = 1 gives V and Lambda shapes, with
    no signed-coefficient machinery and every positivity guarantee untouched. Each
    element still integrates to exactly zero over [0, L] (2 f_m is an integer), which is
    what the ``centered`` rho basis needs, and |phi| <= 2, |phi - 1| <= 1 still hold so
    the omega and rho headroom rescales are unchanged.
    """
    mode = str(value).strip().lower()
    if mode not in CHIRP_BASES:
        raise ValueError(f"Unknown chirp_basis '{value}'. Use one of {CHIRP_BASES}.")
    return mode


def chirp_basis_freqs_signs(basis: str, num_basis: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(freqs, signs) buffers for a basis mode; see :func:`normalize_chirp_basis`."""
    mode = normalize_chirp_basis(basis)
    m = int(num_basis)
    if mode == "integer":
        return torch.linspace(1.0, float(m), m), torch.ones(m)
    # Pairs (f, +1), (f, -1) for f = 0.5, 1.0, 1.5, ... so both signs of every harmonic
    # are available; an odd M drops the trailing negative partner.
    freqs = (torch.arange(m, dtype=torch.float32) // 2 + 1).float() * 0.5
    signs = torch.where(torch.arange(m) % 2 == 0, 1.0, -1.0)
    return freqs, signs


CHIRP_RHO_BASES = ("centered", "nonneg")


def normalize_chirp_rho_basis(value: object) -> str:
    """Basis used for the rho variation.

    ``nonneg`` is the legacy phi = 1 + cos expansion: phi >= 0 guarantees rho > floor,
    but mean_t phi = 1, so ANY rho variation also raises the mean decay by ~sum a^2.
    With a near-zero true decay that damps every oscillatory mode to extinction inside
    the horizon. ``centered`` uses phi - 1 = cos instead: over the window the basis
    frequencies are whole cycles, so its integral is exactly zero and the variation
    contributes NO net decay (rho_bar(L) = rho_floor * L). Positivity is then enforced
    by a headroom rescale instead of by the sign of the basis.
    """
    mode = str(value).strip().lower()
    if mode not in CHIRP_RHO_BASES:
        raise ValueError(f"Unknown chirp_rho_basis '{value}'. Use one of {CHIRP_RHO_BASES}.")
    return mode


def normalize_rho_conditioning_mode(mode: object) -> str:
    value = str(mode).strip().lower()
    if value not in RHO_CONDITIONING_MODES:
        choices = ", ".join(RHO_CONDITIONING_MODES)
        raise ValueError(f"Unknown rho_conditioning_mode '{mode}'. Use one of: {choices}.")
    return value


def normalize_modal_type(mode: object) -> str:
    value = str(mode).strip().lower()
    if value not in MODAL_TYPES:
        choices = ", ".join(MODAL_TYPES)
        raise ValueError(f"Unknown denoiser_modal_type '{mode}'. Use one of: {choices}.")
    return value


class LaplaceTransformEncoder(nn.Module):
    """Modal analysis that maps a time sequence to modal residues.

    Output:
        theta: [B, 2K, D]  (first K cosine residues, last K sine residues)
        rho:   [B, K]
        omega: [B, K]
    """

    def __init__(
        self,
        k: int,
        feat_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        alpha_min: float = 1e-6,
        omega_max: float = math.pi,
        cond_dim: Optional[int] = None,
        attn_cond_dim: Optional[int] = None,
        rho_perturb_scale: float = 0.5,
        omega_perturb_scale: float = 0.5,
        rho_conditioning_mode: str = "raw",
        attn_dropout: float = 0.0,
        rho_init_horizon: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)
        # Horizon used to anchor the rho init so modes survive the forecast window
        # (see reset_parameters). None keeps the legacy horizon-independent range.
        self.rho_init_horizon = None if rho_init_horizon is None else float(rho_init_horizon)
        self.cond_dim = cond_dim
        self.attn_cond_dim = attn_cond_dim
        self.rho_perturb_scale = float(rho_perturb_scale)
        self.omega_perturb_scale = float(omega_perturb_scale)
        self.rho_conditioning_mode = normalize_rho_conditioning_mode(rho_conditioning_mode)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )

        # Base poles
        self._rho_raw = nn.Parameter(torch.empty(self.k))
        self._omega_raw = nn.Parameter(torch.empty(self.k))

        # Conditioned bounded perturbations for poles
        if cond_dim is not None:
            self.to_poles = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * self.k),
            )
            # Start with zero perturbation (so effective poles start at base poles).
            nn.init.zeros_(self.to_poles[-1].weight)
            nn.init.zeros_(self.to_poles[-1].bias)

        # --- Spectral cross-attention path ---
        # Queries from (rho, omega, component_id) where component_id=0 (cos) or 1 (sin)
        self.pole_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.comp_emb = nn.Parameter(torch.zeros(1, 2 * self.k, hidden_dim))
        nn.init.normal_(self.comp_emb, mean=0.0, std=0.02)

        # Keys from time only: k_j = f_k(t_j)
        self.time_key_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Values from x: v_j = f_v(x_j)
        self.value_proj = nn.Linear(feat_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

        # Norms help stability of attention when T is large
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        if attn_cond_dim is not None:
            self.q_cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(attn_cond_dim, hidden_dim),
            )
            self.k_cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(attn_cond_dim, hidden_dim),
            )
            nn.init.normal_(self.q_cond_proj[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.q_cond_proj[-1].bias)
            nn.init.normal_(self.k_cond_proj[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.k_cond_proj[-1].bias)
            self.q_cond_gate = nn.Parameter(torch.tensor(0.0))
            self.k_cond_gate = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # rho init anchored to the horizon when known: the envelope keeps
            # e^{-rho*H} across the forecast, so the legacy horizon-independent
            # (0.01, 0.2) kills oscillatory modes on long horizons (rho*H up to 9.6 at
            # H=48, 33.6 at H=168) and leaves only near-DC modes alive. Scaling by
            # 1/horizon keeps rho*H in [0.1, 2]. Applied to the LTI core too so
            # chirp-vs-LTI stays a fair comparison.
            if self.rho_init_horizon is not None and self.rho_init_horizon > 0:
                lo, hi = 0.1 / self.rho_init_horizon, 2.0 / self.rho_init_horizon
            else:
                lo, hi = 0.01, 0.2
            target_rho = torch.empty_like(self._rho_raw).uniform_(lo, hi)
            y = (target_rho - self.alpha_min).clamp_min(1e-8)
            self._rho_raw.copy_(torch.log(torch.expm1(y)))

            # omega init in [0.01*omega_max, 0.95*omega_max]
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(
                torch.empty_like(self._omega_raw).uniform_(low_log, high_log)
            )
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))

    def _base_poles(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = F.softplus(self._rho_raw.to(device=device, dtype=dtype)) + self.alpha_min
        omega = self.omega_max * torch.sigmoid(
            self._omega_raw.to(device=device, dtype=dtype)
        )
        return rho, omega

    def effective_poles(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample effective poles rho, omega with stability constraints."""
        rho0, omega0 = self._base_poles(dtype, device)  # [K], [K]
        rho = rho0.unsqueeze(0).expand(batch_size, self.k).contiguous()
        omega = omega0.unsqueeze(0).expand(batch_size, self.k).contiguous()

        if self.cond_dim is not None and cond is not None:
            delta = self.to_poles(cond).view(batch_size, 2, self.k)
            d_rho = self.rho_perturb_scale * torch.tanh(delta[:, 0])
            d_omega = self.omega_perturb_scale * torch.tanh(delta[:, 1])

            if self.rho_conditioning_mode == "raw":
                rho_raw = self._rho_raw.to(device=device, dtype=dtype)
                rho = F.softplus(rho_raw.unsqueeze(0) + d_rho) + self.alpha_min
            else:
                rho = F.softplus(rho0.unsqueeze(0) + d_rho) + self.alpha_min

            # Keep omega in (0, omega_max) via logit perturbation of the base sigmoid.
            p0 = (omega0 / self.omega_max).clamp(1e-4, 1 - 1e-4)
            logit0 = torch.log(p0) - torch.log1p(-p0)
            omega = self.omega_max * torch.sigmoid(logit0.unsqueeze(0) + d_omega)

        return rho, omega  # [B,K], [B,K]

    @staticmethod
    def relative_time(
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return model time coordinates with shape [B,T,1].

        Explicit ``t`` values are absolute timestamps and are recentered to the first query.
        Explicit ``dt`` values are already-relative query offsets and are preserved.
        """
        if t is not None:
            t = t.to(device=device, dtype=dtype)
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            return relative_time_offsets(t, time_dim=1, recenter=True)
        if dt is not None:
            dt = dt.to(device=device, dtype=dtype)
            if dt.dim() == 2:
                dt = dt.unsqueeze(-1)
            return relative_time_offsets(dt, time_dim=1, recenter=False)
        return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)

    @staticmethod
    def basis_matrix(
        t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor
    ) -> torch.Tensor:
        """Compute damped cosine/sine basis matrix A_lap, shape [B,T,2K]."""
        rho_ = rho.unsqueeze(1)  # [B,1,K]
        omega_ = omega.unsqueeze(1)  # [B,1,K]
        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_
        cos_basis = decay * torch.cos(angle)
        sin_basis = decay * torch.sin(angle)
        return torch.cat([cos_basis, sin_basis], dim=-1).contiguous()

    @staticmethod
    def chirp_basis_matrix(
        rho_bar: torch.Tensor, omega_bar: torch.Tensor
    ) -> torch.Tensor:
        """Time-varying damped basis from integrated poles, shape [B,T,2K].

        Uses the already-integrated poles rho_bar(t)=int_0^t rho, omega_bar(t)=int_0^t omega
        directly: e^{-rho_bar} [cos(omega_bar), sin(omega_bar)]. Constant poles recover the
        LTI ``basis_matrix`` (rho_bar=rho*t, omega_bar=omega*t).
        """
        decay = torch.exp(-rho_bar)
        cos_basis = decay * torch.cos(omega_bar)
        sin_basis = decay * torch.sin(omega_bar)
        return torch.cat([cos_basis, sin_basis], dim=-1).contiguous()

    def _theta_time_attention(
        self,
        x: torch.Tensor,
        t_rel: torch.Tensor,
        rho: torch.Tensor,
        omega: torch.Tensor,
        attn_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Learned spectral cross-attention to obtain residues."""
        B, _, _ = x.shape
        # Build 2K queries from poles + component id
        rho2 = rho.repeat(1, 2).unsqueeze(-1)  # [B,2K,1]
        omg2 = omega.repeat(1, 2).unsqueeze(-1)  # [B,2K,1]
        comp = torch.cat(
            [
                torch.zeros(B, self.k, 1, device=x.device, dtype=x.dtype),
                torch.ones(B, self.k, 1, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )  # [B,2K,1]
        pole_feat = torch.cat([rho2, omg2, comp], dim=-1)  # [B,2K,3]
        q = self.q_norm(self.pole_embedding(pole_feat) + self.comp_emb)  # [B,2K,H]

        # Keys from time only; values from x only
        k = self.k_norm(self.time_key_proj(t_rel))  # [B,T,H]
        v = self.v_norm(self.value_proj(x))  # [B,T,H]
        if self.attn_cond_dim is not None and attn_cond is not None:
            if attn_cond.dim() != 2 or attn_cond.shape != (B, self.attn_cond_dim):
                raise ValueError(
                    f"attn_cond must be [B,{self.attn_cond_dim}], got {tuple(attn_cond.shape)}"
                )
            attn_cond = attn_cond.to(dtype=q.dtype)
            q = q + torch.tanh(self.q_cond_gate).to(dtype=q.dtype) * self.q_cond_proj(
                attn_cond
            ).unsqueeze(1)
            k = k + torch.tanh(self.k_cond_gate).to(dtype=k.dtype) * self.k_cond_proj(
                attn_cond
            ).unsqueeze(1)

        out, _ = self.attention(q, k, v, need_weights=False)  # [B,2K,H]
        theta = self.out_proj(out)  # [B,2K,D]
        return theta.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        attn_cond: Optional[torch.Tensor] = None,
        poles: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_t_rel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute modal residues and effective poles.

        Args:
            x: [B,T,D]
            dt/t: timing info
            cond: [B,cond_dim]
            poles: optional precomputed (rho, omega), each [B,K]
            return_t_rel: if True, also returns t_rel [B,T,1]

        Returns:
            theta: [B,2K,D]
            rho:   [B,K]
            omega: [B,K]
            t_rel: [B,T,1] if return_t_rel else None
        """
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"Input x must be [B, T, {self.feat_dim}]")
        B, T, _ = x.shape

        t_rel = self.relative_time(B, T, x.dtype, x.device, dt=dt, t=t)
        if poles is None:
            rho, omega = self.effective_poles(B, x.dtype, x.device, cond)
        else:
            rho, omega = poles
            if rho.shape != (B, self.k) or omega.shape != (B, self.k):
                raise ValueError("poles must be (rho, omega) with shape [B,K]")

        theta = self._theta_time_attention(x, t_rel, rho, omega, attn_cond=attn_cond)
        return theta, rho, omega, (t_rel if return_t_rel else None)


class ChirpModalField(nn.Module):
    """Time-varying ("chirp") modal poles in rotation-scaling normal form.

    Produces per-mode instantaneous poles rho_k(t), omega_k(t) > 0 and their exact
    integrals rho_bar_k(t)=int_0^t rho_k, omega_bar_k(t)=int_0^t omega_k via a fixed
    nonnegative Fourier basis with closed-form antiderivative (the "P-exact"
    parameterization):

        phi_m(t) = 1 + cos(2 pi f_m t/L)               in [0, 2]
        Phi_m(t) = t + sin(2 pi f_m t/L)/(2 pi f_m/L)   (antiderivative, Phi_m(0)=0)
        rho_k(t)     = rho_floor_k + sum_m a^rho_km^2 phi_m(t)        (> 0)
        rho_bar_k(t) = rho_floor_k t + sum_m a^rho_km^2 Phi_m(t)

    The basis frequencies f_m are cycles ACROSS the window L (not per unit time): without
    this normalization, at native horizons (L ~ 100-168) the oscillatory part of Phi_m has
    negligible amplitude 1/(2 pi f_m) next to the linear term ~L, so the chirp collapses to
    a constant-slope ramp (LTI). Dividing the frequencies by L restores resolution.

    The conditioning head ``to_coeffs`` is eps-initialized (std 1e-4; exact zero is a
    stationary point of the squared parameterization), so at init the coefficients are
    ~1e-8 and the field is numerically indistinguishable from constant per-mode poles --
    i.e. the chirp synthesizer starts at the LTI (LLapDiff) special case while remaining
    trainable. The integrals are analytic and evaluated in parallel
    over all query times (no ODE solver). Positive instantaneous decay yields the
    contraction ||Phi_k|| = e^{-rho_bar_k} <= 1 used for stability by construction.
    """

    def __init__(
        self,
        k: int,
        cond_dim: int,
        num_basis: int = 8,
        rho_min: float = 1e-4,
        omega_max: float = math.pi,
        time_scale: Optional[float] = None,
        uq_head: bool = False,
        growth_budget: float = 0.0,
        parameterization: str = "p_exact",
        rho_init_horizon: Optional[float] = None,
        rho_max_scale: float = 4.0,
        rho_basis: str = "nonneg",
        omega_basis: str = "nonneg",
        basis: str = "integer",
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.cond_dim = int(cond_dim)
        # Harmonic set for the pole basis. Defaults to the legacy integer-cycle set so a
        # pre-fix checkpoint rebuilt without the kwarg keeps its own function class.
        self.basis = normalize_chirp_basis(basis)
        # Explicit horizon for the rho init; falls back to time_scale (see
        # reset_parameters). Kept separate so the LTI arm can be given the same
        # anchor even though it has no basis time scale.
        self.rho_init_horizon = None if rho_init_horizon is None else float(rho_init_horizon)
        # Upper cap on the instantaneous rho, analogous to the Nyquist cap on omega:
        # rho_max = rho_max_scale / horizon keeps rho*H <= rho_max_scale so a mode's
        # envelope survives the forecast window (see _coeffs). None disables the cap
        # (legacy behaviour, and the only option when no horizon is known).
        self.rho_max_scale = float(rho_max_scale)
        self.rho_basis = normalize_chirp_rho_basis(rho_basis)
        # Same centred/nonneg choice for the omega variation (same validator, same enum).
        # "nonneg" is the legacy phi = 1 + s*cos expansion, whose mean-1 basis couples
        # omega variation to omega MEAN: every unit of a^2 needed for shape also raises
        # the mean frequency, and a redundant common mode within a +/- pair inflates it
        # for free (measured on a trained half-integer model: 63% of the omega coefficient
        # mass was redundant common mode, pushing mean omega to 1.23 vs a truth mean of
        # 0.40). "centered" uses phi - 1 = s*cos, zero-mean over the window, so variation
        # shapes omega(t) around the floor without shifting the mean -- the exact analogue
        # of the rho fix. Defaults to "nonneg" so pre-fix checkpoints are unchanged.
        self.omega_basis = normalize_chirp_rho_basis(omega_basis)
        _rho_h = rho_init_horizon if rho_init_horizon is not None else time_scale
        self.rho_max = (
            float(self.rho_max_scale) / float(_rho_h)
            if (_rho_h is not None and float(_rho_h) > 0 and self.rho_max_scale > 0)
            else None
        )
        self.num_basis = int(num_basis)
        self.rho_min = float(rho_min)
        self.omega_max = float(omega_max)
        self.parameterization = normalize_chirp_parameterization(parameterization)
        # p_grid reads _rho_base through its OWN softplus (see _pgrid_inst) instead of
        # _floor_poles, so re-parameterising the floor there would desynchronise the init
        # from the readout. Bound the floor only for the variants that use _floor_poles.
        self._rho_floor_bounded = self.rho_max is not None and self.parameterization != "p_grid"
        # Window length L that normalizes the basis frequencies to the time axis. A fixed
        # value gives reproducible, checkpoint-comparable chirps; None falls back to a
        # per-sample data-adaptive L = max|t_rel| (robust to units and irregular sampling).
        self.time_scale = None if time_scale is None else float(time_scale)

        # Per-mode floor poles (the constant term; init mirrors
        # LaplaceTransformEncoder.reset_parameters so chirp@init == LTI base poles).
        self._rho_base = nn.Parameter(torch.empty(self.k))
        self._omega_base = nn.Parameter(torch.empty(self.k))

        # Fixed nonnegative basis: phi_m = 1 + s_m cos(2 pi f_m t / L), frequencies in
        # cycles across the window L (see _basis) and signs pairing each harmonic with
        # its negative so the span covers monotone sweeps as well as dips
        # (see normalize_chirp_basis).
        freqs, signs = chirp_basis_freqs_signs(self.basis, self.num_basis)
        self.register_buffer("basis_freqs", freqs, persistent=True)
        # NOT persistent: the signs are a pure function of (basis, num_basis), both of
        # which the checkpoint's model_config already carries. Persisting them would add
        # a state_dict key that pre-fix checkpoints lack, and two load paths are strict
        # (viz/plot_llapdiff_poles.py, train_val_llapdiff._load_diff_model) -- every old
        # chirp checkpoint would fail to load.
        self.register_buffer("basis_signs", signs, persistent=False)

        # The top basis frequency is max(f_m) cycles across L, i.e. max(f_m)/L cycles per
        # step. Anything past L/2 cycles across the window is above the Nyquist limit of a
        # unit-gap grid: it cannot be resolved by the data and only adds jitter to
        # rho(t)/omega(t) (and quadratically many parameters to the coefficient head,
        # 2*K*M outputs). Stated on max(f_m) so it holds for either basis mode.
        max_freq = float(freqs.max())
        if self.time_scale is not None and max_freq > self.time_scale / 2.0:
            warnings.warn(
                f"chirp basis top frequency {max_freq:g} cycles/window exceeds the Nyquist "
                f"bound time_scale/2={self.time_scale / 2.0:.1f} for "
                f"time_scale={self.time_scale}: "
                f"{int((freqs > self.time_scale / 2.0).sum())} of {self.num_basis} basis "
                "functions oscillate faster than the sampling grid can resolve.",
                RuntimeWarning,
                stacklevel=2,
            )

        print(f"self.num_basis: {self.num_basis}")
        print(f"self.rho_min: {self.rho_min}")
        print(f"self.omega_max: {self.omega_max}")
        print(f"self.time_scale: {self.time_scale}")

        if self.parameterization == "p_exact":
            # Conditioned time-varying coefficients (squared -> nonnegative). Near-zero
            # init: the model starts eps-close (a^2 ~ 1e-8) to the constant-pole (LTI)
            # special case. NOT exactly zero — a = 0 is a stationary point of the squared
            # parameterization (d(a^2)/dW = 2a·h = 0), so zero-init would freeze the head
            # forever and the "chirp" would silently train as constant poles.
            self.to_coeffs = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, 2 * self.k * self.num_basis),
            )
            nn.init.normal_(self.to_coeffs[-1].weight, std=1e-4)
            nn.init.zeros_(self.to_coeffs[-1].bias)
        elif self.parameterization == "p_mono":
            # P-mono: monotone integrated poles directly. rho_bar = floor*t +
            # sum_m u_m [softplus(v_m tau + b_m) - softplus(b_m)] with u_m >= 0
            # (softplus of base+delta -> no stationary trap), v_m >= 0 (monotone in t),
            # tau = t/L. Instantaneous poles are the closed-form derivative (positive).
            init_u = math.log(math.expm1(1e-3))  # softplus(base) = 1e-3 -> near-LTI init
            self._mono_u_base = nn.Parameter(torch.full((2, self.k, self.num_basis), init_u))
            rates = torch.linspace(1.0, float(self.num_basis), self.num_basis)
            self._mono_rate_raw = nn.Parameter(
                torch.log(torch.expm1(rates)).view(1, self.num_basis).expand(self.k, -1).clone()
            )
            self._mono_shift = nn.Parameter(
                torch.empty(self.k, self.num_basis).uniform_(-2.0, 2.0)
            )
            self.to_mono = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, 2 * self.k * self.num_basis),
            )
            nn.init.zeros_(self.to_mono[-1].weight)  # linear pre-softplus: no trap
            nn.init.zeros_(self.to_mono[-1].bias)
        else:  # p_grid
            # P-grid: pointwise positive poles from cond + window-scaled basis features
            # psi(t) = [1, phi_1..phi_M]; rho = rho_min + softplus(base + delta·psi),
            # omega = omega_max * sigmoid(base + delta·psi) (Nyquist by construction).
            # Integration is a cumulative trapezoid on the query grid (numerical — the
            # deliberate contrast case for the parameterization ablation). Zero-init
            # deltas start exactly at the LTI floors; the head is linear pre-nonlinearity,
            # so there is no stationary trap.
            self.to_grid = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, 2 * self.k * (self.num_basis + 1)),
            )
            nn.init.zeros_(self.to_grid[-1].weight)
            nn.init.zeros_(self.to_grid[-1].bias)

        # Optional Theorem-C UQ head (isotropic case, constant-in-time noise
        # intensity): per-mode initial variance p0_k and noise intensity q_k,
        # softplus-parameterized around learnable bases with zero-init conditioned
        # deltas, so at init p0 = q = softplus(base) uniformly across the batch.
        # Optional Theorem-B' bounded-growth head: a budgeted excursion
        # gamma_k(t) = c_g [sigma(g_k(t)) - sigma(g_k(0))] subtracted from the
        # integrated decay, so the envelope may genuinely grow but the total
        # multiplicative growth over any subinterval is capped at e^{c_g}.
        # c_g = 0 (default) disables the head and recovers Theorem B verbatim.
        self.growth_budget = float(growth_budget)
        if self.growth_budget < 0:
            raise ValueError(f"growth_budget must be >= 0, got {growth_budget}")
        if self.growth_budget > 0:
            self.to_growth = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.k * self.num_basis),
            )
            nn.init.zeros_(self.to_growth[-1].weight)
            nn.init.zeros_(self.to_growth[-1].bias)

        self.uq_head = bool(uq_head)
        if self.uq_head:
            init_raw = math.log(math.expm1(1e-2))  # softplus(base) = 0.01
            self._p0_base = nn.Parameter(torch.full((self.k,), init_raw))
            self._q_base = nn.Parameter(torch.full((self.k,), init_raw))
            self.to_uq = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.cond_dim, 2 * self.k),
            )
            nn.init.zeros_(self.to_uq[-1].weight)
            nn.init.zeros_(self.to_uq[-1].bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # rho_floor via rho_floor = rho_min + softplus(_rho_base).
            #
            # The decay scale that matters is rho*H (the envelope keeps e^{-rho*H} across
            # the horizon), so a horizon-independent init silently kills modes on long
            # horizons: the legacy (0.01, 0.2) spans rho*H in [0.5, 9.6] at H=48 and
            # [1.7, 33.6] at H=168, i.e. nearly every oscillatory mode decays to nothing
            # before the horizon ends and only near-DC modes survive. Anchoring the draw
            # to 1/time_scale keeps rho*H in [0.1, 2] (envelope 0.9 -> 0.25) so modes stay
            # alive long enough to carry an oscillation. time_scale=None (per-sample
            # adaptive L) has no fixed horizon, so it keeps the legacy range.
            horizon = self.rho_init_horizon
            if horizon is None:
                horizon = self.time_scale
            if horizon is not None and horizon > 0:
                lo, hi = 0.1 / horizon, 2.0 / horizon
            else:
                lo, hi = 0.01, 0.2
            target_rho = torch.empty_like(self._rho_base).uniform_(lo, hi)
            if self._rho_floor_bounded:
                # Bounded floor: invert rho_min + (rho_max - rho_min) * sigmoid(base).
                span = max(self.rho_max - self.rho_min, 1e-12)
                p = ((target_rho - self.rho_min) / span).clamp(1e-4, 1 - 1e-4)
                self._rho_base.copy_(torch.log(p) - torch.log1p(-p))
            else:
                y = (target_rho - self.rho_min).clamp_min(1e-8)
                self._rho_base.copy_(torch.log(torch.expm1(y)))

            # omega_floor in [0.01, 0.95] * omega_max via omega_max * sigmoid(_omega_base)
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(
                torch.empty_like(self._omega_base).uniform_(low_log, high_log)
            )
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_base.copy_(torch.log(p) - torch.log1p(-p))

    def _floor_poles(
        self, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self._rho_base.to(device=device, dtype=dtype)
        if self._rho_floor_bounded:
            # Bounded like omega_floor: rho_floor in (rho_min, rho_max). softplus is
            # unbounded above, which lets the floor alone extinguish a mode at long
            # horizons; sigmoid closes that path and makes the two poles symmetric.
            rho_floor = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(base)
        else:
            rho_floor = self.rho_min + F.softplus(base)
        omega_floor = self.omega_max * torch.sigmoid(
            self._omega_base.to(device=device, dtype=dtype)
        )
        return rho_floor, omega_floor  # [K], [K]

    def _coeffs(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nonnegative per-mode, per-basis coefficients a^2, each [B,K,M].

        The omega coefficients are smoothly rescaled so the instantaneous frequency
        stays below omega_max pointwise: phi_m <= 2 gives sup_t omega <= floor + 2*sum_m a^2,
        and the rescale bounds that sum by (omega_max - floor). The coefficients stay a
        plain linear combination of the basis, so the closed-form antiderivative is
        preserved; at zero coefficients the scale is 1 (LTI equivalence unchanged).

        The decay coefficients get the SAME treatment against ``rho_max``. They are
        squared, so the variation term can only push rho UP from its floor; with no cap
        training inflates rho until the mode's own envelope e^{-rho_bar} extinguishes it
        inside the horizon (measured: floor bounded to <= 0.042 by the horizon-anchored
        init, yet learned rho reached 0.987 -- envelope mass 0.00 -- so the modes at the
        signal frequency contributed nothing and only near-DC modes survived). omega is
        capped by Nyquist; rho is capped by envelope survival over the horizon. Bounding
        rho from ABOVE does not affect the Theorem-B contraction, which needs only
        rho >= rho_min > 0 (guaranteed by the floor).
        """
        c = self.to_coeffs(cond).view(-1, 2, self.k, self.num_basis)
        a_rho2 = c[:, 0].pow(2)
        a_omega2 = c[:, 1].pow(2)
        rho_floor, omega_floor = self._floor_poles(cond.dtype, cond.device)
        if self.omega_basis == "centered":
            # omega(t) = floor + sum a^2 (phi_m - 1), and |phi_m - 1| <= 1, so
            # omega in [floor - S, floor + S] with S = sum a^2. Only the NYQUIST limit is
            # a real constraint here: the sign of omega is a pure gauge, because the
            # synthesis is sum_k e^{-rho_bar}(c_k cos omega_bar + b_k sin omega_bar) and
            # omega -> -omega is absorbed exactly by b_k -> -b_k (free residues). So bound
            # S by the distance to omega_max only.
            #
            # Do NOT also bound S by the floor "to keep omega >= 0": that couples the
            # swing budget to the MEAN frequency, so a model hedging toward DC (small
            # floor) simultaneously loses its ability to sweep -- a self-reinforcing trap.
            # Measured on a K=1 checkpoint trained under that bound: floor 0.027 with the
            # constraint 99.2% saturated, i.e. the sweep was clamped to 0.027 rad/step
            # against a truth swing of 0.48.
            omega_head = (self.omega_max - omega_floor).clamp_min(1e-8).view(1, self.k, 1)
            omega_total = a_omega2.sum(dim=-1, keepdim=True)
            a_omega2 = a_omega2 * (omega_head / (omega_total + omega_head))
        else:
            # Legacy nonneg basis: phi_m <= 2 so sup_t omega = floor + 2*sum a^2.
            headroom = (self.omega_max - omega_floor).view(1, self.k, 1)  # [1,K,1], > 0
            total = 2.0 * a_omega2.sum(dim=-1, keepdim=True)  # [B,K,1]
            a_omega2 = a_omega2 * (headroom / (total + headroom))
        if self.rho_basis == "centered":
            # rho(t) = floor + sum a^2 (phi_m - 1), and |phi_m - 1| <= 1, so
            # rho >= floor - sum a^2. Bounding sum a^2 by (floor - rho_min) keeps
            # rho >= rho_min (Theorem B) and simultaneously caps rho <= 2*floor.
            rho_head = (rho_floor - self.rho_min).clamp_min(1e-8).view(1, self.k, 1)
            rho_total = a_rho2.sum(dim=-1, keepdim=True)
            a_rho2 = a_rho2 * (rho_head / (rho_total + rho_head))
        elif self.rho_max is not None:
            # Legacy nonneg basis: phi_m <= 2 so sup_t rho = floor + 2*sum a^2.
            rho_head = (self.rho_max - rho_floor).clamp_min(1e-8).view(1, self.k, 1)
            rho_total = 2.0 * a_rho2.sum(dim=-1, keepdim=True)
            a_rho2 = a_rho2 * (rho_head / (rho_total + rho_head))
        return a_rho2, a_omega2

    def _time_scale(self, t_rel: torch.Tensor) -> torch.Tensor:
        """Window length L that normalizes the basis frequencies, shape [B,1,1].

        Uses the fixed config constant when set; otherwise a per-sample data-adaptive
        L = max|t_rel| (robust to units / irregular sampling), clamped away from zero.
        """
        if self.time_scale is not None:
            return torch.full((1, 1, 1), self.time_scale, dtype=t_rel.dtype, device=t_rel.device)
        return t_rel.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _basis(
        self, t_rel: torch.Tensor, time_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (phi, Phi), each [B,T,M], for the nonnegative Fourier basis.

        Frequencies are normalized by ``time_scale`` (L) so they resolve a few cycles
        across the window rather than per unit relative-time. The per-basis sign s_m
        keeps phi_m = 1 + s_m cos in [0, 2] (so every bound in ``_coeffs`` is unchanged)
        while letting a nonnegative coefficient reach either sign of the harmonic --
        without it, phi_m(0) = 2 is the maximum for every m and the poles cannot drift
        across the window (see :func:`normalize_chirp_basis`).
        """
        two_pi_f = (2.0 * math.pi) * self.basis_freqs.to(
            device=t_rel.device, dtype=t_rel.dtype
        ) / time_scale  # [M] / [B,1,1] -> [B,1,M]
        signs = self.basis_signs.to(device=t_rel.device, dtype=t_rel.dtype)  # [M]
        wt = t_rel * two_pi_f  # [B,T,1]*[B,1,M] -> [B,T,M]  (= 2 pi f * t/L)
        phi = 1.0 + signs * torch.cos(wt)
        # Phi_m(0)=0; sin term ~ (L/2pi f) sin(2pi f t/L)
        Phi = t_rel + signs * torch.sin(wt) / two_pi_f
        return phi, Phi

    def _pmono_params(
        self, cond: torch.Tensor, time_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """P-mono unit weights (u_rho, u_omega [B,K,M], omega capped), rates v and
        shifts b [K,M]. u = softplus(base + delta(cond)) — alive at init, near-LTI."""
        delta = self.to_mono(cond).view(-1, 2, self.k, self.num_basis)
        u_rho = F.softplus(self._mono_u_base[0].unsqueeze(0) + delta[:, 0])
        u_omega = F.softplus(self._mono_u_base[1].unsqueeze(0) + delta[:, 1])
        v = F.softplus(self._mono_rate_raw)  # [K,M] >= 0 -> monotone in t
        b = self._mono_shift
        # Nyquist cap: sup_t inst omega_var = (1/L) sum_m u v <= omega_max - floor.
        _, omega_floor = self._floor_poles(cond.dtype, cond.device)
        headroom = (self.omega_max - omega_floor).view(1, self.k, 1)
        sup = (u_omega * v.unsqueeze(0)).sum(dim=-1, keepdim=True) / time_scale.view(-1, 1, 1)
        u_omega = u_omega * (headroom / (sup + headroom))
        return u_rho, u_omega, v, b

    def _pmono_poles(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """P-mono instantaneous and integrated variable parts, each [B,T,K]:
        bar_var = sum_m u [softplus(v tau + b) - softplus(b)], inst_var = its
        closed-form t-derivative (>= 0), with tau = t / L."""
        time_scale = self._time_scale(t_rel)
        u_rho, u_omega, v, b = self._pmono_params(cond, time_scale)
        tau = (t_rel / time_scale).unsqueeze(-1)  # [B,T,1,1]
        arg = v.view(1, 1, self.k, self.num_basis) * tau + b.view(1, 1, self.k, self.num_basis)
        sp = F.softplus(arg)  # [B,T,K,M]
        sp0 = F.softplus(b).view(1, 1, self.k, self.num_basis)
        sig = torch.sigmoid(arg)
        inv_l = 1.0 / time_scale.view(-1, 1, 1)
        rho_bar_var = (u_rho.unsqueeze(1) * (sp - sp0)).sum(dim=-1)
        omega_bar_var = (u_omega.unsqueeze(1) * (sp - sp0)).sum(dim=-1)
        rho_inst_var = (u_rho.unsqueeze(1) * sig * v.view(1, 1, self.k, -1)).sum(dim=-1) * inv_l
        omega_inst_var = (u_omega.unsqueeze(1) * sig * v.view(1, 1, self.k, -1)).sum(dim=-1) * inv_l
        return rho_inst_var, omega_inst_var, rho_bar_var, omega_bar_var

    def _pgrid_inst(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """P-grid pointwise poles, each [B,T,K]: rho = rho_min + softplus(base + d·psi),
        omega = omega_max sigmoid(base + d·psi), psi = [1, phi_1..phi_M]."""
        delta = self.to_grid(cond).view(-1, 2, self.k, self.num_basis + 1)
        phi, _ = self._basis(t_rel, self._time_scale(t_rel))  # [B,T,M]
        psi = torch.cat([torch.ones_like(phi[..., :1]), phi], dim=-1)  # [B,T,M+1]
        rho_arg = self._rho_base.view(1, 1, self.k) + torch.einsum(
            "bkm,btm->btk", delta[:, 0], psi
        )
        omega_arg = self._omega_base.view(1, 1, self.k) + torch.einsum(
            "bkm,btm->btk", delta[:, 1], psi
        )
        rho = self.rho_min + F.softplus(rho_arg)
        omega = self.omega_max * torch.sigmoid(omega_arg)
        return rho, omega

    def _pgrid_integrated(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cumulative-trapezoid integration of the pointwise poles over the query grid
        (a virtual node at t=0 anchors the first segment). Numerical by design — the
        contrast case in the P-exact/P-mono/P-grid ablation (integration error grows
        with gap width)."""
        B, T = t_rel.shape[0], t_rel.shape[1]
        t = t_rel.reshape(B, T)
        zero = torch.zeros(B, 1, 1, dtype=t_rel.dtype, device=t_rel.device)
        rho0, omega0 = self._pgrid_inst(cond, zero)  # [B,1,K]
        rho_t, omega_t = self._pgrid_inst(cond, t_rel)
        tt = torch.cat([torch.zeros(B, 1, dtype=t.dtype, device=t.device), t], dim=1)
        seg = (tt[:, 1:] - tt[:, :-1]).clamp_min(0.0).unsqueeze(-1)  # [B,T,1]
        rho_all = torch.cat([rho0, rho_t], dim=1)  # [B,T+1,K]
        omega_all = torch.cat([omega0, omega_t], dim=1)
        rho_bar = (0.5 * seg * (rho_all[:, 1:] + rho_all[:, :-1])).cumsum(dim=1)
        omega_bar = (0.5 * seg * (omega_all[:, 1:] + omega_all[:, :-1])).cumsum(dim=1)
        return rho_bar, omega_bar

    def _growth_terms(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Theorem-B' excursion gamma(t) [B,T,K] and its derivative gamma'(t).

        gamma_k(t) = c_g [sigma(g_k(t)) - sigma(g_k(0))] with g_k a signed expansion in
        the same window-scaled basis: gamma(0)=0, gamma <= c_g (negative values only add
        decay and are harmless). The derivative is closed-form:
        gamma' = c_g sigma(g)(1-sigma(g)) g', with phi_m' = -sin(2 pi f_m t/L) 2 pi f_m/L.
        """
        g_coeff = self.to_growth(cond).view(-1, self.k, self.num_basis)  # [B,K,M], signed
        time_scale = self._time_scale(t_rel)
        phi, _ = self._basis(t_rel, time_scale)  # [B,T,M]
        g_t = torch.einsum("bkm,btm->btk", g_coeff, phi)  # [B,T,K]
        # phi_m(0) = 1 + s_m (2 for a positive harmonic, 0 for its negative partner)
        phi_0 = 1.0 + self.basis_signs.to(device=t_rel.device, dtype=t_rel.dtype)  # [M]
        g_0 = (g_coeff * phi_0).sum(dim=-1)
        sig_t = torch.sigmoid(g_t)
        gamma = self.growth_budget * (sig_t - torch.sigmoid(g_0).unsqueeze(1))

        two_pi_f = (2.0 * math.pi) * self.basis_freqs.to(
            device=t_rel.device, dtype=t_rel.dtype
        ) / time_scale  # [B,1,M]
        # d/dt [1 + s_m cos(2 pi f_m t/L)] = -s_m sin(...) * 2 pi f_m / L
        phi_prime = -(phi_0 - 1.0) * torch.sin(t_rel * two_pi_f) * two_pi_f  # [B,T,M]
        g_prime = torch.einsum("bkm,btm->btk", g_coeff, phi_prime)
        gamma_prime = self.growth_budget * sig_t * (1.0 - sig_t) * g_prime
        return gamma, gamma_prime

    def instantaneous(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Instantaneous poles rho(t), omega(t) at query times, each [B,T,K].

        The pointwise derivative of :meth:`integrated`; used by the pole-trajectory
        tooling (viz / synthetic-recovery figures), not by the training path. Under a
        growth budget (Theorem B') rho(t) may be negative where the envelope grows.
        """
        if self.parameterization == "p_grid":
            rho, omega = self._pgrid_inst(cond, t_rel)
        elif self.parameterization == "p_mono":
            rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
            rho_var, omega_var, _, _ = self._pmono_poles(cond, t_rel)
            rho = rho_floor.view(1, 1, self.k) + rho_var
            omega = omega_floor.view(1, 1, self.k) + omega_var
        else:
            rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
            a_rho2, a_omega2 = self._coeffs(cond)  # [B,K,M]
            phi, _ = self._basis(t_rel, self._time_scale(t_rel))  # [B,T,M]
            # Centred rho basis: phi - 1 = cos, zero-mean over the window, so the
            # variation shapes rho(t) without adding net decay.
            phi_rho = phi - 1.0 if self.rho_basis == "centered" else phi
            phi_omega = phi - 1.0 if self.omega_basis == "centered" else phi
            rho = rho_floor.view(1, 1, self.k) + torch.einsum("bkm,btm->btk", a_rho2, phi_rho)
            omega = omega_floor.view(1, 1, self.k) + torch.einsum("bkm,btm->btk", a_omega2, phi_omega)
        if self.growth_budget > 0:
            _, gamma_prime = self._growth_terms(cond, t_rel)
            rho = rho - gamma_prime
        return rho, omega

    def coefficient_penalty(self, cond: torch.Tensor) -> torch.Tensor:
        """Scalar L2 penalty on the conditioned pole-variation coefficients.

        Penalizes the head outputs that make the poles time-varying and
        condition-dependent, shrinking the field toward its constant-pole (LTI)
        special case — the Tier-2 `CHIRP_COEFF_L2` ablation. Per parameterization:
        p_exact penalizes the raw pre-square outputs ``a`` (mean a^2 = the
        time-variation energy, i.e. the L1 of the nonnegative expansion
        coefficients); p_mono the unit weights ``u``; p_grid the psi-deltas.
        The growth head is deliberately excluded (its excursion is already
        governed by the Theorem-B' budget c_g).
        """
        if self.parameterization == "p_exact":
            raw = self.to_coeffs(cond)
        elif self.parameterization == "p_mono":
            delta = self.to_mono(cond).view(-1, 2, self.k, self.num_basis)
            raw = torch.cat(
                [
                    F.softplus(self._mono_u_base[0].unsqueeze(0) + delta[:, 0]),
                    F.softplus(self._mono_u_base[1].unsqueeze(0) + delta[:, 1]),
                ],
                dim=-1,
            )
        else:  # p_grid
            raw = self.to_grid(cond)
        return raw.pow(2).mean()

    def uq_params(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-mode initial variance p0 and noise intensity q, each [B,K] (> 0)."""
        if not self.uq_head:
            raise RuntimeError("ChirpModalField was built without uq_head=True.")
        delta_p0, delta_q = self.to_uq(cond).view(-1, 2, self.k).unbind(dim=1)
        p0 = F.softplus(self._p0_base.view(1, self.k) + delta_p0)
        q = F.softplus(self._q_base.view(1, self.k) + delta_q)
        return p0, q

    @staticmethod
    def modal_variance(
        rho_bar: torch.Tensor,
        t_rel: torch.Tensor,
        q: torch.Tensor,
        p0: torch.Tensor,
    ) -> torch.Tensor:
        """Per-mode variance scalars s_k(t_r) = e^{-2 rho_bar_r} p0_k + v_k(t_r), [B,T,K].

        v_k(t) = int_0^t e^{-2(rho_bar(t) - rho_bar(s))} q_k ds (Theorem C, Eq. 6, with
        constant-in-time q_k), evaluated by an exponential-integrator recurrence over
        the sorted query times (a virtual node at t=0, rho_bar=0 anchors the first
        segment). With d_rho_r = rho_bar_r - rho_bar_{r-1} the per-segment integral is
        exact for piecewise-constant instantaneous decay:

            v_r = e^{-2 d_rho_r} v_{r-1} + q d_t_r (1 - e^{-2 d_rho_r}) / (2 d_rho_r)

        Without a growth budget rho_bar is nondecreasing, so every exponent is <= 0.
        Under Theorem B' the excursion is bounded (|d_rho| <= c_g on any subinterval),
        so signed increments are supported and exponents stay bounded (a -20 safety
        floor guards float range; e^{40} is finite in float32). For constant poles it
        reproduces the Lyapunov closed form q (1 - e^{-2 rho t}) / (2 rho) to float
        precision. This is the "solver-free 1-D quadrature" realization of the method
        doc's v_k (the integrand has no elementary antiderivative under P-exact).
        """
        B, T, K = rho_bar.shape
        t = t_rel.reshape(B, T)
        v = torch.zeros(B, K, dtype=rho_bar.dtype, device=rho_bar.device)
        prev_t = torch.zeros(B, dtype=rho_bar.dtype, device=rho_bar.device)
        prev_rho = torch.zeros(B, K, dtype=rho_bar.dtype, device=rho_bar.device)
        out = []
        for r in range(T):
            d_t = (t[:, r] - prev_t).clamp_min(0.0)  # [B]
            d_rho = (rho_bar[:, r] - prev_rho).clamp_min(-20.0)  # [B,K], signed under B'
            decay = torch.exp(-2.0 * d_rho)
            # (1 - e^{-2x})/(2x) with the |x| -> 0 limit 1 - x (Taylor) for stability.
            near_zero = d_rho.abs() <= 1e-6
            safe = torch.where(near_zero, torch.ones_like(d_rho), d_rho)
            ratio = torch.where(
                near_zero,
                1.0 - d_rho,
                (-torch.expm1(-2.0 * d_rho)) / (2.0 * safe),
            )
            v = decay * v + d_t.unsqueeze(-1) * q * ratio
            s_r = torch.exp(-2.0 * rho_bar[:, r].clamp_min(-20.0)) * p0 + v
            out.append(s_r)
            prev_t = t[:, r]
            prev_rho = rho_bar[:, r]
        return torch.stack(out, dim=1)  # [B,T,K]

    def integrated(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrated poles rho_bar, omega_bar at query times, each [B,T,K]."""
        if self.parameterization == "p_grid":
            rho_bar, omega_bar = self._pgrid_integrated(cond, t_rel)
        elif self.parameterization == "p_mono":
            rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
            _, _, rho_var, omega_var = self._pmono_poles(cond, t_rel)
            rho_bar = rho_floor.view(1, 1, self.k) * t_rel + rho_var
            omega_bar = omega_floor.view(1, 1, self.k) * t_rel + omega_var
        else:
            rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
            a_rho2, a_omega2 = self._coeffs(cond)  # [B,K,M]
            _, Phi = self._basis(t_rel, self._time_scale(t_rel))  # [B,T,M]
            # Antiderivative of the centred basis: int (phi - 1) = Phi - t. Still closed
            # form (Theorem A); at t = L the basis frequencies are whole cycles so this
            # is exactly zero => rho_bar(L) = rho_floor * L, decay set purely by the floor.
            Phi_rho = Phi - t_rel if self.rho_basis == "centered" else Phi
            Phi_omega = Phi - t_rel if self.omega_basis == "centered" else Phi
            rho_var = torch.einsum("bkm,btm->btk", a_rho2, Phi_rho)
            omega_var = torch.einsum("bkm,btm->btk", a_omega2, Phi_omega)
            rho_bar = rho_floor.view(1, 1, self.k) * t_rel + rho_var
            omega_bar = omega_floor.view(1, 1, self.k) * t_rel + omega_var
        if self.growth_budget > 0:
            gamma, _ = self._growth_terms(cond, t_rel)
            rho_bar = rho_bar - gamma  # Theorem B': rho_bar >= rho_min*t - c_g
        return rho_bar, omega_bar

    def seed_poles(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Instantaneous poles at t=0, used to seed residue extraction; [B,K].

        Growth-budget excursions are deliberately excluded: seeds must stay positive
        for the analysis attention, and gamma only reshapes the synthesis envelope.
        """
        zero = torch.zeros(cond.shape[0], 1, 1, dtype=cond.dtype, device=cond.device)
        if self.parameterization == "p_grid":
            rho0, omega0 = self._pgrid_inst(cond, zero)
            return rho0.squeeze(1), omega0.squeeze(1)
        rho_floor, omega_floor = self._floor_poles(cond.dtype, cond.device)
        if self.parameterization == "p_mono":
            rho_var, omega_var, _, _ = self._pmono_poles(cond, zero)
            return (
                rho_floor.view(1, self.k) + rho_var.squeeze(1),
                omega_floor.view(1, self.k) + omega_var.squeeze(1),
            )
        a_rho2, a_omega2 = self._coeffs(cond)  # [B,K,M]
        # phi_m(0) = 1 + s_m; scale-invariant either way, so no time_scale is needed here.
        # rho reads the same centred basis (phi - 1) that instantaneous/integrated use --
        # otherwise the seeded analysis poles disagree with the poles that synthesize the
        # output, by a factor of 2 at t=0 under rho_basis="centered".
        phi_0 = 1.0 + self.basis_signs.to(device=cond.device, dtype=cond.dtype)  # [M]
        phi_0_rho = phi_0 - 1.0 if self.rho_basis == "centered" else phi_0
        phi_0_omega = phi_0 - 1.0 if self.omega_basis == "centered" else phi_0
        rho0 = rho_floor.view(1, self.k) + (a_rho2 * phi_0_rho).sum(dim=-1)
        omega0 = omega_floor.view(1, self.k) + (a_omega2 * phi_0_omega).sum(dim=-1)
        return rho0, omega0


class LaplacePseudoInverse(nn.Module):
    """Explicit synthesis from residues and effective poles.

    Computes y(t) = A_lap(t; rho, omega) @ theta, then optionally applies a small
    residual MLP to capture transients.
    """

    def __init__(
        self,
        encoder: LaplaceTransformEncoder,
        hidden_dim: Optional[int] = None,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_mlp_residual = bool(use_mlp_residual)
        print(f"self.use_mlp_residual: {self.use_mlp_residual}")
        D = encoder.feat_dim
        H = int(hidden_dim if hidden_dim is not None else D * 2)

        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(D)
            self.mlp_in = spectral_norm(nn.Linear(D, H * 2))
            self.mlp_out = spectral_norm(nn.Linear(H, D))
            # NOTE: spectral_norm + all-zero weight can yield NaNs due to degenerate
            # power iteration. Use a tiny random init for the underlying weight.
            w = getattr(self.mlp_out, "weight_orig", self.mlp_out.weight)
            nn.init.normal_(w, mean=0.0, std=1e-4)
            nn.init.zeros_(self.mlp_out.bias)

    def forward(
        self,
        theta: torch.Tensor,  # [B,2K,D]
        rho: Optional[torch.Tensor] = None,  # [B,K]
        omega: Optional[torch.Tensor] = None,  # [B,K]
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        target_T: Optional[int] = None,
        rho_bar: Optional[torch.Tensor] = None,  # [B,T,K] (chirp / time-varying poles)
        omega_bar: Optional[torch.Tensor] = None,  # [B,T,K]
    ) -> torch.Tensor:
        if theta.dim() != 3:
            raise ValueError("theta must be [B,2K,D]")
        B = theta.shape[0]

        if rho_bar is not None and omega_bar is not None:
            # Chirp path: integrated poles already evaluated at the query times.
            basis = self.encoder.chirp_basis_matrix(rho_bar, omega_bar)  # [B,T,2K]
        else:
            if rho is None or omega is None:
                raise ValueError("Provide (rho, omega) or (rho_bar, omega_bar)")
            if t is not None:
                T = t.shape[1]
            elif dt is not None:
                T = dt.shape[1]
            elif target_T is not None:
                T = int(target_T)
            else:
                raise ValueError("Provide t or dt or target_T to determine output length")
            t_rel = self.encoder.relative_time(B, T, theta.dtype, theta.device, dt=dt, t=t)
            basis = self.encoder.basis_matrix(t_rel, rho, omega)  # [B,T,2K]
        y = torch.bmm(basis, theta)  # [B,T,D]

        if not self.use_mlp_residual:
            return y

        res = self.norm(y)
        gate, val = self.mlp_in(res).chunk(2, dim=-1)
        res = val * F.gelu(gate)
        return y + self.mlp_out(res)
