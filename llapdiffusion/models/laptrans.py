import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from llapdiffusion.models.time_utils import relative_time_offsets

RHO_CONDITIONING_MODES = ("legacy_effective", "raw")
MODAL_TYPES = ("lti", "chirp")

__all__ = [
    "LaplaceTransformEncoder",
    "LaplacePseudoInverse",
    "ChirpModalField",
    "RHO_CONDITIONING_MODES",
    "MODAL_TYPES",
    "normalize_rho_conditioning_mode",
    "normalize_modal_type",
]


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
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)
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
            # rho init in (0.01, 0.2)
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
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

        phi_m(t) = 1 + cos(2 pi f_m t)            in [0, 2]
        Phi_m(t) = t + sin(2 pi f_m t)/(2 pi f_m)  (antiderivative, Phi_m(0)=0)
        rho_k(t)     = rho_floor_k + sum_m a^rho_km^2 phi_m(t)        (> 0)
        rho_bar_k(t) = rho_floor_k t + sum_m a^rho_km^2 Phi_m(t)

    The conditioning head ``to_coeffs`` is zero-initialized, so at init the coefficients
    vanish and the field reduces to constant per-mode poles -- i.e. the chirp synthesizer
    matches the LTI (LLapDiff) model. The integrals are analytic and evaluated in parallel
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
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.cond_dim = int(cond_dim)
        self.num_basis = int(num_basis)
        self.rho_min = float(rho_min)
        self.omega_max = float(omega_max)

        # Per-mode floor poles (the constant term; init mirrors
        # LaplaceTransformEncoder.reset_parameters so chirp@init == LTI base poles).
        self._rho_base = nn.Parameter(torch.empty(self.k))
        self._omega_base = nn.Parameter(torch.empty(self.k))

        # Fixed nonnegative basis frequencies (cycles per unit relative-time).
        freqs = torch.linspace(1.0, float(self.num_basis), self.num_basis)
        self.register_buffer("basis_freqs", freqs, persistent=True)

        # Conditioned time-varying coefficients (squared -> nonnegative). Zero-init so the
        # model starts at the constant-pole (LTI) special case.
        self.to_coeffs = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * self.k * self.num_basis),
        )
        nn.init.zeros_(self.to_coeffs[-1].weight)
        nn.init.zeros_(self.to_coeffs[-1].bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # rho_floor in (0.01, 0.2) via rho_floor = rho_min + softplus(_rho_base)
            target_rho = torch.empty_like(self._rho_base).uniform_(0.01, 0.2)
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
        rho_floor = self.rho_min + F.softplus(self._rho_base.to(device=device, dtype=dtype))
        omega_floor = self.omega_max * torch.sigmoid(
            self._omega_base.to(device=device, dtype=dtype)
        )
        return rho_floor, omega_floor  # [K], [K]

    def _coeffs(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nonnegative per-mode, per-basis coefficients a^2, each [B,K,M]."""
        c = self.to_coeffs(cond).view(-1, 2, self.k, self.num_basis)
        return c[:, 0].pow(2), c[:, 1].pow(2)

    def _basis(self, t_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (phi, Phi), each [B,T,M], for the nonnegative Fourier basis."""
        two_pi_f = (2.0 * math.pi) * self.basis_freqs.to(
            device=t_rel.device, dtype=t_rel.dtype
        )  # [M]
        wt = t_rel * two_pi_f  # [B,T,1]*[M] -> [B,T,M]
        phi = 1.0 + torch.cos(wt)
        Phi = t_rel + torch.sin(wt) / two_pi_f  # Phi_m(0)=0
        return phi, Phi

    def integrated(
        self, cond: torch.Tensor, t_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrated poles rho_bar, omega_bar at query times, each [B,T,K]."""
        rho_floor, omega_floor = self._floor_poles(t_rel.dtype, t_rel.device)
        a_rho2, a_omega2 = self._coeffs(cond)  # [B,K,M]
        _, Phi = self._basis(t_rel)  # [B,T,M]
        rho_var = torch.einsum("bkm,btm->btk", a_rho2, Phi)
        omega_var = torch.einsum("bkm,btm->btk", a_omega2, Phi)
        rho_bar = rho_floor.view(1, 1, self.k) * t_rel + rho_var
        omega_bar = omega_floor.view(1, 1, self.k) * t_rel + omega_var
        return rho_bar, omega_bar

    def seed_poles(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Instantaneous poles at t=0 (phi_m(0)=2), used to seed residue extraction; [B,K]."""
        rho_floor, omega_floor = self._floor_poles(cond.dtype, cond.device)
        a_rho2, a_omega2 = self._coeffs(cond)  # [B,K,M]
        rho0 = rho_floor.view(1, self.k) + 2.0 * a_rho2.sum(dim=-1)
        omega0 = omega_floor.view(1, self.k) + 2.0 * a_omega2.sum(dim=-1)
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
