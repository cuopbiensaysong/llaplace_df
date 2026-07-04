"""Proposition A.1 figure: exp(int A) is exact for the rotation-scaling normal form
but wrong for the companion realization of a varying-frequency oscillator.

Both systems realize the same conjugate pole pair with a linearly chirped frequency
omega(t) = omega0 * (1 + alpha * t) and constant decay rho. The reference transition
Phi_ref(t) solves dPhi/dt = A(t) Phi with a fine-step RK4; the candidate is the naive
matrix exponential exp(int_0^t A(s) ds), which Theorem A proves exact for the normal
form (commuting family) and Proposition A.1 shows is generically wrong for the
non-commuting companion form.

Run:  python -m llapdiffusion.tools.plot_companion_vs_normal_form
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _omega(t: np.ndarray, omega0: float, alpha: float) -> np.ndarray:
    return omega0 * (1.0 + alpha * t)


def _companion_A(t: float, omega0: float, alpha: float) -> np.ndarray:
    w = float(_omega(np.asarray(t), omega0, alpha))
    return np.array([[0.0, 1.0], [-(w**2), 0.0]])


def _normal_A(t: float, omega0: float, alpha: float, rho: float) -> np.ndarray:
    w = float(_omega(np.asarray(t), omega0, alpha))
    return np.array([[-rho, -w], [w, -rho]])


def _rk4_transition(A_fn, t_grid: np.ndarray, dt_ref: float) -> np.ndarray:
    """High-accuracy reference Phi(t) on t_grid via RK4 on dPhi/dt = A(t) Phi."""
    out = np.zeros((t_grid.size, 2, 2))
    phi = np.eye(2)
    t = 0.0
    idx = 0
    if t_grid[0] == 0.0:
        out[0] = phi
        idx = 1
    while idx < t_grid.size:
        t_next = float(t_grid[idx])
        while t < t_next - 1e-12:
            h = min(dt_ref, t_next - t)
            k1 = A_fn(t) @ phi
            k2 = A_fn(t + 0.5 * h) @ (phi + 0.5 * h * k1)
            k3 = A_fn(t + 0.5 * h) @ (phi + 0.5 * h * k2)
            k4 = A_fn(t + h) @ (phi + h * k3)
            phi = phi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[idx] = phi
        idx += 1
    return out


def _naive_exp_companion(t_grid: np.ndarray, omega0: float, alpha: float) -> np.ndarray:
    """exp(int_0^t A_comp): M = [[0, t], [-W2, 0]] with W2 = int omega^2; M^2 = -t*W2*I,
    so exp(M) = cos(s) I + (sin(s)/s) M with s = sqrt(t * W2)."""
    out = np.zeros((t_grid.size, 2, 2))
    for i, t in enumerate(t_grid):
        w2 = (omega0**2) * (t + alpha * t**2 + (alpha**2) * t**3 / 3.0)  # int omega^2
        M = np.array([[0.0, float(t)], [-float(w2), 0.0]])
        s = float(np.sqrt(max(t * w2, 0.0)))
        if s < 1e-12:
            out[i] = np.eye(2) + M
        else:
            out[i] = np.cos(s) * np.eye(2) + (np.sin(s) / s) * M
    return out


def _closed_form_normal(t_grid: np.ndarray, omega0: float, alpha: float, rho: float) -> np.ndarray:
    """exp(int_0^t A_norm) = e^{-rho t} Rot(omega_bar), omega_bar = int omega (Thm A)."""
    out = np.zeros((t_grid.size, 2, 2))
    for i, t in enumerate(t_grid):
        wbar = omega0 * (t + 0.5 * alpha * t**2)
        c, s = np.cos(wbar), np.sin(wbar)
        out[i] = np.exp(-rho * t) * np.array([[c, -s], [s, c]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prop A.1: companion vs normal-form integration error.")
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.15, help="Linear chirp rate.")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=20.0)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--dt-ref", type=float, default=1e-3, help="RK4 reference step.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path.cwd() / "ldt" / "results" / "prop_a1" / "companion_vs_normal_form.pdf"),
    )
    args = parser.parse_args()

    t_grid = np.linspace(0.0, float(args.t_max), int(args.num_points))
    omega0, alpha, rho = float(args.omega0), float(args.alpha), float(args.rho)

    comp_ref = _rk4_transition(lambda t: _companion_A(t, omega0, alpha), t_grid, float(args.dt_ref))
    comp_naive = _naive_exp_companion(t_grid, omega0, alpha)
    comp_err = np.linalg.norm(comp_naive - comp_ref, ord=2, axis=(1, 2))

    norm_ref = _rk4_transition(lambda t: _normal_A(t, omega0, alpha, rho), t_grid, float(args.dt_ref))
    norm_closed = _closed_form_normal(t_grid, omega0, alpha, rho)
    norm_err = np.linalg.norm(norm_closed - norm_ref, ord=2, axis=(1, 2))

    save_path = Path(args.output).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.2))
    plt.semilogy(t_grid, np.maximum(comp_err, 1e-16), label="companion form: ||exp(∫A) − Φ_ref||")
    plt.semilogy(t_grid, np.maximum(norm_err, 1e-16), label="normal form: ||closed form − Φ_ref||")
    plt.xlabel("t")
    plt.ylabel("transition-matrix error (spectral norm)")
    plt.title(f"Prop. A.1 — ω(t)={omega0}·(1+{alpha}t), ρ={rho}")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"companion max error: {comp_err.max():.3e}")
    print(f"normal-form max error: {norm_err.max():.3e}")
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()
