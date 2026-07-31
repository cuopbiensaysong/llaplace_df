"""Search-space definition for the CMD hyperparameter tuner. EDIT GRIDS HERE.

Each training stage is a tuple: (stage_name, kind, knob, candidates)

  kind "cli"    -> forwarded as a llapdiff-train flag (survives the preset
                   re-stamp via the pipeline's REQUESTED_* tracking).
  kind "config" -> stamped onto the config module inside a wrapped
                   apply_dataset_preset, so it works for BOTH preset-stamped
                   knobs (BASE_LR, MINSNR_GAMMA, MODEL_WIDTH, ...) and
                   base-config knobs (LR_SCHEDULE, WARMUP_FRAC, CHIRP_*).

The tuner is a coordinate descent: stages run top to bottom, each stage sweeps
its candidates on top of the current best configuration. Order therefore
matters — put the highest-leverage knobs first.

Candidate values equal to the resolved default are dropped automatically (they
would duplicate the baseline trial), so it is safe to list the default in a
grid for readability.

Tier discipline (cmd_plan_v2.md §7):
  * Tier-1 stages run with IDENTICAL grids on EVERY arm (matched-budget rule).
  * Tier-2 stages apply to the chirp arms (c, d) only.
  * Tier-3 (capacity) is off by default; enable with --include-tier3.

predict_type is NOT swept: v is settled as the best parameterization
(USAGE.md §3.6 — v beats x0 on CRPS across all seven datasets).
"""
from __future__ import annotations

import math

from common import CHIRP_ARMS

# ---- Tier 1: optimizer + loss weighting, identical grid on every arm --------
TIER1_STAGES = [
    # Peak LR for the AdamW diffusion optimizer. Preset-stamped to 1.5e-4, so
    # a plain runtime assignment would be silently reset (DEVELOPER_GUIDE §3);
    # the tuner's wrapped preset handles it.
    ("base_lr", "config", "BASE_LR", [5e-5, 1e-4, 1.5e-4, 3e-4, 5e-4]),
    # LR schedule shape. Only these three names are valid (make_lr_scheduler):
    #   warmup_constant (default) | warmup_cosine (decays to MIN_LR) | constant
    ("lr_schedule", "config", "LR_SCHEDULE", ["warmup_constant", "warmup_cosine", "constant"]),
    # Fraction of total steps spent warming up.
    # ⚠️ COUPLING: when EARLY_STOP_MIN_EPOCHS == 0 (the default), the trainer
    # derives it as ceil(WARMUP_FRAC * EPOCHS) — so a larger warmup also
    # forbids early stopping for longer, i.e. these trials train longer. To
    # decouple, pin EARLY_STOP_MIN_EPOCHS in the same override.
    ("warmup_frac", "config", "WARMUP_FRAC", [0.0, 0.05, 0.095, 0.2]),
    # Min-SNR loss-weighting gamma (preset-stamped: 5.0 for physionet/crypto/
    # us_equity/bms_air, 4.5 for uci_air/noaa_*).
    ("minsnr_gamma", "config", "MINSNR_GAMMA", [3.0, 4.5, 5.0, 6.5]),
]

# ---- Tier 2: chirp-specific (arms c and d only) ------------------------------
#
# The pole-basis size M is the one knob whose useful range depends on the
# horizon, so its grid is COMPUTED per setting rather than written out (see
# chirp_num_basis_grid). M itself is not comparable across horizons: what the
# basis actually buys is a top frequency measured in *cycles across the window*
# L, and the conversion depends on CHIRP_BASIS.
#
#   half_integer (default): f = 0.5, 0.5, 1.0, 1.0, ... — M/2 distinct
#       frequencies, each present with both signs, so the top frequency is
#       M/4 cycles across L.
#   integer (legacy):       f = 1..M, so the top frequency is M cycles across L.
#
# A unit-gap grid resolves at most L/2 cycles (Nyquist), so the resolvable
# ceiling is M <= 2*L for half_integer and M <= L/2 for integer. Past it the
# extra basis functions cannot be resolved by the data, only add jitter to
# rho(t)/omega(t) and quadratically many coefficient-head parameters (2*K*M
# outputs), and trip the model's Nyquist RuntimeWarning.
#
# The ladder below is therefore stated in cycles and converted per setting: the
# same *shape* resolution is offered at every horizon, and the Nyquist clamp
# only bites at short ones. At PRED=12 (ceiling M<=24) it yields the historical
# [4, 8, 16, 24]; at PRED>=32 it opens up to [4, 8, 16, 32, 64].
CHIRP_BASIS_CYCLE_LADDER = (1, 2, 4, 8, 16)

# Reference horizon for the module-level fallback grid (PhysioNet h=12). Callers
# that know their horizon get the scaled grid via stages_for_arm(defaults=...).
FALLBACK_CHIRP_TIME_SCALE = 12

# Tier-2 stages whose grids do not depend on the horizon.
TIER2_FIXED_STAGES = [
    # Minimum decay floor rho_min (the Theorem-B bound constant).
    ("chirp_rho_min", "config", "CHIRP_RHO_MIN", [1e-5, 1e-4, 1e-3]),
    # Theorem-B' growth budget c_g; 0.0 recovers Theorem B exactly (T2 sweep).
    ("growth_budget", "config", "CHIRP_GROWTH_BUDGET", [0.0, math.log(2.0), math.log(5.0)]),
    # L2 shrinkage of the pole-variation coefficients toward the LTI case.
    ("chirp_coeff_l2", "config", "CHIRP_COEFF_L2", [0.0, 1e-4, 1e-2]),
]


def basis_cycles_per_unit_m(basis: str) -> float:
    """Cycles across the window contributed by the top basis function, per unit M."""
    mode = str(basis).strip().lower()
    if mode == "half_integer":
        return 0.25
    if mode == "integer":
        return 1.0
    raise ValueError(f"Unknown CHIRP_BASIS '{basis}'. Use 'half_integer' or 'integer'.")


def nyquist_num_basis_ceiling(time_scale: float, *, basis: str = "half_integer") -> int:
    """Largest M whose top basis frequency stays at or below the Nyquist limit.

    A unit-gap grid resolves L/2 cycles across the window, so the ceiling is
    (L/2) / cycles-per-unit-M: 2*L for the half-integer basis, L/2 for integer.
    """
    scale = float(time_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"chirp time scale must be a positive number, got {time_scale!r}")
    ceiling = int(math.floor((scale / 2.0) / basis_cycles_per_unit_m(basis)))
    return max(1, ceiling)


def chirp_num_basis_grid(
    time_scale: float,
    *,
    basis: str = "half_integer",
    cycles=CHIRP_BASIS_CYCLE_LADDER,
) -> list[int]:
    """Candidate pole-basis sizes M for one horizon, clamped at Nyquist.

    The ladder is specified in cycles across the window (the horizon-invariant
    unit) and converted to M for the active basis, so a longer horizon
    automatically raises the grid instead of leaving it pinned at the value that
    happened to be resolvable at h=12. Clamping — rather than dropping — the
    entries that exceed the ceiling keeps the maximum resolvable M in the grid
    at short horizons, which is what the historical [4, 8, 16, 24] did.
    """
    per_m = basis_cycles_per_unit_m(basis)
    ceiling = nyquist_num_basis_ceiling(time_scale, basis=basis)
    values = [
        min(max(1, int(round(float(count) / per_m))), ceiling)
        for count in cycles
    ]
    return sorted(set(values))


def resolve_chirp_time_scale(defaults) -> float:
    """Window length L the chirp basis frequencies are normalized by.

    Mirrors ``train_val_llapdiff._resolve_chirp_time_scale``: ``None`` (the
    default) resolves to the run's horizon, a number pins L explicitly, and
    ``"adaptive"`` uses a per-sample L = max|t_rel| — which has no fixed value
    here, so the horizon is the right planning proxy for it too.
    """
    value = getattr(defaults, "CHIRP_TIME_SCALE", None)
    if isinstance(value, str):
        if value.strip().lower() != "adaptive":
            raise ValueError(
                f"Unknown CHIRP_TIME_SCALE '{value}'. Use None (horizon), a number, or 'adaptive'."
            )
        value = None
    if value is None:
        return float(getattr(defaults, "PRED"))
    return float(value)


def chirp_num_basis_stage(defaults=None) -> tuple:
    """The ``chirp_num_basis`` sweep stage, scaled to this setting's horizon."""
    if defaults is None:
        time_scale = float(FALLBACK_CHIRP_TIME_SCALE)
        basis = "half_integer"
    else:
        time_scale = resolve_chirp_time_scale(defaults)
        basis = str(getattr(defaults, "CHIRP_BASIS", "half_integer"))
    return ("chirp_num_basis", "config", "CHIRP_NUM_BASIS", chirp_num_basis_grid(time_scale, basis=basis))


# Horizon-agnostic view of Tier 2, kept so the grids stay editable in one place
# (README §7.2). ``stages_for_arm(defaults=...)`` substitutes the horizon-scaled
# chirp_num_basis grid; without ``defaults`` this falls back to the h=12 grid.
TIER2_STAGES = [chirp_num_basis_stage(), *TIER2_FIXED_STAGES]

# ---- Tier 3: capacity (only if Tiers 1-2 leave a gap; --include-tier3) -------
TIER3_STAGES = [
    ("model_width", "config", "MODEL_WIDTH", [192, 256, 384]),
    ("laplace_k", "config", "LAPLACE_K", [128, 256]),
]

# ---- Sampling-time grid (no retraining; identical for every arm) -------------
# Two guidance families (CFG = classifier-free guidance):
#   * scalar  — a CONSTANT guidance weight across the whole reverse process.
#               guidance_power has NO effect on these (the sampler ignores it
#               for scalar guidance, llapdiff.py:328).
#   * ramp    — a SCHEDULED weight (g_min, g_max): it rises across the noise
#               schedule as g_min + (g_max-g_min)*alpha_bar**guidance_power, so
#               guidance_power shapes the ramp and IS swept for these cells.
# The config default (1.0, 2.0) @ power 0.3 is the first ramp cell, so the
# shipped protocol is one of the compared cells (no need for a separate None
# entry — trial *selection* already scores the default protocol). The weights
# source (raw vs EMA) is not swept here; it is fixed by tune.py's
# --select-weights (default: ema).
SAMPLING_GRID = {
    "scalar_guidance": [1.0, 1.25, 1.5, 2.0],
    "ramp_guidance": [(1.0, 2.0), (1.0, 3.0)],
    "guidance_power": [0.3, 1.0, 2.0],   # applies to ramp cells only
    "steps": [32, 64],
}


def stages_for_arm(arm: str, *, include_tier3: bool = False, defaults=None) -> list[tuple]:
    """Sweep stages for one arm.

    ``defaults`` is the preset-applied config namespace for this (dataset,
    horizon); pass it so the chirp_num_basis grid scales with the horizon
    instead of using the h=12 fallback.
    """
    stages = list(TIER1_STAGES)
    if arm in CHIRP_ARMS:
        stages += [chirp_num_basis_stage(defaults), *TIER2_FIXED_STAGES]
    if include_tier3:
        stages += TIER3_STAGES
    return stages


def sampling_cells(weights: str) -> list[dict]:
    """All sampling cells for the sweep. Every cell carries the same four keys
    (guidance, steps, weights, guidance_power) so tune.py's resume/dedup key is
    well defined; guidance_power is None for scalar cells (where it is inert)."""
    cells: list[dict] = []
    for steps in SAMPLING_GRID["steps"]:
        for guidance in SAMPLING_GRID["scalar_guidance"]:
            cells.append({
                "guidance": guidance, "steps": steps,
                "weights": weights, "guidance_power": None,
            })
        for ramp in SAMPLING_GRID["ramp_guidance"]:
            for power in SAMPLING_GRID["guidance_power"]:
                cells.append({
                    "guidance": list(ramp), "steps": steps,
                    "weights": weights, "guidance_power": power,
                })
    return cells
