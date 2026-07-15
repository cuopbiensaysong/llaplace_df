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
TIER2_STAGES = [
    # Fourier basis size for the pole field. Config default is currently 256;
    # the runbook suggests small M at short horizons (8 cycles across 12 steps
    # is near-Nyquist).
    ("chirp_num_basis", "config", "CHIRP_NUM_BASIS", [8, 32, 64, 256]),
    # Minimum decay floor rho_min (the Theorem-B bound constant).
    ("chirp_rho_min", "config", "CHIRP_RHO_MIN", [1e-5, 1e-4, 1e-3]),
    # Theorem-B' growth budget c_g; 0.0 recovers Theorem B exactly (T2 sweep).
    ("growth_budget", "config", "CHIRP_GROWTH_BUDGET", [0.0, math.log(2.0), math.log(5.0)]),
    # L2 shrinkage of the pole-variation coefficients toward the LTI case.
    ("chirp_coeff_l2", "config", "CHIRP_COEFF_L2", [0.0, 1e-4, 1e-2]),
]

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


def stages_for_arm(arm: str, *, include_tier3: bool = False) -> list[tuple]:
    stages = list(TIER1_STAGES)
    if arm in CHIRP_ARMS:
        stages += TIER2_STAGES
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
