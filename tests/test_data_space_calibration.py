"""Data-space calibration in ``evaluate_regression`` — Table 3's missing three columns.

PIT-ECE, coverage @ 0.8 and mean interval width had no producer anywhere in the tree:
``evaluate_regression`` built the ensemble and threw it away at the end of each batch, and the
only calibration numbers that existed were LATENT-space closed-form ones, available for the
analytic arms and structurally impossible for the sampled arm. So A1 and A2/A3 could not be put
in the same column without being scored in different spaces by different estimators.

Two things these tests pin, in order of how much damage getting them wrong would do:

1. **Nothing moves when the flag is off, and CRPS/MAE/MSE do not move when it is on.** Thirteen
   call sites predate this, including the trainer's own model-selection evals, and one of them
   is running right now.
2. **The numbers are right at S = 25**, the ensemble size the pipeline actually runs — not at
   the large-S limit where every PIT estimator looks fine. That distinction is exactly what
   made the first ``ensemble_pit`` wrong, so it is what these tests are aimed at.

The stack is faked so the predictive distribution is chosen, not learned: a perfectly calibrated
ensemble must score ~0, and an over-confident one must be caught by coverage and width — the
failure CRPS is blind to.
"""

from types import SimpleNamespace

import pytest
import torch

from llapdiffusion.trainers import train_val_llapdiff as tv

# Big enough that the binomial noise floor on a 20-bin ECE sits near 0.001, so an assertion at
# 0.004 separates "exact" from the 0.0127 the naive rank form scored at this same S.
B, H, N, C = 64, 16, 8, 1
Z = N * C
NUM_BATCHES = 16
NUM_SAMPLES = 25          # config.NUM_EVAL_SAMPLES, i.e. what U3 runs
LEVEL = 0.8               # Table 3's coverage column


class _FakeStack:
    """Drives ``evaluate_regression`` from precomputed draws, so two runs are comparable.

    ``draw_scale`` multiplies the ensemble's spread while leaving the observations alone:
    0.25 is an over-confident forecaster, 4.0 an under-confident one, 1.0 perfectly calibrated.
    """

    def __init__(self, *, seed: int, draw_scale: float = 1.0, obs_mask=None):
        g = torch.Generator().manual_seed(seed)
        self.y = [torch.randn(B, H, N, C, generator=g) for _ in range(NUM_BATCHES)]
        self.draws = [
            [torch.randn(B, H, Z, generator=g) * draw_scale for _ in range(NUM_SAMPLES)]
            for _ in range(NUM_BATCHES)
        ]
        self.obs_mask = obs_mask
        self._batch = -1
        self._draw = 0

    # --- the pieces evaluate_regression calls ---
    def eval(self):
        return None

    def generate(self, **kwargs):
        draw = self.draws[self._batch][self._draw]
        self._draw += 1
        return draw

    def pack_targets(self, *args, **kwargs):
        self._batch += 1
        self._draw = 0
        obs = (
            self.obs_mask
            if self.obs_mask is not None
            else torch.ones(B, H, N, dtype=torch.bool)
        )
        return torch.zeros(B, H, N, C), torch.zeros(B, N, dtype=torch.bool), obs

    def targets(self, *args, **kwargs):
        return self.y[self._batch]

    def decode(self, vae, x0_norm, **kwargs):
        return x0_norm.reshape(B, H, N, C)


def _batch():
    W = 3
    xb = (torch.ones(B, N, W, 1), torch.zeros(B, N, W, 1))
    yb = torch.ones(B, N, H)
    meta = {
        "entity_mask": torch.ones(B, N, dtype=torch.bool),
        "delta_t": torch.zeros(B, N, W),
        "delta_t_y": torch.zeros(B, N, H),
        "x_obs_mask": torch.ones(B, N, W, 1, dtype=torch.bool),
        "y_obs_mask": torch.ones(B, N, H, dtype=torch.bool),
    }
    return xb, yb, meta


def _evaluate(stack: _FakeStack, **kwargs):
    saved = {
        name: getattr(tv, name)
        for name in (
            "_build_cond_summary_pair", "pack_targets_tokens",
            "targets_to_bhnc", "decode_latents_with_vae",
        )
    }
    tv._build_cond_summary_pair = lambda *a, **k: (torch.zeros(B, 2, 4), torch.zeros(B, 2, 4))
    tv.pack_targets_tokens = stack.pack_targets
    tv.targets_to_bhnc = stack.targets
    tv.decode_latents_with_vae = stack.decode
    # CRPS's second term samples its pairs with `torch.randint` on the GLOBAL RNG, so two
    # otherwise identical calls disagree unless the global seed is reset between them. That is
    # a pre-existing property of the estimator (pinned by the test below), not of calibration —
    # reset it here so this file measures the flag and nothing else.
    torch.manual_seed(20260808)
    try:
        return tv.evaluate_regression(
            stack,
            vae=object(),
            summarizer=object(),
            dataloader=[_batch() for _ in range(NUM_BATCHES)],
            device=torch.device("cpu"),
            mu_mean=torch.zeros(Z),
            mu_std=torch.ones(Z),
            config=SimpleNamespace(NUM_EVAL_SAMPLES=NUM_SAMPLES),
            steps=2,
            crps_pair_samples=8,
            generator_seed=4242,
            **kwargs,
        )
    finally:
        for name, fn in saved.items():
            setattr(tv, name, fn)


# --------------------------------------------------------------------- backward compatibility

def test_off_by_default_and_the_return_shape_is_unchanged():
    """Every pre-existing caller must see exactly the dict it saw before."""
    metrics = _evaluate(_FakeStack(seed=0))

    assert "calibration" not in metrics
    assert set(metrics) == {"crps", "mae", "mse", "pinball", "num_samples", "aggregation"}


def test_enabling_calibration_does_not_move_crps_mae_or_mse():
    """The PIT randomization must draw from its OWN generator.

    Sharing `generator` with the sampler would advance the sampling stream and silently shift
    every forecast metric — a change that would look like a modelling result.
    """
    off = _evaluate(_FakeStack(seed=7))
    on = _evaluate(_FakeStack(seed=7), calibration=True, calibration_level=LEVEL)

    for key in ("crps", "mae", "mse"):
        assert on[key] == pytest.approx(off[key], rel=0, abs=0), key
    assert on["pinball"] == pytest.approx(off["pinball"])


def test_crps_pair_sampling_rides_the_global_rng():
    """Documents WHY the harness above reseeds — and it is a live reporting hazard.

    `crps_pair_samples` pairs are drawn with `torch.randint` on the global RNG, so the same
    command run twice gives two CRPS values. It is small and it hits every arm equally, so it
    cannot manufacture an arm difference; but it lands inside the ± std of a seed sweep and is
    worth knowing about before that std is read as seed noise.
    """
    first = _evaluate(_FakeStack(seed=8))
    torch.manual_seed(999)                       # perturb only the global stream
    second = _evaluate(_FakeStack(seed=8))
    assert first["crps"] == pytest.approx(second["crps"]), "the harness reseed should pin it"

    # Without the reseed the two disagree: same draws, different pairs.
    saved = {
        name: getattr(tv, name)
        for name in ("_build_cond_summary_pair", "pack_targets_tokens",
                     "targets_to_bhnc", "decode_latents_with_vae")
    }
    stack_a, stack_b = _FakeStack(seed=8), _FakeStack(seed=8)
    tv._build_cond_summary_pair = lambda *a, **k: (torch.zeros(B, 2, 4), torch.zeros(B, 2, 4))
    tv.targets_to_bhnc = stack_a.targets
    tv.decode_latents_with_vae = stack_a.decode
    common = dict(
        vae=object(), summarizer=object(), device=torch.device("cpu"),
        mu_mean=torch.zeros(Z), mu_std=torch.ones(Z),
        config=SimpleNamespace(NUM_EVAL_SAMPLES=NUM_SAMPLES),
        steps=2, crps_pair_samples=8, generator_seed=4242,
    )
    try:
        torch.manual_seed(1)
        tv.pack_targets_tokens = stack_a.pack_targets
        a = tv.evaluate_regression(stack_a, dataloader=[_batch() for _ in range(2)], **common)
        torch.manual_seed(2)
        tv.pack_targets_tokens = stack_b.pack_targets
        tv.targets_to_bhnc = stack_b.targets
        tv.decode_latents_with_vae = stack_b.decode
        b = tv.evaluate_regression(stack_b, dataloader=[_batch() for _ in range(2)], **common)
    finally:
        for name, fn in saved.items():
            setattr(tv, name, fn)

    assert a["mae"] == pytest.approx(b["mae"]), "the point forecast is deterministic"
    assert a["crps"] != b["crps"], "CRPS is not — that is the global-RNG dependence"


# ------------------------------------------------------------------------------ the numbers

def test_perfectly_calibrated_ensemble_scores_near_zero_at_the_size_U3_runs():
    """S = 25, not the large-S limit. The naive rank form scored 0.0127 / 0.846 here."""
    metrics = _evaluate(_FakeStack(seed=1), calibration=True, calibration_level=LEVEL)
    cal = metrics["calibration"]

    assert cal["pit_calibration_error"] < 0.004
    assert cal["coverage"] == pytest.approx(LEVEL, abs=0.01)
    assert cal["reliability"]["0.9"] == pytest.approx(0.9, abs=0.01)
    assert cal["num_pit_values"] == NUM_BATCHES * B * H * N * C
    assert cal["coverage_level"] == LEVEL
    # Labelled, so a table can never silently mix it with the latent closed-form number.
    assert (cal["estimator"], cal["space"]) == ("ensemble_pit", "data")


def test_overconfidence_shows_up_in_coverage_and_width_where_crps_barely_moves():
    """The failure the recovery plan's Phase 3 records CRPS as blind to.

    A quarter-width ensemble is badly miscalibrated. Coverage and width say so loudly; CRPS
    moves by a fraction of the seed band (0.036), which is why it cannot be the only column.
    """
    good = _evaluate(_FakeStack(seed=2), calibration=True, calibration_level=LEVEL)
    tight = _evaluate(
        _FakeStack(seed=2, draw_scale=0.25), calibration=True, calibration_level=LEVEL
    )

    assert tight["calibration"]["coverage"] < 0.45, "an over-confident ensemble must under-cover"
    assert tight["calibration"]["pit_calibration_error"] > 0.10
    # Sharper AND wrong: width alone would have preferred it.
    assert tight["calibration"]["mean_interval_width"] < 0.4 * good["calibration"]["mean_interval_width"]


def test_underconfidence_over_covers_and_is_wide():
    wide = _evaluate(
        _FakeStack(seed=3, draw_scale=4.0), calibration=True, calibration_level=LEVEL
    )
    cal = wide["calibration"]

    assert cal["coverage"] > 0.97, "an over-wide ensemble must over-cover"
    assert cal["pit_calibration_error"] > 0.10


def test_mean_interval_width_matches_the_quantile_definition():
    """Pinned against an independent computation, not against itself."""
    stack = _FakeStack(seed=4)
    metrics = _evaluate(stack, calibration=True, calibration_level=LEVEL)

    lo = (1.0 - LEVEL) / 2.0
    total, count = 0.0, 0
    for batch in range(NUM_BATCHES):
        samples = torch.stack(
            [d.reshape(B, H, N, C) for d in stack.draws[batch]], dim=0
        )
        q_lo = torch.quantile(samples, lo, dim=0, interpolation="linear")
        q_hi = torch.quantile(samples, 1.0 - lo, dim=0, interpolation="linear")
        total += float((q_hi - q_lo).sum())
        count += q_hi.numel()

    assert metrics["calibration"]["mean_interval_width"] == pytest.approx(total / count, rel=1e-6)


def test_only_observed_elements_are_scored():
    """Same `valid` mask as MAE/MSE/CRPS — a calibration column over a different element set
    than the accuracy columns would not be comparable to them."""
    obs = torch.ones(B, H, N, dtype=torch.bool)
    obs[:, :, N // 2:] = False        # drop half the entities

    metrics = _evaluate(
        _FakeStack(seed=5, obs_mask=obs), calibration=True, calibration_level=LEVEL
    )

    assert metrics["calibration"]["num_pit_values"] == NUM_BATCHES * B * H * (N // 2) * C
    assert metrics["calibration"]["pit_calibration_error"] < 0.006


# ------------------------------------------------------------------------------------ guards

def test_rejects_an_ensemble_too_small_to_have_a_pit():
    with pytest.raises(ValueError, match="at least 2 draws"):
        _evaluate(_FakeStack(seed=6), calibration=True, num_samples=1)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
def test_rejects_a_nonsense_coverage_level(level):
    with pytest.raises(ValueError, match="calibration_level"):
        _evaluate(_FakeStack(seed=6), calibration=True, calibration_level=level)
