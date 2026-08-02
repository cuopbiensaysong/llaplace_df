"""Gate-0 stage-1 round-trip tool: argument surface and gate wiring.

The heavy path (train a VAE, encode/decode) is validated live against the phase-fix
campaign's checkpoints; here we pin the cheap invariants that broke twice this
session -- the parser producing a namespace that _configure/_build_loaders accept,
and the phase/budget flags actually reaching the config.
"""

import numpy as np
import pytest

from llapdiffusion.tools import run_stage1_roundtrip as rt
from llapdiffusion.tools.run_synthetic_chirp_benchmark import _configure


def _args(**over):
    from types import SimpleNamespace
    base = dict(
        tasks=["synthetic_linear_chirp"], seeds=(0,), window=96, horizon=48,
        series_length=768, num_entities=64, change_point=None,
        gap_distribution="gamma", gap_mean=1.0, gap_shape=4.0,
        sweep_period=144.0, phase_spread=0.393, laplace_k=8, chirp_num_basis=8,
        data_root="/tmp/d", artifact_root="/tmp/a", threshold=0.85,
        json_out=None, verbose=False, debug=False, smoke=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_parser_namespace_drives_configure():
    """The tool's args must satisfy _configure without AttributeError, and the
    phase/K/M flags must land in the built config (the leak-and-crash class)."""
    cfg = _configure("synthetic_linear_chirp", "chirp", 0, _args())
    assert cfg.LAPLACE_K == 8
    assert cfg.CHIRP_NUM_BASIS == 8
    assert "phase-0.393" in cfg.DATA_DIR          # phase-narrowed cache is addressed
    assert cfg.VAE_CKPT.endswith("_entity_elbo.pt")


def test_defaults_are_historical():
    """Default phase spread is the historical 2*pi so nothing silently shifts."""
    ns = _args(phase_spread=2.0 * np.pi, laplace_k=256, chirp_num_basis=None)
    cfg = _configure("synthetic_linear_chirp", "chirp", 0, ns)
    assert "phase-" not in cfg.DATA_DIR            # default omits the tag
    assert cfg.LAPLACE_K == 256


def test_roundtrip_correlation_needs_windows(monkeypatch):
    """Empty test set fails loudly rather than returning a meaningless number."""
    class _EmptyVAE:
        pass
    monkeypatch.setattr(rt, "_build_vae", lambda *a, **k: _EmptyVAE())
    loaders = ([], None, [], None)   # no test batches
    with pytest.raises(RuntimeError, match="No valid entity-windows"):
        rt.roundtrip_correlation(object(), loaders, __import__("torch").device("cpu"))


def test_roundtrip_correlation_selects_the_requested_split(monkeypatch):
    """`split` picks the loader; anything else fails loudly rather than defaulting."""
    monkeypatch.setattr(rt, "_build_vae", lambda *a, **k: object())
    loaders = ([], [], [], None)
    torch = __import__("torch")
    with pytest.raises(ValueError, match="split must be one of"):
        rt.roundtrip_correlation(object(), loaders, torch.device("cpu"), split="valid")
    # train/val/test all reach the empty-window guard, i.e. all three are wired.
    for split in ("train", "val", "test"):
        with pytest.raises(RuntimeError, match="No valid entity-windows"):
            rt.roundtrip_correlation(object(), loaders, torch.device("cpu"), split=split)


def _fake_roundtrip_batch(monkeypatch, y_true, recon):
    """Wire roundtrip_correlation's helpers to serve one canned [B,H,N,1] batch."""
    import torch

    B, H, N = y_true.shape[0], y_true.shape[1], y_true.shape[2]
    mask = torch.ones(B, N, dtype=torch.bool)
    monkeypatch.setattr(rt, "_build_vae", lambda *a, **k: (lambda *_: (None, torch.zeros(B, H, 4), None)))
    monkeypatch.setattr(rt, "pack_targets_tokens", lambda *a, **k: (object(), object(), None))
    monkeypatch.setattr(rt, "decode_latents_with_vae", lambda *a, **k: recon)
    monkeypatch.setattr(rt.tv, "_sanitize_batch", lambda *a, **k: (None, None, mask))
    monkeypatch.setattr(rt.tv, "targets_to_bhnc", lambda *a, **k: y_true)
    return ([], [], [(None, None, {})], None)


def test_constant_windows_are_skipped_and_counted(monkeypatch):
    """A window whose correlation is undefined must not poison the aggregate.

    PhysioNet val is ~71% constant target windows (median true std exactly 0, sparse
    clinical series carried forward). `np.corrcoef` on a constant series is nan, and a
    single nan turned the whole mean into nan while inflating std_ratio to ~1e6 — the
    gate reported `nan / FAIL` instead of a number. Synthetic targets are never
    constant, which is why this only appeared once real datasets were admitted.
    """
    import torch

    H = 8
    ramp = torch.arange(H, dtype=torch.float32)
    y_true = torch.zeros(1, H, 3, 1)
    y_true[0, :, 0, 0] = ramp                 # informative
    y_true[0, :, 1, 0] = 5.0                  # constant target -> below the 1e-8 floor
    y_true[0, :, 2, 0] = ramp                 # informative target...
    recon = torch.zeros(1, H, 3, 1)
    recon[0, :, 0, 0] = ramp * 0.5            # correlates perfectly, half the scale
    recon[0, :, 1, 0] = 5.0
    recon[0, :, 2, 0] = 3.0                   # ...but a FLAT reconstruction -> corr undefined

    loaders = _fake_roundtrip_batch(monkeypatch, y_true, recon)
    stats = rt.roundtrip_correlation(object(), loaders, torch.device("cpu"), split="test")

    assert stats["n"] == 1 and stats["n_degenerate"] == 2
    assert stats["corr"] == pytest.approx(1.0)      # finite, not nan
    assert stats["std_ratio"] == pytest.approx(0.5)  # the 1e6 outlier is gone


def test_dataset_mode_argument_guards():
    """The two modes must not blend: a synthetic geometry flag silently ignored in
    dataset mode would score a different cell than the caller asked for (the B4 class)."""
    import sys

    def parse(*argv):
        saved, sys.argv = sys.argv, ["llapdiff-stage1-roundtrip", *argv]
        try:
            return rt._parse_args()
        finally:
            sys.argv = saved

    with pytest.raises(SystemExit):
        parse("--dataset-key", "noaa_uk")                     # --pred required
    with pytest.raises(SystemExit):
        parse("--dataset-key", "noaa_uk", "--pred", "168", "--tasks", "x")   # exclusive
    with pytest.raises(SystemExit):
        parse("--dataset-key", "noaa_uk", "--pred", "168", "--phase-spread", "0.393")
    with pytest.raises(SystemExit):
        parse("--tasks", "synthetic_linear_chirp", "--pred", "168")  # --pred is dataset-mode

    # Split defaults differ by mode: gates run on val, synthetic keeps its historical test.
    assert parse("--dataset-key", "noaa_uk", "--pred", "168").split == "val"
    assert parse("--tasks", "synthetic_linear_chirp").split == "test"
    assert parse("--tasks", "synthetic_linear_chirp", "--split", "val").split == "val"
    # Task mode keeps its historical default task list.
    assert parse().tasks == ["synthetic_linear_chirp"] and parse().split == "test"


def test_blocked_purged_split_degrades_instead_of_collapsing():
    """Regression: purging three blocks has no lower bound on the fit set.

    `blocked_purged_split(purge=WINDOW)` bans up to `2*len(holdout)*purge` windows. Unlike the
    trailing-20% rule it replaced, that has no floor -- at n_train ~ 2*3*336 the alpha-selection
    fit set collapsed to ZERO and `ridge_reduction` died with
    `TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'` deep inside, because
    the alpha loop never ran. Found on bms_air h=96 by a second agent; the regression came from
    the very fix that agent had proposed.
    """
    import numpy as np
    from llapdiffusion.tools.run_ridge_probe import blocked_purged_split

    # The exact configuration that crashed.
    fit, hold = blocked_purged_split(1420, purge=336)
    assert len(fit) > 0 and len(hold) > 0
    assert not (set(fit.tolist()) & set(hold.tolist()))

    # The sizes our published results used must be untouched by the guard.
    for n, expect_frac in ((14446, 0.66), (30359, 0.73)):
        fit, _ = blocked_purged_split(n, purge=336)
        assert abs(len(fit) / n - expect_frac) < 0.02, f"n={n} fit fraction moved"

    # Degradation keeps the fit set above the floor at every size it can serve...
    for n in (700, 1420, 2500, 5000):
        fit, hold = blocked_purged_split(n, purge=336)
        assert len(fit) >= 0.5 * n and len(hold) >= 2

    # ...and where it genuinely cannot, it says so instead of returning something unusable.
    with pytest.raises(ValueError, match="cannot leave"):
        blocked_purged_split(40, purge=336)
