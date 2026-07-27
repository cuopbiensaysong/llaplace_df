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
