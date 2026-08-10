from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llapdiffusion.models.lapformer import LapFormer
from llapdiffusion.models.laptrans import LaplaceTransformEncoder
from llapdiffusion.models.llapdiff_utils import normalize_cond_per_batch
from llapdiffusion.trainers import train_val_llapdiff as tv
from llapdiffusion.viz import plot_llapdiff_poles


def _encoder(mode: str) -> LaplaceTransformEncoder:
    return LaplaceTransformEncoder(
        k=4,
        feat_dim=2,
        hidden_dim=8,
        num_heads=2,
        cond_dim=6,
        rho_conditioning_mode=mode,
    )


def _cfg(mode: str = "raw") -> SimpleNamespace:
    return SimpleNamespace(
        VAE_LATENT_CHANNELS=3,
        MODEL_WIDTH=16,
        NUM_LAYERS=1,
        NUM_HEADS=2,
        PREDICT_TYPE="v",
        LAPLACE_K=4,
        TIMESTEPS=8,
        SCHEDULE="cosine",
        DROPOUT=0.0,
        ATTN_DROPOUT=0.0,
        SELF_COND=False,
        COND_POOL_MODE="mean",
        COND_POOL_USE_RAW=False,
        BLOCK_SUMMARY_ADALN=False,
        ANALYSIS_SUMMARY_QK=False,
        ANALYSIS_QK_USE_RAW=False,
        RHO_CONDITIONING_MODE=mode,
        SUM_CONTEXT_DIM=16,
        COND_ADAPTER_MODE="none",
        COND_ADAPTER_HIDDEN=8,
        COND_ADAPTER_DROPOUT=0.0,
        COND_ADAPTER_SCALE=0.1,
    )


def test_raw_rho_zero_conditioning_preserves_base_poles():
    encoder = _encoder("raw")
    cond = torch.zeros(3, 6)

    base_rho, base_omega = encoder._base_poles(torch.float32, torch.device("cpu"))
    rho, omega = encoder.effective_poles(3, torch.float32, torch.device("cpu"), cond=cond)

    torch.testing.assert_close(rho, base_rho.unsqueeze(0).expand_as(rho))
    torch.testing.assert_close(omega, base_omega.unsqueeze(0).expand_as(omega))


def test_legacy_effective_zero_conditioning_reproduces_old_rho_formula():
    encoder = _encoder("legacy_effective")
    cond = torch.zeros(2, 6)

    base_rho, _ = encoder._base_poles(torch.float32, torch.device("cpu"))
    rho, _ = encoder.effective_poles(2, torch.float32, torch.device("cpu"), cond=cond)

    expected = F.softplus(base_rho.unsqueeze(0).expand_as(rho)) + encoder.alpha_min
    torch.testing.assert_close(rho, expected)
    assert not torch.allclose(rho, base_rho.unsqueeze(0).expand_as(rho))


def test_invalid_rho_conditioning_mode_raises():
    with pytest.raises(ValueError, match="rho_conditioning_mode"):
        _encoder("effective")


def test_model_builder_propagates_raw_rho_conditioning_mode():
    cfg = _cfg("raw")
    kwargs = tv._llapdiff_model_kwargs(cfg)
    model = tv.build_llapdiff_model(cfg, torch.device("cpu"))

    assert kwargs["rho_conditioning_mode"] == "raw"
    assert model.model.analysis.rho_conditioning_mode == "raw"
    assert tv._llapdiff_model_config(cfg)["llapdiff"]["rho_conditioning_mode"] == "raw"


def test_checkpoint_missing_rho_conditioning_mode_defaults_to_legacy():
    cfg = _cfg("raw")
    payload = {
        "model_config": {
            "llapdiff": {
                "data_dim": 3,
                "hidden_dim": 16,
            }
        }
    }

    kwargs = tv._llapdiff_model_kwargs_from_checkpoint(cfg, payload)

    assert kwargs["data_dim"] == 3
    assert kwargs["hidden_dim"] == 16
    assert kwargs["rho_conditioning_mode"] == "legacy_effective"


def test_checkpoint_nested_model_config_preserves_raw_mode():
    cfg = _cfg("legacy_effective")
    payload = {
        "model_config": {
            "llapdiff": {
                "data_dim": 3,
                "rho_conditioning_mode": "raw",
            },
            "cond_adapter": {"mode": "none"},
        }
    }

    kwargs = tv._llapdiff_model_kwargs_from_checkpoint(cfg, payload)
    plot_kwargs = plot_llapdiff_poles._read_model_config(payload)

    assert kwargs["rho_conditioning_mode"] == "raw"
    assert plot_kwargs["rho_conditioning_mode"] == "raw"
    assert "llapdiff" not in plot_kwargs
    assert "cond_adapter" not in plot_kwargs


def test_plot_checkpoint_config_without_mode_defaults_to_legacy():
    payload = {"model_config": {"llapdiff": {"data_dim": 3}}}

    kwargs = plot_llapdiff_poles._read_model_config(payload)

    assert kwargs["data_dim"] == 3
    assert kwargs["rho_conditioning_mode"] == "legacy_effective"


# --------------------------------------------------------------------------
# B23 — how much conditioning actually reaches the pole field and the AdaLN.
#
# `normalize_cond_per_batch(mode="sample")` subtracts the mean over dims=(1,),
# which is the axis `pool_summary_tokens` then averages, so the pooled summary is
# algebraically zero. Both `make_pole_cond` (-> rho(t), omega(t), the UQ law) and
# `make_block_cond` (-> AdaLN) read that pool, so at the shipped defaults neither
# receives any history. These pin the current behaviour: a fix should flip the
# `sample`/`mean` cell to alive, and the alive cells must stay alive.
# --------------------------------------------------------------------------

_B23_HIDDEN = 32
_B23_TOKENS = 24
_B23_WINDOWS = 4


def _b23_lapformer(
    pool_mode: str,
    block_adaln: bool,
    pool_norm_mode: str = "inherit",
    *,
    uq_head: bool = False,
) -> LapFormer:
    model = LapFormer(
        input_dim=8,
        hidden_dim=_B23_HIDDEN,
        num_layers=1,
        num_heads=4,
        laplace_k=4,
        summary_pool_mode=pool_mode,
        summary_pool_norm_mode=pool_norm_mode,
        block_summary_adaln=block_adaln,
        denoiser_modal_type="chirp",
        chirp_uq_head=uq_head,
    )
    if pool_norm_mode == "global":
        raw = _b23_summary("sample")[1]
        model.set_cond_pool_stats(raw.mean(dim=(0, 1)), raw.std(dim=(0, 1)))
    return model


def _b23_summary(norm_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(cond_summary, cond_summary_raw) at cond_summary_raw's measured scale (std 0.574)."""
    torch.manual_seed(0)
    raw = torch.randn(_B23_WINDOWS, _B23_TOKENS, _B23_HIDDEN) * 0.574
    if norm_mode == "global":
        stats = (raw.mean(dim=(0, 1), keepdim=True), raw.std(dim=(0, 1), keepdim=True))
        return normalize_cond_per_batch(raw, mode="global", stats=stats), raw
    return normalize_cond_per_batch(raw, mode=norm_mode), raw


def test_the_pooled_repair_is_available_but_OFF_by_default():
    """B23's repair exists as a knob and is NOT the default.

    It was shipped as the default on 2026-08-07 and reverted the same day on a pre-registered
    rule: the matched attribution arm measured rescaled `ddim` 4.82 % against the control's
    6.41 %, i.e. -1.59 pt on the axis Phase 1 gates. It improves val CRPS and it is REQUIRED
    for the Theorem-C law to be heteroscedastic, so it is kept and documented rather than
    removed -- the U3 arms must switch it on deliberately.
    """
    from llapdiffusion.configs import config

    assert getattr(config, "COND_NORM_MODE", "sample") == "sample"
    assert getattr(config, "COND_POOL_MODE", "mean") == "mean"
    assert getattr(config, "COND_POOL_NORM_MODE", "inherit") == "inherit"
    # Both remain off: COND_POOL_USE_RAW feeds the pole field an unnormalised summary
    # (std 0.574) and stalled optimisation; BLOCK_SUMMARY_ADALN is closed by the arm-A
    # measurement (oneshot corr 0.12648/0.17021, below control on both seeds).
    assert getattr(config, "COND_POOL_USE_RAW", False) is False
    assert getattr(config, "BLOCK_SUMMARY_ADALN", False) is False


def test_pre_b23_checkpoints_still_rebuild_with_the_legacy_pool():
    """A checkpoint that predates the knob must keep pooling exactly what it trained with --
    otherwise every recorded gate number is rescored under a different model.

    The invariant is that the CHECKPOINT wins over the live config, so this forces the config
    to the opposite value rather than relying on whatever the current default happens to be.
    """
    cfg = _cfg("raw")
    cfg.COND_POOL_NORM_MODE = "global"
    assert tv._llapdiff_model_kwargs(cfg)["summary_pool_norm_mode"] == "global"

    kwargs = tv._llapdiff_model_kwargs_from_checkpoint(
        cfg, {"model_config": {"llapdiff": {"data_dim": 3}}}
    )
    assert kwargs["summary_pool_norm_mode"] == "inherit"


def test_global_pool_refuses_to_run_without_statistics():
    """Silently falling back would restore the exact degeneracy the mode exists to remove."""
    model = LapFormer(
        input_dim=8, hidden_dim=_B23_HIDDEN, num_layers=1, num_heads=4, laplace_k=4,
        summary_pool_norm_mode="global", denoiser_modal_type="chirp",
    )
    cond_summary, raw = _b23_summary("sample")
    t_vec = torch.zeros(_B23_WINDOWS, _B23_HIDDEN)

    with pytest.raises(RuntimeError, match="statistics"):
        model.make_pole_cond(t_vec, cond_summary=cond_summary, cond_summary_raw=raw)

    # And it will not quietly pool the mean-zeroed summary when the raw one is missing.
    model.set_cond_pool_stats(raw.mean(dim=(0, 1)), raw.std(dim=(0, 1)))
    with pytest.raises(RuntimeError, match="cond_summary_raw"):
        model.make_pole_cond(t_vec, cond_summary=cond_summary, cond_summary_raw=None)


def test_the_repair_revives_the_pole_path_at_the_shipped_norm_and_pool_modes():
    """The whole point: 'sample' x 'mean' -- the cell every campaign run used -- goes from an
    algebraic zero to a window-dependent vector, with no change to cross-attention."""
    cond_summary, raw = _b23_summary("sample")
    t_vec = torch.zeros(_B23_WINDOWS, _B23_HIDDEN)

    dead = _b23_lapformer("mean", False).make_pole_cond(
        t_vec, cond_summary=cond_summary, cond_summary_raw=raw)[:, _B23_HIDDEN:]
    repaired = _b23_lapformer("mean", False, pool_norm_mode="global").make_pole_cond(
        t_vec, cond_summary=cond_summary, cond_summary_raw=raw)[:, _B23_HIDDEN:]

    assert dead.std(dim=0).mean() < 1e-6
    assert repaired.std(dim=0).mean() > 1e-2
    # Unit-scaled, unlike COND_POOL_USE_RAW: this is the difference that stalled training.
    assert repaired.abs().max() < 10.0


def test_uq_law_parameters_vary_per_window_once_the_pool_is_repaired():
    """B23's consequence for Table 3: p0 and q come from cond_vec alone, so with a dead pool
    the analytic law's shape is identical for every window -- structurally homoscedastic."""
    cond_summary, raw = _b23_summary("sample")
    t_vec = torch.zeros(_B23_WINDOWS, _B23_HIDDEN)

    def uq_spread(pool_norm_mode: str):
        torch.manual_seed(0)
        model = _b23_lapformer("mean", False, pool_norm_mode=pool_norm_mode, uq_head=True)
        # to_uq's output layer is zero-initialised, so an untrained model returns constant
        # p0/q for ANY conditioning and would "reproduce" the defect in every column.
        for param in model.chirp_field.to_uq[-1].parameters():
            torch.nn.init.normal_(param, std=0.5)
        cond_vec = model.make_pole_cond(
            t_vec, cond_summary=cond_summary, cond_summary_raw=raw)
        p0, q = model.chirp_field.uq_params(cond_vec)
        return p0.std(dim=0).mean().item(), q.std(dim=0).mean().item()

    dead_p0, dead_q = uq_spread("inherit")
    live_p0, live_q = uq_spread("global")

    assert dead_p0 < 1e-8 and dead_q < 1e-8, "expected the shipped law to be window-independent"
    assert live_p0 > 1e-4 and live_q > 1e-4


def test_sample_norm_zeroes_the_token_mean_exactly():
    """The projection B23 rests on: no pooled consumer can see anything under "sample"."""
    cond_summary, _ = _b23_summary("sample")

    assert cond_summary.mean(dim=1).abs().max() < 1e-6


@pytest.mark.parametrize("norm_mode, pool_mode", [("global", "mean"), ("sample", "attn")])
def test_pole_conditioning_is_dead_at_defaults_and_alive_under_either_fix(norm_mode, pool_mode):
    t_vec = torch.zeros(_B23_WINDOWS, _B23_HIDDEN)
    cond_summary, raw = _b23_summary("sample")

    # Shipped defaults: COND_NORM_MODE="sample" x COND_POOL_MODE="mean".
    default_model = _b23_lapformer("mean", False)
    default_half = default_model.make_pole_cond(
        t_vec, cond_summary=cond_summary, cond_summary_raw=raw
    )[:, _B23_HIDDEN:]
    assert default_half.abs().max() < 1e-6, (
        "B23: the pole conditioning vector is expected to be identically zero at the "
        "shipped defaults. If this now carries signal the defaults changed -- update B23."
    )

    # Either knob revives it, and it must vary ACROSS windows, not merely be non-zero.
    fixed_summary, fixed_raw = _b23_summary(norm_mode)
    fixed_half = _b23_lapformer(pool_mode, False).make_pole_cond(
        t_vec, cond_summary=fixed_summary, cond_summary_raw=fixed_raw
    )[:, _B23_HIDDEN:]
    assert fixed_half.std(dim=0).mean() > 1e-4


def test_block_adaln_is_inert_alone_and_revived_by_a_pooling_fix():
    """Debugging lead 2 run alone is a no-op: the widened half is exact zeros."""
    t_vec = torch.zeros(_B23_WINDOWS, _B23_HIDDEN)
    cond_summary, raw = _b23_summary("sample")

    # block_summary_adaln=False: the summary never enters AdaLN at all.
    off = _b23_lapformer("mean", False)
    assert torch.equal(
        off.make_block_cond(t_vec, cond_summary=cond_summary, cond_summary_raw=raw), t_vec
    )

    # block_summary_adaln=True at the defaults: twice the width, all of it zeros.
    on = _b23_lapformer("mean", True)
    block_cond = on.make_block_cond(t_vec, cond_summary=cond_summary, cond_summary_raw=raw)
    assert block_cond.shape[-1] == 2 * _B23_HIDDEN
    assert block_cond[:, _B23_HIDDEN:].abs().max() < 1e-6, (
        "B23: BLOCK_SUMMARY_ADALN=True is inert unless paired with COND_NORM_MODE='global' "
        "or COND_POOL_MODE='attn'."
    )

    # Paired with attention pooling it carries window-dependent signal.
    paired = _b23_lapformer("attn", True).make_block_cond(
        t_vec, cond_summary=cond_summary, cond_summary_raw=raw
    )
    assert paired[:, _B23_HIDDEN:].std(dim=0).mean() > 1e-4
