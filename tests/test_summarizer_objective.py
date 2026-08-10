"""B21 §6 — what the stage-2 pretraining objective can supervise.

`LaplaceAE.forward` returns `context` of shape [B, S, Hc] and that is what the denoiser
consumes, but the legacy decoders read `context.mean(dim=1)`. The reconstruction loss is
therefore a function of ONE token direction: the other S-1 lie in its null space and receive
no gradient from any objective, ever. `decode_mode="token"` decodes per time step through
cross-attention over all S tokens instead.

These tests pin both halves: the legacy mode must keep reproducing the defect (it is what
every shipped artifact was trained under, and the campaign's recorded numbers depend on it),
and the repaired mode must actually reach every token position.
"""

import pytest
import torch

from llapdiffusion.configs.config_utils import (
    clone_config,
    refresh_artifact_paths,
    summarizer_ckpt_path,
)
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.trainers import train_val_summarizer as tvs

B, K, N, D, S, HC = 2, 12, 3, 2, 12, 16
HEADS = ("decoder_net", "v_decoder", "t_decoder", "dt_decoder", "obs_decoder")


def _summarizer(decode_mode: str, *, out_len: int = S) -> LaplaceAE:
    torch.manual_seed(0)
    return LaplaceAE(
        num_entities=N,
        feat_dim=D,
        window_size=K,
        out_len=out_len,
        context_dim=HC,
        n_heads=4,
        dropout=0.0,
        decode_mode=decode_mode,
    ).eval()


def _zero_token_mean_delta(shape, scale: float = 38.35) -> torch.Tensor:
    """A perturbation of `context` that leaves the token-axis mean exactly unchanged."""
    torch.manual_seed(1)
    delta = torch.randn(*shape)
    delta = delta - delta.mean(dim=1, keepdim=True)
    return delta / delta.norm() * scale


def _batch():
    torch.manual_seed(2)
    x = torch.randn(B, K, N, D)
    return {
        "x": x,
        "ctx_diff": torch.diff(x, dim=1, prepend=x[:, :1]),
        "pad_mask": torch.ones(B, N, dtype=torch.bool),
        "dt": torch.arange(K, dtype=torch.float32).view(1, K, 1).expand(B, K, N).contiguous(),
    }


def test_unknown_decode_mode_raises():
    with pytest.raises(ValueError, match="decode_mode"):
        _summarizer("pooled")


def test_mean_decode_cannot_see_anything_but_the_token_mean():
    """The defect itself. If this ever fails, the legacy artifacts are not what B21 measured."""
    model = _summarizer("mean")
    ctx = torch.randn(B, S, HC)
    delta = _zero_token_mean_delta(ctx.shape)

    with torch.no_grad():
        for name in HEADS:
            head = getattr(model, name)
            moved = (head(model._decode_readout(ctx + delta)) - head(model._decode_readout(ctx)))
            assert moved.abs().max() < 1e-6, f"{name} moved by more than float32 noise"


def test_token_decode_reaches_every_head():
    model = _summarizer("token")
    ctx = torch.randn(B, S, HC)
    delta = _zero_token_mean_delta(ctx.shape)

    with torch.no_grad():
        for name in HEADS:
            head = getattr(model, name)
            moved = (head(model._decode_readout(ctx + delta)) - head(model._decode_readout(ctx)))
            assert moved.abs().max() > 1e-3, f"{name} is still blind to the token axis"


@pytest.mark.parametrize("decode_mode, positions_must_differ", [("mean", False), ("token", True)])
def test_gradient_distinguishes_token_positions_only_when_repaired(decode_mode, positions_must_differ):
    """Under "mean" every token position gets the SAME gradient (d/dcontext[s] = 1/S · d/dctx_mean),
    so nothing in the objective can shape one position differently from another."""
    model = _summarizer(decode_mode)
    ctx = torch.randn(B, S, HC, requires_grad=True)

    model.decoder_net(model._decode_readout(ctx)).pow(2).mean().backward()

    per_position = ctx.grad[0]                       # [S, HC]
    spread = (per_position - per_position.mean(dim=0)).abs().max().item()
    assert per_position.abs().sum() > 0
    if positions_must_differ:
        assert spread > 1e-6, "token positions still receive an identical gradient"
    else:
        assert spread < 1e-8, "legacy mode is expected to be rank-1 across the token axis"


@pytest.mark.parametrize("decode_mode", ["mean", "token"])
def test_forward_and_recon_loss_agree_on_shapes(decode_mode):
    """Both modes feed the same five heads, so `aux` and `recon_loss` are unchanged."""
    model = _summarizer(decode_mode)
    batch = _batch()

    context, aux = model(**batch)
    loss = model.recon_loss(aux, batch["pad_mask"], weights=(1.0, 0.1, 0.1, 0.0, 0.0))

    assert context.shape == (B, S, HC)
    assert aux["x_hat"].shape == (B, K, N, D)
    for key in ("v_hat", "t_hat", "dt_hat", "obs_hat"):
        assert aux[key].shape == (B, K, N)
    assert torch.isfinite(loss) and loss.requires_grad


def test_token_decode_handles_a_summary_shorter_than_the_history():
    """S == K only holds because the presets set SUM_CONTEXT_LEN = context_length; the config
    default is 1, so the decoder must not assume one token per reconstructed step."""
    model = _summarizer("token", out_len=3)
    context, aux = model(**_batch())

    assert context.shape == (B, 3, HC)
    assert aux["x_hat"].shape == (B, K, N, D)


def test_the_two_objectives_cannot_share_an_artifact(tmp_path):
    """Every loader is strict=True, so a silent architecture swap must be impossible: the
    state dicts are mutually unloadable AND the two modes resolve to different paths."""
    legacy, repaired = _summarizer("mean"), _summarizer("token")
    path = tmp_path / "summarizer.pt"
    tvs.save_ckpt(path, legacy, {"epoch": 1, "val_loss": 0.5})
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["decode_mode"] == "mean"
    legacy.load_state_dict(payload["model"])          # same architecture: fine
    with pytest.raises(RuntimeError):
        repaired.load_state_dict(payload["model"])

    cfg = clone_config()
    cfg.SUM_DIR, cfg.PRED, cfg.VAE_LATENT_CHANNELS = "/tmp/sum", 168, 16
    cfg.SUM_DECODE_MODE = "mean"
    assert summarizer_ckpt_path(cfg).name == "168-16-summarizer.pt"
    cfg.SUM_DECODE_MODE = "token"
    assert summarizer_ckpt_path(cfg).name == "168-16-summarizer_token.pt"
    refresh_artifact_paths(cfg)
    assert cfg.SUM_CKPT.endswith("168-16-summarizer_token.pt")


def test_repaired_decoder_is_not_bought_with_parameters():
    """The legacy heads emit the whole window from one vector (K·N·D outputs), which is 94.8 %
    of the shipped artifact. Per-time heads must not be a capacity increase in disguise."""
    legacy = sum(p.numel() for p in _summarizer("mean").parameters())
    repaired = sum(p.numel() for p in _summarizer("token").parameters())

    assert repaired <= legacy
