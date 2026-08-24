"""Correctness gate (e): an analytic interval must come from an NLL-trained UQ head.

``q_k`` and ``p0_k`` never enter the mean, so under ``DIFF_LOSS_MODE="mse"`` they receive
**exactly zero** gradient. A checkpoint then carries whatever ``CHIRP_UQ_INIT_VAR`` put
there, and every interval computed from it restates that constant. The failure is silent:
the loss curve is healthy and CRPS moves by <0.001 while coverage goes from 3% to 87%
(measured on physionet h=12), so nothing downstream notices.

Two independent detectors are pinned here, because checkpoints written before the
objective was persisted carry no record of it:
  * the recorded ``diff_loss_mode`` in ``model_config``, for new checkpoints;
  * the untrained-head signature, which works on any checkpoint.
"""

import torch

from llapdiffusion.models.laptrans import ChirpModalField
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.models.llapdiff_utils import NoiseScheduler, diffusion_loss
from llapdiffusion.trainers.train_val_llapdiff import diff_loss_mode_from_checkpoint


def _field(**kw):
    return ChirpModalField(k=4, cond_dim=8, num_basis=4, uq_head=True, **kw)


def test_fresh_uq_head_reports_the_untrained_signature():
    sig = _field().uq_head_untrained_signature()
    assert sig is not None
    # The init is fully determined, so every statistic is exactly zero -- not merely small.
    assert set(sig) == {
        "to_uq_weight_absmax", "to_uq_bias_absmax", "p0_base_spread", "q_base_spread",
    }
    assert all(v == 0.0 for v in sig.values())


def test_signature_clears_once_any_uq_parameter_moves():
    # The perturbation has to be representable where it lands: the bases sit at
    # log(expm1(1e-2)) = -4.6, whose float32 ulp is ~5e-7, so a 1e-8 nudge there is exactly
    # zero and would test nothing. Both values below are still orders of magnitude smaller
    # than any real optimizer step.
    for mutate in (
        lambda f: f.to_uq[-1].weight.add_(1e-8),
        lambda f: f.to_uq[-1].bias.add_(1e-8),
        lambda f: f._p0_base.add_(torch.tensor([0.0, 1e-4, 0.0, 0.0])),
        lambda f: f._q_base.add_(torch.tensor([0.0, 0.0, 1e-4, 0.0])),
    ):
        field = _field()
        with torch.no_grad():
            mutate(field)
        assert field.uq_head_untrained_signature() is None


def test_signature_is_none_without_a_uq_head():
    assert ChirpModalField(k=4, cond_dim=8, num_basis=4).uq_head_untrained_signature() is None


def test_uq_head_gets_no_gradient_under_mse_but_does_under_nll():
    """The mechanism the gate exists to catch, exercised end to end."""
    torch.manual_seed(0)
    model = LLapDiff(
        data_dim=6, hidden_dim=16, num_layers=1, num_heads=2, laplace_k=4, timesteps=20,
        predict_type="x0", denoiser_modal_type="chirp", chirp_uq_head=True,
    )
    sched = NoiseScheduler(timesteps=20, schedule="cosine")
    x0 = torch.randn(2, 5, 6)
    cond = torch.randn(2, 1, 16)
    dt = torch.arange(1.0, 6.0).view(1, 5).expand(2, 5).contiguous()

    grads = {}
    for mode in ("mse", "gaussian_nll"):
        model.zero_grad(set_to_none=True)
        t = torch.full((2,), 5, dtype=torch.long)
        diffusion_loss(
            model, sched, x0, t, cond_summary=cond, predict_type="x0",
            dt=dt, loss_mode=mode,
        ).backward()
        head = model.model.chirp_field
        grads[mode] = max(
            float(p.grad.abs().max()) if p.grad is not None else 0.0
            for p in (head._p0_base, head._q_base, head.to_uq[-1].weight)
        )

    assert grads["mse"] == 0.0, "q_k/p0_k must be exactly untouched by the MSE loss"
    assert grads["gaussian_nll"] > 0.0, "the NLL must actually train the variance head"


def test_diff_loss_mode_from_checkpoint_distinguishes_absent_from_mse():
    """A checkpoint predating the field must read as None, never as 'mse' -- callers have
    to fall back to the signature rather than guess."""
    assert diff_loss_mode_from_checkpoint({"model_config": {}}) is None
    assert diff_loss_mode_from_checkpoint({}) is None
    assert diff_loss_mode_from_checkpoint(None) is None
    assert diff_loss_mode_from_checkpoint(
        {"model_config": {"diff_loss_mode": "mse"}}
    ) == "mse"
    assert diff_loss_mode_from_checkpoint(
        {"model_config": {"diff_loss_mode": "gaussian_nll"}}
    ) == "gaussian_nll"
