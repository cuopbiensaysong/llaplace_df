"""Guards for the speed-up work: the val/test protocol split and its supporting helpers.

The load-bearing property is that the cheap in-training validation protocol (the ``EVAL_*``
keys, which exist only to rank checkpoints) can never influence a reported number, which is
always resolved through ``prefix="TEST"``.
"""

from types import SimpleNamespace

import pytest
import torch

from llapdiffusion.configs.config_utils import dataloader_kwargs
from llapdiffusion.eval_subsets import (
    BatchSubset,
    limit_batches,
    prefix_indices,
    resolve_max_eval_batches,
    strided_indices,
    subset_indices,
)
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.trainers import train_val_llapdiff as tv


def _cfg(**overrides) -> SimpleNamespace:
    base = dict(
        GEN_STEPS=64,
        NUM_EVAL_SAMPLES=25,
        GUIDANCE_STRENGTH=(1.0, 2.0),
        GUIDANCE_POWER=0.3,
        GEN_ETA=0.0,
        DYNAMIC_THRESH_P=0.0,
        DYNAMIC_THRESH_MAX=1.0,
        KARRAS_RHO=7.5,
        EVAL_STEPS=None,
        EVAL_NUM_SAMPLES=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------- protocol split
def test_eval_overrides_do_not_leak_into_the_test_protocol():
    cfg = _cfg(EVAL_STEPS=16, EVAL_NUM_SAMPLES=5)
    evaluation = tv._sampling_kwargs(cfg, prefix="EVAL")
    reported = tv._sampling_kwargs(cfg, prefix="TEST")

    assert (evaluation["steps"], evaluation["num_samples"]) == (16, 5)
    assert (reported["steps"], reported["num_samples"]) == (64, 25)


def test_unset_eval_overrides_fall_back_to_the_reported_protocol():
    """A key present but None counts as unset -- that is how config.py declares them."""
    cfg = _cfg()
    for prefix in ("EVAL", "TEST"):
        resolved = tv._sampling_kwargs(cfg, prefix=prefix)
        assert (resolved["steps"], resolved["num_samples"]) == (64, 25)


def test_missing_keys_also_fall_back():
    cfg = _cfg()
    del cfg.EVAL_STEPS
    del cfg.EVAL_NUM_SAMPLES
    assert tv._sampling_kwargs(cfg, prefix="EVAL")["steps"] == 64


def test_prefix_eval_is_used_only_by_the_in_training_val_evals():
    """If a reported-number call site ever switches to prefix='EVAL', this fails."""
    import pathlib

    root = pathlib.Path(tv.__file__).resolve().parents[2]
    hits = []
    for path in list(root.joinpath("llapdiffusion").rglob("*.py")) + list(
        root.joinpath("finetuning").rglob("*.py")
    ):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "_sampling_kwargs(" in line and 'prefix="EVAL"' in line:
                hits.append(f"{path.relative_to(root)}:{lineno}")
    assert hits, "expected to find the trainer's val-eval call sites"
    assert all(
        h.startswith("llapdiffusion/trainers/train_val_llapdiff.py") for h in hits
    ), hits


def test_evaluate_regression_num_samples_defaults_to_config():
    import inspect

    signature = inspect.signature(tv.evaluate_regression)
    assert signature.parameters["num_samples"].default is None


# --------------------------------------------------------------------- eval subsets
def test_strided_subset_spans_the_split_while_prefix_truncates():
    assert prefix_indices(146, 48) == list(range(48))
    strided = strided_indices(146, 48)
    assert len(strided) == 48
    assert strided[0] == 0
    # The point of striding: the subset must reach the far end of a chronological split.
    assert strided[-1] > 100
    assert strided == sorted(set(strided))


def test_subset_is_a_noop_when_the_cap_is_absent_or_larger_than_the_split():
    loader = [("x", "y", {"i": i}) for i in range(10)]
    assert limit_batches(loader, None) is loader
    assert limit_batches(loader, 0) is loader
    assert limit_batches(loader, 99) is loader


def test_subset_yields_exactly_the_selected_batches_and_reports_its_length():
    loader = [("x", "y", {"i": i}) for i in range(10)]
    subset = limit_batches(loader, 4, mode="stride")
    assert isinstance(subset, BatchSubset)
    assert len(subset) == 4
    assert [b[2]["i"] for b in subset] == subset.source_indices
    assert subset.source_indices == [0, 2, 4, 6]

    prefix = limit_batches(loader, 4, mode="prefix")
    assert [b[2]["i"] for b in prefix] == [0, 1, 2, 3]


def test_subset_is_reiterable_and_stops_early():
    """len() must stay honest across repeated epochs, and iteration must not walk the tail."""
    loader = [("x", "y", {"i": i}) for i in range(10)]
    subset = limit_batches(loader, 3, mode="stride")
    first = [b[2]["i"] for b in subset]
    second = [b[2]["i"] for b in subset]
    assert first == second == [0, 3, 6]


def test_resolve_max_eval_batches_rejects_negative():
    assert resolve_max_eval_batches(None) is None
    assert resolve_max_eval_batches(0) is None
    assert resolve_max_eval_batches(7) == 7
    with pytest.raises(ValueError):
        resolve_max_eval_batches(-1)


def test_unknown_subset_mode_is_rejected():
    with pytest.raises(ValueError, match="eval subset mode"):
        subset_indices(10, 3, mode="random")


# --------------------------------------------------------------------- dataloader kwargs
def test_dataloader_kwargs_omits_worker_only_options_at_zero_workers():
    """PyTorch rejects persistent_workers/prefetch_factor when num_workers == 0."""
    cfg = SimpleNamespace(
        DATALOADER_NUM_WORKERS=0,
        DATALOADER_PERSISTENT_WORKERS=True,
        DATALOADER_PREFETCH_FACTOR=4,
    )
    assert dataloader_kwargs(cfg) == {"num_workers": 0}

    cfg.DATALOADER_NUM_WORKERS = 8
    assert dataloader_kwargs(cfg) == {
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    cfg.DATALOADER_PREFETCH_FACTOR = None
    assert "prefetch_factor" not in dataloader_kwargs(cfg)


# --------------------------------------------------------------------- tf32
def test_tf32_is_opt_in_and_deterministic_overrides_it():
    try:
        set_torch(0)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        set_torch(0, allow_tf32=True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        set_torch(0, allow_tf32=True, deterministic=True)
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        set_torch(0)


# --------------------------------------------------------------------- _flatten_dt
def _meta(grids):
    return {"delta_t_y": torch.tensor(grids, dtype=torch.float32)}


def test_flatten_dt_accepts_a_shared_query_grid():
    meta = _meta([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    mask = torch.tensor([[True, True]])
    out = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]))


def test_flatten_dt_rejects_per_entity_query_grids():
    meta = _meta([[[1.0, 2.0, 3.0], [1.0, 2.0, 9.0]]])
    mask = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="same query grid"):
        tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")


def test_flatten_dt_ignores_masked_entities_and_single_entity_rows():
    """A divergent grid on a masked-out entity was allowed before and must stay allowed."""
    meta = _meta([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]])
    mask = torch.tensor([[True, False]])
    out = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]))


def test_flatten_dt_tolerates_rows_with_no_valid_entity():
    meta = _meta([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    mask = torch.tensor([[False, False]])
    out = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")
    assert torch.allclose(out, torch.zeros(1, 3))


def test_flatten_dt_raises_on_non_finite_values_for_valid_entities():
    meta = _meta([[[1.0, float("nan"), 3.0], [1.0, float("nan"), 3.0]]])
    mask = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="non-finite"):
        tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")


def test_flatten_dt_grid_check_can_be_disabled(monkeypatch):
    meta = _meta([[[1.0, 2.0, 3.0], [1.0, 2.0, 9.0]]])
    mask = torch.tensor([[True, True]])
    monkeypatch.setattr(tv.config, "DIFF_VALIDATE_DT_GRIDS", False, raising=False)
    tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")


# --------------------------------------------------------------------- grad finiteness
def test_grads_finite_flag_defers_the_device_sync():
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    for p in params:
        p.grad = torch.zeros(2)
    flag = tv._grads_finite_flag(params)
    assert isinstance(flag, torch.Tensor) and flag.dtype == torch.bool
    assert bool(flag) is True

    params[1].grad = torch.tensor([float("inf"), 0.0])
    assert bool(tv._grads_finite_flag(params)) is False
    assert tv._grads_are_finite_params(params) is False


def test_grads_finite_flag_is_none_without_gradients():
    params = [torch.nn.Parameter(torch.zeros(2))]
    assert tv._grads_finite_flag(params) is None
    assert tv._grads_are_finite_params(params) is True
