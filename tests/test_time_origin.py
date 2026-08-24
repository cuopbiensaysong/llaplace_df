"""The query-time origin is part of the model, and it decides whether the predictive law
is a stochastic process at all.

Two things are pinned here.

1. **The shipped origin is already query-independent.** The loaders anchor ``delta_t_y`` at
   the LAST HISTORY timestamp (``fin_dataset._compute_time_offsets_from_anchor(times,
   Tf[e-1], ...)``) and ``relative_time`` preserves it (``recenter=False`` on the ``dt``
   path), so the first lead gap ``t^q_1 - t_i`` survives and inserting or dropping a query
   moves no other offset. Measured on noaa_uk h=168: first offset 1.0, not 0.0; subset
   discrepancy exactly 0.

   This matters beyond tidiness: the alternative anchoring is what breaks subset
   consistency, and a silent regression to it would invalidate every joint-law claim
   without changing a loss curve.

2. **The legacy anchoring still exists and still fails**, deliberately. It is the positive
   control for the marginalization-consistency test, so a test that only checked the good
   path would let the control rot.
"""

import pytest
import torch

from llapdiffusion.models.laptrans import (
    TIME_ORIGINS,
    LaplaceTransformEncoder,
    normalize_time_origin,
)

_rel = LaplaceTransformEncoder.relative_time


def _offsets(dt, **kw):
    """t_rel for a [B,T] tensor of query offsets, squeezed back to [B,T]."""
    B, T = dt.shape
    return _rel(B, T, torch.float32, torch.device("cpu"), dt=dt, **kw)[..., 0]


# A lead gap of 3, then irregular spacing -- exactly the shape delta_t_y has.
LEAD_GAP = 3.0
QUERIES = torch.tensor([[3.0, 4.0, 7.0, 8.0, 13.0]])


def test_normalize_time_origin_round_trips_and_rejects_junk():
    for mode in TIME_ORIGINS:
        assert normalize_time_origin(mode) == mode
        assert normalize_time_origin(mode.upper()) == mode
    with pytest.raises(ValueError, match="Unknown time_origin"):
        normalize_time_origin("first_query")


def test_dt_path_preserves_the_lead_gap():
    """The default path must NOT recenter: offset 0 is the lead gap, not zero."""
    out = _offsets(QUERIES)
    assert out[0, 0].item() == pytest.approx(LEAD_GAP)
    torch.testing.assert_close(out, QUERIES)


def test_default_and_explicit_last_history_agree():
    """TIME_ORIGIN='last_history' must be a no-op relabelling of the shipped behaviour."""
    torch.testing.assert_close(_offsets(QUERIES), _offsets(QUERIES, time_origin="last_history"))


@pytest.mark.parametrize("origin", [None, "last_history"])
def test_subset_consistency_holds_for_the_shipped_origin(origin):
    """Dropping the first query must not move any remaining offset.

    This is the concrete content of projective consistency (Prop. C.1) at the level of the
    time map: the same physical query time gets the same t_rel whatever else was asked for.
    """
    full = _offsets(QUERIES, time_origin=origin)
    for drop in range(QUERIES.shape[1]):
        keep = [i for i in range(QUERIES.shape[1]) if i != drop]
        sub = _offsets(QUERIES[:, keep], time_origin=origin)
        torch.testing.assert_close(sub, full[:, keep])


def test_inserting_a_query_before_the_first_changes_nothing():
    """Only insertion BEFORE the first point exercises the origin -- so test exactly that."""
    base = _offsets(QUERIES)
    grown = _offsets(torch.cat([torch.tensor([[1.0]]), QUERIES], dim=1))
    torch.testing.assert_close(grown[:, 1:], base)


def test_query_first_origin_breaks_subset_consistency():
    """The legacy control must keep failing, and fail by the amount the geometry implies."""
    full = _offsets(QUERIES, time_origin="query_first")
    assert full[0, 0].item() == pytest.approx(0.0)  # lead gap destroyed

    sub = _offsets(QUERIES[:, 1:], time_origin="query_first")
    shift = (full[:, 1:] - sub).abs().max().item()
    # Dropping query 0 moves the origin from t=3 to t=4, so every offset shifts by 1.
    assert shift == pytest.approx(QUERIES[0, 1].item() - QUERIES[0, 0].item())


def test_window_start_origin_is_query_independent_and_needs_its_anchor():
    """(I1): the anchor is a property of the window, so subsets must still agree."""
    w = torch.tensor([2.0])
    full = _offsets(QUERIES, time_origin="window_start", window_origin=w)
    torch.testing.assert_close(full, QUERIES - 2.0)
    sub = _offsets(QUERIES[:, 2:], time_origin="window_start", window_origin=w)
    torch.testing.assert_close(sub, full[:, 2:])

    with pytest.raises(ValueError, match="needs an explicit window_origin"):
        _offsets(QUERIES, time_origin="window_start")


def test_last_history_refuses_absolute_timestamps():
    """Absolute `t` carries no history anchor; recentering it would silently give
    `query_first` while the caller believes it asked for `last_history`."""
    B, T = QUERIES.shape
    with pytest.raises(ValueError, match="needs `dt`"):
        _rel(B, T, torch.float32, torch.device("cpu"), t=QUERIES, time_origin="last_history")


def test_t_argument_without_an_explicit_origin_keeps_legacy_recentering():
    """Back-compat: the historical argument-driven behaviour is unchanged."""
    B, T = QUERIES.shape
    out = _rel(B, T, torch.float32, torch.device("cpu"), t=QUERIES)[..., 0]
    torch.testing.assert_close(out, QUERIES - QUERIES[:, :1])


def test_no_time_metadata_falls_back_to_the_integer_grid():
    out = _rel(2, 4, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(out[..., 0], torch.arange(4.0).expand(2, 4))
