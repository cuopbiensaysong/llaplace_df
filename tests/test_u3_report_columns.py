"""Table 3's calibration columns must survive the driver's aggregation, for EVERY arm.

The producer lives in `evaluate_regression` (see `test_data_space_calibration.py`); this pins
the other half — that `run_u3_uq.py report` carries PIT-ECE, coverage and mean interval width
from the per-seed eval JSONs into the table, and that **A1 gets them too**. A1 is the arm the
old pre-registration excluded from calibration entirely, so "the sampled arm has a PIT-ECE" is
the specific thing worth a regression test.

It also pins the guard that stops two arms scored at different nominal levels from being read
off one column, which is silent otherwise: 0.9 coverage under an 0.8 heading looks like an arm
that over-covers.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

# `finetuning/` is a script directory, not an installed package (see test_campaign_state_merge).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))

import run_u3_uq as u3  # noqa: E402


def _eval_json(*, crps, pit, coverage, width, level=0.8, sampled=True):
    """One arm's data-space block, in the shape run_analytic_uq_eval writes."""
    block = {
        "crps": crps, "mae": 0.4, "mse": 0.3,
        "wall_seconds": 12.0,
        "pit_calibration_error": pit,
        "coverage": coverage,
        "coverage_level": level,
        "mean_interval_width": width,
        "calibration_estimator": "ensemble_pit",
        "calibration_space": "data",
    }
    out = {"data_space_analytic": dict(block, denoiser_passes_per_batch=65)}
    if sampled:
        out["data_space_sampled"] = dict(block, denoiser_passes_per_batch=1600)
    return out


def _campaign(tmp_path, monkeypatch, seeds=(0, 1)):
    monkeypatch.setattr(u3, "combo_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(u3, "state_path", lambda *a, **k: tmp_path / "state.json")
    args = SimpleNamespace(
        dataset_key="noaa_uk", pred=168, seeds=list(seeds), split="val",
        smoke=False, num_samples=None, dry_run=False, gpu=None,
    )
    return u3.Campaign(args)


def _write_evals(tmp_path, seeds, *, s2_kwargs, s3_kwargs):
    out = tmp_path / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        (out / f"seed{seed}_s2_diffusion_uq_val.json").write_text(
            json.dumps(_eval_json(**s2_kwargs[seed]))
        )
        (out / f"seed{seed}_s3_oneshot_uq_val.json").write_text(
            json.dumps(_eval_json(sampled=False, **s3_kwargs[seed]))
        )


def test_every_arm_including_the_sampled_one_carries_the_three_columns(tmp_path, monkeypatch):
    seeds = (0, 1)
    _write_evals(
        tmp_path, seeds,
        s2_kwargs={
            0: dict(crps=0.30, pit=0.020, coverage=0.78, width=1.10),
            1: dict(crps=0.32, pit=0.030, coverage=0.82, width=1.30),
        },
        s3_kwargs={
            0: dict(crps=0.35, pit=0.050, coverage=0.70, width=0.90),
            1: dict(crps=0.37, pit=0.070, coverage=0.74, width=1.10),
        },
    )
    campaign = _campaign(tmp_path, monkeypatch, seeds)
    campaign.report()

    payload = json.loads((tmp_path / "U3_val.json").read_text())
    assert payload["coverage_level"] == 0.8

    for arm in ("A1_diffusion_sampled", "A2_diffusion_analytic", "A3_oneshot_analytic"):
        entry = payload["arms"][arm]
        assert entry["n_seeds"] == 2, arm
        for column in ("pit_calibration_error", "coverage", "mean_interval_width"):
            assert entry[column] is not None, f"{arm} lost {column}"
            assert entry[column]["n"] == 2

    # Aggregation is over seeds, not a copy of seed 0.
    a2 = payload["arms"]["A2_diffusion_analytic"]
    assert a2["pit_calibration_error"]["mean"] == pytest.approx(0.025)
    assert a2["coverage"]["mean"] == pytest.approx(0.80)
    assert a2["mean_interval_width"]["mean"] == pytest.approx(1.20)
    assert a2["pit_calibration_error"]["std"] > 0

    # The efficiency column separates the arms by construction, not by wall-clock noise.
    assert payload["arms"]["A1_diffusion_sampled"]["denoiser_passes_per_batch"]["mean"] == 1600
    assert payload["arms"]["A2_diffusion_analytic"]["denoiser_passes_per_batch"]["mean"] == 65


def test_the_rendered_table_names_its_nominal_level_and_keeps_the_spaces_apart(tmp_path, monkeypatch):
    seeds = (0,)
    _write_evals(
        tmp_path, seeds,
        s2_kwargs={0: dict(crps=0.30, pit=0.02, coverage=0.78, width=1.1, level=0.9)},
        s3_kwargs={0: dict(crps=0.35, pit=0.05, coverage=0.70, width=0.9, level=0.9)},
    )
    campaign = _campaign(tmp_path, monkeypatch, seeds)
    campaign.report()

    md = (tmp_path / "RESULTS.md").read_text()
    assert "## Data-space calibration" in md
    assert "coverage @ 0.9" in md, "the table must state the level it was scored at"
    # Every arm has a row in the calibration table itself, sampled arm included. Sliced to
    # that section rather than counted over the whole document, so adding a table elsewhere
    # cannot break this and, more importantly, cannot satisfy it either.
    section = md.split("## Data-space calibration", 1)[1].split("\n## ", 1)[0]
    for label in ("A1 diffusion + sampled UQ", "A2 diffusion + analytic UQ (Thm C)",
                  "A3 one-shot NLL, no diffusion"):
        assert f"| {label} |" in section, label
    # The latent table must not be presentable as the same column.
    assert "exact-Gaussian cross-check" in md


def test_arms_scored_at_different_levels_are_refused(tmp_path, monkeypatch):
    seeds = (0,)
    _write_evals(
        tmp_path, seeds,
        s2_kwargs={0: dict(crps=0.30, pit=0.02, coverage=0.78, width=1.1, level=0.8)},
        s3_kwargs={0: dict(crps=0.35, pit=0.05, coverage=0.88, width=0.9, level=0.9)},
    )
    campaign = _campaign(tmp_path, monkeypatch, seeds)

    with pytest.raises(ValueError, match="different coverage levels"):
        campaign.report()


def test_an_older_eval_json_without_calibration_still_reports(tmp_path, monkeypatch):
    """Back-compat: a JSON written before the columns existed must not crash the driver."""
    out = tmp_path / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed0_s2_diffusion_uq_val.json").write_text(json.dumps({
        "data_space_analytic": {"crps": 0.30, "mae": 0.4, "mse": 0.3, "wall_seconds": 9.0},
        "data_space_sampled": {"crps": 0.31, "mae": 0.4, "mse": 0.3, "wall_seconds": 90.0},
    }))
    campaign = _campaign(tmp_path, monkeypatch, (0,))
    campaign.report()

    payload = json.loads((tmp_path / "U3_val.json").read_text())
    assert payload["coverage_level"] is None
    assert payload["arms"]["A2_diffusion_analytic"]["pit_calibration_error"] is None
    assert payload["arms"]["A2_diffusion_analytic"]["crps"]["mean"] == pytest.approx(0.30)
    assert "coverage @ —" in (tmp_path / "RESULTS.md").read_text()
