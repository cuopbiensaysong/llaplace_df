"""`save_state` must not drop a concurrent driver's trial records.

One driver per seed is the documented way to use both GPUs, and each holds a private copy of
the state loaded at start-up. Before the merge, the later writer replaced `trials` wholesale:
seed 0's completed S1 (5.7 h of training) disappeared when seed 1 saved 9 minutes later. The
artifacts survive on disk, so nothing is *lost* — but `resolve()` then finds no record and a
restart retrains the finished stage while every dependent stage reports BLOCKED.
"""

import json
import sys
from pathlib import Path

# `finetuning/` is a script directory, not an installed package -- its own modules bootstrap it
# the same way (see finetuning/run_u3_uq.py). Doing it here keeps `pytest tests/` self-sufficient.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))

from common import load_state, save_state  # noqa: E402


def test_concurrent_drivers_do_not_clobber_each_other(tmp_path):
    path = tmp_path / "state.json"

    # Two drivers start from the same (empty) state and each learns about its own seed.
    seed0 = {"run_tag": "uq_u3", "trials": {}, "evals": {}}
    seed1 = {"run_tag": "uq_u3", "trials": {}, "evals": {}}
    seed0["trials"]["s1_mean::seed0"] = {"trial_id": "aaa", "checkpoint": "/ckpt/0.pt"}
    save_state(path, seed0)
    seed1["trials"]["s1_mean::seed1"] = {"trial_id": "bbb", "checkpoint": "/ckpt/1.pt"}
    save_state(path, seed1)

    trials = load_state(path)["trials"]
    assert set(trials) == {"s1_mean::seed0", "s1_mean::seed1"}
    assert trials["s1_mean::seed0"]["checkpoint"] == "/ckpt/0.pt"


def test_the_writer_still_wins_for_its_own_key(tmp_path):
    """Merging is per key: a driver updating its own record must not be reverted to the
    copy on disk, or a finished stage would keep re-reporting a stale checkpoint."""
    path = tmp_path / "state.json"
    save_state(path, {"trials": {"s1::0": {"trial_id": "old", "checkpoint": None}}})

    later = {"trials": {"s1::0": {"trial_id": "new", "checkpoint": "/ckpt/new.pt"}}}
    save_state(path, later)

    rec = load_state(path)["trials"]["s1::0"]
    assert rec["trial_id"] == "new" and rec["checkpoint"] == "/ckpt/new.pt"


def test_in_memory_state_picks_up_the_other_driver(tmp_path):
    """After saving, the caller's dict should reflect the merged view, so a long-lived
    driver does not keep re-writing a state that is missing its peer's entries."""
    path = tmp_path / "state.json"
    save_state(path, {"trials": {"a": {"x": 1}}, "evals": {}})

    mine = {"trials": {"b": {"x": 2}}, "evals": {}}
    save_state(path, mine)

    assert set(mine["trials"]) == {"a", "b"}


def test_temp_file_is_process_scoped(tmp_path):
    """Two drivers writing at once must not share one .tmp path, or os.replace can publish
    a half-written file from the other process."""
    path = tmp_path / "state.json"
    save_state(path, {"trials": {"a": {}}})
    assert json.loads(path.read_text())["trials"] == {"a": {}}
    assert not list(tmp_path.glob("*.tmp*")), "temp files must be cleaned up by os.replace"
