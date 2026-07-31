"""Shared helpers for the CMD hyperparameter tuning harness.

Everything here is deliberately environment-free: no llapdiffusion imports, so
the orchestrator can plan/dry-run without touching torch. Paths are derived
from this file's location, so renaming the folder keeps everything working.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

FINETUNING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FINETUNING_DIR.parent
RESULTS_ROOT = FINETUNING_DIR / "results"
# Stage-3 artifacts for tuning trials live under ldt/tuning/ so they can never
# collide with the default (paper) runs in ldt/output/.
TUNING_ARTIFACT_ROOT = REPO_ROOT / "ldt" / "tuning"

# Arm letters follow the G3 factorial in cmd_plan_v2.md §2 / CMD_RUNBOOK.md §1.
ARM_FLAGS = {
    "a": (),                                                # lti + head  (in-harness LLapDiff)
    "b": ("--output-head", "off"),                          # lti − head  (control)
    "c": ("--modal-type", "chirp", "--output-head", "on"),  # chirp + head (redundancy probe)
    "d": ("--modal-type", "chirp"),                         # chirp − head (CMD)
}
ARM_LABELS = {
    "a": "lti + head (in-harness LLapDiff)",
    "b": "lti − head (control)",
    "c": "chirp + head (redundancy probe)",
    "d": "chirp − head (CMD)",
}
CHIRP_ARMS = ("c", "d")


def values_equal(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def canonical_overrides(overrides: dict | None) -> dict:
    """Sorted, minimal copy of {"cli": {...}, "config": {...}} (empty sections dropped)."""
    out: dict = {}
    for section in ("cli", "config"):
        values = (overrides or {}).get(section) or {}
        if values:
            out[section] = {str(k): values[k] for k in sorted(values)}
    return out


def trial_id(dataset: str, pred: int, arm: str, seed: int, overrides: dict | None) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "pred": int(pred),
            "arm": arm,
            "seed": int(seed),
            "overrides": canonical_overrides(overrides),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def cell_key(cell: dict) -> str:
    return json.dumps({k: cell[k] for k in sorted(cell)}, sort_keys=True)


def combo_dir(run_tag: str, dataset: str, pred: int, arm: str) -> Path:
    return RESULTS_ROOT / run_tag / f"{dataset}_h{int(pred)}" / arm


def state_path(run_tag: str, dataset: str, pred: int, arm: str) -> Path:
    return combo_dir(run_tag, dataset, pred, arm) / "state.json"


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=False, default=str))
    os.replace(tmp, path)


def run_logged(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess from the repo root, appending stdout+stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    return proc.returncode


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
