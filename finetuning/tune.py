"""CMD hyperparameter tuner — orchestrator. See finetuning/README.md.

Per (dataset, horizon, arm) it runs up to three phases, all resumable (state
lives in finetuning/results/<run-tag>/<ds>_h<H>/<arm>/state.json; finished
trials/cells are never re-run):

  train    - coordinate-descent sweep over training knobs (search_spaces.py).
             Every trial trains stage-3 only (shared frozen VAE/summarizer),
             then is scored by CRPS on the SELECTION split with the SELECTION
             weights (--select-split / --select-weights; default test + ema).
             Incumbent = lowest selection CRPS.
  sampling - guidance x DDIM-steps grid on the incumbent checkpoint, scored the
             same way. No retraining.
  final    - NOT part of --phase all. Retrains the winning configuration across
             --final-seeds and runs the full checkpoint-eval protocol (forecast
             + regular-keep + random-mask imputation) on test per seed.

After every invocation the reports are refreshed:
  finetuning/results/<run-tag>/RESULTS.md          (human summary)
  finetuning/results/<run-tag>/.../BEST.json       (winning hyperparameters)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ARM_FLAGS,
    ARM_LABELS,
    FINETUNING_DIR,
    REPO_ROOT,
    RESULTS_ROOT,
    canonical_overrides,
    cell_key,
    combo_dir,
    load_state,
    read_json,
    run_logged,
    save_state,
    state_path,
    trial_id,
    values_equal,
)
from search_spaces import sampling_cells, stages_for_arm  # noqa: E402

DEFAULT_FINAL_SEEDS = (0, 1, 2, 3, 4)

# Fields that identify a sampling cell for resume/dedup. guidance_power MUST be
# here: ramp cells that differ only in power share guidance/steps/weights, so
# omitting it would collide them (one overwriting the other on resume).
SAMPLING_KEY_FIELDS = ("guidance", "steps", "weights", "guidance_power")


def sampling_cell_key(cell: dict) -> str:
    return cell_key({field: cell.get(field) for field in SAMPLING_KEY_FIELDS})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning orchestrator for the CMD arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--preds", nargs="+", type=int, required=True,
                        help="Horizons to tune (one tuning campaign per horizon).")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARM_FLAGS), default=["d"],
                        help="Factorial arms: a=lti+head b=lti-head c=chirp+head d=CMD.")
    parser.add_argument("--phase", choices=("train", "sampling", "final", "all", "report"),
                        default="all",
                        help="'all' = train + sampling. 'final' (multi-seed + imputation) "
                             "must be requested explicitly.")
    parser.add_argument("--select-split", choices=("test", "val"), default="test",
                        help="Split whose CRPS selects hyperparameters. NOTE: 'test' means the "
                             "reported test CRPS is selection-biased (cmd_plan_v2.md §1 "
                             "pre-registers 'val'); use 'val' for a clean held-out number.")
    parser.add_argument("--select-weights", choices=("ema", "raw"), default="ema",
                        help="Weight source scored during selection: the checkpoint's EMA shadow "
                             "(default) or its raw weights.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for tuning trials.")
    parser.add_argument("--final-seeds", nargs="+", type=int, default=None,
                        help=f"Seeds for --phase final (default {list(DEFAULT_FINAL_SEEDS)}).")
    parser.add_argument("--run-tag", default="v1",
                        help="Namespace for this campaign (results + ldt/tuning routing).")
    parser.add_argument("--stages", nargs="*", default=None,
                        help="Subset of training stages to sweep (names from search_spaces.py); "
                             "default = all tiers applicable to the arm.")
    parser.add_argument("--include-tier3", action="store_true",
                        help="Also sweep Tier-3 capacity knobs (width/K).")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override eval ensemble size (keep 25 for reported numbers).")
    parser.add_argument("--imputation-random-mask-ratio", type=float, default=0.30)
    parser.add_argument("--allow-upstream-training", action="store_true",
                        help="Let the first trial train a missing VAE/summarizer instead of "
                             "requiring llapdiff-artifact-prep beforehand.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned trials/cells without running anything.")
    parser.add_argument("--smoke", action="store_true",
                        help="3-epoch trials + tiny ensembles to validate the plumbing. "
                             "Uses run-tag 'smoke' unless --run-tag is set explicitly.")
    return parser.parse_args()


# --------------------------------------------------------------------------- defaults
_DEFAULTS_CACHE: dict = {}


def resolve_defaults(dataset: str, pred: int):
    """Preset-applied config namespace = the harness default for this setting."""
    key = (dataset, int(pred))
    if key not in _DEFAULTS_CACHE:
        from llapdiffusion.configs.config_utils import clone_config
        from llapdiffusion.configs.dataset_defaults import apply_dataset_preset

        cfg = clone_config()
        apply_dataset_preset(cfg, dataset, pred=int(pred))
        _DEFAULTS_CACHE[key] = cfg
    return _DEFAULTS_CACHE[key]


def normalize_overrides(overrides: dict, defaults) -> dict:
    """Drop override entries equal to the harness default so equivalent
    configurations always hash to the same trial."""
    out: dict = {"cli": {}, "config": {}}
    for knob, value in (overrides.get("cli") or {}).items():
        if knob == "predict_type" and str(value) == "v":
            continue
        out["cli"][knob] = value
    for knob, value in (overrides.get("config") or {}).items():
        if values_equal(value, getattr(defaults, knob, None)):
            continue
        out["config"][knob] = value
    return canonical_overrides(out)


def describe_overrides(overrides: dict) -> str:
    parts = [f"{k}={v}" for k, v in (overrides.get("cli") or {}).items()]
    parts += [f"{k}={v}" for k, v in (overrides.get("config") or {}).items()]
    return ", ".join(parts) if parts else "(defaults)"


def describe_cell(cell: dict) -> str:
    guidance = cell.get("guidance")
    if guidance is None:
        g = "default-ramp"
    elif isinstance(guidance, (list, tuple)):
        g = f"ramp{tuple(guidance)}@pow{cell.get('guidance_power')}"
    else:
        g = f"const{guidance}"
    return f"guidance={g}, steps={cell.get('steps')}, weights={cell.get('weights')}"


# --------------------------------------------------------------------------- one combo
class Combo:
    def __init__(self, args: argparse.Namespace, dataset: str, pred: int, arm: str):
        self.args = args
        self.dataset = dataset
        self.pred = int(pred)
        self.arm = arm
        self.dir = combo_dir(args.run_tag, dataset, pred, arm)
        self.state_file = state_path(args.run_tag, dataset, pred, arm)
        self.state = load_state(self.state_file)
        self.state.setdefault("dataset", dataset)
        self.state.setdefault("pred", self.pred)
        self.state.setdefault("arm", arm)
        self.state.setdefault("arm_label", ARM_LABELS[arm])
        self.state.setdefault("run_tag", args.run_tag)
        self.state.setdefault("tune_seed", int(args.seed))
        self.state.setdefault("smoke", bool(args.smoke))
        self.state["selection"] = {"split": args.select_split, "weights": args.select_weights}
        self.state.setdefault("trials", {})
        self._planned: set[str] = set()

    def save(self) -> None:
        save_state(self.state_file, self.state)

    def log(self, message: str) -> None:
        print(f"[{self.dataset} h{self.pred} arm-{self.arm}] {message}", flush=True)

    def _selection_matches(self, rec: dict) -> bool:
        select = rec.get("select") or {}
        return (
            select.get("split") == self.args.select_split
            and select.get("weights") == self.args.select_weights
        )

    # ---------------- trials
    def ensure_trial(self, overrides: dict, stage: str, seed: int, *, need_score: bool) -> str | None:
        """Train (if needed) and score (if needed) one configuration.
        Returns the trial id, or None on failure / dry-run."""
        tid = trial_id(self.dataset, self.pred, self.arm, seed, overrides)
        rec = self.state["trials"].get(tid)
        scored = rec and self._selection_matches(rec)
        if rec and rec.get("checkpoint") and (scored or not need_score):
            return tid
        if self.args.dry_run:
            if tid not in self._planned:
                self._planned.add(tid)
                self.log(f"PLAN trial {tid} seed={seed} stage={stage}: {describe_overrides(overrides)}")
            return None

        if rec is None:
            rec = {
                "overrides": overrides,
                "stage": stage,
                "seed": int(seed),
                "status": "pending",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.state["trials"][tid] = rec

        trial_dir = self.dir / "trials" / tid
        trial_dir.mkdir(parents=True, exist_ok=True)

        if not rec.get("checkpoint"):
            summary_path = trial_dir / "summary.json"
            train_log = trial_dir / "train.log"
            cmd = [
                sys.executable, str(FINETUNING_DIR / "run_trial.py"),
                "--dataset-key", self.dataset,
                "--pred", str(self.pred),
                "--arm", self.arm,
                "--seed", str(seed),
                "--trial-id", tid,
                "--run-tag", self.args.run_tag,
                "--overrides-json", json.dumps(overrides),
                "--summary-json", str(summary_path),
            ]
            if self.args.smoke:
                cmd.append("--smoke")
            self.log(f"train trial {tid} seed={seed} stage={stage}: {describe_overrides(overrides)}")
            started = time.time()
            rc = run_logged(cmd, train_log)
            rec["train_wall_s"] = round(time.time() - started, 1)
            rec["train_log"] = str(train_log)
            rec["summary_json"] = str(summary_path)
            summary = read_json(summary_path) or {}
            entry = (summary.get("results") or {}).get(str(self.pred)) or {}
            checkpoint = entry.get("loaded_checkpoint")
            if rc != 0 or not checkpoint:
                rec["status"] = "failed_train"
                self.save()
                self.log(f"trial {tid} FAILED in training (rc={rc}); log: {train_log}")
                return None
            rec["checkpoint"] = checkpoint
            llapdiff_stats = entry.get("llapdiff") or {}
            rec["trainer_best_primary_metric"] = llapdiff_stats.get("best_primary_metric")
            rec["trainer_best_primary_metric_name"] = llapdiff_stats.get("best_primary_metric_name")
            rec["status"] = "trained"
            self.save()

        if need_score and not self._selection_matches(rec):
            split = self.args.select_split
            weights = self.args.select_weights
            cells = [{"weights": weights}]
            out_path = trial_dir / f"select_{split}_{weights}.json"
            cmd = [
                sys.executable, str(FINETUNING_DIR / "eval_sampling.py"),
                "--dataset-key", self.dataset,
                "--pred", str(self.pred),
                "--checkpoint", rec["checkpoint"],
                "--cells-json", json.dumps(cells),
                "--split", split,
                "--out-json", str(out_path),
            ]
            num_samples = 4 if self.args.smoke else self.args.num_samples
            if num_samples is not None:
                cmd += ["--num-samples", str(num_samples)]
            self.log(f"score trial {tid} on {split} split ({weights} weights, default sampling)")
            rc = run_logged(cmd, trial_dir / "select_eval.log")
            payload = read_json(out_path) or {}
            ok_rows = [
                r for r in payload.get("rows", [])
                if r.get("status") == "ok" and r.get("crps") is not None
            ]
            if rc != 0 or not ok_rows:
                rec["status"] = "failed_score"
                self.save()
                self.log(f"trial {tid} FAILED in scoring (rc={rc}); "
                         f"log: {trial_dir / 'select_eval.log'}")
                return None
            row = ok_rows[0]
            rec["select"] = {
                "split": split,
                "weights": weights,
                "crps": float(row["crps"]),
                "mae": row.get("mae"),
                "mse": row.get("mse"),
                "file": str(out_path),
            }
            rec["status"] = "done"
            self.save()
            self.log(f"trial {tid}: {split} CRPS {rec['select']['crps']:.4f} ({weights} weights)")

        return tid

    def pick_incumbent(self) -> str | None:
        candidates = [
            (float(rec["select"]["crps"]), tid)
            for tid, rec in self.state["trials"].items()
            if self._selection_matches(rec) and int(rec.get("seed", -1)) == int(self.args.seed)
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    # ---------------- phases
    def phase_train(self) -> None:
        defaults = resolve_defaults(self.dataset, self.pred)
        stages = stages_for_arm(self.arm, include_tier3=self.args.include_tier3)
        if self.args.stages is not None:
            wanted = set(self.args.stages)
            unknown = wanted - {s[0] for s in stages}
            if unknown:
                raise SystemExit(f"Unknown/inapplicable stages for arm {self.arm}: {sorted(unknown)}")
            stages = [s for s in stages if s[0] in wanted]

        baseline = self.ensure_trial({}, "baseline", self.args.seed, need_score=True)
        if baseline is None and not self.args.dry_run:
            self.log("baseline trial failed — aborting this arm")
            return

        for stage_name, kind, knob, candidate_values in stages:
            incumbent = self.pick_incumbent()
            base_overrides = (
                self.state["trials"][incumbent]["overrides"] if incumbent else {}
            )
            for value in candidate_values:
                candidate = copy.deepcopy(base_overrides) or {}
                candidate.setdefault(kind, {})[knob] = value
                candidate = normalize_overrides(candidate, defaults)
                self.ensure_trial(candidate, stage_name, self.args.seed, need_score=True)

        incumbent = self.pick_incumbent()
        if incumbent:
            self.state["incumbent"] = incumbent
            rec = self.state["trials"][incumbent]
            self.log(
                f"incumbent {incumbent}: {rec['select']['split']} CRPS "
                f"{rec['select']['crps']:.4f} ({describe_overrides(rec['overrides'])})"
            )
            self.save()

    def phase_sampling(self) -> None:
        incumbent = self.state.get("incumbent") or self.pick_incumbent()
        if not incumbent:
            self.log("no incumbent trial yet — run --phase train first")
            return
        self.state["incumbent"] = incumbent
        checkpoint = self.state["trials"][incumbent]["checkpoint"]
        split = self.args.select_split
        weights = self.args.select_weights

        sampling = self.state.get("sampling")
        if (
            not sampling
            or sampling.get("checkpoint_trial") != incumbent
            or sampling.get("split") != split
            or sampling.get("weights") != weights
        ):
            sampling = {
                "checkpoint_trial": incumbent,
                "split": split,
                "weights": weights,
                "rows": {},
                "best": None,
            }
        self.state["sampling"] = sampling

        cells = sampling_cells(weights)
        remaining = [c for c in cells if sampling_cell_key(c) not in sampling["rows"]]
        if self.args.dry_run:
            self.log(f"PLAN sampling sweep on {incumbent}: {len(remaining)}/{len(cells)} cells "
                     f"on {split} ({weights} weights)")
            return
        if remaining:
            sweep_dir = self.dir / "sampling"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            out_path = sweep_dir / f"{incumbent}_{split}_{weights}.json"
            cmd = [
                sys.executable, str(FINETUNING_DIR / "eval_sampling.py"),
                "--dataset-key", self.dataset,
                "--pred", str(self.pred),
                "--checkpoint", checkpoint,
                "--cells-json", json.dumps(remaining),
                "--split", split,
                "--out-json", str(out_path),
            ]
            num_samples = 4 if self.args.smoke else self.args.num_samples
            if num_samples is not None:
                cmd += ["--num-samples", str(num_samples)]
            self.log(f"sampling sweep on {incumbent}: {len(remaining)} cells "
                     f"({split} split, {weights} weights)")
            rc = run_logged(cmd, sweep_dir / "sweep.log")
            payload = read_json(out_path) or {}
            for row in payload.get("rows", []):
                sampling["rows"][sampling_cell_key(row)] = row
            self.save()
            if rc != 0:
                self.log(f"sampling sweep exited rc={rc}; kept {len(sampling['rows'])} finished cells "
                         f"(rerun to resume); log: {sweep_dir / 'sweep.log'}")

        ok_rows = [
            r for r in sampling["rows"].values()
            if r.get("status") == "ok" and r.get("crps") is not None
        ]
        if ok_rows:
            best = min(ok_rows, key=lambda r: float(r["crps"]))
            sampling["best"] = best
            self.log(f"best sampling cell: {describe_cell(best)} -> {split} CRPS {best['crps']:.4f}")
            self.save()

    def phase_final(self) -> None:
        incumbent = self.state.get("incumbent")
        best_cell = (self.state.get("sampling") or {}).get("best")
        if not incumbent or not best_cell:
            self.log("need finished train + sampling phases before --phase final")
            return
        overrides = self.state["trials"][incumbent]["overrides"]
        chosen_sampling = {
            k: best_cell.get(k)
            for k in ("guidance", "steps", "weights", "guidance_power", "dynamic_thresh_p")
        }
        final = self.state.setdefault("final", {})
        final["training_overrides"] = overrides
        final["sampling"] = chosen_sampling
        per_seed = final.setdefault("per_seed", {})

        seeds = self.args.final_seeds
        if seeds is None:
            seeds = [self.args.seed] if self.args.smoke else list(DEFAULT_FINAL_SEEDS)
        for seed in seeds:
            tid = self.ensure_trial(overrides, "final_seed", seed, need_score=False)
            if tid is None:
                continue
            if per_seed.get(str(seed), {}).get("forecast"):
                continue
            checkpoint = self.state["trials"][tid]["checkpoint"]
            final_dir = self.dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            out_path = final_dir / f"seed{seed}_test.json"
            cmd = [
                sys.executable, str(FINETUNING_DIR / "final_eval.py"),
                "--dataset-key", self.dataset,
                "--pred", str(self.pred),
                "--checkpoint", checkpoint,
                "--sampling-json", json.dumps(chosen_sampling),
                "--imputation-random-mask-ratio", str(self.args.imputation_random_mask_ratio),
                "--out-json", str(out_path),
            ]
            if self.args.smoke:
                cmd.append("--smoke")
            self.log(f"FINAL test eval seed={seed} (forecast + imputation) trial {tid}")
            rc = run_logged(cmd, final_dir / f"seed{seed}_test.log")
            payload = read_json(out_path)
            if rc != 0 or not payload:
                self.log(f"final eval seed={seed} FAILED (rc={rc}); log: {final_dir}/seed{seed}_test.log")
                continue
            forecast = payload.get("forecast_test") or {}
            per_seed[str(seed)] = {
                "trial": tid,
                "checkpoint": checkpoint,
                "forecast": {k: forecast.get(k) for k in ("crps", "mae", "mse")},
                "regular_keep25_hidden_crps": (payload.get("regular_keep25") or {}).get("hidden_crps"),
                "random_mask_hidden_crps": (payload.get("random_mask") or {}).get("hidden_crps"),
                "file": str(out_path),
            }
            self.save()

        crps_values = [
            float(entry["forecast"]["crps"])
            for entry in per_seed.values()
            if entry.get("forecast", {}).get("crps") is not None
        ]
        if crps_values:
            final["aggregate"] = {
                "n_seeds": len(crps_values),
                "test_crps_mean": statistics.mean(crps_values),
                "test_crps_std": statistics.stdev(crps_values) if len(crps_values) > 1 else 0.0,
            }
            self.log(
                f"FINAL: test CRPS {final['aggregate']['test_crps_mean']:.4f} "
                f"± {final['aggregate']['test_crps_std']:.4f} over {len(crps_values)} seed(s)"
            )
        self.save()


# --------------------------------------------------------------------------- reports
def write_reports(run_tag: str) -> None:
    root = RESULTS_ROOT / run_tag
    states = sorted(root.glob("*/*/state.json"))
    if not states:
        return

    selections = {
        f"{(read_json(s) or {}).get('selection', {}).get('split')}"
        f"/{(read_json(s) or {}).get('selection', {}).get('weights')}"
        for s in states
    }
    selection_note = ", ".join(sorted(selections))
    biased = any(s.startswith("test/") for s in selections)

    lines = [
        f"# CMD hyperparameter tuning — run tag `{run_tag}`",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')} by finetuning/tune.py._",
        "",
        f"**Selection rule:** lowest CRPS on the **{selection_note}** "
        "(split/weights) at the default sampling protocol.",
    ]
    if biased:
        lines += [
            "",
            "> ⚠️ Hyperparameters were selected on the **test** split, so the test CRPS below is "
            "selection-biased and is NOT a clean held-out number. `cmd_plan_v2.md` §1 "
            "pre-registers validation-split selection; rerun with `--select-split val` "
            "(new `--run-tag`) for the paper's headline table.",
        ]
    lines += [
        "",
        "| Dataset | H | Arm | Trials | Best training config | Best sampling | Select CRPS | Final test CRPS (mean±std) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    details: list[str] = []
    for state_file in states:
        state = read_json(state_file) or {}
        trials = state.get("trials", {})
        incumbent = state.get("incumbent")
        inc = trials.get(incumbent) if incumbent else None
        best_cell = (state.get("sampling") or {}).get("best")
        final = state.get("final") or {}
        aggregate = final.get("aggregate") or {}
        selection = state.get("selection") or {}
        smoke_mark = " (SMOKE)" if state.get("smoke") else ""

        select_crps = f"{inc['select']['crps']:.4f}" if inc and inc.get("select") else "—"
        test_crps = (
            f"{aggregate['test_crps_mean']:.4f} ± {aggregate['test_crps_std']:.4f} "
            f"(n={aggregate['n_seeds']})"
            if aggregate else "—"
        )
        n_scored = sum(1 for t in trials.values() if t.get("select"))
        lines.append(
            f"| {state.get('dataset')}{smoke_mark} | {state.get('pred')} | "
            f"{state.get('arm')} | {n_scored} | "
            f"{describe_overrides(inc['overrides']) if inc else '—'} | "
            f"{describe_cell(best_cell) if best_cell else '—'} | {select_crps} | {test_crps} |"
        )

        details.append(f"\n## {state.get('dataset')} h{state.get('pred')} — arm "
                       f"{state.get('arm')} ({state.get('arm_label')}){smoke_mark}\n")
        details.append(f"- Selection: CRPS on **{selection.get('split')}** split, "
                       f"**{selection.get('weights')}** weights; tuning seed "
                       f"{state.get('tune_seed')}")
        if inc:
            details.append(f"- **Incumbent trial** `{incumbent}` — {describe_overrides(inc['overrides'])}; "
                           f"select CRPS **{inc['select']['crps']:.4f}**")
            details.append(f"- Checkpoint: `{inc.get('checkpoint')}`")
        if best_cell:
            details.append(f"- **Best sampling**: {describe_cell(best_cell)}; "
                           f"select CRPS **{float(best_cell['crps']):.4f}**")
        ranked = sorted(
            ((t["select"]["crps"], tid, t) for tid, t in trials.items() if t.get("select")),
            key=lambda x: float(x[0]),
        )
        if ranked:
            details.append("- Trials (select CRPS, ascending):")
            for crps, tid, t in ranked:
                details.append(f"  - `{tid}` [{t.get('stage')}] {describe_overrides(t['overrides'])}: "
                               f"{float(crps):.4f}")
        per_seed = final.get("per_seed") or {}
        if per_seed:
            details.append("- **Final test (forecast + imputation)**, sampling "
                           f"{describe_cell(final.get('sampling') or {})}:")

            def fmt(value):
                return f"{float(value):.4f}" if value is not None else "—"

            for seed in sorted(per_seed, key=int):
                entry = per_seed[seed]
                fc = entry.get("forecast") or {}
                details.append(
                    f"  - seed {seed}: CRPS {fmt(fc.get('crps'))}, MAE {fmt(fc.get('mae'))}, "
                    f"MSE {fmt(fc.get('mse'))}, imput. hidden CRPS (reg/rand) "
                    f"{fmt(entry.get('regular_keep25_hidden_crps'))}/{fmt(entry.get('random_mask_hidden_crps'))}"
                )
            if aggregate:
                details.append(f"  - **aggregate: {aggregate['test_crps_mean']:.4f} "
                               f"± {aggregate['test_crps_std']:.4f} (n={aggregate['n_seeds']})**")

        best_payload = {
            "dataset": state.get("dataset"),
            "pred": state.get("pred"),
            "arm": state.get("arm"),
            "arm_label": state.get("arm_label"),
            "smoke": state.get("smoke", False),
            "selection": {
                **selection,
                "crps": inc["select"]["crps"] if inc and inc.get("select") else None,
                "mae": inc["select"].get("mae") if inc and inc.get("select") else None,
                "mse": inc["select"].get("mse") if inc and inc.get("select") else None,
            },
            "training_overrides": inc["overrides"] if inc else None,
            "sampling": {
                k: best_cell.get(k)
                for k in ("guidance", "steps", "weights", "guidance_power", "dynamic_thresh_p")
            } if best_cell else None,
            "checkpoint": inc.get("checkpoint") if inc else None,
            "final": final or None,
        }
        (state_file.parent / "BEST.json").write_text(json.dumps(best_payload, indent=2, default=str))

    (root / "RESULTS.md").write_text("\n".join(lines + details) + "\n")
    print(f"[report] {root / 'RESULTS.md'}")


# --------------------------------------------------------------------------- main
def check_upstream(dataset: str, pred: int, allow_training: bool) -> None:
    defaults = resolve_defaults(dataset, pred)
    missing = [p for p in (defaults.VAE_CKPT, defaults.SUM_CKPT) if not Path(p).exists()]
    if missing and not allow_training:
        raise SystemExit(
            f"Missing shared stage-1/2 artifacts for {dataset} h{pred}:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\nRun once (all arms/trials reuse them):\n"
            f"  llapdiff-artifact-prep --datasets {dataset} "
            f"--summary-json ldt/results/prep_{dataset}.json\n"
            "or pass --allow-upstream-training to let the first trial train them."
        )


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    if args.smoke and args.run_tag == "v1":
        args.run_tag = "smoke"

    for pred in args.preds:
        if args.phase in ("train", "final", "all") and not args.dry_run:
            check_upstream(args.dataset_key, pred, args.allow_upstream_training)
        for arm in args.arms:
            combo = Combo(args, args.dataset_key, pred, arm)
            if args.phase in ("train", "all"):
                combo.phase_train()
            if args.phase in ("sampling", "all"):
                combo.phase_sampling()
            if args.phase == "final":
                combo.phase_final()
            combo.save()

    if not args.dry_run:
        write_reports(args.run_tag)


if __name__ == "__main__":
    main()
