"""U2/U3 — the matched-synthesizer UQ ablation (cmd_plan_v2.md §4).

Three arms hold the chirp-modal synthesizer fixed and vary only how the predictive
law is obtained:

  A1  diffusion + sampled UQ      25-draw DDIM ensemble          -> data_space_sampled
  A2  diffusion + analytic UQ     Thm-C N(m,S) around DDIM x0    -> data_space_analytic (ddim)
  A3  one-shot Gaussian NLL       Thm-C N(m,S), no diffusion     -> data_space_analytic (oneshot)

A1 and A2 are the SAME checkpoint, two inference modes. A3 needs its own training
run (`TRAIN_T_SAMPLER="max_only"`), which is why this driver trains three stages:

  S1  x0 chirp-MSE mean            the warm-start source for both UQ arms
  S2  + CHIRP_UQ_HEAD + NLL        -> A1 and A2
  S3  S2 + TRAIN_T_SAMPLER=max_only -> A3

Everything goes through ``finetuning/run_trial.py``, which applies config overrides
*inside* a wrapped ``apply_dataset_preset`` (the double-preset footgun of
DEVELOPER_GUIDE §3) and keeps the VAE/summarizer paths shared across arms — the
parity requirement. **No edit to ``configs/config.py`` is needed or made.**

Two traps this driver exists to prevent:

* Training Gaussian-NLL from scratch diverges (CMD_RUNBOOK §4: latent RMSE 10.8,
  predicted std 0.16, coverage ~0). S2/S3 always warm-start from that seed's S1
  checkpoint via ``DIFF_INIT_CKPT``.
* ``data_space_sampled`` is meaningless on the S3 checkpoint — reverse DDIM would
  walk timesteps the model never trained on. The A3 eval always passes
  ``--skip-sampled``.

Usage (set CUDA_VISIBLE_DEVICES, or pass --gpu):

    python finetuning/run_u3_uq.py plan   --seeds 0 1 2 3 4
    python finetuning/run_u3_uq.py train  --seeds 0 --smoke --gpu 1
    python finetuning/run_u3_uq.py gates  --gpu 1
    python finetuning/run_u3_uq.py train  --seeds 0 1 2 3 4 --gpu 1
    python finetuning/run_u3_uq.py eval   --seeds 0 1 2 3 4 --split val --gpu 1
    python finetuning/run_u3_uq.py eval   --seeds 0 1 2 3 4 --split test --gpu 1   # once, after PREREG
    python finetuning/run_u3_uq.py report --split test
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FINETUNING_DIR,
    combo_dir,
    load_state,
    read_json,
    run_logged,
    save_state,
    state_path,
    trial_id,
)

RUN_TAG = "uq_u3"
# Smoke runs get their own tag: --smoke is not part of the overrides hash, so without
# this a 3-epoch smoke checkpoint would land under the same trial id and be silently
# reused as if it were a real run.
SMOKE_RUN_TAG = "uq_u3_smoke"
ARM = "d"  # chirp - head (CMD); chirp_uq_head requires the certified no-head path

# Tuned incumbent from finetuning/results/fixed_bugs/RESULTS.md, identical across all
# three stages so the training budget is matched. NOTE: these were selected under
# v-prediction; U3 forces x0 (chirp_uq_head requires it), so absolute CRPS will sit
# above the v-pred headline table. The three-way comparison is internally matched.
BASE_CONFIG = {
    "BASE_LR": 3e-4,
    "LR_SCHEDULE": "warmup_cosine",
    "WARMUP_FRAC": 0.05,
    "MINSNR_GAMMA": 4.5,
    "CHIRP_NUM_BASIS": 4,
    "CHIRP_RHO_MIN": 1e-3,
    "CHIRP_COEFF_L2": 0.01,
    "GUIDANCE_POWER": 2.0,  # the tuned ramp shape; config default is 0.3
}

# CHIRP_UQ_INIT_VAR is NOT cosmetic. At the library default (1e-2) the initial
# predicted variance on physionet h=12 is 0.0011 against a squared error of 2.675, so
# the NLL mean gradient (pred-target)/sigma^2 is inflated ~2415x and destroys the
# warm-started mean: measured latent val MSE 2.8 (MSE arm) -> 93+ (NLL arm), with data
# space CRPS almost unchanged, i.e. CRPS does not reveal the damage. 25.0 puts the
# initial variance at the residual scale. It is scale-dependent -- re-measure with
# scratch diag_uqinit.py when moving to another dataset/horizon.
UQ_CONFIG = {
    "CHIRP_UQ_HEAD": True,
    "DIFF_LOSS_MODE": "gaussian_nll",
    "CHIRP_UQ_INIT_VAR": 25.0,
}

STAGE_ORDER = ("s1_mean", "s2_diffusion_uq", "s3_oneshot_uq")
STAGE_LABELS = {
    "s1_mean": "S1  x0 chirp-MSE mean (warm-start source)",
    "s2_diffusion_uq": "S2  diffusion + UQ head + NLL  -> arms A1, A2",
    "s3_oneshot_uq": "S3  one-shot NLL (max_only)    -> arm A3",
}
# Which stage's checkpoint each reported arm is evaluated on, and how.
ARM_EVAL = {
    "A1_diffusion_sampled": ("s2_diffusion_uq", "ddim"),  # read from data_space_sampled
    "A2_diffusion_analytic": ("s2_diffusion_uq", "ddim"),  # read from data_space_analytic
    "A3_oneshot_analytic": ("s3_oneshot_uq", "oneshot"),
}


def stage_overrides(stage: str, init_ckpt: str | None) -> dict:
    """The {"cli":…, "config":…} payload for one stage. x0 everywhere: chirp_uq_head
    requires predict_type='x0' (models/llapdiff.py:60-64), and S1 must match so the
    warm start transfers a mean trained for the same target."""
    config = dict(BASE_CONFIG)
    if stage in ("s2_diffusion_uq", "s3_oneshot_uq"):
        config.update(UQ_CONFIG)
        if not init_ckpt:
            raise ValueError(f"{stage} requires DIFF_INIT_CKPT (train s1_mean first).")
        config["DIFF_INIT_CKPT"] = str(init_ckpt)
    if stage == "s3_oneshot_uq":
        config["TRAIN_T_SAMPLER"] = "max_only"
    return {"cli": {"predict_type": "x0"}, "config": config}


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset = args.dataset_key
        self.pred = int(args.pred)
        self.run_tag = SMOKE_RUN_TAG if args.smoke else RUN_TAG
        self.dir = combo_dir(self.run_tag, self.dataset, self.pred, ARM)
        self.state_file = state_path(self.run_tag, self.dataset, self.pred, ARM)
        self.state = load_state(self.state_file) or {
            "run_tag": self.run_tag,
            "dataset": self.dataset,
            "pred": self.pred,
            "arm": ARM,
            "experiment": "U3 matched-synthesizer UQ ablation",
            "trials": {},
            "evals": {},
        }
        self.state.setdefault("trials", {})
        self.state.setdefault("evals", {})

    # ---------------- plumbing
    def log(self, msg: str) -> None:
        print(f"[u3] {msg}", flush=True)

    def save(self) -> None:
        save_state(self.state_file, self.state)

    def trial_key(self, stage: str, seed: int) -> str:
        return f"{stage}::seed{int(seed)}"

    def resolve(self, stage: str, seed: int) -> tuple[dict | None, str | None, str | None]:
        """(overrides, trial_id, checkpoint) for the CURRENT config.

        The checkpoint is returned only when the stored record was produced by these
        exact overrides. Keying purely on (stage, seed) would silently serve a
        checkpoint trained under a different config after any knob change — which is
        how a stale CHIRP_UQ_INIT_VAR arm nearly got reported as the fixed one.
        """
        init_ckpt = None
        if stage != "s1_mean":
            _, _, init_ckpt = self.resolve("s1_mean", seed)
            if not init_ckpt:
                return None, None, None
        overrides = stage_overrides(stage, init_ckpt)
        tid = trial_id(self.dataset, self.pred, ARM, seed, overrides)
        rec = self.state["trials"].get(self.trial_key(stage, seed)) or {}
        ckpt = rec.get("checkpoint")
        fresh = rec.get("trial_id") == tid and ckpt and Path(ckpt).exists()
        return overrides, tid, (ckpt if fresh else None)

    def checkpoint_for(self, stage: str, seed: int) -> str | None:
        return self.resolve(stage, seed)[2]

    # ---------------- training
    def train_stage(self, stage: str, seed: int) -> str | None:
        """Train one (stage, seed) if not already done. Returns the checkpoint path."""
        overrides, tid, existing = self.resolve(stage, seed)
        if existing:
            self.log(f"{stage} seed={seed}: reusing {existing}")
            return existing

        if overrides is None:  # the S1 warm-start source is missing
            if not self.args.dry_run:
                self.log(f"{stage} seed={seed}: BLOCKED — s1_mean seed={seed} has no checkpoint")
                return None
            # Planning only: show the stage with a placeholder warm-start path.
            overrides = stage_overrides(stage, f"<s1_mean seed={seed} checkpoint>")

        if self.args.dry_run:
            shown = {k: v for k, v in overrides["config"].items() if k not in BASE_CONFIG}
            self.log(f"PLAN {stage} seed={seed}")
            self.log(f"      + {json.dumps(shown, sort_keys=True) if shown else '(base config only)'}")
            return None

        init_ckpt = overrides["config"].get("DIFF_INIT_CKPT")
        trial_dir = self.dir / "trials" / tid

        trial_dir.mkdir(parents=True, exist_ok=True)
        summary_path = trial_dir / "summary.json"
        cmd = [
            sys.executable, str(FINETUNING_DIR / "run_trial.py"),
            "--dataset-key", self.dataset,
            "--pred", str(self.pred),
            "--arm", ARM,
            "--seed", str(seed),
            "--trial-id", tid,
            "--run-tag", self.run_tag,
            "--overrides-json", json.dumps(overrides),
            "--summary-json", str(summary_path),
        ]
        if self.args.smoke:
            cmd.append("--smoke")

        self.log(f"train {stage} seed={seed} trial={tid}"
                 f"{' [SMOKE]' if self.args.smoke else ''}")
        started = time.time()
        rc = run_logged(cmd, trial_dir / "train.log")
        wall = round(time.time() - started, 1)

        summary = read_json(summary_path) or {}
        entry = (summary.get("results") or {}).get(str(self.pred)) or {}
        checkpoint = entry.get("loaded_checkpoint")
        rec = {
            "stage": stage,
            "seed": int(seed),
            "trial_id": tid,
            "overrides": overrides,
            "train_wall_s": wall,
            "train_log": str(trial_dir / "train.log"),
            "summary_json": str(summary_path),
            "smoke": bool(self.args.smoke),
        }
        if rc != 0 or not checkpoint:
            rec["status"] = "failed_train"
            self.state["trials"][self.trial_key(stage, seed)] = rec
            self.save()
            self.log(f"FAILED {stage} seed={seed} (rc={rc}); log: {trial_dir / 'train.log'}")
            return None

        stats = entry.get("llapdiff") or {}
        rec.update(
            status="trained",
            checkpoint=checkpoint,
            best_primary_metric=stats.get("best_primary_metric"),
            best_primary_metric_name=stats.get("best_primary_metric_name"),
        )
        if init_ckpt:
            rec["warm_start_from"] = init_ckpt
            rec["warm_start_confirmed"] = _warm_start_took(trial_dir / "train.log")
        self.state["trials"][self.trial_key(stage, seed)] = rec
        self.save()
        self.log(f"done  {stage} seed={seed} in {wall}s -> {checkpoint}")
        return checkpoint

    def train(self) -> None:
        for seed in self.args.seeds:
            for stage in STAGE_ORDER:
                if self.train_stage(stage, seed) is None and not self.args.dry_run:
                    self.log(f"stopping seed={seed}: {stage} did not produce a checkpoint")
                    break

    # ---------------- evaluation
    def eval_one(self, stage: str, seed: int, *, split: str, latent_only: bool = False,
                 tag: str | None = None) -> dict | None:
        ckpt = self.checkpoint_for(stage, seed)
        if not ckpt:
            self.log(f"eval {stage} seed={seed}: no checkpoint, skipping")
            return None

        mean_source = "oneshot" if stage == "s3_oneshot_uq" else "ddim"
        label = tag or f"{stage}_{split}"
        out_dir = self.dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed{seed}_{label}.json"

        cmd = [
            sys.executable, "-m", "llapdiffusion.tools.run_analytic_uq_eval",
            "--dataset-key", self.dataset,
            "--pred", str(self.pred),
            "--checkpoint", ckpt,
            "--mean-source", mean_source,
            "--weights", "ema",          # parity: every other phase of this project uses EMA
            "--split", split,
            "--out-json", str(out_path),
        ]
        if latent_only:
            cmd.append("--latent-only")
        else:
            # The sampled-diffusion baseline is invalid on a max_only checkpoint.
            if stage == "s3_oneshot_uq":
                cmd.append("--skip-sampled")
            if self.args.num_samples is not None:
                cmd += ["--num-samples", str(self.args.num_samples)]

        self.log(f"eval  {stage} seed={seed} split={split} mean-source={mean_source}"
                 f"{' [latent-only]' if latent_only else ''}")
        rc = run_logged(cmd, out_dir / f"seed{seed}_{label}.log")
        if rc != 0:
            self.log(f"FAILED eval {stage} seed={seed}; log: {out_dir / f'seed{seed}_{label}.log'}")
            return None
        report = read_json(out_path)
        if report is not None:
            self.state["evals"][f"{stage}::seed{seed}::{label}"] = str(out_path)
            self.save()
        return report

    def evaluate(self) -> None:
        for seed in self.args.seeds:
            # One call covers A1 (data_space_sampled) and A2 (data_space_analytic).
            self.eval_one("s2_diffusion_uq", seed, split=self.args.split)
            # A3: separate checkpoint, sampled baseline skipped.
            self.eval_one("s3_oneshot_uq", seed, split=self.args.split)

    # ---------------- gates
    def gates(self) -> None:
        """G-b (NLL health) and G-c (one-shot conditioning), seed 0, val, latent-only."""
        seed = self.args.seeds[0]
        verdicts = {}

        s2 = self.eval_one("s2_diffusion_uq", seed, split="val", latent_only=True, tag="gate_b")
        if s2:
            verdicts["G-b_nll_health"] = _gate_b(s2)
        s3 = self.eval_one("s3_oneshot_uq", seed, split="val", latent_only=True, tag="gate_c")
        if s3:
            verdicts["G-c_oneshot_conditioning"] = _gate_c(s3)

        self.state["gates"] = verdicts
        self.save()
        print()
        print("=" * 78)
        for name, v in verdicts.items():
            print(f"{name}: {v['verdict']}")
            for line in v["detail"]:
                print(f"    {line}")
        print("=" * 78)
        if not verdicts:
            self.log("no gate ran — train seed 0 first")

    # ---------------- report
    def report(self) -> None:
        split = self.args.split
        rows: dict[str, list[dict]] = {arm: [] for arm in ARM_EVAL}
        latent: dict[str, list[dict]] = {"A2_diffusion_analytic": [], "A3_oneshot_analytic": []}

        for seed in self.args.seeds:
            s2 = read_json(self.dir / "eval" / f"seed{seed}_s2_diffusion_uq_{split}.json")
            s3 = read_json(self.dir / "eval" / f"seed{seed}_s3_oneshot_uq_{split}.json")
            if s2:
                if s2.get("data_space_sampled"):
                    rows["A1_diffusion_sampled"].append(s2["data_space_sampled"])
                if s2.get("data_space_analytic"):
                    rows["A2_diffusion_analytic"].append(s2["data_space_analytic"])
                latent["A2_diffusion_analytic"].append(s2)
            if s3:
                if s3.get("data_space_analytic"):
                    rows["A3_oneshot_analytic"].append(s3["data_space_analytic"])
                latent["A3_oneshot_analytic"].append(s3)

        payload = {
            "experiment": "U3 matched-synthesizer UQ ablation",
            "dataset": self.dataset,
            "pred": self.pred,
            "arm": ARM,
            "split": split,
            "weights": "ema",
            "seeds": list(self.args.seeds),
            "gates": self.state.get("gates", {}),
            "arms": {},
        }
        for arm, entries in rows.items():
            payload["arms"][arm] = {
                "n_seeds": len(entries),
                **{m: _agg([e.get(m) for e in entries]) for m in ("crps", "mae", "mse", "wall_seconds")},
                "latent": {
                    m: _agg([e.get(m) for e in latent.get(arm, [])])
                    for m in ("latent_gaussian_nll", "pit_calibration_error",
                              "latent_rmse", "mean_predicted_std")
                } if arm in latent else None,
            }

        out_json = self.dir / f"U3_{split}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        (self.dir / "RESULTS.md").write_text(_render_markdown(payload))
        self.log(f"wrote {out_json}")
        self.log(f"wrote {self.dir / 'RESULTS.md'}")
        print()
        print(_render_markdown(payload))


# ---------------- helpers

def _warm_start_took(log_path: Path) -> bool | None:
    """The trainer prints this only when DIFF_INIT_CKPT loaded but lacked the UQ head."""
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    return "DIFF_INIT_CKPT lacks the UQ head" in text or "loaded diffusion weights from" in text


def _agg(values: list) -> dict | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return {
        "mean": statistics.fmean(nums),
        "std": statistics.stdev(nums) if len(nums) > 1 else 0.0,
        "n": len(nums),
    }


def _gate_b(report: dict) -> dict:
    """NLL health on S2. The documented from-scratch failure is RMSE ~10.8,
    predicted std ~0.16, PIT error ~0.22, coverage ~0 (CMD_RUNBOOK §4)."""
    rmse = float(report.get("latent_rmse", float("nan")))
    base = float(report.get("baseline_rmse_predict_mean", float("nan")))
    std = float(report.get("mean_predicted_std", float("nan")))
    tgt = float(report.get("latent_target_std", float("nan")))
    pit = float(report.get("pit_calibration_error", float("nan")))
    cov90 = (report.get("reliability") or {}).get("0.9")
    cov90 = float(cov90) if cov90 is not None else float("nan")
    # Calibration is the gate's job; the mean is only required not to be catastrophic
    # (the collapse signature is RMSE many multiples of the trivial baseline).
    ok = rmse < 3.0 * base and 0.2 * tgt < std < 5.0 * tgt and pit < 0.10 and cov90 > 0.75
    return {
        "verdict": (
            "PASS" if ok else
            "FAIL — the NLL arm is overconfident/collapsed. Check CHIRP_UQ_INIT_VAR "
            "BEFORE suspecting the warm start: a variance init below the residual "
            "scale inflates the mean gradient (measured 2415x at the 1e-2 default) and "
            "destroys even a correctly warm-started mean. Confirm with the trainer's "
            "own val_diag_mse_raw, which CRPS does not reveal."
        ),
        "detail": [
            f"latent_rmse           = {rmse:.4f}   (predict-mean baseline {base:.4f})",
            f"mean_predicted_std    = {std:.4f}   (target std {tgt:.4f}; collapse <0.2*target)",
            f"pit_calibration_error = {pit:.4f}   (collapse signature ~0.22)",
            f"coverage @ nominal 0.9= {cov90:.4f}   (collapse signature ~0.0)",
        ],
        "metrics": {"latent_rmse": rmse, "baseline_rmse_predict_mean": base,
                    "mean_predicted_std": std, "latent_target_std": tgt,
                    "pit_calibration_error": pit, "coverage_0.9": cov90},
    }


def _gate_c(report: dict) -> dict:
    """One-shot conditioning on S3. CMD_BUG_REPORT ('the denoiser barely uses its
    conditioning', OPEN) measured std 0.09 vs target 0.62 and corr ~0 at steps=1 —
    but on a *uniformly* trained model. S3 trains only at t=T-1, so this is the fair
    test. RMSE ~1.0 on standardized latents == no better than predicting the mean."""
    rmse = float(report.get("latent_rmse", float("nan")))
    base = float(report.get("baseline_rmse_predict_mean", float("nan")))
    std = float(report.get("mean_predicted_std", float("nan")))
    corr = report.get("latent_corr")
    corr = float(corr) if corr is not None else float("nan")
    informative = rmse < base
    return {
        "verdict": (
            f"PASS — the one-shot mean beats predict-the-mean ({rmse:.3f} < {base:.3f})"
            if informative else
            f"CONFOUNDED — the one-shot mean ({rmse:.3f}) does NOT beat predict-the-mean "
            f"({base:.3f}); a 'diffusion wins' reading of U3 is not clean "
            "(see CMD_BUG_REPORT, the open conditioning issue)"
        ),
        "detail": [
            f"latent_rmse        = {rmse:.4f}   (predict-mean baseline {base:.4f})",
            f"latent_corr        = {corr:+.4f}   (~0 means the mean carries no signal)",
            f"mean_predicted_std = {std:.4f}",
        ],
        "metrics": {"latent_rmse": rmse, "baseline_rmse_predict_mean": base,
                    "latent_corr": corr, "mean_predicted_std": std},
    }


def _fmt(agg: dict | None, places: int = 4) -> str:
    if not agg:
        return "—"
    return f"{agg['mean']:.{places}f} ± {agg['std']:.{places}f}"


def _render_markdown(p: dict) -> str:
    lines = [
        "# U3 — matched-synthesizer UQ ablation",
        "",
        f"_{p['dataset']} h={p['pred']}, arm {p['arm']} (chirp − head), "
        f"**{p['split']}** split, **{p['weights']}** weights, seeds {p['seeds']}._",
        "",
        "All three arms share the same chirp-modal synthesizer and differ only in how the",
        "predictive law is obtained. A1/A2 are the same checkpoint (S2); A3 is a separately",
        "trained one-shot model (S3, `TRAIN_T_SAMPLER=\"max_only\"`).",
        "",
        "> ### ⚠️ Read this before quoting any number below",
        ">",
        "> **This is a 5-window estimate.** On physionet h=12 the diffusion stage receives",
        "> **13 train / 5 val / 5 test windows** (bug **B19**): the set-VAE pools all 445",
        "> entities into one latent per window, so the denoiser sees 2 496 training scalars",
        "> against 11.5 M parameters and memorizes (train loss 8.23 → 0.075 while the val",
        "> latent MSE never beats predict-zero). Latent correlation with the target is ≈ 0 in",
        "> **every** arm, including a plain-MSE control — so the arm-to-arm differences below",
        "> are **not** evidence about diffusion. Treat this table as an end-to-end validation",
        "> of the U3 pipeline. The statistical claim belongs on noaa_uk h=168",
        "> (16 286 / 2 183 / 4 702 windows).",
        "",
        "| Arm | CRPS | MAE | MSE | wall (s) |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "A1_diffusion_sampled": "A1 diffusion + sampled UQ",
        "A2_diffusion_analytic": "A2 diffusion + analytic UQ (Thm C)",
        "A3_oneshot_analytic": "A3 one-shot NLL, no diffusion",
    }
    for arm, label in labels.items():
        a = p["arms"].get(arm) or {}
        lines.append(
            f"| {label} | {_fmt(a.get('crps'))} | {_fmt(a.get('mae'))} | "
            f"{_fmt(a.get('mse'))} | {_fmt(a.get('wall_seconds'), 1)} |"
        )
    lines += [
        "",
        "## Latent-space calibration (where the Thm-C law is exactly Gaussian)",
        "",
        "| Arm | Gaussian NLL | PIT cal. error | latent RMSE | mean predicted std |",
        "|---|---|---|---|---|",
    ]
    for arm in ("A2_diffusion_analytic", "A3_oneshot_analytic"):
        lat = ((p["arms"].get(arm) or {}).get("latent")) or {}
        lines.append(
            f"| {labels[arm]} | {_fmt(lat.get('latent_gaussian_nll'))} | "
            f"{_fmt(lat.get('pit_calibration_error'))} | {_fmt(lat.get('latent_rmse'))} | "
            f"{_fmt(lat.get('mean_predicted_std'))} |"
        )
    if p.get("gates"):
        lines += ["", "## Pre-flight gates", ""]
        for name, v in p["gates"].items():
            lines.append(f"- **{name}**: {v.get('verdict')}")
            for d in v.get("detail", []):
                lines.append(f"  - `{d}`")
    lines += [
        "",
        "## Scope (all documented; none optional)",
        "",
        "- UQ is **aleatoric only**, conditional on the predicted parameters — no epistemic",
        "  component (`chirp_modal_method_update.md` §Honest limitations).",
        "- `q_k` is **constant in time**; the method doc permits `q_k(t̃)`.",
        "- `v_k` is an exponential-integrator **quadrature**, not a closed form: the P-exact",
        "  integrand `e^{2ρ̄(s)}q(s)` has no elementary antiderivative.",
        "- Propagation to data space is **decoder Monte Carlo**, not the delta method the plan",
        "  named — deliberately, so the analytic arm shares the sampled arm's CRPS estimator.",
        "- These are **x0** arms (forced by `chirp_uq_head`). `USAGE.md` §3.6 records x0 as",
        "  ≈0.03 CRPS worse than v on PhysioNet h=12, so absolute CRPS sits above the",
        "  v-prediction headline table. The three-way comparison is internally matched; it is",
        "  **not** comparable to `finetuning/results/fixed_bugs/RESULTS.md`.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U3 matched-synthesizer UQ ablation driver.")
    parser.add_argument("command", choices=("plan", "train", "gates", "eval", "report"))
    parser.add_argument("--dataset-key", default="physionet")
    parser.add_argument("--pred", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Ensemble size for BOTH data-space arms (default: config's 25).")
    parser.add_argument("--smoke", action="store_true", help="3-epoch plumbing check.")
    parser.add_argument("--gpu", default=None, help="Sets CUDA_VISIBLE_DEVICES for children.")
    args = parser.parse_args()
    args.dry_run = args.command == "plan"
    return args


def main() -> None:
    args = _parse_args()
    if args.gpu is not None:
        # Children (run_trial.py, the uq-eval module) inherit this.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # run_logged redirects child stdout to a file, so Python block-buffers it and a
    # multi-hour training run shows nothing until it exits. Line-buffer the children.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    campaign = Campaign(args)
    if args.command in ("plan", "train"):
        if args.command == "plan":
            print(f"[u3] {len(args.seeds)} seeds x {len(STAGE_ORDER)} stages = "
                  f"{len(args.seeds) * len(STAGE_ORDER)} training runs")
            for stage in STAGE_ORDER:
                print(f"[u3]   {STAGE_LABELS[stage]}")
        campaign.train()
    elif args.command == "gates":
        campaign.gates()
    elif args.command == "eval":
        campaign.evaluate()
    elif args.command == "report":
        campaign.report()


if __name__ == "__main__":
    main()
