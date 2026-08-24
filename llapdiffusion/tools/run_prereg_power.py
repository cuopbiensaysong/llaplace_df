"""Seed statistics and the power calculation the pre-registration requires before test.

`w_docs/noaa_uk_h168/PREREG_U3_noaa_uk_h168.md` §6 fixes a floor of 5 seeds and defers the exact count to a
power calculation on **val** seeds, computed before the test split is touched. This tool does
that computation from the campaign's result JSONs, so the number in the freeze record is
derived rather than asserted.

For each pre-registered directional read it reports, per metric:

* the per-arm across-seed mean and standard deviation;
* the **paired** per-seed difference, which is what the one-sided Wilcoxon signed-rank test in
  the pre-registration actually consumes -- pairing by seed removes the shared seed effect and
  is usually far tighter than the per-arm spreads suggest;
* the required `n` for 80 % power at alpha = 0.05, one-sided, from the paired difference.

The required-`n` formula is the normal approximation `n >= (z_{1-a} + z_{1-b})^2 s_d^2 / d^2`
for a paired one-sided test. It is reported alongside a Wilcoxon-specific caveat: the
signed-rank test cannot reach p < 0.05 one-sided with fewer than **5** seeds whatever the
effect size, because 1/2^5 = 0.031 is the smallest attainable one-sided p-value. That floor,
not the variance, is what sets the minimum here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Z_ALPHA = 1.6448536269514722      # one-sided 0.05
Z_BETA = 0.8416212335729143       # 80 % power

# Smallest attainable one-sided p for the sign/Wilcoxon signed-rank test at n seeds.
WILCOXON_FLOOR = {1: 0.5, 2: 0.25, 3: 0.125, 4: 0.0625, 5: 0.03125, 6: 0.015625}


def _mean_sd(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def required_n(diffs: Sequence[float]) -> Dict[str, float]:
    """Paired one-sided power calculation from per-seed differences."""
    d, sd = _mean_sd(diffs)
    out: Dict[str, float] = {"paired_mean": d, "paired_sd": sd, "n_seeds": len(diffs)}
    if len(diffs) < 2 or not math.isfinite(sd) or d == 0:
        out["required_n_80pct"] = float("nan")
        return out
    if sd == 0.0:
        out["required_n_80pct"] = 2.0        # any n; variance is zero
        return out
    n = ((Z_ALPHA + Z_BETA) ** 2) * (sd ** 2) / (d ** 2)
    out["required_n_80pct"] = math.ceil(max(n, 2.0))
    out["effect_size_dz"] = abs(d) / sd
    return out


def _load(pattern: Path, seeds: Sequence[int]) -> Dict[int, dict]:
    out = {}
    for s in seeds:
        p = Path(str(pattern).replace("{seed}", str(s)))
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def one_pass_reads(runs: Dict[int, dict]) -> Dict[str, object]:
    """Reads T3/T4/T5: CMD-GP against the same-encoder diagonal head."""
    # (metric, arm_a, arm_b, hypothesis). `diff = b - a`, and `hypothesis` states the sign of
    # that difference which CONFIRMS the pre-registered read -- spelled out per read rather
    # than assumed uniform, because T4 deliberately registers a direction adverse to the
    # method (CMD-GP is expected to be WORSE on the marginal axis).
    metrics = {
        "T3_VS": ("VS_p0.5", "chirp_gp", "diag_gaussian", "positive",
                  "CMD-GP has the LOWER variogram score"),
        "T4_VE": ("variance_explained_vs_train_mean", "chirp_gp", "diag_gaussian", "positive",
                  "CMD-GP explains LESS variance (registered adverse)"),
        "T4_ES": ("ES_unbiased", "chirp_gp", "diag_gaussian", "negative",
                  "CMD-GP has the HIGHER energy score (registered adverse)"),
        "T5_PIT": ("PIT_ECE", "chirp_gp", "diag_gaussian", "positive",
                   "CMD-GP has the LOWER PIT-ECE"),
    }
    rows: Dict[str, object] = {}
    seeds = sorted(runs)
    for read, (metric, a, b, want, prose) in metrics.items():
        va = [runs[s]["heads"][a][metric] for s in seeds]
        vb = [runs[s]["heads"][b][metric] for s in seeds]
        diffs = [y - x for x, y in zip(va, vb)]          # b - a
        holds = all(d > 0 for d in diffs) if want == "positive" else all(d < 0 for d in diffs)
        rows[read] = {
            "metric": metric, "arm_a": a, "arm_b": b,
            "hypothesis": prose, "confirming_sign_of_b_minus_a": want,
            "per_seed_a": dict(zip(seeds, va)), "per_seed_b": dict(zip(seeds, vb)),
            "per_seed_diff_b_minus_a": dict(zip(seeds, diffs)),
            "arm_a_mean_sd": _mean_sd(va), "arm_b_mean_sd": _mean_sd(vb),
            **required_n(diffs),
            "hypothesis_holds_on_all_seeds": holds,
            "sign_consistent": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        }
    return rows


def frontier_reads(runs: Dict[int, dict]) -> Dict[str, object]:
    """Read T2: the CMD-D(25) mixture against the conventional 25-sample ensemble."""
    seeds = sorted(runs)
    rows: Dict[str, object] = {}
    for read, metric in (("T2_VS", "VS_p0.5"), ("T2_ES", "ES_unbiased"),
                         ("T2_CRPS", "CRPS_unbiased")):
        va = [runs[s]["arm_scores"]["mixture_M25"][metric] for s in seeds]
        vb = [runs[s]["arm_scores"]["sampled_M25"][metric] for s in seeds]
        diffs = [y - x for x, y in zip(va, vb)]      # b - a; >0 means the mixture is better
        rows[read] = {
            "metric": metric, "arm_a": "mixture_M25", "arm_b": "sampled_M25",
            "hypothesis": "the mixture scores LOWER than the sampled ensemble",
            "confirming_sign_of_b_minus_a": "positive",
            "hypothesis_holds_on_all_seeds": all(d > 0 for d in diffs),
            "per_seed_a": dict(zip(seeds, va)), "per_seed_b": dict(zip(seeds, vb)),
            "per_seed_diff_b_minus_a": dict(zip(seeds, diffs)),
            "arm_a_mean_sd": _mean_sd(va), "arm_b_mean_sd": _mean_sd(vb),
            **required_n(diffs),
            "sign_consistent": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        }
    return rows


def gate_read(runs: Dict[int, dict]) -> Dict[str, object]:
    seeds = sorted(runs)
    rs = [runs[s]["R_at_max_M"] for s in seeds]
    m, sd = _mean_sd(rs)
    return {"per_seed_R": dict(zip(seeds, rs)), "mean": m, "sd": sd,
            "all_above_0.25": all(r > 0.25 for r in rs),
            "all_below_0.10": all(r < 0.10 for r in rs)}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Results are namespaced by cell, so a second dataset cannot silently be read as the
    # first. `--cell` moves all four defaults together; the individual paths still override
    # it for one-off layouts.
    p.add_argument("--cell", default="noaa_uk_h168",
                   help="results subdirectory, '<dataset_key>_h<pred>'. Default: noaa_uk_h168.")
    args_cell, _ = p.parse_known_args()
    root = f"finetuning/results/cmd_gp/{args_cell.cell}"
    p.add_argument("--one-pass", default=f"{root}/R10_option_a_s{{seed}}_grid5.json")
    p.add_argument("--frontier", default=f"{root}/frontier_seed{{seed}}_val.json")
    p.add_argument("--gate", default=f"{root}/R9_gate_seed{{seed}}_val.json")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--out-json", default=f"{root}/prereg_power.json")
    return p.parse_args()


def main() -> Dict[str, object]:
    args = _parse_args()
    one = _load(Path(args.one_pass), args.seeds)
    fro = _load(Path(args.frontier), args.seeds)
    gat = _load(Path(args.gate), args.seeds)

    report: Dict[str, object] = {
        "seeds_found": {"one_pass": sorted(one), "frontier": sorted(fro), "gate": sorted(gat)},
        "wilcoxon_one_sided_floor": WILCOXON_FLOOR,
        "note": (
            "required_n_80pct is the normal-approximation paired one-sided calculation. The "
            "binding constraint at small n is instead the signed-rank test's own floor: with "
            "fewer than 5 seeds no one-sided p can reach 0.05 whatever the effect size."
        ),
    }
    if gat:
        report["T1_gate"] = gate_read(gat)
    if fro:
        report["frontier"] = frontier_reads(fro)
    if one:
        report["one_pass"] = one_pass_reads(one)

    # Console summary
    print(f"seeds: one-pass {sorted(one)}  frontier {sorted(fro)}  gate {sorted(gat)}\n")
    if "T1_gate" in report:
        g = report["T1_gate"]
        print(f"T1 gate  R = {g['mean']:.4f} +/- {g['sd'] if g['sd']==g['sd'] else float('nan'):.4f}"
              f"   all>0.25: {g['all_above_0.25']}")
    for block in ("frontier", "one_pass"):
        if block not in report:
            continue
        print(f"\n{block}:")
        print(f"  {'read':10s}{'metric':34s}{'paired mean':>13s}{'sd':>11s}"
              f"{'d_z':>7s}{'n@80%':>7s}  H holds")
        for read, r in report[block].items():
            dz = r.get("effect_size_dz", float("nan"))
            n80 = r.get("required_n_80pct", float("nan"))
            print(f"  {read:10s}{r['metric']:34s}{r['paired_mean']:13.4f}"
                  f"{r['paired_sd']:11.4f}{dz:7.2f}{n80:7.0f}  "
                  f"{r['hypothesis_holds_on_all_seeds']}")

    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {args.out_json}")
    return report


if __name__ == "__main__":
    main()
