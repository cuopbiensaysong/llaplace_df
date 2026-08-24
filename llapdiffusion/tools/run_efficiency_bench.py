"""Appendix A5: end-to-end efficiency, in the units that actually differ.

The paper reports cost three ways because the three disagree, and conflating them is how a
"1600x" headline gets written for a speedup nobody measures:

* **passes** -- denoiser evaluations. The FLOP claim.
* **depth** -- sequential network depth per forecast. The latency claim: 64 DDIM steps are
  serial, so `M` trajectories at depth 64 are not 64M deep, they are 64 deep and M wide.
* **ms** -- synchronized batched wall-clock on pinned hardware. What a user feels, and the
  only one that includes the parts that do not cancel.

It also separates the costs the closed form is supposed to make cheap, because they scale
differently and the paper's own §cost table distinguishes them:

* mean and per-time marginals: `O(K h d_z)`, fully parallel in the query index, depth 1;
* the variance recurrence: `O(K h)`, sequential in the query index;
* explicit cross-time kernel materialisation: `O(K h^2 d_z)`;
* exact joint sampling by the Markov recursion: `O(K h d_z)`, no Cholesky;
* joint sampling through a Cholesky of the materialised kernel: `O(h^3)` per channel.

"One pass" is reported honestly: the summarizer runs once and the decoder runs once per drawn
path in **every** arm alike, so those costs cancel in a comparison and the measured speedup is
well below the ratio of denoiser evaluations. Both are timed separately so the cancellation is
visible rather than asserted.

Runs on existing checkpoints. No training, no test split.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional

import torch

from llapdiffusion.configs.config_utils import refresh_artifact_paths
from llapdiffusion.models.chirp_gp import (
    cross_time_kernel,
    marginal_law,
    modal_increments,
    sample_joint_markov,
)
from llapdiffusion.models.llapdiff_utils import set_torch
from llapdiffusion.tools import llapdiff_checkpoint_eval as ce
from llapdiffusion.tools.llapdiff_checkpoint_eval import build_eval_config
from llapdiffusion.tools.run_analytic_uq_eval import prepare_eval_stack
from llapdiffusion.trainers import train_val_llapdiff as tv


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timeit(fn, *, device, repeats: int, warmup: int = 2) -> Dict[str, float]:
    """Synchronized timing with warm-up and repeats, reported as median and p95.

    Warm-up matters more than repeats here: the first call pays autotuning and allocator
    growth, and on a short kernel that dominates. A single un-warmed trial is the classic way
    to publish a speedup that does not exist.
    """
    for _ in range(warmup):
        fn()
    _sync(device)
    ts: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"ms_median": ts[len(ts) // 2], "ms_p95": ts[min(len(ts) - 1, int(0.95 * len(ts)))],
            "ms_min": ts[0], "repeats": repeats}


def _peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / 2 ** 20


@torch.no_grad()
def bench(diff_model, cfg, batch, device, *, horizons, components, repeats) -> List[Dict]:
    """One row per (h, M) plus the closed-form component costs."""
    (V, T), _, mask_bn = batch
    summarizer = bench.summarizer
    cond_summary, cond_summary_raw = tv._build_cond_summary_pair(
        summarizer, diff_model, V, T, mask_bn, device,
        dt=bench.meta.get("delta_t"), x_obs_mask=bench.meta.get("x_obs_mask"),
    )
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    steps = int(sampling["steps"])
    B = cond_summary.shape[0]
    d_z = int(getattr(cfg, "VAE_LATENT_CHANNELS", 16))
    K = int(getattr(cfg, "LAPLACE_K", 128))
    rows: List[Dict] = []

    # The summarizer cost, which every arm pays exactly once and which therefore cancels.
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t_sum = _timeit(
        lambda: tv._build_cond_summary_pair(
            summarizer, diff_model, V, T, mask_bn, device,
            dt=bench.meta.get("delta_t"), x_obs_mask=bench.meta.get("x_obs_mask"),
        ),
        device=device, repeats=repeats,
    )
    rows.append({"component": "summarizer_S_phi", "note": "paid once by EVERY arm; cancels",
                 "batch": B, **t_sum})

    for h in horizons:
        dt_model = torch.arange(1.0, h + 1.0, device=device).view(1, h).expand(B, h).contiguous()
        shape = (B, h, d_z)

        # --- one denoiser forward: the unit both cost claims are quoted in ---
        x = torch.randn(shape, device=device)
        t_b = torch.full((B,), steps - 1, device=device, dtype=torch.long)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_fwd = _timeit(
            lambda: diff_model(x, t_b, cond_summary=cond_summary,
                               cond_summary_raw=cond_summary_raw, dt=dt_model),
            device=device, repeats=repeats)
        rows.append({"component": "denoiser_forward", "h": h, "passes": 1, "depth": 1,
                     "peak_mem_mb": _peak_mem_mb(device), **t_fwd})

        # --- the closed-form pieces, which is where the O() distinctions live ---
        cap: Dict[str, torch.Tensor] = {}
        diff_model.generate(shape=shape, steps=2, guidance_strength=1.0, guidance_power=1.0,
                            eta=0.0, cond_summary=cond_summary,
                            cond_summary_raw=cond_summary_raw, dt=dt_model, modal_capture=cap)
        if {"theta", "rho_bar", "omega_bar", "t_rel", "p0", "q"} <= set(cap):
            th, rb, ob = cap["theta"], cap["rho_bar"], cap["omega_bar"]
            inc = modal_increments(rb, cap["t_rel"], cap["q"], cap["p0"])
            lam = inc["lam"]
            for name, fn, order in (
                ("mean_and_marginals", lambda: marginal_law(th, rb, ob, lam), "O(K h d_z), depth 1"),
                ("variance_recurrence", lambda: modal_increments(rb, cap["t_rel"], cap["q"], cap["p0"]),
                 "O(K h), sequential in r"),
                ("kernel_materialisation", lambda: cross_time_kernel(th, rb, ob, lam),
                 "O(K h^2 d_z)"),
                ("joint_sampling_markov",
                 lambda: sample_joint_markov(th, rb, ob, inc["incr"], inc["d_rho"], cap["p0"],
                                             num_samples=25), "O(K h d_z), no Cholesky"),
            ):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                try:
                    r = _timeit(fn, device=device, repeats=max(3, repeats // 2))
                    rows.append({"component": name, "h": h, "K": K, "d_z": d_z,
                                 "asymptotic": order, "peak_mem_mb": _peak_mem_mb(device), **r})
                except RuntimeError as exc:                     # e.g. OOM at large h
                    rows.append({"component": name, "h": h, "error": str(exc)[:120]})

            # Cholesky of the materialised kernel: the O(h^3) route we deliberately avoid.
            try:
                ker = cross_time_kernel(th, rb, ob, lam)
                eye = torch.eye(h, device=device) * 1e-6
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                r = _timeit(lambda: torch.linalg.cholesky(ker[:, :, :, 0] + eye),
                            device=device, repeats=max(3, repeats // 2))
                rows.append({"component": "joint_sampling_cholesky", "h": h,
                             "asymptotic": "O(h^3) per channel",
                             "peak_mem_mb": _peak_mem_mb(device), **r})
            except RuntimeError as exc:
                rows.append({"component": "joint_sampling_cholesky", "h": h,
                             "error": str(exc)[:120]})

        # --- the one-pass arm, which is the paper's headline cost claim ---
        # Timing does not depend on the weights, so this is built fresh rather than loaded:
        # the probe does not persist a head, and a cost measurement needs the shapes, not the
        # fit. Without this row the table would compare CMD-D against its own internals and
        # never actually measure "1 pass, depth 1".
        try:
            from llapdiffusion.models.dist_heads import ChirpGPHead
            head = ChirpGPHead(
                cond_dim=int(bench.cond_dim), d_z=d_z, k=32, hidden=256,
                num_basis=4, horizon=float(h), anchor_nodes=128,
            ).to(device).eval()
            E = torch.randn(B, int(bench.cond_dim), device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            r = _timeit(lambda: head(E, dt_model), device=device, repeats=repeats)
            rows.append({"component": "CMD-GP_one_pass", "h": h, "passes": 1, "depth": 1,
                         "note": "head only; add summarizer_S_phi, which every arm pays",
                         "peak_mem_mb": _peak_mem_mb(device), **r})
        except Exception as exc:                               # never abort the whole table
            rows.append({"component": "CMD-GP_one_pass", "h": h, "error": str(exc)[:120]})

        # --- the decoder, paid once per drawn path by EVERY arm, so it cancels ---
        if bench.vae is not None:
            try:
                mu = torch.randn(B, h, d_z, device=device)
                pad = torch.zeros(B, bench.n_entities, dtype=torch.bool, device=device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                r = _timeit(lambda: bench.vae.decode_mu(mu, pad), device=device, repeats=repeats)
                rows.append({"component": "vae_decoder_per_path", "h": h,
                             "note": "paid once per drawn path by EVERY arm; cancels",
                             "peak_mem_mb": _peak_mem_mb(device), **r})
            except Exception as exc:
                rows.append({"component": "vae_decoder_per_path", "h": h, "error": str(exc)[:120]})

        # --- whole arms, in passes / depth / ms ---
        for M in components:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            r = _timeit(
                lambda: [diff_model.generate(
                    shape=shape, steps=steps, guidance_strength=1.0, guidance_power=1.0,
                    eta=0.0, cond_summary=cond_summary, cond_summary_raw=cond_summary_raw,
                    dt=dt_model) for _ in range(M)],
                device=device, repeats=max(1, repeats // 3), warmup=1)
            rows.append({"component": f"CMD-D(M={M})", "h": h, "M": M,
                         "passes": steps * M, "depth": steps,
                         "peak_mem_mb": _peak_mem_mb(device), **r})
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-key", default="noaa_uk")
    p.add_argument("--pred", type=int, default=168)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--horizons", type=int, nargs="*", default=[24, 48, 96, 168])
    p.add_argument("--components", type=int, nargs="*", default=[1, 4, 8, 16, 25])
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--weights", default="ema", choices=("raw", "ema"))
    p.add_argument("--vae-entity-encode", action="store_true", default=True)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main() -> Dict[str, object]:
    args = _parse_args()
    cfg = build_eval_config(args.dataset_key, int(args.pred))
    if args.vae_entity_encode:
        cfg.VAE_ENTITY_ENCODE = True
    refresh_artifact_paths(cfg)
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=False,
                       allow_tf32=bool(getattr(cfg, "ALLOW_TF32", True)))

    loaders, stack = prepare_eval_stack(cfg, args.checkpoint, device=device)
    diff_model, vae, summarizer, _, _ = stack
    if args.weights == "ema":
        ce._apply_ema_weights(diff_model, torch.load(args.checkpoint, map_location="cpu"))

    for xb, yb, meta in loaders[1]:
        parts = tv._sanitize_batch(xb, yb, meta, device)
        if parts[2].any():
            bench.summarizer, bench.meta, bench.vae = summarizer, meta, vae
            # The one-pass head consumes a strided 16-token view of the summary, matching
            # run_option_a_probe -- mean-pooling it would understate its input width.
            bench.cond_dim = 16 * int(getattr(cfg, 'SUM_CONTEXT_DIM', 256))
            bench.n_entities = int(parts[2].shape[1])
            rows = bench(diff_model, cfg, parts, device,
                         horizons=args.horizons, components=args.components,
                         repeats=int(args.repeats))
            break
    else:
        raise RuntimeError("no usable batch")

    report = {
        "table": "A5 (end-to-end efficiency)",
        "dataset_key": args.dataset_key, "pred": int(args.pred),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": args.checkpoint, "weights": args.weights,
        "note": ("depth is SEQUENTIAL network depth: M trajectories at 64 DDIM steps are "
                 "64 deep and M wide, not 64M deep. Summarizer and decoder are paid once per "
                 "arm and per drawn path respectively, identically across arms, so they "
                 "cancel in a comparison -- they are timed separately here to show it."),
        "rows": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=float)
    print(text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            fh.write(text + "\n")
    return report


if __name__ == "__main__":
    main()
