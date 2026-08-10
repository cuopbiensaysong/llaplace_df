# agents_communication — agent-A ↔ agent-B  🔒 **CLOSED 2026-08-03**

A two-agent parallel investigation of the B21 conditioning gap. **Condensed 2026-08-10** from
1 674 lines of exchange to the findings and the protocol; the full record is at
`archive/2026-08-uq-campaign/agents_communication.md`. Detailed results:
`results_nonlinear_probe.md`. Durable ledger: `CMD_BUG_REPORT.md` (B20, B21, B5).

---

## 1. What the channel settled

| finding | detail |
|---|---|
| **The B21 gap is genuine information LOSS**, not nonlinear re-encoding | an oracle-selected MLP (config *and* epoch chosen on val) reproduces the linear gap to **0.3 pt**: 50.0 % vs 49.7 % |
| **335 of 336 summarizer token directions are unsupervised** | every `LaplaceAE` pretraining head reads `ctx_mean = context.mean(dim=1)`; `COND_NORM_MODE="sample"` reduces over the same axis, deleting the one direction that *was* supervised |
| `cond_summary`'s honest figure is **~14 %**, not 7.7 % | the 2 048-dim view is impoverished; nonlinearity recovers strided-away tokens, not hidden structure |
| Nonlinearity and level-restoration are **substitutes**, not additive | both agents predicted additive; both were wrong. Best number anywhere is plain ridge on the rich view, 14.23 % |
| **`"global"` @8192 → raw target is 13.69 %, not 16.4 %** | a published B21 number, corrected. It helps the poor view (+4.54) but **not** the rich one (−0.54) |
| **`COND_NORM_MODE="global"` is unmeasured, not small** | 5 seeds: mean −0.0086, 4× inside the 0.036 band, sign test p ≈ 0.19 — **and 8 of 10 runs were still improving at the final eval**, so `EPOCHS=60` truncated rather than converged. A faster-converging arm necessarily wins a truncated budget, so the 4/5 majority is what the artefact predicts |
| **bms_air is VOID at every horizon** | best A anywhere +1.8 % (h=24), declining 1.8 → 0.5 → 0.0 → −0.3. The target is PM2.5; the cause is physical, ruled out as a level-shift artefact three ways |
| **noaa_us h=168 replaces it as the Phase-2 cross-check** | A = 23.4 %, B/A = **11.1 %** — B21 replicates on an independent network, and harder |
| 🔴 **Entity pooling is the DOMINANT loss on wide panels** — retracts a B20 claim | noaa_uk: 28.6 → 25.3 (pooling, −3.3) → 14.2. noaa_us: 23.4 → **6.0 (pooling, −17.4)** → 2.6. The two losses are separate terms whose relative size **flips between cells**, and B5 is promoted from parked to live |
| The mediator is **shared variance**, and it saturates | leave-one-out: noaa_uk 76.5 %, noaa_us 46.8 %, crypto 42.4 %, us_equity 34.2 %. ~24 % of the apparent noaa_uk/noaa_us gap was panel-size arithmetic (a 4-panel carries a 1/4 self-contribution) |
| `ridge_reduction`'s alpha holdout must be **blocked + purged** | the literal last-20 % band is a shifted seasonal regime here: ridge scores **−8.3 % on it while scoring +28.8 % on val**. A selector that anti-correlates with the target metric is not a selector. Fixed in `run_ridge_probe.py` (`ef0d679`), then guarded against small-`n` collapse |

**One discrepancy never resolved:** the two agents' noaa_us shared-variance scripts read 42 % vs
49.0 %, with drop rates 11.4 % vs 1.7 %. They filter differently and the cause was never found.
agent-B's column is the one of record; **do not mix the two.**

---

## 2. The protocol — reuse this if a second agent runs again

It worked: across six tasks, agent-B found **two defects in committed code, one gap in the
handoff, one failure mode in an instruction it was given, and one structural defect in an
experiment agent-A designed**. Four predictions across both agents were falsified.

- **Append only; never edit another agent's entry.** Re-read the whole file immediately before
  appending — an entry you have not seen may change your task. Disagree by appending a correction
  that quotes what you disagree with. (Entries *did* cross here; nothing was lost because of this.)
- **Numbers carry their protocol.** Split, seeds, feature construction, how hyperparameters were
  selected. A bare percentage is not a result.
- **Register predictions before the numbers exist, and record failures as failures.** Both agents
  declined credit offered wrongly, and both insisted on owning their own errors — the record is
  more useful for it.
- **Choose the honest *measure* before seeing which is kinder.** agent-A's `"global"` prediction
  held on the row it named and failed on the ceiling measure; it accepted the ceiling, because
  that was the measure already established as honest.
- **Falsification criteria must condition on the test being informative.** "B ≈ A falsifies the
  mechanism" was badly written: on bms_air B ≈ A held **vacuously, because A ≈ 0**. You cannot
  measure whether stage 2 discards signal on a cell that has none. The result was VOID, not
  negative.
- **Verify instructions rather than executing them.** agent-B checked that the precompute cache
  is normalisation-agnostic before reusing one build across both arms — had it stored *normalised*
  summaries, agent-A's own instruction would have fed `"sample"` conditioning into the `"global"`
  runs and produced a clean-looking null.
- **Apply validity gates blind.** Runs were voided by a train-loss criterion fixed *before*
  unblinding, executed by a script that never reads CRPS. Note when an exclusion helps the
  excluder's case — one here *hurt* it, which is worth stating.

---

## 3. Hazards specific to multi-agent work on this tree

- 🔴 **`ldt/vae/**` and `ldt/summarizer/**` are NOT re-rooted by `--run-tag`**, and
  `run_single_pred` retrains them **whenever the checkpoint file is missing**. So one agent's run
  can silently retrain a shared artifact underneath another's in-flight campaign and invalidate
  every seed on both sides — no error, no warning. **Confirm both files exist before any training
  run**, and never pass `--recompute-vae` / `--recompute-summarizer` on a shared tree. This was a
  gap in the original handoff, caught by pre-flight rather than by damage.
- **Give each agent its own `DIFF_PRECOMPUTE_DIR`.** "Match my protocol exactly" and "use your own
  cache" only look contradictory: the cache holds frozen VAE latents and summaries, so a separate
  build differs by ~1e-6 against a 0.036 band.
- **`model_config["cond_norm_mode"]` is top-level**, a sibling of `"llapdiff"`, not nested inside
  it. Misreading the nesting nearly produced a false "the flag is inert" conclusion.
- **Never `git stash` / `checkout` / `restore` / `clean` / `reset` on the shared tree.** Large
  uncommitted diffs live only there.
- **Pin GPUs per agent** (`CUDA_VISIBLE_DEVICES`); contention perturbs `cudnn.benchmark`, the
  documented 0.036 CRPS noise source.
- **Put probes in standalone scripts outside the repo** that import `llapdiffusion` read-only,
  when another agent owns that surface.
