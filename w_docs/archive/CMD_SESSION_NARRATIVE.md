<!-- HANDOFF COPY. Body below the 2026-07-27 section is the verbatim export of the
     assistant's project memory (~/brain/.copilot1/.../cmd-paper-status.md), copied
     2026-07-23. A second handoff section was PREPENDED 2026-07-27 for a further account
     transfer; everything under "State at handoff (2026-07-23)" is unchanged history. -->

# CMD paper — full working narrative (handoff export)

**What this is.** The running log of the CMD (Chirp-Modal Dynamics) paper work: what was
built, what broke, what was measured, and which claims were later retracted. Entries are
roughly chronological; later entries correct earlier ones, so **read to the end before
acting on any single claim** — or rather, read the 2026-07-27 section first, since it
supersedes much of what follows.

---

# ⭐ STATE AT HANDOFF — 2026-07-27 (read this first)

**Read `w_docs/CMD_BUG_REPORT.md` before anything else.** It is now the canonical document:
17 bugs, what each broke, how it was fixed, what results it invalidates, plus the results
ledger, retracted claims, and audit recipes. *This narrative is history; the bug report is
current.*

## 1. Code state

| item | value |
|---|---|
| working branch | **`fixed_bugs`** @ `84b92e1`, **pushed** to `origin/fixed_bugs` |
| uncommitted on top | `cond_norm_mode` checkpoint persistence, Tier-2 M grid retune, `CHIRP_NUM_BASIS` default 256→8, bug-report edits |
| reference branch | **`update_method` @ `347cd8a` — deliberately frozen pre-fix.** Do not move it. |
| tests | **379 passing** (`conda activate llapdiff && python -m pytest tests/ -q`) |
| deleted branches | `phase1`, `phase2U`, `chirp_fix_basis_scale` (0 unique commits); `phase0` preserved as tag **`archive/phase0`** (local only — `git push origin archive/phase0` to back it up) |

⚠️ **Security:** the `origin` remote URL embeds a GitHub PAT in plaintext in `.git/config`,
and `PAT.txt` (gitignored) holds the same token. It was **never rotated**. Revoke and reissue,
then `git remote set-url origin https://github.com/cuopbiensaysong/llaplace_df.git` + a
credential helper.

## 2. The finding that reframes everything

**The H2 denoiser barely uses its conditioning.** Measured on the p32 chirp benchmark, K=1,
converged:

| quantity | R² on the true latent trajectory |
|---|---|
| ridge probe on the model's **own** `cond_summary` | **+0.747** |
| the trained denoiser's 25-draw ensemble mean | **−0.30** (before B15) → **+0.21** (after) |

At `steps=1` (the one-shot conditional mean) the output had **std 0.09 vs the target's 0.62**,
corr ≈ 0 — it predicts the *marginal* mean. **The target is sound**: the true latent carrier
sits at dominant ω = **0.360** against a ground truth of **0.362**.

⇒ **Poles cannot track because the FORECAST does not track.** This one upstream failure
explains every H2 observation: flat pole slopes, CRPS pinned at 0.19–0.20 across *all* arms /
K / basis / ρ settings, and insensitivity to every intervention.
⇒ **No prior H2 result — including the 5-seed "confirmed negative" below — ever actually
tested the CMD tracking claim.** That question is *reopened*, not closed.

## 3. Fig-2 status

Three structural obstacles found and fixed this session (details = B9–B11 in the bug report):
representability (`CHIRP_BASIS="half_integer"`, R² **0.03 → 0.99** on the real truth), ω-level
inflation (`CHIRP_OMEGA_BASIS="centered"`), and an ω-headroom trap **I introduced myself**
(bounding the swing by the floor made "hedge to DC" and "cannot sweep" the same event; K=1 was
99.2% constraint-saturated). Post-fix, K=1 *chooses* a swing of **0.544 vs truth 0.48**.

**Tracking is still ~0 — but it is NOISE-DOMINATED at single seed.** Aggregate slopes observed
across configurations: **+0.170, −0.035, +0.069, +0.054**; per-window values from **−0.44 to
+0.79**. With 4 windows × 3 draws these are indistinguishable from each other and from zero.
**Do not read any single-run slope as signal** — I briefly did, and was wrong.

**Remaining blocker = §2** (conditioning used at 0.21 of the 0.75 achievable). Best untried
lever: `block_summary_adaln` is currently **`False`** — the summary reaches the trunk only via
cross-attention into K modal tokens whose residues come from `x_t`, so at high noise it must
overwrite noise-dominated tokens. Enabling AdaLN injection targets exactly the measured gap.
**Evaluate that on conditional R² (a stable number), not on the noisy pole slope.**

Deliverables that exist and stand on their own: `ldt/results/fig2_diagnosis.pdf` (3 panels:
representable ✓ / present in conditioning ✓ / recovered ✗) and
`ldt/results/fig_representability.pdf`.

## 4. Running right now

`ldt/scripts/run_tune_fixed_version.sh` (background, GPU wait-loop, 24 h cap):
physionet h=12 arm d → smoke, then the campaign at `--run-tag fixed_version`.
**Smoke succeeded** — trials completing, incumbent test CRPS **0.3495** with
`CHIRP_NUM_BASIS=4` — which confirms the new defaults (incl. `COND_NORM_MODE="sample"`) do not
break physionet. Logs: `ldt/results/tune_runner.log`, `tune_smoke.log`, `tune_fixed_version.log`.

**Two caveats to carry with any number it produces:** six model defaults changed at once with
**no legacy control arm**, so the result is *not* attributable to the bug fixes and is *not*
comparable to G3's 0.3709 or v1's 0.3469; and selection is on the **test** split, so the CRPS
is selection-biased and is not a clean test estimate.

## 5. Do NOT retrain the VAE or the summarizer

Verified 2026-07-27 (bug report §1b): stages 1–2 have **zero** references to the changed
symbols; `SUM_FT_MODE="none"` keeps the summarizer frozen in stage 3; and the diffusion
precompute cache stores `summary_raw` *un-normalised*, so **caches stay valid too**. Only
stage-3 denoiser checkpoints need retraining. (Exception: if `SUM_FT_MODE` is ever set to
`pool`/`top`/`all`, stage 3 fine-tunes the summarizer through the conditioning path.)

## 6. Environment reality

The H100 is **shared and usually full** — other tenants (`saturn2`, `respi`,
`kienptMedSigLIP`) hold 70–80 GB of 81.5 GB; free memory is often 0.5–5 GB. Our jobs need
~1.5–6 GB. **Always use the GPU wait-loop pattern** from `ldt/scripts/run_*.sh` rather than
launching directly; a smoke run already died of OOM this session. `ldt/` is gitignored, so
scripts and results there are untracked.

## 7. Ranked next actions

1. **Close the conditioning gap** (§3): try `block_summary_adaln=True`; score on conditional
   R² vs the 0.747 probe ceiling. This gates Fig 2 and probably H1 too.
2. **Fix B16** (open): `_best.pt` and `_best_ema.pt` hold identical weights and `_load_stack`
   never applies `payload["ema"]`, so CRPS from `llapdiff-checkpoint-eval` / the H2 benchmark
   is raw-weight labelled EMA. (`finetuning/` is unaffected — it applies EMA correctly.)
3. **Re-run Fig 2 multi-seed** only after (1) — single-seed slopes are noise.
4. **H1 NOAA-UK h=168 has never been run** (`ldt/output/noaa_uk/` does not exist). It inherits
   every fix, so it starts clean — but do (1) first.
5. Commit the uncommitted follow-ups; consider `git push origin archive/phase0`.

## 8. Things already settled — do not re-derive

Four claims are **retracted** (18% CRPS win; ~12× stability advantage; "the model hedges to DC
because phase is uncertain"; "LTI is spectrally static"). Full reasoning in bug report §4.
Single-run CRPS comparisons are untrustworthy (`cudnn.benchmark=True`; a 0.036 swing from pure
nondeterminism, larger than every margin ever quoted). Estimator traps (single-sinusoid fits,
turning-point windows, the noise-fooled `forecast_freq_*`) are catalogued in bug report §5.

---

**Companion documents in this folder**
| file | contents |
|---|---|
| `pole_rho_init_horizon_bug.md` | Root-cause report: horizon-blind rho init (both cores), the uncapped rho variation, and §11 the history-only pole finding. Includes an audit recipe. |
| `fig2_signature_figure_strategy.md` | Fig-2 campaign plan, gates, and the amendments made as causes were found. |
| `h2_pole_recovery_problems_fixes.md` | The earlier (separate) pole-recovery tooling bug, P1-P10, all landed. |
| `CMD_RUNBOOK.md` | Operational guide: every CLI knob, red-box warnings, reproduction commands. |

**State at handoff (2026-07-23)**
- 345 tests passing.
- 🔴 **The Fig-2 "confirmed negative" below is RETRACTED (2026-07-23, later same day).**
  The pole basis could not represent the target: `basis_freqs = linspace(1, M, M)` gives
  every basis function a whole number of cycles across the window, so with squared
  (nonnegative) coefficients `omega(0) == omega(L) == sup_t omega(t)` for every mode --
  the instantaneous frequency can only dip and return, and a monotone sweep is outside
  the function class. Fitted to the real truth over 120 test windows with the model's own
  basis and constraint: **R^2 = 0.119 +- 0.266** (legacy) vs **0.990 +- 0.008** with
  half-integer harmonics; net endpoint drift is 73% of the within-window range. The five
  "independent" designs varied sweep steepness and period but never the shape class, so
  they shared one confound. Fix landed (`CHIRP_BASIS`, default `half_integer`); re-run
  pending. Full account in `w_docs/fig2_pole_basis_cannot_drift.md`.
- Fig-2 *tracking* half (SUPERSEDED, kept for the record): read as **confirmed negative**
  across 5 benchmark designs and a full
  5-seed run (slope -0.0032 +- 0.0113 vs a >=0.6 gate; CRPS diff +0.0010 +- 0.0043 se, not
  significant), all known confounds eliminated. Architectural reason in root-cause §11.
- Two claims made mid-session were **retracted** on further evidence (an 18% CRPS win, and
  a ~12x stability advantage). Both retractions and their causes are recorded below.
- **Highest-value outstanding work: re-run the h=168 NOAA-UK headline.** Those runs were
  trained under the horizon-blind rho init (rho*H up to 33.6 => nearly all oscillatory
  modes dead at initialisation). This affects a headline number, not just a figure.
- Uncommitted at handoff: model/tool/test changes across `llapdiffusion/` and `tests/`,
  plus these docs. `ldt/` is gitignored, so run scripts under `ldt/scripts/` are untracked.

---

---
name: cmd-paper-status
description: "CMD (chirp-modal) paper — code done incl. H2 pole-recovery rewrite (2026-07-20); G2/G3 branch = 0.36 match+certify; next: PREREG freeze then H1 NOAA-UK h=168 + reswept H2 retrains"
metadata: 
  node_type: memory
  type: project
  originSessionId: e1732273-9b45-4cbd-81a4-aa30805dbfad
  modified: 2026-07-22T06:25:21.093Z
---

🔴 **STALE-CHECKPOINT TRAP (found 2026-07-20, fixed).** The v3 sweep-144 campaign
"failed" for a pure tooling reason: 11/30 runs — ALL synthetic_linear_chirp + 5/6
synthetic_quadratic_chirp, i.e. exactly the two ω-sweep tasks Fig 2 needs — were
EVALUATED on a 07-14 `llapdiff_pred-48_best_raw.pt` trained on the legacy
near-constant cache, while that run's fresh checkpoint sat unused. Chain:
`_best_raw.pt` is written only when `EMA_COMPARE_EVERY > 0` (the benchmark sets 0),
the trainer reported it via a bare `.exists()`, and `_train_or_reuse_stack` preferred
`best_checkpoint_raw` first. Fixed: trainer reports only checkpoints written THIS run
(mtime guard + `written=` filter on `_select_eval_checkpoint_path`); tool now takes
the trainer's `loaded_checkpoint`. Second, independent contamination: stage-1/2
artifacts are keyed by (task, seed) with NO cache tag, so both chirp tasks reused
07-14 VAE+summarizer (user omitted `--recompute-artifacts`) — now guarded by an
`artifact_cache.json` fingerprint that fails loudly. Lesson: **always check the
`checkpoint` column's mtime in the raw CSV before believing an H2 result.**
Verified-good diagnostics (retrain-free, worth reusing): the VAE latent carrier
MATCHES the data-space truth ω (0.48 vs 0.467, 96-100% of channels) and the truth
poles explain 83-92% of latent variance → the Fig-2 target is fair and well-posed.
Open concern: at K=256 the modal decomposition is non-identifiable (~8 modes give
recon R²=0.99, and *which* 8 changes between training runs), so recovered per-mode
poles may be meaningless regardless of M — hence the K arm.
✅ **ROOT CAUSE + FIX (2026-07-21): H2's entity phases vs the VAE's entity mean-pool.**
`latent_vae.py` mean-pools across the entity axis before `mu_head`; the generator draws
`phase0 ~ U(0,2π)` per entity, so 64 differently-phased sinusoids cancel and the carrier
never reaches the latent. Ruled out by measurement (all ~0.6): input dropout 0.20→0.0
(0.620→0.586), latent 24→64 (0.586→0.591), 8 entities (0.479 — WORSE). Fix = narrow the
phase draw: spread 0 → corr 0.863/val_recon 0.0188; **spread π/8≈0.393 → corr 0.857,
val_recon 0.0202 (3.2× better than 0.0611), recommended** (keeps entity diversity).
Exposed as `--phase-spread` (tags cache `_phase-0.393`, in result rows + summary key).
Gate 0 confirmed end-to-end 2026-07-21: FULL retrained stack (VAE+summ) on the
phase-0.393 cache passes at corr 0.857 (matches isolated probe). New committed tool
`llapdiff-stage1-roundtrip` runs this check (PASS/FAIL vs threshold, exits non-zero).
Phase-fixed Fig-2 campaign RUNNING: `ldt/scripts/run_fig2_phasefix.sh` →
`ldt/results/fig2_phase_fixed/k-{8,32}`, linear_chirp, both arms, K∈{8,32} M=8,
sweep-144 + phase-0.393, w=1 capture 3 draws, fresh artifact-root
`chirp_benchmark_phasefix`. First clean test of the K-scan + Option 3 (forecast IF).
🔴 **DON'T CALL THE FAILURE BRANCH YET — the phase-fixed denoiser was UNDERTRAINED.**
K=8 phase-fixed (Gate 0 = 0.857, sound stage 1): pole slope +0.026 (flat), forecast↔target
corr 0.24, CRPS chirp 0.197 < lti 0.201 (chirp wins, 1st time). BUT the denoiser
early-stopped at **epoch 19** (val v-MSE 0.60, selected on the cheap val_diag_mse_raw
proxy, DOWNSTREAM_EVAL_EVERY=10) — the benchmark default EPOCHS=80/EARLY_STOP=8 is a
short schedule. So flat poles + weak forecast are CONFOUNDED by undertraining. Also: my
"forecast 4x too fast" was a metric artifact — zero-crossings 24.5 vs truth 11.3 (~2x,
not 4x) + phase drift (early-half corr 0.21, late-half 0.07); `forecast_freq_slope` is
computed on the raw LATENT and is unreliable (noise-fooled), drop or gate it. Fix: added
`--epochs`/`--early-stop` (auto CRPS selection when >80, `_ep-NNN` denoiser tag so it
won't clobber). Convergence rerun RUNNING: `ldt/scripts/run_fig2_k8_long.sh` (400 ep,
ES30, CRPS-selected, reuses Gate-0 stage 1) → `ldt/results/fig2_phase_fixed_long`. This
is the real test of whether CMD tracks the chirp; failure-branch decision waits on it.
**CONVERGED K=8 RESULT (2026-07-21, all confounds removed):** chirp best_epoch 40 (val CRPS
0.199), lti best_epoch 115 (0.202). Test CRPS chirp 0.212 vs lti 0.206. Forecast↔target corr
chirp 0.253 vs lti 0.265. Pole slope chirp +0.099 but carried entirely by the LEAST-swept
window (582: swing 0.071, slope +0.394); the 3 real-swept windows read -0.12/+0.07/+0.06
(flat). Selection valid at 0.85-0.98 → decomposition trustworthy and says NO tracking.
**Verdict: at this sweep difficulty (Δω ~0.07-0.18 rad/step over h=48), CMD shows NO
advantage over LTI — neither pole (Opt 2) nor forecast (Opt 3) tracking.** KEY caveat before
declaring the prereg failure branch: LTI's OWN recovery is valid and its w_eff_rmse 0.121
BEATS chirp's 0.192, CRPS ties → the benchmark doesn't discriminate; the sweep is too gentle
for LTI to structurally fail, so it can't test the CMD claim either way. Last test before the
failure branch is real: STEEPER sweep to build the LTI-failure regime H2 was meant to have.
User said keep trying Fig 2 (don't pivot). **STEEPER-SWEEP TEST RUNNING 2026-07-22**: added
`--freq-multiplier` flag (default 2.0, tags cache `_fm-N`, in rows/summary). Phase-desync
math: LTI desyncs from a chirp by ~Δω·H/12 over the horizon; fm-2 gives only 0.5 rad (0.09
cyc, LTI fine), **fm-4 gives Δω~0.4 rad/step → 1.6 rad desync (0.26 cyc)** where a constant
pole breaks (ω stays ≤1.04, ~6 samples/cyc, reconstruction-safe; fm-5 risks Gate 0).
`ldt/scripts/run_fig2_steep.sh` → `ldt/results/fig2_steep_fm4`: linear_chirp K=8 M=8,
sweep-144 phase-0.393 fm-4, converged (400ep ES30 CRPS), fresh artifacts
`chirp_benchmark_fm4`, Gate 0 rechecked after. SUCCESS = LTI's w_eff_rmse rises (constant
pole can't match steep truth) + chirp CRPS/slope separates. Also added `--epochs`/
`--early-stop` (CRPS-select when >80, `_ep-NNN` tag, `best_epoch` in rows) and the committed
`llapdiff-stage1-roundtrip` Gate-0 tool. Runbook documents all new knobs. 331 tests green.
**LR LEVER (user hint 2026-07-22): "higher LR makes chirp better on real data."** Checked the
converged fm-2 model: coefficients DID grow (to_coeffs absmax 0.036 vs 1e-4 eps-init; model's
own omega sweeps up to 0.44 rad/step) — so converged chirp is NOT stuck at LTI. BUT the mode
that chirps at the right freq (mode 3: mean 0.35, range 0.37) may not carry the output energy,
and on real data with a normal budget higher LR grows coeffs faster — so the hint is
well-motivated. Added `--base-lr` flag (default 1.5e-4, tags `_lr-N`, in rows). Running fm-4 at
BOTH LRs: default (`ldt/results/fig2_steep_fm4`) and 5e-4 (`ldt/results/fig2_steep_fm4_lr5e4`,
reuses fm-4 stage 1). Comparison at hard difficulty: does higher LR make chirp separate from
LTI? If yes, LR is the missing lever; try 1e-3 next. All runs single-seed still.
**FM-4 RESULTS + MECHANISM (2026-07-22).** Gate 0 PASSED 0.851 on fm-4. Difficulty lever
WORKED: LTI w_eff_rmse 0.121 (fm-2) → 0.420 (fm-4) — the benchmark finally penalises constant
poles. BUT chirp degraded too and is WORSE than LTI (0.582 @1.5e-4, 0.376 @5e-4 vs LTI
0.420/0.310); slopes still ~0; CRPS tied. **Mechanism found: the model HEDGES TO DC.** With
truth ω=0.881, the dominant output mode is ω≈0.039 holding **92%** of energy at lr1.5e-4;
at lr5e-4 that drops to 53% and energy shifts to ω=0.335/0.588 modes. I.e. when future phase
is uncertain the MSE-optimal forecast is damped toward flat → flat poles + weak corr. **Higher
LR measurably pulls energy OUT of the DC hedge — the user's hint acts on the right mechanism**
(chirp w_eff_rmse -35%, best_epoch 115→265 = still learning). **Design tension (derived):**
desync≈Δω·H/12 and cycles≈ω_max·H/2π with ω_max≥Δω ⇒ cycles ≳ 1.91·desync. fm-2 desync 0.54/
4.0cyc (LTI fine, too easy); fm-4 desync 1.62/8.0cyc (LTI fails but phase unpredictable →
both hedge). Sweet spot = max desync per cycle: **bf=1/64 mult=6 period=96 → ω[0.10,0.59],
desync 1.41, 4.5 cyc**. Added `--base-frequency` flag (pins the draw, tags `_bf-N`; also on
the roundtrip tool). RUNNING: fm-4@lr1e-3 (`fig2_steep_fm4_lr1e3`) and sweet-spot@lr5e-4
(`fig2_sweetspot`, own artifacts `chirp_benchmark_sweet`, Gate 0 after). 331 tests green.

🔴🔴 **ROOT CAUSE (user hypothesis, CONFIRMED 2026-07-22): rho INIT IS HORIZON-BLIND — the
oscillating modes are DEAD before the horizon ends.** User said "rho(t) is very large vs
ground truth → flat trajectories". Verified on trained models: truth rho=0.0032 (rho*H=0.16,
envelope keeps 0.86), but every mode at a real signal frequency (ω=0.34/0.59/0.96) learned
rho≈0.10-0.13 = **21-76x truth**, rho*H=3-12, **envelope_mass 0.03-0.12 → DEAD**. The only
survivors are near-DC (ω=0.019, rho=0.019, envelope 0.41) — so the "DC hedging" I diagnosed
is NOT the model hedging, it's that decay kills every oscillatory mode and DC is all that's
left. Cause: BOTH cores hardcoded `uniform_(0.01, 0.2)` with NO horizon term
(`laptrans.py` chirp ~:514, LTI ~:161). rho*H at init: h=12 → [0.12,2.4] (fine, where it was
tuned); **h=48 → [0.5,9.6]; h=168 (NOAA-UK HEADLINE) → [1.7,33.6] = nearly all modes dead.**
⇒ this silently damages the long-horizon headline runs too, not just Fig 2.
FIX (landed, 335 tests green): rho init anchored to the horizon, `uniform(0.1, 2.0)/horizon`
⇒ rho*H∈[0.1,2] (envelope 0.9→0.25). New `pole_init_horizon` kwarg threaded
LLapDiff→LapFormer→BOTH cores (LTI too — `_resolve_chirp_time_scale` returns None for lti, so
using chirp_time_scale alone would have biased the comparison); trainer sets it to PRED for
both arms; `setdefault(None)` keeps pre-fix checkpoints on the legacy init.
`tests/test_pole_init_horizon.py` pins it. A/B RUNNING: `fig2_sweetspot` (pre-fix rho) vs
`fig2_sweetspot_rhofix` (fixed rho, identical copied stage 1) — isolates the rho effect.
Full root-cause write-up: `w_docs/pole_rho_init_horizon_bug.md` (+ red box in CMD_RUNBOOK).
2026-07-22: fm-4@lr1e-3 died of OOM (other tenants took ~74GB; our jobs only ~1.5GB each) —
NOT worth resurrecting: it was launched pre-rho-fix, so it measured LR effects inside a model
whose oscillating modes were already dead. **Redo the LR sweep (5e-4 vs 1e-3) AFTER the rho
fix is validated.** Both A/B runs survived the OOM.
⇒ ALL H2 caches and models trained on them must be regenerated; prior chirp-vs-LTI H2
comparisons ran through a handicapped stage 1. New Gate 0 before any Fig-2 claim:
stage-1 round-trip corr > ~0.85 on the run's own cache. Re-test Option 3 after the fix
(it failed only because the forecast was uncorrelated with the target), and re-run the
latent-capacity sweep (width was useless only because info died upstream).

🔴 **H2 BENCHMARK HAS A PIPELINE CEILING (found 2026-07-21).** Encode→decode of the
TRUE targets through the set-VAE reaches only **corr 0.62, std ratio 0.68** — the upper
bound for ANY denoiser here. Cause is visible in the config: `VAE_LATENT_CHANNELS=24`
vs `--num-entities 64`, with `shared_poles=True` giving every entity the same ω(t) but an
independent random phase, so 64 numbers/timestep are squeezed into 24 channels. Measured
downstream: the h=48 forecast correlates with the target at only **+0.06…+0.32** (std
ratio 0.66) and its instantaneous frequency is ~2 rad/step vs truth ~0.48 — i.e. the model
does not reproduce the oscillation at all. ⇒ **Option 3 (forecast-instantaneous-frequency
figure) is DEAD in this configuration**; "the forecast chirps" is false. Pole recovery
stays well-posed though: the latent still carries a clean carrier at the truth frequency
(0.48 vs 0.477) and truth poles explain 83–92% of latent variance — the SHARED pole
function survives the bottleneck, only per-entity phase is lost. Before any further H2
compute, fix the geometry (fewer entities and/or larger VAE_LATENT_CHANNELS) and re-check
the round-trip. NOTE the short-time IF estimator itself is validated (0.472 vs 0.477 truth,
RMSE 0.03 on real decoded targets) — don't blame the metric.

**Fig-2 Stage-0 scan LAUNCHED 2026-07-21** (`ldt/scripts/run_fig2_stage0_scan.sh` →
`ldt/results/fig2_stage0/k-{256,32,8}`, log `ldt/results/fig2_stage0_scan.log`):
linear_chirp seed 0, both arms, K∈{256,32,8} at M=8, sweep-144, w=1 capture, 3 draws.
Decision rule: `omega_trend_slope`→1 as K falls ⇒ Option 2 (rank-matched budget) is the
figure; `forecast_freq_slope` separating arms at any K ⇒ Option 3 (forecast instantaneous
frequency, identifiable by construction); neither ⇒ pre-registered failure branch.
⚠️ The lab H100 is shared (5-day llava + LaMed jobs, ~76 GB resident, 100% util) — our
runs add ~3 GB and are slow. Deleted 2026-07-21: 29 stale 07-14 .pt files, the two legacy
non-sweep chirp caches, and their precompute caches (~7 GB).

**H2 pole-recovery rewrite DONE 2026-07-20** (reviewer doc `h2_pole_recovery_problems_fixes.md`,
P1–P10 all implemented; status block at its top; 316 tests green). Diagnostic verdict: the
model recovered poles fine — the old figure ranked modes by coefficient variation and plotted
zero-output-share junk (ρ≈100/step). New path: `modal_capture` hook (final DDIM step) →
`modal_contributions` E_k ranking → ω_eff/ρ_eff over all modes as primary metric, both arms,
stratified windows, `--recovery-share-threshold` validity gate + watermark, `--sweep-period`
triangle re-sweep (legacy ramp = only ~6% excursion per window; confirmed LTI matches truth
within-window on legacy caches → sweep is REQUIRED for the "LTI fails structurally" claim).
Prereg amendment (third "selection invalid" category) in `cmd_plan_v2.md` §H2 BEFORE freeze.
Fixed-tool regeneration of the 5 trained pairs: `ldt/results/chirp_benchmark/recovery_v2_2026-07-20/`
(originals untouched). ⚠️ The user's H2 checkpoints were trained with the leaked base-config
`CHIRP_NUM_BASIS=256` (committed by user; benchmark inherits base CHIRP_*; ~10× Nyquist at
H=48) — recovery JSON now records chirp_num_basis; paper H2 runs should retrain with
`--sweep-period 144` caches and a horizon-sized num_basis (8–16). Model findings the fixed
figures expose: ρ_eff overestimated (~0.1–0.2 vs truth 0.003), ω_eff under-tracks late windows.
Note: plan/method docs now live at repo root (`cmd_plan_v2.md`, `chirp_modal_method_update.md`),
not `w_plan/`. User is running their own num-basis campaigns (g3_chirp_num_basis_{64,256,512}
results dirs; laptrans/lapformer have their debug prints — left untouched).

**G2/G3 EXECUTED 2026-07-05 on this box's H100** (the old CPU-only note is obsolete —
the driver now supports CUDA; physionet h=12 trains in ~110 s/run). Results:
`ldt/results/g3/G3_RESULTS.md`. 2×2 CRPS, 5 seeds: (a) 0.3710±0.0117,
(b) 0.3719±0.0125, (c) 0.3706±0.0024, (d) 0.3709±0.0018.
**§8 branch DECIDED: ≈0.36 reproduction-gap → "match + certify + strictly generalize"**
(0.320 anchor retired; the released-checkpoint reference 0.367 IS reproduced).
Reads: (d)≈(a) holds exactly; (c)≈(d) kill-shot holds (head redundant once poles vary);
(d)>(b) not observed at h=12 (risk-register-expected; headline regime is h=168).
Bonus: chirp arms show ~6× lower seed variance — attribution is the pole
parameterization, not head removal ((b) is head-free but loose).
Next: freeze PREREG.md, then H1 NOAA-UK h=168 (2×2 + repro, 10 seeds) + H2 benchmark.

The LLapDiffusion repo is being extended into the **CMD paper** (Chirp-Modal Dynamics),
driven by `w_plan/chirp_modal_method_update.md` (Theorems A–D + B′) and
`w_plan/cmd_plan_v2.md` (pre-registered experiment plan). Full feasibility review +
implementation map: `~/brain/.copilot1/plans/fancy-herding-micali.md`.

**As of 2026-06-25:** Phase-0 code gates are implemented and verified on branch
`update_method` (241 tests green): `--seed` (3 stages seeded, `seed-<n>/` routing),
`--output-head {auto,on,off}` decoupled from `--modal-type` (all 2×2 factorial cells
buildable, `head-<mode>/` routing, `|s|≤1` clamp), `CHIRP_TIME_SCALE=None` → resolves to
the run's `PRED` (`"adaptive"` = per-sample opt-in), instantaneous ω capped below π.

**Phase-1 code landed (user backed up Phase-0 to another branch first):** H2 synthetic
chirp benchmark (`llapdiff-synthetic-chirp`, 5 ground-truth pole tasks + `shared_poles`
caches + persisted `pole_truth/*.npz`), pole-trajectory tooling
(`ChirpModalField.instantaneous`, `extract_chirp_pole_trajectories`,
`*_pole_trajectories.pdf` in llapdiff-plot-poles), Prop-A.1 figure script
(companion err ~4.6 vs normal-form ~2e-11). 256 tests green; e2e smoke passed.
**Known pre-existing bug (documented in CMD_RUNBOOK.md H3, tool NOT fixed):**
`llapdiff-synthetic-regime` crashes at BOTH protocol defaults (288/216 and 432/373):
empty-val purged split (val band < horizon). Verified workaround via
--validate-split-only: `--series-length 768 --change-point 606` (12 crossing
windows/asset; cp must sit a full lookback inside the test region — 576 (the 3/4
point) is WRONG, it lands in val). The chirp tool defaults to 768 and validates.

**Phase-2 code landed (user backed up Phase 1 first):** U2 Theorem-C UQ stack
(`CHIRP_UQ_HEAD` p0/q heads, `modal_variance` exponential-integrator quadrature —
exact for constant poles, overflow-free; `return_variance` diagonal Eq.-7 readout;
`DIFF_LOSS_MODE="gaussian_nll"`; `models/uq_metrics.py` PIT/reliability/NLL;
`llapdiff-uq-eval` tool with `prepare_eval_stack` mirroring evaluate_checkpoint's
target-metadata preamble), U3 arm (`TRAIN_T_SAMPLER="max_only"` + `--mean-source
oneshot` eval), U1 (`llapdiff-u1-sweep` on val + clip_stats through
generate/evaluate_regression). 267 tests green; e2e UQ smoke passed (train+eval).
⚠️ NLL-from-scratch is a trap (smoke: latent RMSE 10.8, std 0.16, coverage ≈0 —
tiny init variance → huge mean gradients). Two-stage recipe documented prominently
in CMD_RUNBOOK.md §4: train chirp-MSE first, then CHIRP_UQ_HEAD + gaussian_nll +
DIFF_INIT_CKPT=<mse ckpt>. `_load_diff_init_state` (added 2026-06-25) tolerates
exactly the missing UQ-head keys — before that, DIFF_INIT_CKPT into a UQ model
crashed on strict load.

**Phase-3 code landed 2026-07-05:** T2 Theorem-B′ growth budget (`CHIRP_GROWTH_BUDGET`,
γ-excursion head with closed-form derivative, signed-increment variance quadrature),
T1 `llapdiff-t1-poles` (cross-regime trajectory distance + gap moments + Eq.-8
multipliers), T4 `llapdiff-t4-timing`. 276+ tests green.

🔴 **CRITICAL BUG found & fixed 2026-07-05 (via the T1 smoke):** `to_coeffs` was
zero-init AND squared → d(a²)/dW = 2a·h = 0 at a=0 → the pole-coefficient head got
EXACTLY ZERO gradient and never trained. Every chirp checkpoint trained before
2026-07-05 had frozen constant condition-independent poles — chirp-vs-lti comparisons
from those runs are meaningless; ALL chirp cells must be retrained. Fixed by eps-init
(std 1e-4) + `test_chirp_coeffs_receive_gradient_at_init` regression guard. Documented
in CMD_RUNBOOK.md §0 red box. (to_uq/to_growth/to_poles heads are alive — verified.)

**Phase-4 code landed 2026-07-05:** `CHIRP_PARAMETERIZATION` = p_exact (default) /
p_mono (monotone integrated poles, closed-form derivative, softplus heads — no
stationary trap) / p_grid (pointwise positive poles + trapezoid on the query grid,
the deliberate numerical-contrast ablation case). Near-LTI at init, ω-capped,
composes with growth/UQ heads, checkpoint metadata + setdefault. 290 tests green.
**ALL cmd_plan_v2 experiments (Phases 0–4) are now runnable from the combined branch.**

**§C method-doc corrections APPLIED 2026-07-05** to `chirp_modal_method_update.md`
(user re-posted it at the repo root): (1) §4.5 v_k = solver-free exponential-integrator
quadrature, closed form only for constant poles; (2) Thm-B remark now names the
LayerNorm output head as the genuinely uncertified term (the synthesis MLP is
spectral-normed) and the retained clamped scaling |s|<=1; (3) new "Implementation
notes (reproduction traps)" bullet: eps-init for squared coefficients (stationary
trap), window-scaled basis frequencies, NLL warm-start, diagonal readout of Eq. 7.

Remaining (non-code): user's GPU G2/G3 cells decide the cmd_plan_v2 §8 branch. All
cmd_plan_v2 code (Phases 0-4) is implemented; every experiment is runnable per
CMD_RUNBOOK.md. **Canonical compliance audit: repo-root `plan_v2_implementation.md`**
(2026-07-05) — verified matches table, 4 deliberate deviations, open gaps ranked by
paper impact. Gap 1 (H2 irregular renewal sampling) CLOSED 2026-07-05: Gamma gaps
with tunable Var(Δ)=mean²/shape, gap-aware discretization (regular = unit-gap special
case, bit-compatible), shared grid per cache, moments in meta, tool defaults gamma
m1/k4; GPU e2e smoke passed. Gap 2 (U2 data-space) CLOSED 2026-07-06:
`AnalyticLawSampler` duck-types LLapDiff inside the unchanged evaluate_regression
(N(mean,Var) draws, law cached per batch, one decode per draw) → analytic vs
sampled CRPS at matched ensemble/seed with wall-clock (`analytic_speedup_x`, 8.2×
in smoke); U3 three-way now apples-to-apples. Gap 3 (CHIRP_COEFF_L2) CLOSED
2026-07-06: coefficient_penalty (all 3 parameterizations, growth head excluded) →
diffusion_loss(coeff_l2) at the 2 training sites only; verified unit + e2e spy
(loss inflated by λ·penalty in real GPU training). NOTE: penalty is quadratically
small at ε-init — weight-absmax after 1 epoch is not a valid observable; it bites
as coefficients grow (soft constraint). Gap 4 (U1 p=0.995 clip check) CLOSED
2026-07-06: `--dynamic-thresh-p/--dynamic-thresh-max` on llapdiff-u1-sweep, p
recorded per row; first real data point on G3 cell-(d) ckpt: clip fraction 0.23%
at p=0.995 (z0 barely brushes the threshold post-head-removal). 299 tests green.
Still open: (5) T1 gap-blind arm + Fig 3/4/5 plotters (incl. data-space PIT
diagrams). Plus the x0-vs-v parameterization tension to decide.

Debug/lab numbers 0.469 and pre-fix 0.367 (PhysioNet CRPS) are excluded from paper tables
per the plan's table-hygiene rule.

🔴 **CRPS SINGLE-RUN COMPARISONS ARE NOT TRUSTWORTHY (2026-07-22).** The benchmark trains with
`set_torch(seed, deterministic=False)` → `cudnn.benchmark=True`, so runs are NOT reproducible at
fixed seed. Proof: the LTI arm has byte-identical config between `fig2_sweetspot_rhofix` and
`fig2_sweetspot_rhocap` (the rho cap is chirp-only; same LR/K/best_epoch=55) yet CRPS = 0.2243 vs
0.1886 — a **0.036 swing from pure nondeterminism, LARGER than every chirp-vs-LTI margin quoted**
(0.005 cap run, 0.040 init-fix run). ⇒ The earlier "chirp beats LTI by 18%" claim is RETRACTED;
no CRPS conclusion survives single-run evidence. **Any CRPS claim needs multi-seed.**
BUT a robust pattern: chirp CRPS range 0.1836–0.1869 (0.003) vs LTI 0.1886–0.2243 (0.036) —
**~12x lower variance for chirp**, echoing the earlier G3 finding ("chirp arms ~6x lower seed
variance"). Two independent observations ⇒ *stability*, not mean accuracy, may be the defensible
claim. RHO PROGRESS (structural, large effect, trustworthy): chirp osc-mode envelope
0.00-0.08 (dead) → **0.14-0.15 (alive)** with the cap; rho 300x → 17x truth.

🔴 **MULTI-SEED VERDICT (2026-07-23) — BOTH positive claims RETRACTED; Fig-2 tracking is a
confirmed negative.** Seeds {0,3,4} at the best config (centred rho, cap 1/H, sweet-spot,
converged, lr 5e-4), 5-seed campaign (1,2 pending):
  CRPS      chirp 0.1953±0.0097 vs lti 0.1970±0.0086 → diff +0.0017±0.0075 = NOT significant
  slope     chirp 0.0020±0.0013 (flat, tight across seeds) vs lti 0.0000
  w_eff     chirp 0.158±0.022 vs lti 0.182±0.028 (overlapping)
1. **CRPS win RETRACTED** — noise, as the nondeterminism caveat predicted.
2. **"chirp has ~12x lower variance" RETRACTED — my comparison error.** Those numbers came
   from runs differing in CONFIG (pre-fix/init-fix/cap/centred), not seeds; LTI varied more
   across config changes and I misread it as seed stability. At fixed config across seeds the
   **variance ratio is 0.9x** (equal). The G3 "~6x lower seed variance" was physionet h=12 and
   gets NO support here — do not cite it for H2.
3. **Fig-2 tracking = confirmed negative.** Slope ~0 across 5 benchmark designs (fm-2, fm-4,
   sweet-spot, period-96, period-32) with EVERY confound removed (Gate 0 0.85+, converged,
   K=8 identifiable, modes alive at 2-7x truth / envelope 0.36-0.73) and now across seeds.
   Architectural reason: poles are conditioned on the HISTORY summary only (see
   `w_docs/pole_rho_init_horizon_bug.md` §11), so they are a pure extrapolation and nothing
   in the forecasting loss forces them to match the true omega(t).
⇒ RECOMMENDATION: take the prereg failure branch for the tracking half. Keep certificate +
analytic UQ + Theorem D + the demonstrable "LTI structurally fails" half (3.5x recovery
degradation under a steeper sweep) + the two methodological findings (horizon-scaled rho;
history-only poles). **The rho init bug still requires re-running h=168 headline.**

✅ **FINAL 5-SEED AGGREGATE (2026-07-23, seeds 0-4 complete, best config: centred rho,
cap 1/H, sweet-spot difficulty, converged 400ep ES30, lr 5e-4).** Supersedes the 3-seed
interim above; conclusions unchanged and now fully powered.
  CRPS       chirp 0.1951±0.0069  vs  lti 0.1961±0.0068
             diff (lti-chirp) = +0.0010 ± 0.0043 se  → **NOT significant**
  pole slope chirp **-0.0032 ± 0.0113**  vs  lti 0.0000   (tracking gate: ≥0.6) → **FLAT**
  w_eff_rmse chirp 0.1737±0.0360 vs lti 0.1776±0.0262 → overlapping
  variance ratio (lti sd / chirp sd) = **0.98x** → no stability advantage (claim stays retracted)
⇒ Fig-2 tracking half: **negative, multi-seed confirmed, fully powered.** Both mid-session
positive claims (18% CRPS win; ~12x stability) remain retracted. Recommendation stands:
prereg failure branch for the tracking half; keep certificate + analytic UQ + Theorem D +
the demonstrable "LTI structurally fails" half + the two methodological findings
(horizon-scaled rho; history-only poles). Highest-value remaining work: **re-run h=168
NOAA-UK headline on the fixed rho init.**
