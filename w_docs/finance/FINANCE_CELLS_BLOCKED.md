# crypto and us_equity — blocked at stage 1, and why

**Status 2026-08-18: PARKED.** Neither cell should be taken into diffusion training as it
stands. Everything below is measured, on **val**, and reproducible from the commands at the end.
Nothing here is deleted — the artifacts trained during this investigation are on disk and the
work resumes from them.

---

## 1. The verdict in one table

| cell | context | pred | train_w | **val_w** | test_w | ent/win | params | supervision | Gate 0 round-trip |
|---|---|---|---|---|---|---|---|---|---|
| crypto | 200 | 100 | 1 225 | **90** | 379 | 118 | 11.75 M | 1.96 M | **0.624** |
| us_equity | 200 | 100 | 1 099 | **72** | 343 | 221 | 11.75 M | 1.32 M | **0.690** |
| *noaa_uk (reference)* | *336* | *168* | *16 286* | *2 183* | *4 702* | *4* | *10.17 M* | *43.8 M* | *0.9952* |

Two independent blockers, either of which is on its own sufficient.

---

## 2. Blocker A — statistical power, at every horizon

The Phase-0 audit rejected both cells at h=100 (val windows below the 500 stop condition, model
over-parameterised). The natural hope was that a shorter horizon would rescue them, the way
bms_air's mode-1 problem at h=168 disappeared at h=24. **Measured across all four horizons the
presets offer, it does not:**

| cell | h | train_w | **val_w** | supervision scalars | vs 11.75 M params |
|---|---|---|---|---|---|
| crypto | 5 | 1 386 | 194 | 110 880 | 106× over |
| crypto | 20 | 1 361 | 178 | 435 520 | 27× over |
| crypto | 60 | 1 293 | 134 | 1 241 280 | 9.5× over |
| crypto | 100 | 1 225 | **90** | 1 960 000 | **6.0× over** |
| us_equity | 5 | 1 260 | 176 | 75 600 | 155× over |
| us_equity | 20 | 1 235 | 160 | 296 400 | 39.6× over |
| us_equity | 60 | 1 167 | 116 | 840 240 | 14× over |
| us_equity | 100 | 1 099 | **72** | 1 318 800 | **8.9× over** |

The best val count available anywhere is **194**, against a threshold of 500. And a shorter
horizon makes the parameter ratio *worse* (106× against 6×), because the window count barely
moves while the supervision target shrinks. **h=100 is the least bad choice on both axes**, and
it is what the rest of this document uses.

Note this limits what can be claimed, not whether a run is possible. Seeds — not windows — are
the unit of the pre-registered tests, so a Wilcoxon read is still *computable*; each seed's
score would simply rest on 72–90 windows.

---

## 3. Blocker B — stage 1 cannot represent the target, and we now know it is not the pooling

Gate 0 measures how well the frozen stage-1 VAE reconstructs its own targets. It is a ceiling
on everything downstream: the denoiser is asked to forecast a latent, and whatever the latent
cannot represent is unreachable regardless of the model. Threshold 0.85.

### B20's fix barely moves these cells

| cell | shipped VAE | entity-encoded (B20) | Δ |
|---|---|---|---|
| crypto h100 | 0.617 | 0.624 | **+0.007** |
| us_equity h100 | 0.643 | 0.690 | **+0.047** |
| *noaa_uk h168* | *0.9222* | *0.9952* | *+0.073* |
| *noaa_us h168* | *0.669* | *0.800* | *+0.131* |

🔴 **This falsifies B20's stated prediction.** That entry claims the gain is larger where the
panel is wider, from +0.073 at 4 entities and +0.131 at 40. crypto has **118** entities per
window and us_equity **221** — far wider — so the predicted gains were above +0.131. The
measured gains are the two smallest in the table. **Permutation invariance is not what caps
these cells.** B20's mechanism is real on the noaa cells and is not the binding constraint here.

### Nor is it latent capacity

The surviving hypothesis was width: the preset channel counts were chosen when the encoder could
not tell entities apart, and `channels per entity` tracked the round-trips closely
(noaa_uk 4.000 → 0.9952; noaa_us 0.600 → 0.800; crypto 0.136 → 0.624; us_equity 0.054 → 0.690).
Measured directly:

| cell | ch | ch/entity | round-trip | std_ratio | gate |
|---|---|---|---|---|---|
| crypto | 16 | 0.136 | 0.624 | 0.81 | FAIL |
| crypto | 48 | 0.407 | **0.596** | 0.83 | FAIL |
| crypto | 96 | 0.814 | **0.603** | 0.94 | FAIL |
| us_equity | 12 | 0.054 | 0.690 | 0.73 | FAIL |
| us_equity | 48 | 0.217 | **0.714** | 0.77 | FAIL |
| us_equity | 96 | 0.434 | **0.683** | 0.74 | FAIL |

Widening crypto **6×** moves the round-trip from 0.624 to 0.603 — *down*. us_equity peaks at
0.714 and falls again at 96. The curve is flat inside noise across a 6–8× range of width.
**Capacity is not the constraint either.**

### What the `std_ratio` column says the constraint actually is

This is the informative part. As the latent widens, `std_ratio` **rises** (crypto 0.81 → 0.94)
while the correlation does not. The wider code reconstructs more of the target's *amplitude*
and none of its *structure* — the extra capacity is spent encoding variance uncorrelated with
the target, i.e. noise.

That is the signature of a **rate/SNR** limit rather than an architectural one. On financial
return series most of the variance is unpredictable, and the VAE's KL term makes that component
expensive to encode, so it is discarded. Widening the code does not change what is worth
encoding.

**Do not "fix" this by lowering `--vae-beta`.** That would loosen the bottleneck until noise
fits through and the round-trip clears 0.85 — passing the gate by defeating its purpose. The
gate exists to certify that the latent carries the target's *signal*.

---

## 4. Consequence, and the rule already on the books

B20 states it directly: *"any crypto/us_equity result is bounded by a ~0.66 stage-1 round-trip,
i.e. the denoiser is being asked to forecast a latent that cannot represent half the target. A
'ties baseline' outcome there is not attributable to the model."*

That is now measured rather than estimated, and it survives both repairs. A diffusion seed costs
≈26 h. Spending it here buys a number that cannot be interpreted either way: a win would be
suspect and a tie would be unattributable.

**Decision: parked.** Not abandoned — see §6.

---

## 5. 🔴 CORRECTED 2026-08-19 — crypto is not irregular, and the first measurement was wrong

**An earlier version of this section claimed us_equity was genuinely irregular and crypto
supplied missingness. Half of that was an artefact and is retracted here.**

The time tensors are `[B, N_entity, T]` with inactive entity slots zero-filled. The first pass
flattened them without the entity mask, so on cells where most slots are padding the statistics
described the padding, not the data: crypto's "gaps 0.0–1.0" and "grid deviation 100.0", and its
"observed fraction 0.946", were all slot occupancy. Re-measured on **active entity rows only**:

| cell | gap min | gap max | distinct | mean gap | share of gaps = 1 | grid deviation between windows |
|---|---|---|---|---|---|---|
| **us_equity h100** | 1.0 | 4.0 | 4 | **1.458** | **78.2 %** | 6.0 |
| bms_air h168 | 1.0 | 18.0 | 4 | 1.011 | 99.76 % | 20.0 |
| **crypto h100** | 1.0 | 1.0 | **1** | 1.000 | **100 %** | **0.0** |
| noaa_uk h168 | 1.0 | 1.0 | 1 | 1.000 | 100 % | 0.0 |

* **crypto is exactly as regular as noaa_uk** — every gap is 1.0, every window's grid identical.
  It contributes nothing on the irregularity axis and nothing on missingness. The claim that it
  did was a measurement error.
* **us_equity is the only genuinely irregular cell in the project.** Only 78 % of its gaps are
  unit; the mean gap is 1.458 and the calendar produces 2, 3 and 4. This survives the
  correction.
* **bms_air is 99.76 % regular.** Its max gap of 18 is real but rare, and its value on this axis
  is the *variable lead gap* (grid deviation 20 against noaa_uk's 0), not irregular spacing.

### The consequence is uncomfortable and should be stated in the paper

| | passes Gate 0 | genuinely irregular |
|---|---|---|
| noaa_uk | ✅ | ❌ |
| bms_air | ✅ | ❌ (99.76 % unit gaps) |
| us_equity | ❌ (0.690) | ✅ |
| crypto | ❌ (0.624) | ❌ |

**No cell in the project is both usable and genuinely irregular.** The one cell that exercises
the paper's central capability is the one whose stage 1 cannot represent its target. That is a
stronger reason to unblock us_equity than anything in §3, and it also means §10's framing
problem is *not* solved by bms_air.

The zero-width-gap concern raised earlier is withdrawn for the same reason: with padding
excluded, the minimum gap on every cell is exactly 1.0. **No cell feeds a zero-width segment to
the anchored-variance path**, so the `log(0) → NaN` guard is not exercised by real data here.

## 6. If this is picked up again, in order

1. **Establish an SNR baseline before any modelling.** Fit a train-mean and a ridge on raw
   history for these cells and report variance explained. If the forecastable share is a few
   percent, no stage-1 repair will make the round-trip meaningful and the cells should be
   reported as data-side evidence only (§5).
2. **Score the round-trip against the *predictable* component, not the raw target.** The current
   gate asks the VAE to reconstruct noise it is right to discard. A gate conditioned on the
   predictable part would say whether stage 1 is losing anything that matters.
3. **Only then consider a wider or lower-`β` code**, with the caveat in §3.
4. Do **not** re-run the width sweep — §3 covers 6–8× and is flat.

## Artifacts already on disk (do not delete)

```
ldt/vae/saved_model/crypto/pred-100_ch-16_entity_entenc_{elbo,recon}.pt
ldt/vae/saved_model/crypto/pred-100_ch-{48,96}_entity_entenc_{elbo,recon}.pt
ldt/vae/saved_model/us_equity/pred-100_ch-12_entity_entenc_{elbo,recon}.pt
ldt/vae/saved_model/us_equity/pred-100_ch-{48,96}_entity_entenc_{elbo,recon}.pt
finetuning/results/finance/          # every Gate 0 json/log quoted above
```

The shipped (non-`entenc`) artifacts were not touched; the channel count and the `entenc` tag
are both part of the filename, so nothing overwrote anything.

## Reproduce

```bash
source /vol/dl-nguyenb5-solar/users/cuopbiensaysong/llaplace/bin/activate
export LLAPDIFF_SRC=/vol/dl-nguyenb5-solar/users/cuopbiensaysong/fixed_bugs/llaplace_df
export PYTHONPATH=$LLAPDIFF_SRC

# Gate 0 at any width (--latent-channels selects the artifact AND builds the model to match;
# without it the width comes from the preset and the strict load fails)
python -m llapdiffusion.tools.run_stage1_roundtrip \
  --dataset-key us_equity --pred 100 --split val --max-batches 200 \
  --vae-entity-encode --latent-channels 48

# the whole width sweep, both cells, with Gate 0 after each
bash finetuning/latent_width_sweep.sh
```

Always read `n` and the skipped-window count beside `corr`, per B20's audit rule: crypto drops
6 constant windows of 10 112, us_equity none of 15 912, so neither number is a filtering
artefact.
