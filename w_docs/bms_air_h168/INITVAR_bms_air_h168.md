# `CHIRP_UQ_INIT_VAR` for bms_air h=168 — measured 2026-08-19

**Value passed to s2/s3: `LLAPDIFF_CHIRP_UQ_INIT_VAR=1.64`.**

Measured on **seed 2's** `s1_mean` checkpoint (`805f27a425`, trial dir under
`ldt/tuning/uq_u3/bms_air_h168/d/`). This is a property of the cell, not the seed, so one
seed suffices. Raw JSON + the driver script: `finetuning/results/uq_u3/bms_air_h168/d/initvar/`.

## Protocol

```
python -m llapdiffusion.tools.run_analytic_uq_eval \
  --dataset-key bms_air --pred 168 --checkpoint <s1 seed2 best_ema> \
  --weights ema --split val --latent-only \
  --vae-entity-encode --vae-ckpt ldt/vae/saved_model/bms_air/pred-168_ch-24_entity_entenc_elbo.pt \
  --guidance 1.0 --guidance-power 0.3 --mean-source {oneshot,ddim}
```

`--guidance 1.0 --guidance-power 0.3` matches what the campaign's own `eval_one` passes
(`finetuning/run_u3_uq.py:422`); without it `_sampling_kwargs` resolves the shipped `(1.0, 2.0)`
CFG ramp, which sharpens the law. The console script `llapdiff-uq-eval` is the sibling-checkout
hazard — use `python -m`.

## Result — full val split, all 70 083 456 latent elements

| read | `latent_mse` | predict-mean baseline | `latent_rmse` | `latent_corr` |
|---|---|---|---|---|
| `oneshot`, full val | 0.9331 | 0.8881 | 0.9660 | +0.0220 |
| **`ddim`, full val** | **1.6387** | 0.8881 | 1.2801 | +0.0143 |
| `oneshot`, first 200 batches | 1.2441 | 1.1888 | 1.1154 | +0.0011 |

`ddim` is the mean source that produced noaa_uk's value of record, so **1.64** is the
protocol-matched choice. `oneshot` would give 0.93. Both are within 2x of noaa_uk's 1.0, so the
26x cross-cell spread the guard exists to catch is not in play on this cell.

## Two things to know about the comparison to noaa_uk

1. **noaa_uk's 1.0 came from `finetuning/results/table3/initvar_entenc.json`**: `latent_mse`
   0.8236, `--mean-source ddim`, **967 680 elements (~1/6 of val)**, at the default `(1.0, 2.0)`
   CFG ramp — it was run 2026-08-08 11:34, before the D2 guidance fix landed in `eval_one`.
   bms_air's number above is full val at guidance 1.0. The protocols are therefore *not*
   identical; the mean source is.
2. **`--max-batches` truncation is biased here.** The date-batching loader does not shuffle
   (`dataloader_kwargs` docstring), so a capped read is a contiguous *early*-val chunk, not a
   sample. Measured on this cell: an 8-batch read reported `latent_target_std` 1.77 and
   `latent_corr` **-0.098**; the full split reports 0.94 and **+0.022**. Do not read a capped
   number as a split estimate. noaa_uk's value of record is a capped read.

## The finding that is not about init var

`latent_corr` is **+0.014 to +0.022** on this cell against **+0.363** for noaa_uk's equivalent
read: the s1 diffusion mean carries essentially no forecast signal on bms_air h=168, and
`latent_mse` (0.9331 oneshot) is 5.1% *worse* than predicting the val mean. Worse-than-baseline
is a documented property of this pipeline (the Phase-1b read on noaa_uk was also worse than
baseline), but the near-zero correlation is specific to this cell.

This does not contradict Family A's `chirp_gp` VE of +2.67% — different model family, and not
this quantity. It does mean Table 3's (`tab:decomp`) bms_air row will be measured on a cell
whose conditional mean is close to uninformative, which needs saying wherever that row is read.
