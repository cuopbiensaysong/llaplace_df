# w_docs — what is where

Reorganised 2026-08-18, when the campaign moved from one dataset to several.

> **Continuing this work?** Read `HANDOFF_2026-08-19.md` first. It carries the live state,
> what is running, the immediate next step, and the corrections that are not yet folded into
> the per-cell documents.

> **Latest results: `CAMPAIGN_RESULTS_2026-08-21.md`.** Every measurement since the
> 2026-08-19 handoff, for Tables 1-3. It supersedes the Family A numbers in
> `noaa_uk_h168/CMD_RESULTS_FOR_PAPER.md` and the cross-cell table in the handoff.

## Cell-specific

```
noaa_uk_h168/
  PREREG_U3_noaa_uk_h168.md   the pre-registration; §6.1 holds the frozen power table,
                              §11 the freeze record (6 of 8 boxes ticked)
  CMD_RESULTS_FOR_PAPER.md    every experimental result for the paper, self-contained
  CMD_PHASE1B_GATE.md         the Phase-1b latent gate read
finance/
  FINANCE_CELLS_BLOCKED.md    crypto + us_equity: PARKED at stage 1, with the measurements
ushcn_24_12/
  USHCN_RESULTS.md            the only genuinely irregular cell; Family A complete on val,
                              chirp_gp wins VS 8/8. Artifacts live OUTSIDE the checkout in
                              fixed_bugs/_ushcn_experiments/.
```

## Global — apply to every cell

| file | what it is |
|---|---|
| `CMD_BUG_REPORT.md` | the bug ledger B1–B23. Read before changing anything. |
| `CMD_PHASE0_WINDOW_AUDIT.md` | window counts and the stop-condition verdicts per cell |
| `CMD_UQ_RECOVERY_PLAN.md` | standing rules and the U3 phase spec |
| `CMD_UQ_TABLE3_PLAN.md` | plan of record for Table 3 |
| `PREREG_U3.md` | superseded, PhysioNet-scoped; retired by B19, kept as the record |
| `DEVELOPER_GUIDE.md`, `USAGE.md` | the codebase |
| `results_nonlinear_probe.md`, `HANDOFF_nonlinear_probe.md` | the B21 conditioning probe |

## Adding a cell

Put its pre-registration and its results write-up under `w_docs/<dataset>_h<pred>/`. Keep bug
findings in the global ledger — they are almost never cell-specific, and B20 is the cautionary
case: its mechanism is real on the noaa cells and turned out **not** to be the binding
constraint on the finance cells, which is only visible if both live in one document.

Results and locks follow the same `<dataset_key>_h<pred>` convention; see
`finetuning/results/README.md`.
