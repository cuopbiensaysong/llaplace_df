# Archive — superseded working documents

These are the **original working documents**, kept verbatim as the raw record. They have been
consolidated into **`../CMD_BUG_REPORT.md`**, which is the document to read and maintain.

Nothing here should be updated. Consult it only when you need the primary source: the exact
wording of a finding, the chronological order in which things were discovered, or a detail the
consolidated report condensed.

| file | what it holds | where it went |
|---|---|---|
| `fig2_pole_basis_cannot_drift.md` | Root-cause report: the integer-cycle pole basis could not represent a swept ω, plus the companion recovery-metric defect | `CMD_BUG_REPORT.md` §1 (B9, B14) |
| `pole_rho_init_horizon_bug.md` | Root-cause report: horizon-blind ρ init (both cores), the uncapped ρ variation, and §11 the history-only pole-conditioning finding | `CMD_BUG_REPORT.md` §2 (B6, B7, B8) and §5 (the §11 design rule) |
| `CMD_SESSION_NARRATIVE.md` | The running project log to 2026-07-23: what was built, what broke, what was measured, which claims were retracted | `CMD_BUG_REPORT.md` §2 (B1–B5), §4 (results ledger + retractions), §5 (traps) |

## Read this before citing anything here

1. **These files predate the 2026-07-24/25 findings.** In particular
   `fig2_pole_basis_cannot_drift.md` §6 and `pole_rho_init_horizon_bug.md` §11 present the
   history-only pole conditioning as *the* remaining obstacle. That is no longer the leading
   explanation: the denoiser was later measured to barely use its conditioning at all
   (linear probe R² **+0.747** vs the trained model's **−0.30**), which is upstream of the
   pole head entirely. See `CMD_BUG_REPORT.md` §3.
2. **`CMD_SESSION_NARRATIVE.md` contains claims that were later retracted**, including the
   "confirmed negative" 5-seed Fig-2 verdict, an 18% CRPS win, and a ~12× stability
   advantage. Its own header warns that later entries correct earlier ones. The retraction
   record lives in `CMD_BUG_REPORT.md` §4 — read that first.
3. The **implementation history** (Phases 0–4, what landed when, the G2/G3 branch decision)
   survives only in `CMD_SESSION_NARRATIVE.md`. `CMD_BUG_REPORT.md` §4 keeps the G2/G3
   numbers and the branch decision, but not the full phase-by-phase chronology — that is the
   main reason this archive exists rather than a deletion.
