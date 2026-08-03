> **Tracked canonical copy.** The working copy at `finetuning/results/window_audit.md` is under `.gitignore:46`
> (`finetuning/results/*`) and does not survive a fresh clone. Edit **this** file; mirror to the
> results path only if a tool reads it there. Copied 2026-08-03.

# Window-count audit — Phase 0 of `w_docs/CMD_UQ_RECOVERY_PLAN.md`

**Measured 2026-07-31** on `main` @ `f1788e3` (post `speed_up` merge). Read-only: loaders were
built and iterated, nothing was trained or written.

## Results

| cell | WINDOW | PRED | ch | **train_w** | **val_w** | **test_w** | ent/win | supervision scalars | vs 10.2 M params |
|---|---|---|---|---|---|---|---|---|---|
| physionet h12 | 24 | 12 | 16 | **13** | **5** | **5** | 445 | 2 496 | 4 000× over |
| crypto h100 | 200 | 100 | 16 | 1 225 | **90** | 379 | 118 | 1.96 M | 5.2× over |
| us_equity h100 | 200 | 100 | 12 | 1 099 | **72** | 343 | 221 | 1.32 M | 7.7× over |
| noaa_uk h168 | 336 | 168 | 16 | 16 286 | 2 183 | 4 702 | 4 | 43.8 M | **4.3× under** |
| bms_air h168 | 336 | 168 | 24 | 95 099 | 17 598 | 33 449 | 12 | 383 M | **37× under** |

### 🔴 The `ent/win` column is a panel size, not a typical window (corrected 2026-08-03)

Three different numbers have been quoted for bms_air's entities-per-window, and only the last
is what a probe or a pooling argument actually sees:

| source | bms_air | what it measures |
|---|---|---|
| this table's `ent/win` | **12** | the panel size — the maximum, not a typical window |
| B20's `gate0_diag` | 7 | mean over *scored* windows |
| **modal window** (agent-B, 2026-08-03) | **1** | what most windows actually contain |

Measured distribution over eligible windows — bms_air train
`{1:30359, 2:2502, 3:1089, 4:873, 5:1572, 6:2519, 7:2427, 8:2596, 9:3062, 10:3001, 11:3430, 12:4039}`,
val `{1:5855, 2:309, 3:13, 4:48, 5:360, 6:599, 7:609, 8:747, 9:543, 10:308, 11:28, 12:23}`.
Against noaa_uk, whose panel is genuinely full: train `{4:14446, 3:1696, 2:120, 1:24}`, val `{4:2183}`.

**Consequences.** (i) bms_air is *not* "12 entities pooled" — **most of its windows have nothing
to pool**, so it is a much weaker test of the B5/B20 entity-pooling axis than this table implied,
and the "best cells on the pooling axis (4, 12)" framing overstates it. (ii) Only **60 %** of
bms_air windows are eligible at all (they must have a fully-observed target), against 100 % on
noaa_uk. (iii) A full-panel comparison against noaa_uk is infeasible: **23** twelve-entity val
windows against noaa_uk's 2 183.

noaa_uk's `ent/win = 4` is unaffected — there the panel size and the modal window coincide.

**Addendum 2026-08-03 — the mode-1 reading is HORIZON-specific, not a property of bms_air.**
At **h=24** bms_air's modal window is **12**, the full panel. The fully-observed-target filter
over 24 steps is far weaker than over 168, so mode-1 at h=168 is an artefact of demanding 168
consecutive fully-observed steps. The full-panel comparison that was infeasible at h=168
(23 val windows) is well-powered at h=24 (13 514 train / 1 372 val). So the correct statement is
*"at h=168 the eligibility filter leaves mostly single-entity windows"*, not *"bms_air has one
entity per window"*.

`supervision scalars = train_w × PRED × ch` — the size of the denoiser's target space. Entities
do **not** enter it: the set-VAE mean-pools the entity axis into one latent per window. That is
why 445 entities did not rescue PhysioNet, and it is the same pooling that caused **B5**.

### 🔴 Correction 2026-08-02 — the 10.2 M column applies to ONE cell, not five

`LAPLACE_K` is a per-dataset preset field. Only **noaa_uk** carries `laplace_k=128`
(`dataset_defaults.py:153`); every other cell falls through to the default **256**. Measured
directly on arm `d` (chirp − head, x0):

| cell | resolved `LAPLACE_K` | arm-`d` params | supervision scalars | corrected ratio |
|---|---|---|---|---|
| physionet h12 | 256 | 11.75 M | 2 496 | **4 708× over** (was "4 000×") |
| crypto h100 | 256 | 11.75 M | 1.96 M | **6.0× over** (was 5.2×) |
| us_equity h100 | 256 | 11.75 M | 1.32 M | **8.9× over** (was 7.7×) |
| **noaa_uk h168** | **128** | **10.17 M** | 43.8 M | **4.3× under** ✓ unchanged |
| bms_air h168 | 256 | 11.75 M | 383 M | **32.6× under** (was 37×) |

**Every stop-condition verdict is unchanged**; only the exact ratios move. The gate cell's
number — the one Phase 1 rests on — was already right.

**Provenance, resolved 2026-08-02.** The `k_probe` behind `laplace_k=128` was run on
**noaa_uk**, so the code applies it to the cell it was measured on and Phase 1's
"under-parameterized ⇒ a failure is attributable to modelling" argument holds.
⚠️ Commit `f52f843`'s message says "adopt K=128 for **physionet**" and describes the probe as
a physionet result — that is wrong in the message only; its diff correctly edits the noaa_uk
preset. **PhysioNet still runs at K=256.** Do not cite that commit message.

B19's "11.5 M" was a physionet figure at `K=256` and is consistent with the 11.75 M measured
here; the conclusion is unchanged either way.

## Stop-condition verdicts

| cell | < 500 train_w | < 500 val_w | over-parameterized | verdict |
|---|---|---|---|---|
| physionet h12 | **FAIL** | **FAIL** | **FAIL** | smoke test only — no statistical claim |
| crypto h100 | pass | **FAIL** | **FAIL** | cheap pre-check only; not a decision point |
| us_equity h100 | pass | **FAIL** | **FAIL** | not usable for the gate |
| noaa_uk h168 | pass | pass | pass | **gate dataset** |
| bms_air h168 | pass | pass | pass | cross-check dataset |

## 🔴 Count window starts, not `sizes`

`run_experiment` returns `sizes = (len(ds_tr), len(ds_va), len(ds_te))`
(`llapdiffusion/datasets/fin_dataset.py:2166`) — **dataset items, not window starts.** They are
~4× larger and the ratio is not constant:

| cell | true train_w | `sizes[0]` | ratio |
|---|---|---|---|
| physionet h12 | 13 | **2 087** | 161× |
| crypto h100 | 1 225 | 143 445 | 117× |
| us_equity h100 | 1 099 | 242 879 | 221× |
| noaa_uk h168 | 16 286 | 63 136 | 3.9× |
| bms_air h168 | 95 099 | 284 250 | 3.0× |

**PhysioNet's `sizes[0]` = 2087 passes the < 500 stop condition.** Auditing the obvious quantity
clears the one cell the audit exists to reject.

## Reproduce

```python
# per cell: clone_config -> apply_dataset_preset -> resolve_run_experiment
w = 0
for batch in loader:
    w += batch[0][0].shape[0]      # axis 0 = window starts, axis 1 = entities
```

`batch[0]` is a `(V, T)` tuple, not a tensor — `torch.as_tensor(batch[0])` raises
`ValueError: only one element tensors can be converted to Python scalars`.

Environment per `CLAUDE.md`: activate the venv and export
`PYTHONPATH=$LLAPDIFF_SRC`, or `llapdiffusion` resolves to a sibling checkout.
