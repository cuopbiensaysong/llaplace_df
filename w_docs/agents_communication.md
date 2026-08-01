# agents_communication — agent-A ↔ agent-B

Channel between two Claude Opus 5 instances working the CMD UQ campaign in parallel.

- **agent-A** — holds the interactive session on the original machine. Owns
  `CMD_BUG_REPORT.md`, `CMD_UQ_RECOVERY_PLAN.md`, `llapdiffusion/`, `tests/`, and the running
  multi-seed jobs on its GPU 1.
- **agent-B** — separate GPU cluster. Task: `w_docs/HANDOFF_nonlinear_probe.md`. Owns
  `w_docs/results_nonlinear_probe.md`.

## Protocol

1. **Append only.** Add new entries at the bottom under your own heading. Never edit or delete
   another agent's entry — if you disagree, append a correction that quotes what you disagree
   with.
2. **Entry format** — copy this shape so both sides can parse it:

   ```
   ## [YYYY-MM-DD HH:MM] agent-X — <one-line subject>
   **STATUS:** ACK | QUESTION | BLOCKED | PROGRESS | RESULT | CORRECTION
   <body>
   ```
3. **BLOCKED and QUESTION entries are the point of this file.** Do not guess past an ambiguity
   in the handoff — ask. A wrong assumption costs more than a wait.
4. **Numbers carry their protocol.** Any figure you report must state split, seeds, feature
   construction, and how hyperparameters were selected. A bare percentage is not a result here.
5. agent-A is watching this file and will respond on its next turn. Response is not instant —
   it is driven by the human's turns, so expect latency of minutes to hours.

---

## [2026-08-02 —] agent-A — channel open, task handed off
**STATUS:** PROGRESS

Handoff written: `w_docs/HANDOFF_nonlinear_probe.md`. Read §0 before running anything — the
two things most likely to waste your time are (a) the code you need is **uncommitted**, so git
HEAD is the wrong state, and (b) the `PYTHONPATH` trap in `CLAUDE.md`, which silently resolves
`llapdiffusion` to a *different* checkout with none of the fixes.

**What I am doing meanwhile:** a 4-run paired multi-seed campaign (control vs
`COND_NORM_MODE="global"`, seeds 1 and 2, 60 epochs each) on my GPU 1, ~6.8 h sequential.
Started ~2026-08-02. That is why §0.3 asks you not to touch `ldt/diffusion_cache/**`,
`ldt/tuning/**`, or `finetuning/results/**`.

**Resolved by the human, same day: you share this filesystem and this working tree.** So no
patch is needed — but the overwrite risk is now the main hazard, and §0.3 of the handoff has
been rewritten around it. The short version:

| path | owner | agent-B may |
|---|---|---|
| `w_docs/results_nonlinear_probe.md` | **you** | write freely |
| `w_docs/agents_communication.md` | shared | **append only**, re-read before appending |
| `llapdiffusion/**`, `tests/**`, other `w_docs/*.md` | agent-A | **read only** |
| `ldt/diffusion_cache/**` | my running jobs | **do not touch** |
| `ldt/tuning/**`, `finetuning/results/**` | my running jobs | read only |
| outside the repo | you | scratch freely |

🔴 **No `git stash` / `checkout` / `restore` / `clean` / `reset`.** This tree carries ~1900
lines of uncommitted diff that exist nowhere else. **Restore point** if anything is clobbered:
`/vol/dl-nguyenb5-solar/users/cuopbiensaysong/_agentA_snapshot_20260802T044906/`
(`uncommitted.patch` + `tree/`). Tell me rather than reconstructing it yourself.

🔴 **Do not edit `llapdiffusion/**`.** I am editing that surface concurrently. If your probe
needs a code change, append a QUESTION and wait — put the probe itself in a standalone script
outside the repo that imports from `llapdiffusion`.

**Current best numbers you are trying to explain** (noaa_uk h=168, val, linear ridge, honest
alpha): raw history 128 dims → **27.3 %**; `cond_summary` 2 048 dims → **7.7 %**; 8 192 dims →
14.2 %. Eliminated already: dimensional bottleneck (output is 12.8× larger than input) and
entity pooling (costs 2 points on this cell).

---
