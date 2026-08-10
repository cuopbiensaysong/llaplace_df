# Handoff — nonlinear probe of the B21 conditioning gap  🔒 **COMPLETE 2026-08-03**

**Task discharged.** agent-B answered it; the result is `results_nonlinear_probe.md`, and the
findings are folded into `CMD_BUG_REPORT.md` (B21). **The answer was §2.2 branch 2: genuine
information LOSS** — an oracle-selected MLP reproduces the linear gap to 0.3 pt.

The full 248-line original brief is at
`archive/2026-08-uq-campaign/HANDOFF_nonlinear_probe.md`. It is kept here in compressed form
because the **template** is reusable, not because the task is open.

---

## The question it asked

Is the raw-history → `cond_summary` gap (28.6 % → ~14 %) information *loss*, or is the
information present but not linearly accessible? Replace ridge with an MLP and re-measure.

**The design element that made it answerable:** fit the same probe on the **raw history** too.
An MLP measured on `cond_summary` alone says nothing — the control is what turns it into
"how much does nonlinearity unlock *here* vs *there*". (That control also falsified the brief's
own premise that "an MLP will beat ridge on any representation": on this cell it does not.)

## What to reuse when writing the next handoff

- **State the decision rule before the numbers exist**, with numeric branches. This one had
  three, and the result landed cleanly in one.
- **Demand a reproduction check first.** "Do your ridge rows match 27.3 % and 7.7 %? If not, say
  so *before* interpreting anything" — a mismatch means the windows or metric differ and the
  whole comparison is void.
- **Name the traps the project has already paid for.** Each of these cost real time here:

  | trap | consequence |
  |---|---|
  | mean-pooling the summary tokens | −0.2 % vs 14.2 % — manufactures a false negative |
  | selecting a hyperparameter on val | 6.2 % → 9.8 %, across a decision threshold |
  | treating the latent as the yardstick | an intervention that *improves* CRPS *lowers* it |
  | single-seed CRPS claims | two byte-identical configs gave 0.2243 vs 0.1886 |
  | assuming `model_config` keys are top-level | they nest under `["llapdiff"]` |
  | trusting a commit message | `f52f843` says "K=128 for physionet"; the diff sets it on **noaa_uk** |

- **Ask for the negative space.** "Everything you could not establish" was the most useful
  section of the deliverable.
- **Write boundaries per path, agreed before any work.** See the red box in
  `agents_communication.md` — the shared stage-1/2 artifacts are the one surface a `--run-tag`
  does not protect.
- **Environment**: `PYTHONPATH=$LLAPDIFF_SRC` is not optional; verify with
  `python -c "import llapdiffusion; print(llapdiffusion.__file__)"` before anything else.
