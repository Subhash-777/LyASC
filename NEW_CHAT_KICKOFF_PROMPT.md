# New-Chat Kickoff Prompt — LyASC Implementation

Paste this into the new chat, with the files listed in Section A attached.
This prompt is fully self-contained — it does not assume the new chat has
any memory of prior conversation. Do not treat anything not explicitly
stated here or verifiable in an attached file as established fact.

---

## A. Files to attach to this new chat

1. **`base_model/plan.md`** — the completed base-paper reproduction's
   governing blueprint. Read this first for full context on what has
   already been built, tested, and verified.
2. **`base_model/results/logs/decisions.log`** — the complete, authoritative
   record of every bug found and fixed during the base-model reproduction
   (LayerNorm-axis bug, weight-decay-scope bug, the `wSkip+LN` two-attempt
   investigation, the report-fabrication incident and its correction, the
   verdict-logic bug and its correction). Treat this as ground truth over
   any summary, including this prompt.
3. **`base_model/results/FINAL_REPORT.md`** (the corrected version, 14/32
   PASS under the symmetric ±0.4pp rule) — the final, verified base-model
   results.
4. **`base_model/results/tables/reproduced_table1_FINAL.csv`,
   `reproduced_table3_FINAL.csv`, `reproduced_table4_FINAL.csv`,
   `reproduced_table6_FINAL.csv`** — the exact numeric baselines every
   LyASC result will be compared against.
5. **`LyASC_Revised_Blueprint_v2.docx`** — the journal-project narrative:
   concept, novelty argument, and how the base-model reproduction's
   findings (especially `wSkip+LN`'s failure) motivate LyASC's design.
6. **`LyASC_ARCHITECTURE.md`** — the exact technical specification of
   `LyASCSkip` and `StabilityEstimator`, including explicitly-flagged open
   design questions (Section 8) that are NOT yet resolved.
7. **`LyASC_plan.md`** — the implementation blueprint: workspace structure,
   acceptance criteria, step-by-step build order, and the "what NOT to do"
   rules carried over from the base-model project's hard-won lessons.
8. **`base_model/src/models/skip_variants.py`, `preact_resnet.py`,
   `base_model/src/training/train.py`, `config.py`** — the actual, current
   source code `LyASCSkip` must integrate with. Read these before writing
   any new code; do not assume any interface not directly visible in these
   files.
9. *(Optional but recommended)* `Base_paper.pdf` — the original paper
   (Liu et al., *Rethinking Skip Connection with Layer Normalization in
   Transformers and ResNets*) for direct citation fidelity if the journal
   submission needs precise quotes or equation references.

---

## B. Mission

Implement and evaluate LyASC (`LyASCSkip` + `StabilityEstimator`) per
`LyASC_ARCHITECTURE.md` and `LyASC_plan.md`, building on top of the
already-complete, already-verified base-model reproduction. The base-model
work is DONE — do not re-run, re-verify, or second-guess its results
unless a specific inconsistency is found in an attached file (in which
case, follow the same investigate-before-fixing discipline documented in
`decisions.log`, don't just assume something is wrong).

---

## C. Absolute ground rules — carried over from the base-model project, non-negotiable

1. **No fabricated results.** Every number reported must trace to an
   actual result JSON on disk. If no such file exists, the correct
   response is "this hasn't been run yet," never an invented plausible
   number. This exact failure mode already happened once during the
   base-model project (a fabricated Table 4 section) and was only caught
   by manual cross-verification against raw JSONs — do not repeat it.
2. **No silent assumptions on open design questions.** `LyASC_ARCHITECTURE.md`
   Section 8 lists 6 explicitly unresolved decisions. Each must be
   deliberately decided, justified in one sentence, and logged via
   `log_decision(...)` before being implemented — not defaulted to
   silently.
3. **Validate cheaply before scaling up.** One seed, checked for NaN/Inf
   and basic sanity, before any 3-seed batch. This is not optional
   process overhead — it is what prevented multiple multi-hour wasted
   training runs throughout the base-model project.
4. **A stable-but-unremarkable or negative result is a valid, reportable
   outcome**, not a failure to fix. Do not reframe, re-tolerance, or
   selectively re-run to manufacture a more favorable-looking number.
5. **Acceptance criteria are exactly as defined in `LyASC_plan.md` Section
   0** — comparison against the specific verified base_model baselines
   listed there, not against an invented external target.
6. **Reuse, don't rebuild.** The base_model backbone, training loop,
   gradient clipping, weight-decay-exclusion logic, and seeding are
   already correct and tested — integrate with them per
   `LyASC_ARCHITECTURE.md` Section 3.4, do not reimplement them.

---

## D. First actions in the new chat, in order

1. Read all attached files in full, in the order listed in Section A.
2. Confirm understanding by summarizing, in your own words: what
   `wSkip+LN`'s reproduced failure demonstrated, and why it is the direct
   competitor baseline for `LyASCSkip` (this checks that the motivation
   from `LyASC_Revised_Blueprint_v2.docx` Section 6 has actually been
   absorbed, not skimmed).
3. Begin at `LyASC_plan.md` Section 3, Step 1 (unit tests first) — do not
   skip ahead to training code before the estimator's correctness is
   verified against a hand-checkable test case.
