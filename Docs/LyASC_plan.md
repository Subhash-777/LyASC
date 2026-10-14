# plan.md — LyASC Implementation Blueprint

**Workspace (new, sibling to base_model/):** `LyASC/lyasc_model/`
**Depends on:** `LyASC/base_model/` (complete, verified, read-only reference
— reuse its checkpoints, results, and code conventions; do not duplicate or
retrain any of its baselines).
**Status:** implementation not yet started. This document is the plan, not
a record of results. No claim in this document should be read as a
completed or measured outcome unless explicitly marked "verified
(base_model)".

---

## 0. Scope, Goal, and Acceptance Criteria

**Goal:** implement and evaluate `LyASCSkip` (per `LyASC_ARCHITECTURE.md`)
on the same architecture, datasets, and protocol as the base-model
reproduction, so every LyASC result is directly comparable to an existing,
verified number.

**Scope for Phase 1:** CIFAR-10, PreAct-ResNet-110, matching
`base_model/configs/cifar10_table1.yaml`'s protocol exactly (SGD momentum
0.9, weight decay 0.0002, batch 128, LR 0.1→0.01→0.001 at iterations
32000/48000 of 64000 total, gradient clipping max_norm=5.0, 3 seeds per
configuration).

**Acceptance criteria — stated precisely, no invented pass/fail number:**
Since there is no "paper target" for LyASC, results are reported as direct
3-seed-mean comparisons against the base_model baseline family (Section 5
of `LyASC_ARCHITECTURE.md`), not against a fixed external number. A result
is reported as:
- **"Matches `rSkip+LN`"** if the 3-seed mean is within ±0.4pp of
  `rSkip+LN`'s verified 6.15% (the same tolerance convention used
  throughout the base-model reproduction, applied here for consistency,
  not because it is externally mandated).
- **"Beats `wSkip+LN`"** if the 3-seed mean is below `wSkip+LN`'s verified
  8.16%, AND training is stable without requiring `wSkip+LN`'s LR-multiplier
  mitigation (i.e., LyASC's own stability signal does the job the
  hand-tuned mitigation did for the naive gate).
- **Any other outcome is reported exactly as measured, with no attempt to
  reframe a miss as a success.** A LyASC run that is stable but does not
  beat `rSkip+LN` is still a valid, reportable, and potentially
  publishable negative/neutral result — it is not a failure of the
  documentation process to report it plainly.

---

## 1. Environment

Reuse the existing `lyasc-base` conda environment from `base_model/` —
same PyTorch/CUDA versions, no new dependencies expected for Phase 1
(power iteration and EMA smoothing use only core `torch` operations).
```bash
conda activate lyasc-base
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If any new dependency becomes necessary, add it to a new
`lyasc_model/environment_delta.yml` documenting only the addition, rather
than duplicating `base_model/environment.yml`.

---

## 2. Workspace Tree

```
LyASC/
├── base_model/                        <- existing, complete, read-only reference
└── lyasc_model/
    ├── plan.md                        <- this file
    ├── README.md
    ├── src/
    │   ├── models/
    │   │   ├── stability_estimator.py <- StabilityEstimator (new)
    │   │   ├── lyasc_skip.py          <- LyASCSkip (new)
    │   │   └── __init__.py
    │   ├── training/
    │   │   ├── train.py               <- thin wrapper: imports base_model's
    │   │   │                             train_one_run, injects LyASCSkip
    │   │   │                             into the variant registry
    │   │   └── config.py              <- extends base_model's config schema
    │   │                                 with k, N, ema_decay, lambda0,
    │   │                                 cold_start_strategy
    │   └── evaluation/
    │       └── compare_to_baseline.py <- reads lyasc_model results +
    │                                     base_model/results/tables/*_FINAL.csv,
    │                                     produces the comparison report
    │                                     defined in Section 0's acceptance
    │                                     criteria
    ├── configs/
    │   ├── lyasc_table1_equiv.yaml    <- 1xSkip, rSkip+LN, wSkip+LN, LyASC
    │   ├── lyasc_coldstart_ablation.yaml
    │   ├── lyasc_table3_equiv.yaml    <- catastrophic/bimodal stress test
    │   └── lyasc_table4_equiv.yaml    <- LN vs BN vs no-G pairing ablation
    ├── results/
    │   ├── logs/                      <- decisions.log lives here, same
    │   │                                 discipline as base_model
    │   ├── checkpoints/
    │   └── tables/
    └── tests/
        ├── test_stability_estimator.py
        └── test_lyasc_skip.py
```

---

## 3. Implementation Order (do not skip steps or reorder)

### Step 1 — Unit tests first, before any training code
Write `tests/test_lyasc_skip.py` covering: `LyASCSkip.forward()` produces
`alpha_l * x + (1-alpha_l) * F_x` exactly given a fixed, mocked
`estimator.current_value()`; `beta_l` initializes to 0 (giving
`alpha_l=0.5` at a fixed `lambda_l=lambda0`); gradients flow to both
`beta_l` and through `x`/`F_x`. Write `tests/test_stability_estimator.py`
covering: the power-iteration loop converges toward the correct dominant
singular value on a small, hand-constructed linear layer with a known
Jacobian (a solvable, checkable test case — do not skip this, it is the
only way to catch a sign or normalization error in the estimator before
it silently produces meaningless gate values); the recompute-every-N-steps
logic actually skips recomputation on non-multiple steps and returns the
cached EMA value.

### Step 2 — Resolve Section 8's open design questions, in order, cheaply
For each of the 6 open questions in `LyASC_ARCHITECTURE.md` Section 8:
state the decision made, the one-sentence rationale, and log it via the
same `log_decision(key=..., value=..., rationale=..., log_dir=log_dir)`
pattern used throughout `base_model/src/training/train.py`. Do this before
writing the full training loop integration — these are architectural
decisions, not tuning choices to discover mid-run.

### Step 3 — Single-seed smoke test on the smallest viable case
Before any 3-seed batch: one seed, `1xSkip`-scale settings (i.e., verify
`LyASCSkip` with `beta_l=0` fixed and untrainable reduces to a sane,
non-diverging 50/50 blend — a literal sanity check that the mechanism
doesn't break basic training before testing whether it helps). Confirm no
NaN/Inf, confirm the power-iteration overhead is measurable and roughly
matches the ~2% estimate in `LyASC_ARCHITECTURE.md` Section 2.4 (report
the actual measured number, do not just assume the estimate was right).

### Step 4 — `lyasc_table1_equiv.yaml`: full 4-variant, 3-seed comparison
```bash
cd LyASC/lyasc_model
for VARIANT in "1xSkip" "rSkip+LN" "wSkip+LN" "LyASC"; do
    for SEED in 0 1 2; do
        python -m src.training.train \
            --config configs/lyasc_table1_equiv.yaml \
            --variant "$VARIANT" --seed $SEED --depth 110 \
            --log_dir results/logs --checkpoint_dir results/checkpoints \
            --num_workers 4 --base_dir .
    done
done
```
Note: `1xSkip`, `rSkip+LN`, and `wSkip+LN` do not need to be retrained here
if reusing `base_model`'s already-completed, verified checkpoints is
sufficient for the comparison — retrain only if an apples-to-apples
same-workspace/same-run comparison is specifically required for the
journal submission's reproducibility section. Decide and document this
choice explicitly rather than defaulting to either option silently.

### Step 5 — Apply the general debugging protocol to any instability
Reuse the exact methodology from `base_model/plan.md` Section 4 /
`decisions.log`, unmodified: characterize before touching code, grep for
NaN/Inf first, split gradients by parameter group before guessing (the
`beta_l`-vs-rest split, following the `w_grad_norm`/`other_grad_norm`
pattern already implemented and validated), validate any fix on one cheap
seed before a full batch, cap at 3 distinct hypotheses per issue, log
everything.

### Step 6 — Cold-start ablation (`lyasc_coldstart_ablation.yaml`)
LyASC with strategy A (warm-up) vs. strategy B (damped influence) vs. no
handling at all (baseline exposure to the raw cold-start transient),
3 seeds each. This directly answers Section 6's open question with data.

### Step 7 — Table 3 / Table 4 equivalents
Only after Steps 4-6 produce a stable, documented LyASC configuration.
These are the harder stress tests (Section 5/7 of `LyASC_ARCHITECTURE.md`)
and should not be attempted with an unresolved cold-start strategy.

---

## 4. What NOT to do (carried over, unmodified, from the base-model project's hard-won rules)

- Do not retrain any base_model baseline "just to be sure" without a
  specific, stated reason — they are complete and verified.
- Do not report a result as a success by silently changing the comparison
  metric or tolerance after seeing the number.
- Do not skip the single-seed smoke test to save time before a 3-seed batch.
- Do not leave an open design question (Section 8 of the architecture doc)
  unresolved-but-implemented-anyway — a silent default is exactly the kind
  of undocumented assumption the entire base-model project's `decisions.log`
  discipline exists to prevent.
- Do not claim an experiment ran, or ran successfully, without a
  corresponding result JSON and log file on disk to point to.

---

## 5. Reporting

Every LyASC result, whether a clean success, a stable-but-unremarkable
result, or a genuine failure, gets the same treatment `base_model`'s
FINAL_REPORT.md used at its best (not its worst, fabricated draft): a
table of raw per-seed values, means, comparison targets, and diffs,
generated programmatically from result JSONs — never hand-typed into a
markdown table.
