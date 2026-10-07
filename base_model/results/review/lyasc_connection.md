# LyASC Connection: What the Base-Paper Reproduction Teaches

*Extracted from LyASC Revised Blueprint v2, Section 6 — for mentor review 2026-08-22*

---

## 6.1 The Naive-Gate Failure is Now Primary Evidence, Not Motivation-by-Analogy

`wSkip+LN`'s investigation trail is a complete, reusable case study:
- `grad_norm` spike ≈ 1100 at iteration 1000
- Diagnostic split: `w_grad_norm ≈ 70` vs. `other_grad_norm ≈ 1100` — spike is not gate-specific
- Confirmed as universal LN cold-start transient via a `1xSkip+LN` control run
- Mitigated but not resolved via a 0.1× LR multiplier on the gate

LyASC's own gate is structurally similar (a learned, per-block modulation of skip strength) and should be evaluated with the same diagnostic discipline: **split gradients between the Lyapunov-estimator-driven gate parameters and the rest of the network** before assuming any instability is gate-specific.

---

## 6.2 A Falsifiable, Pre-Registered Comparison

LyASC's headline claim should be stated as:

> *A stability-signal-driven gate trains at least as stably as the best fixed schedule (`2rSkip+LN`) while requiring no manual λ search, and trains more stably and to a better final result than an unconstrained learned gate (`wSkip+LN`) under matched conditions.*

This is now **directly testable against verified numbers** (6.15% for 2rSkip+LN, 8.16% for wSkip+LN) rather than paper-reported ones.

---

## 6.3 Normalization Pairing is Now an Explicit, Motivated Ablation

Since BN measurably underperforms LN as the `G()` function in every verified comparison:

- LyASC's **default configuration pairs the gate with LN**
- **BN-paired LyASC is included as an ablation** to test whether a stability-aware gate narrows the BN-vs-LN gap

This is a stronger, more specific claim than the base paper's own BN-independence framing, and directly testable using the existing Table 4 infrastructure.

---

## 6.4 The Catastrophic Rows are LyASC's Sharpest Test Case

`LN(x+2F)` reproduced as **genuinely bimodal** — one seed escapes cleanly to a strong result, two seeds plateau in a degenerate state — rather than merely noisy.

This is the single most informative stress test available: **if a stability-aware gate can detect the divergence between the escaping and plateauing trajectories early and intervene**, that is a concrete, falsifiable, and highly legible result for a journal submission. Succeeding here is a much stronger claim than any incremental accuracy gain on a well-behaved variant.

---

## 6.5 The Comparison Baseline is Unusually Strong — Which is Good for the Paper

Because this reproduction's modern implementation details (Kaiming init, corrected weight-decay scoping, gradient clipping, channel-wise `LayerNorm2d`) produced results that **systematically outperform the original paper**, any LyASC improvement measured against this baseline:

- Is measured against a **harder target** than the paper's own numbers
- **Strengthens any positive result** and pre-empts the obvious reviewer objection ("is the gain just from a stronger baseline?") by construction

---

*Source: LyASC Revised Blueprint v2, `/home/subhash/projects/LyASC/LyASC_Revised_Blueprint_v2.docx`*
