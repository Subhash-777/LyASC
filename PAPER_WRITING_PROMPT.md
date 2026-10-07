# Antigravity Agent Prompt — LyASC IEEE Conference Paper (Partial Draft, LaTeX)

Save this file as `LyASC/PAPER_WRITING_PROMPT.md` in the workspace, then paste
its contents into the Antigravity Agent panel. Self-contained: treat only
this prompt and the files it references as ground truth. This is a
**writing task**, not an experiment task — no training, no GPU, no touching
`base_model/` or any result-producing code. If Phase E1/E2 training is still
running in the background, this work runs alongside it without interfering.

---

## 0. What this paper is, and what it explicitly is not

This is a **partial draft** of an IEEE two-column conference paper covering
the project's work **up through the current point** — base-model
reproduction, the LyASC architecture, Phase C, the LayerNorm fix, Phase D,
and Phase E1. It is not a claim that the project is finished, and it must
not read like one.

- Do **not** write results for Phase E2, Phase F, or Phase G — none of that
  work exists yet. Anything from those phases belongs in a short, concrete
  "planned future work" paragraph, not in the Results section.
- Do **not** upgrade a mixed or partial result into a clean win. Where LyASC
  underperforms the best fixed baseline (Phase C, normal conditions), say so
  plainly. Where a finding has a real complication (Phase E1's `stress_3.0`
  two-cluster outcome), report the complication, not a smoothed average.
- Every number in the paper must trace to a specific file: a result JSON, a
  verified CSV, or a `decisions.log` entry. If you cannot point to the
  source file for a number, do not write it down — flag it and ask instead
  of estimating or rounding to something plausible.

---

## 1. Source material — read all of this before writing anything

1. `LyASC_ARCHITECTURE.md` — the exact architecture spec (`LyASCSkip`,
   `StabilityEstimator`, the gate formula) — this is the primary source for
   the Methodology section, not a paraphrase of it from memory.
2. `LyASC_plan.md` and `base_model/plan.md` — experimental protocol details
   (optimizer, LR schedule, seeds, tolerance rule) for the Experimental
   Setup section.
3. `base_model/results/FINAL_REPORT.md` and
   `base_model/results/tables/reproduced_table{1,3,4,6}_FINAL.csv` — the
   verified base-model reproduction numbers. These are the only source for
   any "ours vs. paper" comparison number in the reproduction section.
4. `lyasc_model/results/logs/decisions.log` and
   `lyasc_model/results/logs/phase_e_stresstest/decisions.log` — the
   authoritative record of what was actually found at each phase, including
   the corrected/final framings (e.g., the `stress_3.0` two-cluster
   breakdown and the early-escape threshold-effect finding, NOT the earlier
   pooled-mean framing that was walked back). When a decisions.log entry
   contains both an early draft of a finding and a later correction, use
   the later one.
5. `lyasc_model/dashboard/dashboard_data.json` — the consolidated,
   already-verified numeric summary across all phases; cross-check every
   number you plan to cite against this file as well as its underlying
   result JSON.
6. The literature review content (the 20 literature-review slides in
   `0th_Review_PPT` or wherever the reviewed-papers table now lives) — this
   is the source for the Related Work section and its citations. Use only
   these already-vetted sources; do not invent additional citations or
   fabricate DOIs to fill out the bibliography.
7. `LyASC_Architecture_Research_Plan.docx` — background and the ten
   proposed novel extensions. Useful for framing the Discussion/Future Work
   section, but it is a working document full of speculative material, not
   polished prose — do not lift sentences from it directly.

If any of these files are missing or contradict each other (e.g. a stale
copy vs. the live one), stop and flag it rather than picking one silently.

---

## 2. Paper structure (IEEE two-column conference format)

Use the standard `IEEEtran` LaTeX class, conference mode, two-column, 10pt.
Set up the workspace as:

```
LyASC/paper/
├── main.tex
├── references.bib
├── figures/
│   └── (reuse existing diagrams where they fit, e.g. the architecture
│        diagram image already in the deck — do not regenerate it)
└── README.md   (compile instructions: pdflatex, bibtex, pdflatex, pdflatex)
```

Section plan, with the source material each one draws from:

1. **Abstract** — one paragraph. State the idea, state honestly that this
   is a partial/in-progress report, and state the two things that are
   actually established: the normal-condition result and the stress-test
   result.
2. **I. Introduction** — motivation (fixed vs. freely-learned skip gating,
   why neither is satisfying), the paper's contribution list stated
   precisely (do not claim more than what Sections IV-VI support).
3. **II. Related Work** — drawn only from the literature review table's
   entries. Group them the way the existing "Research Gap" slide already
   does (fixed-schedule methods, feature/activation-driven gates,
   edge-of-stability / Lyapunov-adjacent theory work) rather than a flat
   list.
4. **III. Proposed Method** — the LyASC formulation, the stability
   estimator (power iteration, EMA smoothing), the gate equation, and how
   it integrates into `PreActBasicBlock` with zero interface changes. Pull
   the formula and design rationale directly from `LyASC_ARCHITECTURE.md`.
5. **IV. Experimental Setup** — dataset, architecture (PreAct-ResNet-110,
   CIFAR-10), optimizer/schedule, the ±0.4pp tolerance convention, and the
   `k=1, N=20` hyperparameters actually used, with the one-line rationale
   for why (measured overhead, logged decision).
6. **V. Results** — subsections, each citing its exact source file:
   - V-A Base-model reproduction verification (cite the FINAL_REPORT.md
     pass/fail counts)
   - V-B Initial LyASC result and the diagnosed cause of the early gap
     (Phase C)
   - V-C Effect of adding LayerNorm after the gate (Ablation 1)
   - V-D Cold-start strategy comparison (Phase D) — report `none` and
     `damped` as statistically indistinguishable, exactly as
     `decisions.log` states it, not as a clean win for `none`.
   - V-E Stress-test results (Phase E1) — report `stress_0.5` and
     `stress_2.0` as clean, consistent rescues; report `stress_3.0` as a
     two-cluster outcome with the early-escape threshold-effect hypothesis,
     clearly labeled as a hypothesis, not a confirmed mechanism.
7. **VI. Discussion and Limitations** — the real limitations: LyASC does
   not yet beat the best fixed baseline under normal conditions, the
   overhead cost of the estimator, the single-GPU/small-seed-count
   constraints, and the unresolved `stress_3.0` threshold mechanism.
8. **VII. Conclusion and Future Work** — summarize only what is actually
   done. Name Phase E2, the proposed extensions, and the `stress_3.0`
   follow-up seeds as specific, concrete next steps — not a vague
   "promising directions" paragraph.
9. **References** — BibTeX entries built from the literature review
   table's actual author/journal/year/DOI fields.

---

## 3. Writing requirements — humanized, no AI-writing tells

This is the part to take literally, not as a vague style note. The draft
must not contain any of the following patterns. Treat this as a checklist
to run against the text before presenting it, the same way a plagiarism
checker would be run — not a one-time reminder.

**Banned vocabulary** (these words are statistically over-represented in
AI-generated text and must not appear): delve, boast/boasts, leverage,
crucial, pivotal, intricate/intricacies, tapestry, testament, underscore
(as a verb), showcase (as a verb), foster/fostering, enhance, garner,
landscape (as an abstract noun, e.g. "the research landscape"), key (as a
vague adjective), robust (unless used in its precise technical sense),
vibrant, align with, interplay.

**Banned constructions:**
- "Not only... but also..." and "It's not just X, it's Y" negative
  parallelisms.
- Sentences that avoid a plain "is/are" in favor of "serves as," "stands
  as," "represents a," "boasts a."
- Present-participle tack-on phrases for fake depth ("...,
  highlighting/underscoring/demonstrating the importance of...").
- Rule-of-three padding — forcing findings into groups of three when the
  real count is two or four. Report however many there actually are.
- Inflated-significance framing ("marks a pivotal moment," "represents a
  significant step," "underscores the broader importance of..."). State
  what happened and what it means technically; do not editorialize about
  its importance.
- Vague attributions ("researchers have shown," "it is widely believed")
  — every claim needs its specific source (a cited paper, or "we found,"
  or "the reproduction confirmed").
- Generic "Challenges and Future Work" padding — the Future Work section
  must name specific, concrete next steps (already listed in Section 2
  above), not vague optimism about the field.
- Filler phrases: "in order to" (use "to"), "due to the fact that" (use
  "because"), "it is important to note that" (just state the fact).
- Excessive hedging stacked in one sentence ("could potentially possibly
  suggest").
- Em dashes and en dashes. None, anywhere in the final draft. Use a
  period, comma, colon, or parentheses instead. Scan the finished text for
  em dashes and en dashes before calling it done; either character
  appearing means the draft is not finished.
- Bold-header inline lists ("**Performance:** Performance was improved by
  ..."). Write it as normal prose sentences instead.

**What good looks like here**: this is formal technical writing, not a
blog post, so the fix is not to inject personality or first-person asides
— plain, direct, technically precise prose *is* the correct voice for an
IEEE paper. The goal is removing padding and inflation, not adding
color. Match the tone already used in this project's own `decisions.log`
entries and the corrected walkthrough language (e.g. "none and damped are
statistically indistinguishable at n=3" is exactly the right register:
specific, plain, unafraid to state an unglamorous result).

**Required self-check before presenting any section**: after drafting a
section, reread it and ask "does this contain any pattern from the banned
list above." Fix what you find, then check again. Do this for every
section, not once at the end for the whole document — it is much easier to
catch these while a section is still short.

---

## 4. Process

1. Read all source material in Section 1.
2. Propose the paper's exact section-by-section outline, with which source
   file backs each planned claim, and the approximate page budget (a
   partial draft covering this much work is likely 4 to 6 pages at IEEE
   two-column density). **Stop and get approval on the outline before
   writing full prose** — restructuring is much cheaper before the LaTeX
   exists.
3. Draft section by section, running the Section 3 self-check on each one
   before moving to the next.
4. Build `references.bib` from the literature review table and confirm it
   compiles (no missing fields, no fabricated entries).
5. Compile the full document (`pdflatex` twice plus `bibtex`) and confirm
   it produces a clean PDF with no undefined references or overfull hbox
   warnings worth caring about.
6. Present the compiled PDF and the page count, along with a short list of
   any claims where the source data was ambiguous or where you had to
   choose a specific phrasing for a nuanced result (e.g. how you phrased
   the `stress_3.0` finding) so that can be checked against intent before
   this goes anywhere.
