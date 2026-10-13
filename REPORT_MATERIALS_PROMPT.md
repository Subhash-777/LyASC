# Antigravity Agent Prompt: LyASC Report Materials Pack

Save as `LyASC/REPORT_MATERIALS_PROMPT.md`, then paste into the Antigravity Agent panel.
Self-contained. If a file disagrees with a claim here, the file wins.

---

## 0. What this task is, and the limits on it

Collect and generate everything needed to write the full project report: key
points, terminal screenshots, figures, tables. All of it goes into one new
folder, `LyASC/report_materials/`.

Hard limits:

1. **Read-only on everything that exists.** Create files only under
   `report_materials/`. Do not edit `base_model/`, `lyasc_model/`, `paper/`,
   the READMEs, the dashboard files, or MLflow data. Do not commit or push to git.
2. **No training, no GPU use.** All figures are CPU-only (matplotlib). Check
   `nvidia-smi` first; if a run is active, leave it alone.
3. **Run everything inside the `lyasc-experiments` conda environment**, and
   show that in the first terminal capture.
4. **No fabricated or retouched evidence.** Terminal images must show the
   verbatim output of commands actually run during this task. Figures must be
   drawn from real result files. Never type a number into a figure by hand.
5. **MLflow safety.** Copy `lyasc_model/results/mlflow.db` to a temp location
   and run any query or UI against the copy, so the real store is never touched.
6. **Prose style** for every text file you write: plain, specific, no inflated
   language, no em dashes or en dashes (same rule as the paper).

---

## 1. Folder layout

```
LyASC/report_materials/
├── 00_MANIFEST.md                  (every asset: file, what it shows, source, date made)
├── 01_key_points/
│   ├── KEY_POINTS.md
│   ├── numbers_index.csv
│   ├── claims_register.md
│   └── issues_and_resolutions.md
├── 02_terminal_screenshots/
│   ├── raw/        (verbatim .txt of each command's output)
│   ├── png/
│   └── RUN_AND_SCREENSHOT.md       (same commands, for taking real window screenshots by hand)
├── 03_figures/
│   ├── png/ (300 dpi)   pdf/ (vector)   tables/ (csv + tex)
│   └── source_scripts/             (one script per figure, re-runnable)
├── 04_ui_screenshots/              (dashboard, MLflow UI)
├── 05_architecture_and_workflow/   (copies of the existing diagrams)
└── REPORT_SECTION_MAPPING.md       (which asset belongs in which report chapter, plain caption each)
```

---

## 2. Step 0: audit before collecting (stop and report if anything fails)

Figures and key points must not carry forward anything unresolved. Check first:

1. **Phase E2 provenance.** The `LyASC_noG` values reported for Phase E2
   (10.21% and 8.82%, alpha_mean 0.5982 and 0.7179) match Phase C's LyASC
   (no normalization) seeds 0 and 1 exactly, down to alpha_mean. Find out
   whether these are the same runs reused, or separate runs that happened to
   reproduce bit for bit. Either is acceptable, but every asset must describe
   it correctly ("reused from Phase C" if reused). Then confirm every Phase E2
   per-seed value (noG, +BN, and the LyASC+LN baseline of 7.17%) against its
   result JSON, and confirm the baseline row is Phase D's three seeds
   (7.33, 7.25, 6.93) and not any other numbers.
2. **Cross-check these expected values against the files** (a sanity list,
   not a source of truth; the files decide):
   Phase C LyASC 9.66 (10.21, 8.82, 9.95); Ablation 1 7.22 (single seed);
   Phase D none 7.17, damped 7.46, warmup 7.95 (2 seeds); Phase E1
   stress_0.5 8.38 (7.98, 8.84, 8.31), stress_2.0 8.44 (8.13, 8.81, 8.38),
   stress_3.0 seeds 27.90, 32.73, 19.12, 20.88, 8.75; base-paper failures
   37.14, 37.59, 49.27; base reproduction 9/10 Table 1 within tolerance.
   List every mismatch.
3. **Use the latest framing in each decisions.log entry**, not an earlier
   draft that a later entry corrected.

Report the audit result in a short list before building anything that depends
on a flagged item. Items that pass can proceed.

---

## 3. Key points (01_key_points/)

**KEY_POINTS.md**, organised by report chapter, each point one or two plain
sentences with its source file in brackets:
problem and motivation; base paper and why it was reproduced; the LyASC idea
and gate equation; implementation (stack, environment, estimator, tests,
MLflow, resume checkpoints); the experiment sequence (Phase A to E2) with
what each phase asked and found; limitations; planned future work.

**numbers_index.csv**: one row per number used anywhere in the pack.
Columns: id, value, unit, description, n_seeds, source_file, how_extracted,
status (verified / flagged). Every number in a figure or key point must have a row.

**claims_register.md**, four lists:
- Supported (what the data shows, with n)
- Hypotheses (for example, the early-escape threshold explanation for stress_3.0)
- Untested (for example, BN behaviour under stress, the Phase F extensions)
- Earlier claims that were corrected, and what replaced them (the 0.5^54
  gradient claim, the saturated-ReLU explanation for lambda spread, the
  "bit-identical warmup and damped" result and its cause, the pooled
  stress_3.0 mean as a headline, the BN "proves superior" wording)

**issues_and_resolutions.md**: built from `decisions.log` and the
git history. For each problem: what happened, how it was detected, the fix,
the evidence file. Include at least the import-precedence bug, the
base_model config edit that was reverted, the device mismatch, the MLflow
path problem, the wall-clock audit, the compute-plan unit error, and the
Phase E2 baseline mix-up. Do not invent entries; use only what the logs show.

---

## 4. Terminal screenshots (02_terminal_screenshots/)

Method: run the command, save stdout and stderr verbatim to `raw/Txx.txt`,
then render that text to `png/Txx.png` in a dark terminal style showing the
working directory, the command line, a `date` timestamp, and the full output.
If output is long, limit it with `head` or `tail` visibly in the command, never
by silently cutting lines. Keep the raw file next to every PNG.

Captures to make:

| ID | Command or content |
|---|---|
| T01 | `conda info --envs` and `which python` showing `lyasc-experiments` active |
| T02 | torch and CUDA check, plus `nvidia-smi` (GPU model and memory) |
| T03 | `tree -L 3` of the workspace (exclude data, checkpoints, `__pycache__`) |
| T04 | `pytest lyasc_model/tests/ base_model/tests/` summary line |
| T05 | base-model reproduction summary (`base_model/scripts/summarize_all.py`) |
| T06 | head and tail of one Phase C training log (schedule header, LR drops, final eval) |
| T07 | the resume-fromcheckpoint log lines from a resumed run |
| T08 | one LyASC result JSON (`cat`), and one stress_3.0 seed JSON |
| T09 | selected `decisions.log` entries (device fix, baseline reuse, Phase D selection) |
| T10 | `mlflow.search_runs` table for the core experiments, sorted by start time (against the copied db) |
| T11 | launcher environment guard rejecting `/usr/bin/python3` (run it; it fails before any import) |
| T12 | `git log --oneline` and `git status` |
| T13 | `pdflatex` and `bibtex` tail from compiling `paper/` into a temp directory, plus the page count |
| T14 | `nvidia-smi` during a training run, only if one is already running; otherwise skip and say so |

Also write `RUN_AND_SCREENSHOT.md`: the same commands in order, so any of them
can be re-run by hand and captured as a real window screenshot if the report
requires that instead.

---

## 5. Figures (03_figures/)

Rules for every chart: data read from result JSONs or the verified CSVs by a
script saved in `source_scripts/`; n stated on the chart or caption; per-seed
points shown over any bar; y axis starting at 0 unless a break is labelled;
one fixed colour per variant across all figures (colour-blind safe); text at
least 9 pt at print size; width 3.5 in for single column, 7 in for full width;
PNG at 300 dpi plus a PDF copy.

| ID | Figure | Source |
|---|---|---|
| F01 | Base reproduction, Table 1: ours vs paper, tolerance band marked | `reproduced_table1_FINAL.csv` |
| F02 | Progression: LyASC without norm (Phase C), with LayerNorm, against 2rSkip+LN, 1xSkip, wSkip+LN | Phase C and D JSONs, Table 1 CSV |
| F03 | Cold-start strategies (none, damped, warmup) with per-seed points and spread | Phase D JSONs |
| F04 | Stress rescue: base-paper failure vs LyASC for 0.5x, 2x, 3x, per-seed points | Phase E1 JSONs, Table 3 CSV |
| F05 | stress_3.0 outcomes, all five seeds, drawn as two clusters (no pooled mean as the headline) | Phase E1 JSONs |
| F06 | Training loss curves for the five stress_3.0 seeds with LR drops at 32k and 48k marked | logs or MLflow copy |
| F07 | alpha_mean over iterations for stress_0.5, 2.0 and 3.0 (all seeds) | logs or MLflow copy |
| F08 | Normalization after the gate (none, LayerNorm, BatchNorm), n labelled, "reused from Phase C" noted where true | Phase E2 JSONs, only after the Step 0 audit passes |
| F09 | Estimator overhead and seconds per iteration by configuration (k=2,N=10 vs k=1,N=20; noG, LN, BN) | timing JSONs and decisions.log |
| F10 | Experiment timeline from MLflow run start times, by phase | MLflow copy |
| F11 | Phase C training loss and accuracy curves, three seeds | Phase C logs |
| F12 | One summary table figure: every phase, variant, n, mean, spread, comparison target | `numbers_index.csv` |

Also export the key tables as CSV and LaTeX (`tables/`).

---

## 6. UI screenshots (04_ui_screenshots/) and existing diagrams (05_)

- Serve the existing dashboard as it is and capture the hero section, Phase D,
  Phase E1, and the decisions trail (a headless browser script already exists
  in `lyasc_model/dashboard/`; reuse it with a different output folder). Do not
  re-run the data exporter, because it overwrites `dashboard_data.json`. If the
  dashboard data looks stale against the result JSONs, report that instead.
- Run the MLflow UI against the copied db and capture: the experiments list,
  the run table for one core experiment sorted by start time, and a metric
  chart or compare view for the stress_3.0 seeds.
- Copy (do not move) the existing architecture and workflow diagrams into
  `05_architecture_and_workflow/`.

---

## 7. Manifest, mapping, and final self-check

1. `00_MANIFEST.md`: table of every asset with filename, what it shows,
   source files or command, and the time it was created.
2. `REPORT_SECTION_MAPPING.md`: for each report chapter, the assets that
   belong in it and a one-line plain caption for each.
3. Self-check before reporting:
   - every PNG opens and is readable at its intended size
   - every number in every figure or key point has a row in `numbers_index.csv`
   - search the whole pack for superseded values used as current: a 6.3x
     figure as the LyASC+LN baseline, the pooled stress_3.0 mean as a headline,
     "29.5pp" as the headline rescue, "15 of 16 runs", "proves superior";
     fix or remove any hit
   - no em dashes or en dashes in any prose file

Report back with: audit findings, the count of assets by folder, total size on
disk, anything skipped and why, and any discrepancy you found between files.
