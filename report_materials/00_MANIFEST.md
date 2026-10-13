# Report Materials Asset Manifest

This manifest catalogs all assets generated for the LyASC report materials pack.

---

## 01 Key Points Index (`01_key_points/`)

| Filename | Description | Source File / Command | Date Created |
|---|---|---|---|
| `KEY_POINTS.md` | Key technical and experimental points organized by report chapter | Source JSONs, logs, and decisions.log | 2026-10-09 |
| `numbers_index.csv` | Master index of all 54 quantitative values used in the report pack | `build_numbers_index.py` | 2026-10-09 |
| `claims_register.md` | Categorized register of supported, hypothetical, untested, and corrected claims | Project results and decisions.log | 2026-10-09 |
| `issues_and_resolutions.md` | Detailed audit log of 7 critical technical issues and resolutions | decisions.log and git commit history | 2026-10-09 |

---

## 02 Terminal Screenshots (`02_terminal_screenshots/`)

| ID | Raw Text File | PNG Screenshot | Description | Source Command |
|---|---|---|---|---|
| T01 | `raw/T01.txt` | `png/T01.png` | Active conda environment verification | `conda info --envs && which python` |
| T02 | `raw/T02.txt` | `png/T02.png` | PyTorch CUDA & GPU status check | `python -c "import torch..." && nvidia-smi` |
| T03 | `raw/T03.txt` | `png/T03.png` | Directory tree of workspace (depth 3) | `tree -L 3 -I 'data\|checkpoints\|...'` |
| T04 | `raw/T04.txt` | `png/T04.png` | Test suite execution summary (78 passed) | `pytest lyasc_model/tests/ base_model/tests/` |
| T05 | `raw/T05.txt` | `png/T05.png` | Base model reproduction summary table | `python base_model/scripts/summarize_all.py` |
| T06 | `raw/T06.txt` | `png/T06.png` | Phase C training log header and tail | `head/tail .../phase_c_LyASC_seed1.log` |
| T07 | `raw/T07.txt` | `png/T07.png` | Checkpoint saving and resume confirmation lines | `grep -C 3 'Resumed from checkpoint' ...` |
| T08 | `raw/T08.txt` | `png/T08.png` | Raw JSON result file payloads | `cat lyasc_table1_equiv...json & stress_3.0...json` |
| T09 | `raw/T09.txt` | `png/T09.png` | Selected decisions.log entries | `grep selected decisions.log entries` |
| T10 | `raw/T10.txt` | `png/T10.png` | MLflow runs table sorted chronologically | `sqlite3 SELECT from copied mlflow.db` |
| T11 | `raw/T11.txt` | `png/T11.png` | Environment guard rejection of /usr/bin/python3 | `/usr/bin/python3 run_phase_e_stresstest.py` |
| T12 | `raw/T12.txt` | `png/T12.png` | Git commit log history and working tree status | `git log --oneline -n 12 && git status` |
| T13 | `raw/T13.txt` | `png/T13.png` | Paper LaTeX source compilation log output | `pdflatex paper/main.tex` |
| T14 | `raw/T14.txt` | `png/T14.png` | Active GPU training check verification | `nvidia-smi` |
| Doc | `RUN_AND_SCREENSHOT.md` | - | Re-run instructions for manual window screenshots | Manual capture documentation |

---

## 03 Figures (`03_figures/`)

| ID | PNG File (300 dpi) | PDF Vector File | Description | Source Script |
|---|---|---|---|---|
| F01 | `png/F01_base_reproduction_table1.png` | `pdf/F01_base_reproduction_table1.pdf` | Base paper reproduction vs targets with tolerance band | `generate_all_figures.py` |
| F02 | `png/F02_architectural_progression.png` | `pdf/F02_architectural_progression.pdf` | Architectural progression from base skip to LyASC+LN | `generate_all_figures.py` |
| F03 | `png/F03_cold_start_strategies.png` | `pdf/F03_cold_start_strategies.pdf` | Phase D cold-start strategies with per-seed scatter | `generate_all_figures.py` |
| F04 | `png/F04_stress_rescue_comparison.png` | `pdf/F04_stress_rescue_comparison.pdf` | Phase E1 stress rescue vs base paper failures | `generate_all_figures.py` |
| F05 | `png/F05_stress3_cluster_outcomes.png` | `pdf/F05_stress3_cluster_outcomes.pdf` | Stress 3.0 outcome breakdown into two distinct clusters | `generate_all_figures.py` |
| F06 | `png/F06_stress3_loss_curves.png` | `pdf/F06_stress3_loss_curves.pdf` | Training loss curves for 5 stress 3.0 seeds with LR drops | `generate_all_figures.py` |
| F07 | `png/F07_alpha_mean_trajectories.png` | `pdf/F07_alpha_mean_trajectories.pdf` | Mean gate alpha (α) trajectories across stress levels | `generate_all_figures.py` |
| F08 | `png/F08_phase_e2_normalization_pairing.png` | `pdf/F08_phase_e2_normalization_pairing.pdf` | Phase E2 post-gate normalization pairing comparison | `generate_all_figures.py` |
| F09 | `png/F09_estimator_timing_overhead.png` | `pdf/F09_estimator_timing_overhead.pdf` | Estimator timing overhead (s/iter) by configuration | `generate_all_figures.py` |
| F10 | `png/F10_experiment_timeline.png` | `pdf/F10_experiment_timeline.pdf` | Experiment execution timeline across project phases | `generate_all_figures.py` |
| F11 | `png/F11_phase_c_training_curves.png` | `pdf/F11_phase_c_training_curves.pdf` | Phase C training loss and accuracy progression curves | `generate_all_figures.py` |
| F12 | `png/F12_master_summary_table_fig.png` | `pdf/F12_master_summary_table_fig.pdf` | Master summary table figure across all phases | `generate_all_figures.py` |

### Figure Tables (`03_figures/tables/`)

- `table1_reproduction.csv` & `table1_reproduction.tex`: Table 1 base reproduction values.
- `table3_stress_rescue.csv` & `table3_stress_rescue.tex`: Phase E1 stress rescue comparison values.
- `master_summary.csv` & `master_summary.tex`: Master summary table across all project phases.

---

## 04 UI Screenshots (`04_ui_screenshots/`)

| Filename | Description | Source / Capture Script | Date Created |
|---|---|---|---|
| `dashboard_hero.png` | Showcase dashboard hero section and KPI summary | Puppeteer capture of served index.html | 2026-10-09 |
| `dashboard_headline_finding.png` | Showcase dashboard headline finding section | Puppeteer capture of served index.html | 2026-10-09 |
| `dashboard_phase_e.png` | Showcase dashboard Phase E stress rescue section | Puppeteer capture of served index.html | 2026-10-09 |
| `dashboard_decisions.png` | Showcase dashboard decision trail section | Puppeteer capture of served index.html | 2026-10-09 |
| `dashboard_full_page.png` | Full-page screenshot of interactive showcase dashboard | Puppeteer capture of served index.html | 2026-10-09 |
| `mlflow_experiments_list.png` | MLflow UI experiments list overview | Puppeteer capture of MLflow UI (port 5050) | 2026-10-09 |
| `mlflow_runs_table.png` | MLflow UI core experiment runs table (Exp 7) | Puppeteer capture of MLflow UI (port 5050) | 2026-10-09 |
| `mlflow_stress3_compare.png` | MLflow UI stress 3.0 run comparison view | Puppeteer capture of MLflow UI (port 5050) | 2026-10-09 |

---

## 05 Architecture and Workflow (`05_architecture_and_workflow/`)

| Filename | Description | Source File | Date Created |
|---|---|---|---|
| `architect.png` | High-resolution architectural block diagram | `architect.png` (Workspace root) | 2026-10-09 |
| `workflow.png` | Experimental workflow flowchart | `workflow.png` (Workspace root) | 2026-10-09 |
| `LyASC_architecture_overview.html` | Interactive HTML architecture overview document | `LyASC_architecture_overview.html` | 2026-10-09 |
