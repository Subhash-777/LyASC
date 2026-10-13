# Report Section Mapping

This document maps each asset in the report materials pack to its corresponding chapter in the final project report, providing a plain one-line caption for each item.

---

## Chapter 1: Introduction and Background

- **`05_architecture_and_workflow/architect.png`**
  *Caption*: Architectural diagram of the LyASC gated block showing shortcut, residual, and stability estimator components.
- **`05_architecture_and_workflow/workflow.png`**
  *Caption*: Experimental workflow diagram illustrating the sequential progression from base paper reproduction to normalization ablations.
- **`05_architecture_and_workflow/LyASC_architecture_overview.html`**
  *Caption*: Interactive HTML reference document detailing LyASC block mechanics and mathematical equations.

---

## Chapter 2: Base Model Reproduction

- **`03_figures/png/F01_base_reproduction_table1.png`**
  *Caption*: Base paper reproduction test errors compared to paper targets across 10 Table 1 variants on CIFAR-10 at depth 110.
- **`03_figures/tables/table1_reproduction.tex`**
  *Caption*: LaTeX table summarizing reproduced test error means and standard deviations against base paper targets for Table 1.
- **`02_terminal_screenshots/png/T05.png`**
  *Caption*: Terminal output of the automated base model summary script verifying reproduction metrics across all tables.

---

## Chapter 3: LyASC Architectural Formulation

- **`03_figures/png/F02_architectural_progression.png`**
  *Caption*: Test error progression comparing standard skip connections, paper baselines, un-normalized LyASC (Phase C), and LyASC+LN (Phase D).
- **`02_terminal_screenshots/png/T04.png`**
  *Caption*: Summary of 78 passing unit tests validating stability estimator updates, skip connection logic, and model integration.
- **`02_terminal_screenshots/png/T08.png`**
  *Caption*: Raw JSON result file output illustrating recorded evaluation metrics, wall-clock time, and gate diagnostic statistics.

---

## Chapter 4: System Implementation and Environment

- **`02_terminal_screenshots/png/T01.png`**
  *Caption*: Verification of active lyasc-experiments conda environment and associated Python interpreter path.
- **`02_terminal_screenshots/png/T02.png`**
  *Caption*: System hardware inspection verifying PyTorch CUDA availability and GPU device parameters.
- **`02_terminal_screenshots/png/T03.png`**
  *Caption*: Directory tree of the workspace repository structure showing module organization.
- **`02_terminal_screenshots/png/T11.png`**
  *Caption*: Environment guard execution error demonstrating automated rejection of unauthorized system Python environments.
- **`02_terminal_screenshots/png/T12.png`**
  *Caption*: Version control history showing recent git commit logs and clean working tree status.
- **`02_terminal_screenshots/png/T07.png`**
  *Caption*: Logging output confirming checkpoint saving and automatic multi-session training resumption at iteration 32,000.

---

## Chapter 5: Experimental Sequence and Results

### Section 5.1: Phase C Un-normalized LyASC
- **`03_figures/png/F11_phase_c_training_curves.png`**
  *Caption*: Phase C training loss and test accuracy trajectories across 3 random seeds for un-normalized LyASC at depth 110.
- **`02_terminal_screenshots/png/T06.png`**
  *Caption*: Phase C training log header and tail highlighting learning rate schedule drops and final evaluation accuracy.

### Section 5.2: Phase D Cold-Start Strategies
- **`03_figures/png/F03_cold_start_strategies.png`**
  *Caption*: Comparison of Phase D cold-start strategies (none, damped, warmup) showing per-seed test errors and spread.
- **`02_terminal_screenshots/png/T09.png`**
  *Caption*: Structured decision log entries documenting the pragmatic selection of the 'none' initialization strategy.

### Section 5.3: Phase E1 Stress Testing
- **`03_figures/png/F04_stress_rescue_comparison.png`**
  *Caption*: LyASC stress rescue performance compared to base paper catastrophic failures across 0.5x, 2.0x, and 3.0x scaling.
- **`03_figures/png/F05_stress3_cluster_outcomes.png`**
  *Caption*: Individual seed test error outcomes under 3.0x stress scaling grouped into Partial Rescue and Full Rescue clusters.
- **`03_figures/png/F06_stress3_loss_curves.png`**
  *Caption*: Log-scale training loss trajectories for the 5 stress 3.0 seeds with scheduled learning rate drop iterations marked.
- **`03_figures/png/F07_alpha_mean_trajectories.png`**
  *Caption*: Mean Lyapunov gate alpha (α) trajectories over training iterations across 0.5x, 2.0x, and 3.0x stress levels.
- **`03_figures/tables/table3_stress_rescue.tex`**
  *Caption*: LaTeX table presenting test error reductions achieved by LyASC under extreme residual scaling conditions.
- **`04_ui_screenshots/dashboard_phase_e.png`**
  *Caption*: Interactive showcase dashboard section illustrating Phase E1 stress rescue findings and KPI cards.
- **`04_ui_screenshots/mlflow_stress3_compare.png`**
  *Caption*: MLflow tracking interface comparing metric trajectories for stress 3.0 experiment runs.

### Section 5.4: Phase E2 Normalization Pairing
- **`03_figures/png/F08_phase_e2_normalization_pairing.png`**
  *Caption*: Post-gate normalization pairing comparison evaluating Identity (noG), LayerNorm (LN), and BatchNorm (BN).

---

## Chapter 6: System Performance and Timing Analysis

- **`03_figures/png/F09_estimator_timing_overhead.png`**
  *Caption*: Computational speed in seconds per iteration and total 64,000-iteration training time across model configurations.
- **`03_figures/png/F10_experiment_timeline.png`**
  *Caption*: Chronological execution timeline of experiment runs across project phases tracked in MLflow.
- **`04_ui_screenshots/mlflow_runs_table.png`**
  *Caption*: MLflow experiment runs table displaying start times, durations, and statuses sorted chronologically.

---

## Chapter 7: Summary and Future Directions

- **`03_figures/png/F12_master_summary_table_fig.png`**
  *Caption*: Comprehensive master summary table displaying test errors, sample sizes, and comparison targets across all project phases.
- **`03_figures/tables/master_summary.tex`**
  *Caption*: LaTeX master summary table documenting quantitative results for inclusion in final report publication.
- **`04_ui_screenshots/dashboard_hero.png`**
  *Caption*: Showcase dashboard hero section summarizing headline research findings and overall project KPIs.
- **`04_ui_screenshots/dashboard_decisions.png`**
  *Caption*: Interactive showcase dashboard trail tracking architectural decisions and resolution rationale.
