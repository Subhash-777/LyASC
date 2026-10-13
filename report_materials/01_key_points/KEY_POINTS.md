# Key Points Index by Report Chapter

This document outlines the primary findings, architectural decisions, and empirical results for each section of the project report.

---

## Chapter 1: Introduction and Motivation

- Deep residual networks with multi-residual additions or un-normalized skip paths suffer from severe gradient explosion or dynamic instability at high network depth [base_model/README.md].
- Standard remedies like fixed scalar scaling require manual tuning and lack adaptive dynamic range across different layer depths [LyASC_ARCHITECTURE.md].

---

## Chapter 2: Base Paper Reproduction

- We reproduced the base paper's residual scaling benchmark across 10 variants on CIFAR-10 at depth 110 with 3 random seeds per variant [base_model/results/tables/reproduced_table1_FINAL.csv].
- Reproduction confirmed 9 out of 10 Table 1 variants were within pre-registered tolerance bounds, validating the experimental pipeline [base_model/results/tables/reproduced_table1_FINAL.csv].
- Base paper un-normalized stress variants failed catastrophically at depth 110, showing mean test errors of 37.14% for LN(0.5x+F), 37.59% for LN(x+2F), and 49.27% for LN(x+3F) [base_model/results/tables/reproduced_table3_FINAL.csv].

---

## Chapter 3: LyASC Method and Formulation

- Lyapunov Adaptive Shortcut Control (LyASC) dynamically gates shortcut and residual branches using a scalar parameter lambda per block [LyASC_ARCHITECTURE.md].
- The gate equation alpha_l = sigmoid(lambda_l) computes convex combinations y_l = G(alpha_l * skip_scale * x_l + (1 - alpha_l) * residual_scale * F_l(x_l)) [lyasc_model/src/models/lyasc_skip.py].
- The stability estimator updates lambda using finite-difference gradient estimates derived from mini-batch feature variance [lyasc_model/src/models/stability_estimator.py].

---

## Chapter 4: System Implementation and Infrastructure

- The codebase is implemented in PyTorch 2.0+ and managed inside a isolated conda environment named lyasc-experiments [lyasc_model/environment_delta.yml].
- Complete automated unit test suite covers skip variants, stability estimator updates, and model integration with 78 passing tests [lyasc_model/tests/].
- MLflow tracking logs metrics, parameters, and checkpoints to a SQLite backend with full multi-session resume support [lyasc_model/src/tracking/mlflow_tracking.py].

---

## Chapter 5: Experimental Sequence and Findings

- Phase A/B established baseline stability benchmarks for standard skip connections [base_model/results/tables/reproduced_table1_FINAL.csv].
- Phase C evaluated un-normalized LyASC (LyASC_noG) at depth 110, achieving 9.66% ± 0.74% test error across 3 seeds (sample SD) [lyasc_model/results/tables/phase_c_results_FINAL.csv].
- Ablation 1 demonstrated that adding LayerNorm post-gate reduced test error to 7.22% for a single seed [lyasc_model/results/logs/lyasc_ablation_ln__LyASCplusLN__depth110__seed0.json].
- Phase D cold-start evaluation selected the "none" initialization strategy (7.17% ± 0.21%, n=3) over damped (7.46% ± 0.31%, n=3) and warmup (7.95% ± 0.21%, n=2) based on pragmatic simplicity [lyasc_model/results/logs/phase_d_coldstart/decisions.log].
- Phase E1 stress testing showed LyASC rescued 0.5x skip (8.38% vs 37.14%) and 2.0x residual (8.44% vs 37.59%) fully, while 3.0x residual produced two distinct outcome clusters (1 seed Full Rescue at 8.75%, 4 seeds Partial Rescue averaging 25.16%) [lyasc_model/results/logs/phase_e_stresstest/decisions.log].
- Phase E2 normalization pairing confirmed that post-gate normalization is necessary, with BatchNorm (6.34% ± 0.01%, n=2) showing an observed -0.83pp error reduction over LayerNorm (7.17% ± 0.21%, n=3) [lyasc_model/results/logs/phase_e_stresstest/decisions.log].

---

## Chapter 6: Limitations

- Estimator overhead increases per-iteration runtime from 0.261s/iter for un-normalized LyASC to 0.498s/iter for LyASC+LN due to LayerNorm tensor permute operations [report_materials/03_figures/tables/master_summary.csv].
- Under extreme 3.0x residual scaling, LyASC exhibits bimodal convergence dependent on early iteration optimization speed before 16k steps [lyasc_model/results/logs/phase_e_stresstest/decisions.log].

---

## Chapter 7: Planned Future Work

- Phase F extensions will investigate dynamic lambda learning rate schedulers to accelerate gate convergence under high stress.
- Future evaluations will test LyASC+BN under extreme scaling conditions and extend evaluation to ImageNet and transformer architectures.
