# Issues and Resolutions

This document logs critical technical issues encountered during the project, detailing how each was detected, the root cause, the resolution, and the evidence file.

---

## 1. Import-Precedence Bug

- **What Happened**: Python module import precedence loaded system or base_model modules instead of local lyasc_model modules when running experiment scripts across subdirectories.
- **Detection**: Unintended attribute errors and module resolution failures during cross-folder script invocations.
- **Fix**: Added explicit `sys.path` prepend logic (`sys.path.insert(0, ...)`)) and environment guards at all script entry points.
- **Evidence File**: `lyasc_model/scripts/run_phase_e_stresstest.py` [L10-L25].

---

## 2. base_model Config Edit Reversion

- **What Happened**: A base_model configuration file was modified during local debugging, risking non-reproducibility of standard baseline experiments.
- **Detection**: Pre-flight git status audit flagged modified YAML files under `base_model/configs/`.
- **Fix**: Reverted the modified config file back to verbatim base paper hyperparameter values.
- **Evidence File**: `base_model/configs/cifar10_table1.yaml`.

---

## 3. PyTorch Device Mismatch

- **What Happened**: PyTorch tensors inside the stability estimator module were instantiated on CPU by default while model parameters resided on CUDA GPU.
- **Detection**: Runtime exception: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`.
- **Fix**: Updated `StabilityEstimator` to bind state buffers directly to the target device via explicit `.to(device)` calls during block initialization and forward pass.
- **Evidence File**: `lyasc_model/src/models/stability_estimator.py`.

---

## 4. MLflow Database Relative Path Problem

- **What Happened**: Relative path specification for `mlflow.db` caused new experiment scripts to instantiate separate SQLite database files inside local subdirectories.
- **Detection**: Audit of MLflow tracking runs revealed missing runs on the central dashboard.
- **Fix**: Enforced absolute URI path resolution (`sqlite:////home/subhash/projects/LyASC/lyasc_model/results/mlflow.db`) across all experiment launchers.
- **Evidence File**: `lyasc_model/src/tracking/mlflow_tracking.py`.

---

## 5. Wall-Clock Time Audit & Session Interruption

- **What Happened**: Recorded wall-clock time for LyASC+LN Seed 0 in log metadata showed 38,315 seconds, which appeared incongruous with single-session timing estimates.
- **Detection**: Audit of training log timestamps revealed that Task 1580 ran iterations 0 to 8000 (~5473s) before a system restart, saving a resume checkpoint. Task 1596 then resumed at iteration 8000 and ran to 64000 (56,000 iterations), recording 38,315 seconds for the post-resume session.
- **Fix**: Documented true cumulative wall-clock time (~43,789s / ~12.16h) and identified per-iteration time increase from 0.26s/iter to 0.68s/iter due to LayerNorm tensor operations.
- **Evidence File**: `lyasc_model/results/logs/decisions.log`.

---

## 6. Compute-Plan Unit Error Correction

- **What Happened**: Draft Phase E plan executive summary stated ~3.8h runtime per 64,000-iteration training run instead of ~12.16h.
- **Detection**: Cross-audit of s/iter benchmark measurements (0.6842 s/iter) revealed a unit conversion error in the original formula calculation.
- **Fix**: Corrected per-run wall-clock formula `64000 * (s/iter) / 3600` and updated budget estimates across all planning documents.
- **Evidence File**: `lyasc_model/results/logs/phase_e_stresstest/decisions.log`.

---

## 7. Phase E2 Baseline Mix-up and Provenance Restoration

- **What Happened**: Draft Phase E2 table figures (6.31%, 6.32%, 6.39%) were pulled from base paper reference values instead of the ground-truth Phase D LyASC+LN baseline.
- **Detection**: Baseline audit and provenance verification cross-checked Phase E2 draft tables against closed-out Phase D JSON results.
- **Fix**: Restored the true Phase D 3-seed baseline result (7.17% ± 0.21%, seeds 7.33%, 7.25%, 6.93%) and updated all Phase E2 comparisons without retraining.
- **Evidence File**: `lyasc_model/results/logs/phase_e_stresstest/decisions.log`.
