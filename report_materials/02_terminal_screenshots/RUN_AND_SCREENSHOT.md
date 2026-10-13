# Terminal Command Capture Instructions for Manual Screenshots

This document lists the exact sequence of commands used to generate the terminal screenshot captures T01 through T14. These commands can be executed in an active terminal window to capture native GUI screenshots if required.

---

## Prerequisites

Activate the project conda environment prior to running commands:

```bash
source /home/subhash/miniconda3/bin/activate lyasc-experiments
cd /home/subhash/projects/LyASC
```

---

## Command Sequence

### T01 - Conda Environment Verification
```bash
conda info --envs && which python
```
*Expected Output*: Displays list of conda environments with `lyasc-experiments` active and its Python path.

---

### T02 - PyTorch CUDA & GPU Status
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" && nvidia-smi
```
*Expected Output*: PyTorch version, CUDA availability, GPU device name, and `nvidia-smi` status.

---

### T03 - Workspace Structure
```bash
tree -L 3 -I 'data|checkpoints|__pycache__|*.pyc|*.png|*.pdf|*.db|report_materials'
```
*Expected Output*: Directory tree of the workspace up to 3 levels, excluding large data and checkpoint directories.

---

### T04 - Test Suite Summary
```bash
pytest lyasc_model/tests/ base_model/tests/
```
*Expected Output*: Execution summary of all unit and integration tests (78 passed).

---

### T05 - Base Model Reproduction Summary
```bash
python base_model/scripts/summarize_all.py
```
*Expected Output*: Formatted summary table of base paper reproduction results across Tables 1, 3, 4, and 6.

---

### T06 - Phase C Training Log Header & Tail
```bash
head -n 25 lyasc_model/results/logs/phase_c_LyASC_seed1.log
echo "... [TRUNCATED] ..."
tail -n 25 lyasc_model/results/logs/phase_c_LyASC_seed1.log
```
*Expected Output*: Schedule configuration header, learning rate step drops, and final evaluation error.

---

### T07 - Resume from Checkpoint Log
```bash
grep -C 3 'Resumed from checkpoint' lyasc_model/results/logs/lyasc_table1_equiv__LyASC__seed2.log
```
*Expected Output*: Log entries showing checkpoint saving and resume confirmation at iteration 32000.

---

### T08 - Result JSON Files
```bash
cat lyasc_model/results/logs/lyasc_table1_equiv__LyASC__depth110__seed0.json
echo ""
cat lyasc_model/results/logs/phase_e_stresstest/lyasc_phase_e_stresstest__LyASC_stress_3.0__depth110__seed0.json
```
*Expected Output*: Raw JSON payload containing configuration, test error, wall clock time, and Lyapunov diagnostics.

---

### T09 - Selected decisions.log Entries
```bash
echo "=== PHASE D COLD-START SELECTION ==="
grep -A 15 "DECISION: phase_d_cold_start_strategy_selection" lyasc_model/results/logs/phase_d_coldstart/decisions.log
echo ""
echo "=== PHASE E2 NORMALIZATION ABLATION ==="
grep -A 10 "DECISION: phase_e2_normalization_pairing_ablation_complete" lyasc_model/results/logs/phase_e_stresstest/decisions.log
```
*Expected Output*: Structured decision entries detailing cold-start strategy selection and Phase E2 baseline provenance.

---

### T10 - MLflow Runs Table
```bash
python -c "import sqlite3, pandas as pd; conn = sqlite3.connect('/tmp/lyasc_mlflow_copy/mlflow.db'); pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); df = pd.read_sql_query('SELECT run_uuid, experiment_id, name, status, start_time, end_time FROM runs ORDER BY start_time ASC', conn); df['start_time'] = pd.to_datetime(df['start_time'], unit='ms'); df['end_time'] = pd.to_datetime(df['end_time'], unit='ms'); print(df.to_string(index=False))"
```
*Expected Output*: Table of experiment runs sorted chronologically from the copied MLflow database.

---

### T11 - Environment Guard Rejection
```bash
/usr/bin/python3 lyasc_model/scripts/run_phase_e_stresstest.py
```
*Expected Output*: RuntimeError exception thrown by the environment guard before any third-party imports.

---

### T12 - Git Log & Working Tree Status
```bash
git log --oneline -n 12 && echo "" && git status
```
*Expected Output*: Recent git commit history and clean git working tree status.

---

### T13 - Paper LaTeX Compilation
```bash
pdflatex paper/main.tex
```
*Expected Output*: Compilation log output or status check for paper LaTeX source compilation.

---

### T14 - Active GPU Training Verification
```bash
nvidia-smi
```
*Expected Output*: GPU process table verifying no background training runs are active.
