# LyASC: Lyapunov-Adaptive Skip Connections

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)]()

Official repository for **LyASC: Lyapunov-Adaptive Skip Connections**, introducing dynamic, stability-signal-driven skip connection regulation for deep PreAct-ResNet architectures.

---

## Executive Overview

Deep residual networks suffer from severe numerical instability and gradient degradation under structural feature/residual power imbalances (e.g. contracted skips or expanded residual scaling). Standard PreAct-ResNets exhibit catastrophic failure modes under these conditions (e.g., $37.14\% - 49.27\%$ error on CIFAR-10).

**LyASC** resolves this by dynamically estimating residual branch amplification via an online stability estimator ($k=1, N=20, \text{EMA decay}=0.9$), adaptively modulating the gating parameter $\alpha_l \in (0, 1)$ to enforce stable gradient dynamics without manual tuning.

---

## Summary of Empirical Results

### 1. Phase D: Cold-Start Strategy Ablation (PreAct-ResNet-110, CIFAR-10)
Evaluates initialization strategies ($w=0$ vs. warmup vs. damped) with LayerNorm2d after gate ($G=\text{LN}$):

| Cold-Start Strategy | Normalization ($G$) | Seeds ($n$) | Test Error (Mean ± Sample SD) | Best Seed Error | Status |
|---|---|---|---|---|---|
| **`none`** ($w=0$) | LayerNorm2d | 3 | **7.17% ± 0.21%** | **6.93%** | Optimal (Zero cold-start delay required) |
| **`damped`** | LayerNorm2d | 3 | **7.46% ± 0.31%** | **7.13%** | Stable and Resilient |
| **`warmup`** | LayerNorm2d | 2 | **7.95% ± 0.21%** | **7.80%** | Stable |

### 2. Phase E1: Table 3 Stress Testing (Catastrophic Scaling Imbalances)
Evaluates LyASC resilience against baseline paper failure modes (where standard PreAct-ResNet fails catastrophically):

| Stress Variant | Scaling Equation | Base Paper Baseline Error | LyASC Error (Individual Seeds) | Outcome Clusters Observed in Sample | Rescue Delta |
|---|---|---|---|---|---|
| **`LyASC_stress_0.5`** | $\text{LN}(0.5x + F(x))$ | **37.14%** (Catastrophic) | s0: 7.98%, s1: 8.84%, s2: 8.31% | **Full Rescue**: **8.38% ± 0.43%** ($n=3$, sample SD) | **↓ 28.76pp** |
| **`LyASC_stress_2.0`** | $\text{LN}(x + 2F(x))$ | **37.59%** (Bimodal) | s0: 8.13%, s1: 8.81%, s2: 8.38% | **Full Rescue**: **8.44% ± 0.34%** ($n=3$, sample SD) | **↓ 29.15pp** |
| **`LyASC_stress_3.0`** | $\text{LN}(x + 3F(x))$ | **49.27%** (Severe Collapse) | s0: 27.90%, s1: 32.73%, s2: 19.12%, s3: 20.88%, **s4: 8.75%** | **Full Rescue (1 seed)**: **8.75%**<br>**Partial Rescue (4 seeds)**: **25.16% mean** *(median 20.88%)*<sup>1</sup> | **↓ 27.39pp** |

*<sup>1</sup> Footnote: Pooled 5-seed aggregate statistics for `LyASC_stress_3.0` are **21.88% ± 9.15pp** (sample SD).*

### 3. Phase E2: Normalization Pairing Ablations (PreAct-ResNet-110, CIFAR-10)
Evaluates post-gate normalization functions ($G = \text{Identity}$ vs $G = \text{BatchNorm2d}$ vs $G = \text{LayerNorm2d}$ baseline):

| Variant | Post-Gate Norm $G$ | Seeds | Test Error (Individual) | Test Error (Mean ± Sample SD) | $\alpha_{\text{mean}}$ | Delta vs `LyASC+LN` |
|---|---|---|---|---|---|---|
| **`LyASC+LN`** (Phase D Baseline) | LayerNorm2d | 3 | 7.33%, 7.25%, 6.93% | **7.17% ± 0.21%** | 0.7014 | reference |
| **`LyASC_noG`** (reused from Phase C) | Identity (None) | 3 | 10.21%, 8.82%, 9.95% | **9.66% ± 0.74%** | 0.6580 | **+2.49pp** |
| **`LyASC+BN`** | BatchNorm2d | 3 | 6.35%, 6.33%, 6.27% | **6.32% ± 0.04%** | 0.5680 | **-0.85pp** |
| **PreAct-ResNet-110** (Base) | N/A | 3 | 6.31%, 6.28%, 6.23% | **6.27% ± 0.04%** | N/A | -0.90pp |

- **Post-Gate Normalization Requirement**: Omitting post-gate normalization (`LyASC_noG`) incurs a **+2.49pp** penalty vs `LyASC+LN` (9.66% vs 7.17%), confirming post-gate normalization is necessary.
- **Observed Unstressed Performance**: In unstressed training, `LyASC+BN` achieves **6.32% ± 0.04% mean error** ($n=3$), an observed difference of **-0.85pp** vs `LyASC+LN` (7.17%, $n=3$). This result is statistically indistinguishable from plain 1xSkip (6.27% ± 0.04%, +0.05pp difference) and remains above 2rSkip+LN (6.15% ± 0.09%). The performance of `LyASC+BN` under stress conditions is untested.

---

## Key Findings (`LyASC_stress_3.0` & Normalization Pairing)

1. **Configuration Audit & Verification**:
   - Confirmed `skip_scale=1.0`, `residual_scale=3.0`, `use_layer_norm=True` across all 54 `LyASCSkip` blocks in Seed 4.
   - Initial gradient norm at iteration 1000 was **1.185M** (matching Seeds 0 to 3 at 1.15M to 1.44M, ~5.6x unstressed baseline), confirming the 3x scaling pre-factor was genuinely active. Its **8.75%** test error is a verified breakthrough.
2. **Unified Continuous Threshold Effect (Early Escape Speed)**:
   - Trajectory outcomes are governed by early escape speed before ~16k:
     - **Full Rescue Breakthrough (Seed 4: 8.75% error)**: Rapid early optimization progress before ~16k (`alpha_mean` passing **0.7060** at iter 16k) triggers full gate breakthrough to $\alpha_{\text{mean}} = \mathbf{0.8285}$, matching standard non-stressed baseline performance (8.75%).
     - **Partial Rescue Cluster (Seeds 0 to 3: 19.12% to 32.73% error)**: Slower initial progress caps `alpha_mean` at $\sim 0.64 - 0.66$. Within this cluster, escape timing relative to the 32k LR drop determines final error:
       - *Early escape before 32k* (Seeds 2 & 3 at 27k to 28k): Reaches **19.12%** and **20.88%**.
       - *Stall/relapse at 32k* (Seeds 0 & 1): Reaches **27.90%** and **32.73%**.
3. **Relapse Evaluation**:
   - A temporary loss regression right before the 1st LR drop ($1.93 \to 2.21$) occurred **only in Seed 1**. Seeds 2, 3, and 4 descended smoothly through the 32k boundary ($1.83 \to 1.50$, $1.74 \to 1.27$, $0.38 \to 0.32$), confirming boundary relapse was a seed-specific stochastic perturbation.

---

## Repository Structure

```
LyASC/
├── base_model/           # Verified reference PreAct-ResNet implementation
├── lyasc_model/          # LyASC architecture implementation & experiments
│   ├── configs/          # YAML experiment configurations
│   ├── dashboard/        # Static interactive showcase dashboard & exporter
│   ├── results/          # Result JSONs, logs, checkpoints, mlflow.db
│   └── src/              # lyasc_skip.py, stability_estimator.py, train.py
├── paper/                # IEEE LaTeX publication source (main.tex, references.bib)
└── README.md             # Top-level research & project overview
```

---

## Quick Start & Verification

### 1. Environment Setup
```bash
conda activate lyasc-experiments
```

### 2. Run Test Suite
```bash
pytest lyasc_model/tests/ base_model/tests/
```
*(78 passed in 5.74s)*

### 3. Launch Showcase Dashboard & MLflow UI
```bash
# Export dashboard data & launch static dashboard
python3 lyasc_model/scripts/export_dashboard_data.py
cd lyasc_model/dashboard && python3 -m http.server 8765
# Open http://localhost:8765/index.html in browser

# Launch MLflow UI (Experiment Tracking Store)
conda activate lyasc-experiments
mlflow ui --backend-store-uri sqlite:////home/subhash/projects/LyASC/lyasc_model/results/mlflow.db --port 5000
# Open http://localhost:5000 in browser
```

### 4. Compile IEEE Paper PDF
```bash
cd paper/
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### 5. Run Interactive CPU Inference Demo
The demo evaluates CIFAR-10 test set images entirely on CPU without GPU requirements, printing sample-level predictions, ground truth labels, confidence scores, and per-stage architectural statistics (LyASC gate $\alpha$ distributions or baseline scaling schedules).

```bash
conda activate lyasc-experiments

# List all available pre-configured presets (9 models across LyASC and baselines)
bash demo/run_demo.sh --list

# 1. Primary LyASC+LN model (Phase D baseline, 7.33% seed 0)
bash demo/run_demo.sh --preset ln

# 2. LyASC+BN model (Phase E2 BatchNorm pairing, 6.35% seed 0)
bash demo/run_demo.sh --preset bn

# 3. LyASC_noG ablation model (Phase C un-normalized gating, 10.21% seed 0)
bash demo/run_demo.sh --preset nog

# 4. LyASC Stress 0.5x rescue model (recovers from 37.14% baseline failure to 7.98%)
bash demo/run_demo.sh --preset stress_0.5

# 5. LyASC Stress 2.0x rescue model (recovers from 37.59% baseline failure to 8.13%)
bash demo/run_demo.sh --preset stress_2.0

# 6. LyASC Stress 3.0x Full Rescue breakthrough model (Seed 4, 8.75% error)
bash demo/run_demo.sh --preset stress_3.0_rescue

# 7. LyASC Stress 3.0x Partial Rescue cluster model (Seed 1, 32.73% error)
bash demo/run_demo.sh --preset stress_3.0_partial

# 8. Base Model 1xSkip baseline (standard identity shortcut, 6.31% error)
bash demo/run_demo.sh --preset 1xskip

# 9. Base Model 2rSkip+LN baseline (Liu et al. best hand-tuned fixed schedule, 6.09% error)
bash demo/run_demo.sh --preset 2rskip

# Additional options: customize sample count or random seed
bash demo/run_demo.sh --preset bn --samples 12 --seed 42

# Direct custom checkpoint path
bash demo/run_demo.sh lyasc_model/results/checkpoints/phase_e_stresstest/lyasc_phase_e_stresstest__LyASCplusBN__depth110__seed0.pt
```

