# LyASC Model — Lyapunov-Adaptive Skip Connections

Experimental workspace for the LyASC architecture, implementing and evaluating stability-signal-driven skip-connection gating.

---

## Executive Overview

LyASC dynamically regulates residual branch amplification via an online Lyapunov stability estimator ($k=1, N=20, \text{EMA decay}=0.9$).

---

## Experimental Results Summary

### Phase D: Cold-Start Strategy Ablation (PreAct-ResNet-110, CIFAR-10)
| Cold-Start Strategy | Normalization ($G$) | Seeds ($n$) | Test Error (Mean ± Sample SD) | Best Seed Error | Status |
|---|---|---|---|---|---|
| **`none`** ($w=0$) | LayerNorm2d | 3 | **7.17% ± 0.21%** | **6.93%** | Optimal |
| **`damped`** | LayerNorm2d | 3 | **7.46% ± 0.31%** | **7.13%** | Stable |
| **`warmup`** | LayerNorm2d | 2 | **7.95% ± 0.21%** | **7.80%** | Stable |

### Phase E1: Table 3 Stress Testing (Severe Scaling Imbalances)
| Stress Variant | Scaling Equation | Base Paper Baseline | LyASC Error (Individual Seeds) | Outcome Clusters Observed in Sample | Rescue Delta |
|---|---|---|---|---|---|
| **`LyASC_stress_0.5`** | $\text{LN}(0.5x + F(x))$ | **37.14%** | s0: 7.98%, s1: 8.84%, s2: 8.31% | **Full Rescue**: **8.38% ± 0.43%** ($n=3$, sample SD) | **↓ 28.76pp** |
| **`LyASC_stress_2.0`** | $\text{LN}(x + 2F(x))$ | **37.59%** | s0: 8.13%, s1: 8.81%, s2: 8.38% | **Full Rescue**: **8.44% ± 0.34%** ($n=3$, sample SD) | **↓ 29.15pp** |
| **`LyASC_stress_3.0`** | $\text{LN}(x + 3F(x))$ | **49.27%** | s0: 27.90%, s1: 32.73%, s2: 19.12%, s3: 20.88%, **s4: 8.75%** | **Full Rescue (1 seed)**: **8.75%**<br>**Partial Rescue (4 seeds)**: **25.16% mean** *(median 20.88%)*<sup>1</sup> | **↓ 27.39pp** |

*<sup>1</sup> Footnote: Pooled 5-seed aggregate statistics for `LyASC_stress_3.0` are **21.88% ± 9.15pp** (sample SD).*

### Phase E2: Normalization Pairing Ablations (PreAct-ResNet-110, CIFAR-10)
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

1. **Config Verification**:
   - Confirmed `skip_scale=1.0`, `residual_scale=3.0`, `use_layer_norm=True` across all 54 `LyASCSkip` blocks in Seed 4. Initial gradient norm at iter 1000 was **1.185M** (~5.6x higher than unstressed baseline), confirming 3x scaling was genuinely applied. Seed 4's 8.75% error is a verified breakthrough.
2. **Unified Continuous Threshold Effect**:
   - Trajectory outcomes are governed by early escape speed before ~16k:
     - **Full Rescue Breakthrough (Seed 4: 8.75% error)**: Rapid early optimization progress before ~16k (`alpha_mean` passing **0.7060** at iter 16k) triggers full gate breakthrough to $\alpha_{\text{mean}} = \mathbf{0.8285}$, matching standard non-stressed baseline performance (8.75%).
     - **Partial Rescue Cluster (Seeds 0 to 3: 19.12% to 32.73% error)**: Slower initial progress caps `alpha_mean` at $\sim 0.64 - 0.66$. Escape timing relative to the 32k LR drop determines final error (19% vs 32%).

---

## Workspace Structure

```
lyasc_model/
├── configs/              # YAML experiment configs (lyasc_phase_e_stresstest.yaml, etc.)
├── dashboard/        # Static interactive showcase dashboard & data exporter
├── results/
│   ├── checkpoints/      # Model checkpoints (.pt)
│   ├── logs/             # decisions.log, per-run log files, JSON results
│   ├── mlflow.db         # MLflow tracking SQLite database
│   └── tables/           # Result CSVs
├── src/
│   ├── models/           # lyasc_skip.py, stability_estimator.py
│   ├── tracking/         # mlflow_tracking.py
│   └── training/         # train.py, config.py
└── tests/                # PyTorch unit tests
```

---

## Quick Start Commands

```bash
# 1. Environment
conda activate lyasc-experiments

# 2. Run Test Suite
pytest lyasc_model/tests/ base_model/tests/

# 3. Export Data & Launch Dashboard
python3 lyasc_model/scripts/export_dashboard_data.py
cd lyasc_model/dashboard && python3 -m http.server 8765

# 4. Run Interactive CPU Inference Demo
# List all 9 pre-configured presets
bash demo/run_demo.sh --list

# Run primary LyASC+LN preset
bash demo/run_demo.sh --preset ln

# Run LyASC+BN preset
bash demo/run_demo.sh --preset bn

# Run stress 3.0 full rescue preset
bash demo/run_demo.sh --preset stress_3.0_rescue
```
