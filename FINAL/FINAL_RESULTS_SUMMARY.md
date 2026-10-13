# LyASC Final Results Summary

Master empirical summary across all experimental phases for PreAct-ResNet-110 on CIFAR-10.
All values trace to verified result JSONs and MLflow runs. Standard deviations are sample SD (ddof=1).

---

## 1. Experimental Matrix & Comparison Summary

| Phase | Variant | Post-Gate Norm | n | Seeds (%) | Mean (%) | Sample SD | Baseline Target | Target Value (%) | Delta vs Target (pp) | Status / Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **Phase C** | `LyASC_noG` | Identity (None) | 3 | 10.21, 8.82, 9.95 | **9.66** | ± 0.74 | `1xSkip` | 6.27 | +3.39 | Un-normalized baseline |
| **Phase D** | `LyASC+LN (none)` | LayerNorm2d | 3 | 7.33, 7.25, 6.93 | **7.17** | ± 0.21 | `2rSkip+LN` | 6.15 | +1.02 | Optimal cold-start strategy |
| **Phase D** | `LyASC+LN (damped)` | LayerNorm2d | 3 | 7.80, 7.21, 7.37 | **7.46** | ± 0.31 | `LyASC+LN (none)` | 7.17 | +0.29 | Stable alternative |
| **Phase D** | `LyASC+LN (warmup)` | LayerNorm2d | 2 | 8.10, 7.80 | **7.95** | ± 0.21 | `LyASC+LN (none)` | 7.17 | +0.78 | Inferior to none |
| **Phase E1** | `LyASC_stress_0.5` | LayerNorm2d | 3 | 7.98, 8.84, 8.31 | **8.38** | ± 0.43 | Base LN(0.5x+F) | 37.14 | -28.76 | Full Rescue (↓ 28.76pp) |
| **Phase E1** | `LyASC_stress_2.0` | LayerNorm2d | 3 | 8.13, 8.81, 8.38 | **8.44** | ± 0.34 | Base LN(x+2F) | 37.59 | -29.15 | Full Rescue (↓ 29.15pp) |
| **Phase E1** | `LyASC_stress_3.0 (Full)` | LayerNorm2d | 1 | 8.75 | **8.75** | N/A | Base LN(x+3F) | 49.27 | -40.52 | Full Breakthrough (↓ 40.52pp) |
| **Phase E1** | `LyASC_stress_3.0 (Partial)` | LayerNorm2d | 4 | 27.90, 32.73, 19.12, 20.88 | **25.16** | ± 6.31 | Base LN(x+3F) | 49.27 | -24.11 | Partial Rescue (↓ 24.11pp) |
| **Phase E1** | `LyASC_stress_3.0 (Pooled)` | LayerNorm2d | 5 | 27.90, 32.73, 19.12, 20.88, 8.75 | **21.88** | ± 9.15 | Base LN(x+3F) | 49.27 | -27.39 | Bimodal pooled aggregate |
| **Phase E2** | `LyASC_noG` (reused Phase C) | Identity (None) | 3 | 10.21, 8.82, 9.95 | **9.66** | ± 0.74 | `LyASC+LN` | 7.17 | +2.49 | Confirms G is required |
| **Phase E2** | `LyASC+BN` | BatchNorm2d | 2 | 6.35, 6.33 | **6.34** | ± 0.01 | `2rSkip+LN` | 6.15 | +0.19 | Matches 2rSkip+LN / 1xSkip (seed 2 running in background) |

---

## 2. Key Acceptance Criteria Summary

1. **Matches Best Fixed Baseline (`2rSkip+LN` at 6.15% ± 0.40pp)**:
   - `LyASC+LN`: 7.17% ± 0.21% (+1.02pp gap) -> **NOT MET**.
   - `LyASC+BN`: 6.34% ± 0.01% (+0.19pp gap) -> **MET** (indistinguishable from plain 1xSkip at 6.27%; this is "matches", not "beats"; stress behavior untested).

2. **Beats Unconstrained Learned Gate (`wSkip+LN` at 8.16%)**:
   - `LyASC+LN` (7.17%) and `LyASC+BN` (6.34%) strictly beat 8.16% without requiring LR multiplier mitigations -> **MET**.
   - `LyASC_noG` (9.66%) underperforms -> **NOT MET** (confirms normalization requirement).

3. **Catastrophic Stress Recovery**:
   - `stress_0.5` (8.38% vs 37.14%) -> **MET** (Full Rescue).
   - `stress_2.0` (8.44% vs 37.59%) -> **MET** (Full Rescue).
   - `stress_3.0`: 1 seed Full Rescue (8.75%), 4 seeds Partial Rescue (25.16% mean) -> **MET / PARTIALLY RESOLVED** (governed by continuous threshold early escape dynamics).

---

## 3. Runtime Overhead & Computational Efficiency

| Normalization / Mode | Estimator Settings | Timing (s/iter) | Slowdown vs Plain ResNet (~0.18s/iter) | Memory Peak (4GB GPU) |
|---|---|---|---|---|
| Frozen baseline | k=2, N=10 | 0.1922 | 1.07x | ~3.8 GB |
| Active LyASC_noG | k=1, N=20 | 0.2614 | 1.45x | ~3.8 GB |
| Active LyASC+BN | k=1, N=20 | 0.3475 | 1.93x | ~3.8 GB |
| Active LyASC+LN | k=1, N=20 | 0.4979 | 2.76x | ~3.8 GB |

---

## 4. Test Suite and Checkpoint Integrity

- **Unit tests**: 78 passed in 5.74s (`lyasc_model/tests/` and `base_model/tests/`).
- **Checkpoint test re-evaluation**: 9 saved checkpoints re-evaluated on 10,000 CIFAR-10 test set images on CPU. All checkpoints reproduce recorded JSON test errors within 0.02 percentage points (criterion: <= 0.05pp).
- **Gate parameter maps**: Extracted across 54 blocks for 7 model variants (`report_materials/03_figures/png/F13_checkpoint_gate_maps.png` and `tables/checkpoint_gate_maps.csv`).
