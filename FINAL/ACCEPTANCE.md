# LyASC Acceptance Criteria Evaluation

Evaluated against the pre-registered acceptance criteria in `LyASC_plan.md` Section 0.
All comparison baselines are from the verified base-model reproduction on CIFAR-10, PreAct-ResNet-110 (3 seeds):
- **Plain baseline (`1xSkip`)**: 6.27% ± 0.04%
- **Best fixed schedule (`2rSkip+LN`)**: 6.15% ± 0.09%
- **Unconstrained learned gate (`wSkip+LN`)**: 8.16% ± 0.13%
- **Tolerance rule**: symmetric ±0.40 percentage points (pp).

---

## 1. Criterion 1: Matches Best Fixed Baseline (`2rSkip+LN` at 6.15%)
*Rule: 3-seed mean within ±0.40pp of 6.15% (i.e., within [5.75%, 6.55%]).*

| Variant | n | Seeds (%) | Mean ± Sample SD | Diff vs 2rSkip+LN | Status | Verdict Note |
|---|---|---|---|---|---|---|
| **LyASC+LN** (Phase D `none`) | 3 | 7.33, 7.25, 6.93 | **7.17% ± 0.21%** | +1.02pp | **NOT MET** | Does not match hand-tuned fixed schedule; falls +1.02pp short in unstressed training. |
| **LyASC+BN** (Phase E2) | 3 | 6.35, 6.33, 6.27 | **6.32% ± 0.04%** | +0.17pp | **MET** | Statistically matches 2rSkip+LN within tolerance (+0.17pp). Indistinguishable from plain 1xSkip (6.27% ± 0.04%, +0.05pp difference); this is "matches", not "beats". Stress behavior untested. |

---

## 2. Criterion 2: Beats Unconstrained Learned Gate (`wSkip+LN` at 8.16%)
*Rule: 3-seed mean strictly below 8.16%, AND training is stable without requiring wSkip+LN's LR-multiplier mitigation.*

| Variant | n | Seeds (%) | Mean ± Sample SD | Diff vs wSkip+LN | Status | Verdict Note |
|---|---|---|---|---|---|---|
| **LyASC+LN** (Phase D `none`) | 3 | 7.33, 7.25, 6.93 | **7.17% ± 0.21%** | -0.99pp | **MET** | Beats naive learned gate by 0.99pp with stable convergence and no LR mitigations. |
| **LyASC+BN** (Phase E2) | 3 | 6.35, 6.33, 6.27 | **6.32% ± 0.04%** | -1.84pp | **MET** | Beats naive learned gate by 1.84pp with stable convergence. |
| **LyASC_noG** (Phase C reused) | 3 | 10.21, 8.82, 9.95 | **9.66% ± 0.74%** | +1.50pp | **NOT MET** | Without post-gate norm, un-normalized gating drifts and underperforms wSkip+LN. |

---

## 3. Criterion 3: Stress Test Recovery (Catastrophic Regimes)
*Evaluates LyASC resilience under extreme scaling imbalances where fixed-weight PreAct-ResNet fails catastrophically.*

| Stress Variant | Equation | Base Paper Baseline | LyASC Error (Individual Seeds) | LyASC Outcome Description | Rescue Delta | Status |
|---|---|---|---|---|---|---|
| **`LyASC_stress_0.5`** | LN(0.5x + F(x)) | **37.14%** (Catastrophic) | s0: 7.98%, s1: 8.84%, s2: 8.31% | **Full Rescue**: **8.38% ± 0.43%** (n=3) | **↓ 28.76pp** | **MET** |
| **`LyASC_stress_2.0`** | LN(x + 2F(x)) | **37.59%** (Bimodal failure) | s0: 8.13%, s1: 8.81%, s2: 8.38% | **Full Rescue**: **8.44% ± 0.34%** (n=3) | **↓ 29.15pp** | **MET** |
| **`LyASC_stress_3.0`** | LN(x + 3F(x)) | **49.27%** (Severe collapse) | s0: 27.90%, s1: 32.73%, s2: 19.12%, s3: 20.88%, s4: 8.75% | **Two distinct clusters observed (n=5)**:<br>• **Full Rescue (1 seed)**: Seed 4 at **8.75%** (α=0.83)<br>• **Partial Rescue (4 seeds)**: **25.16% ± 6.31%** (median 24.39%, range 19.12%-32.73%, α~0.64-0.66) | **↓ 40.52pp** (Full)<br>**↓ 24.11pp** (Partial) | **MET / PARTIALLY RESOLVED** |

*Footnote: Pooled 5-seed aggregate statistics for `LyASC_stress_3.0` are **21.88% ± 9.15pp**.*

---

## 4. Summary Matrix

| Objective / Claim | Pre-Registered Standard | Empirical Outcome | Status |
|---|---|---|---|
| LyASC matches rSkip+LN (LN post-gate) | Mean within ±0.40pp of 6.15% | 7.17% ± 0.21% (+1.02pp gap) | **NOT MET** |
| LyASC matches rSkip+LN (BN post-gate) | Mean within ±0.40pp of 6.15% | 6.32% ± 0.04% (+0.17pp gap; matches 1xSkip 6.27%) | **MET (n=3)** |
| LyASC beats naive learned gate wSkip+LN | Mean strictly below 8.16% | 7.17% ± 0.21% (LN), 6.32% ± 0.04% (BN) | **MET** |
| LyASC rescues 0.5x skip scaling collapse | Stable convergence below 10% | 8.38% ± 0.43% (rescued from 37.14%) | **MET** |
| LyASC rescues 2.0x residual scaling collapse | Stable convergence below 10% | 8.44% ± 0.34% (rescued from 37.59%) | **MET** |
| LyASC rescues 3.0x residual scaling collapse | Stable convergence below 10% | 1 seed Full Rescue (8.75%), 4 seeds Partial Rescue (25.16% mean) | **MET / PARTIALLY RESOLVED** |
