# Phase C Gate Report — Multi-Seed LyASC Evaluation

## Consolidated Comparison Table

| Variant | Seeds | Mean Test Error (%) | Std (%) | Source | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 2rSkip+LN | 3 | 6.15 | 0.09 | base_model (verified) | PASS |
| 1xSkip | 3 | 6.27 | 0.04 | base_model (verified) | PASS |
| wSkip+LN | 3 | 8.16 | 0.13 | base_model (verified) | FAIL |
| **LyASC** | 3 | **9.66** | 0.74 | trained lyasc_model | **FAIL** |

## Gate Analysis

- **LyASC mean error**: `9.66%` (seeds: [10.21, 8.82, 9.95])
- **vs 2rSkip+LN** (best target baseline): `+3.51pp`
- **vs wSkip+LN** (learnable gate target baseline): `+1.50pp`
- **Acceptance threshold**: $\le 6.55\%$ (2rSkip+LN + 0.40pp tolerance)
- **Phase C Gate Outcome**: **FAIL**
