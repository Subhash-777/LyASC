# Claims Register

This document maintains a clear register of claims, categorizing empirical findings, active hypotheses, untested territory, and corrected earlier claims.

---

## 1. Supported Claims (Empirical Data-Backed, with n)

- **Base Paper Reproduction**: 9 out of 10 Table 1 variants reproduced within pre-registered tolerance bounds (n=3 seeds per variant) [base_model/results/tables/reproduced_table1_FINAL.csv].
- **Base Paper Stress Failure**: Un-normalized residual scaling (0.5x skip, 2.0x residual, 3.0x residual) leads to severe instability and high test errors (37.14%, 37.59%, 49.27% respectively, n=3 seeds each) [base_model/results/tables/reproduced_table3_FINAL.csv].
- **LyASC Stress Rescue**: LyASC gates fully stabilize 0.5x skip (8.38% ± 0.43%, n=3) and 2.0x residual (8.44% ± 0.34%, n=3) variants, reducing test error by ~28-29pp compared to base paper failure [lyasc_model/results/logs/phase_e_stresstest/decisions.log].
- **Phase D Cold-Start Strategy Selection**: The "none" initialization strategy (7.17% ± 0.21%, n=3) provides statistically equivalent performance to "damped" (7.46% ± 0.31%, n=3) while outperforming "warmup" (7.95% ± 0.21%, n=2), making "none" the pragmatic baseline choice [lyasc_model/results/logs/phase_d_coldstart/decisions.log].
- **Post-Gate Normalization Requirement**: Omitting post-gate normalization (LyASC_noG) incurs a +2.35pp test error penalty vs LyASC+LN (9.52% ± 0.98%, n=2 vs 7.17% ± 0.21%, n=3), confirming post-gate normalization is necessary for optimal performance [lyasc_model/results/logs/phase_e_stresstest/decisions.log].
- **LyASC+BN Unstressed Performance**: In unstressed training, LyASC+BN achieves 6.34% ± 0.01% test error (n=2), representing a -0.83pp observed difference vs LyASC+LN (7.17%), which is statistically indistinguishable from plain 1xSkip (6.27%) [lyasc_model/results/logs/phase_e_stresstest/decisions.log].

---

## 2. Hypotheses (Plausible Explanations Pending Further Proof)

- **Stress 3.0 Early-Escape Threshold**: The bimodal outcome distribution in stress 3.0 (1 seed Full Rescue at 8.75% vs 4 seeds Partial Rescue averaging 25.16%) is governed by optimization progress before iteration 16,000. Seeds that pass alpha_mean = 0.70 by iteration 16,000 escape the loss plateau early and achieve full rescue [lyasc_model/results/logs/phase_e_stresstest/decisions.log].
- **Alpha Movement Rate**: The slower movement of alpha in stress 3.0 in early iterations reflects optimization difficulty rather than a fundamental mathematical breakdown of the Lyapunov estimator [lyasc_model/results/logs/phase_e_stresstest/decisions.log].

---

## 3. Untested Claims (Future Directions / Open Questions)

- **BatchNorm Under High Stress**: The performance and dynamic stability of LyASC+BN under extreme scaling conditions (such as stress 2.0x or 3.0x) remains untested, as Phase E2 evaluated BN only under standard unstressed conditions [lyasc_model/results/logs/phase_e_stresstest/decisions.log].
- **Phase F Extensions**: Dynamic lambda learning rate schedulers and adaptive EMA decay rates remain untested extensions for future evaluation.
- **Large-Scale Architectures**: LyASC behavior on ImageNet-scale models or multi-head attention Transformer residual blocks remains untested.

---

## 4. Corrected Claims Register (Earlier Draft Claims & Replacement Realities)

- **Original Claim**: The theoretical gradient scale factor without normalization is 0.5^54.
  - **Correction**: Replaced by empirical gradient norm measurements showing active gradient clipping at max_norm=5.0 and explicit L2-norm tracking.
- **Original Claim**: Lambda value spread across layers was caused by saturated ReLU activations.
  - **Correction**: Replaced by stability estimator variance ratio dynamics tracking feature variance preservation across block layers.
- **Original Claim**: Warmup and damped cold-start strategies yielded bit-identical test errors.
  - **Correction**: Identified as a script parameter logging mix-up; distinct runs confirmed warmup (7.95%) is inferior to damped (7.46%) and none (7.17%).
- **Original Claim**: Headline stress 3.0 performance reported as a single pooled mean of 21.88%.
  - **Correction**: Replaced by two-cluster framing distinguishing Full Rescue (Seed 4: 8.75%) from Partial Rescue (Seeds 0-3: mean 25.16%), avoiding misleading pooled averages.
- **Original Claim**: LyASC+BN "proves superior" to LyASC+LN overall.
  - **Correction**: Replaced by nuanced framing: LyASC+BN shows an observed -0.83pp difference in unstressed training (6.34% vs 7.17%), but its behavior under stress conditions remains untested.
