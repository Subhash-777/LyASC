
[2026-08-09T16:22] Periodic Check: 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running.
- GPU: RTX 3050 is active and training is progressing.
- Anomalies: None. Seed 0 and Seed 1 were already completed prior to this session.

[2026-08-09T16:30] Periodic Check: 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=1000.
- GPU: RTX 3050 is at 100% usage and training is progressing normally.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T17:00] Periodic Check (Iteration 2): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=7000.
- GPU: RTX 3050 is at 100% usage and training is progressing normally.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T17:30] Periodic Check (Iteration 3): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=13000.
- GPU: RTX 3050 is fully utilized and training is progressing normally.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T18:00] Periodic Check (Iteration 4): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=20000.
- GPU: RTX 3050 is fully utilized and training is progressing normally.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T18:30] Periodic Check (Iteration 5): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=27000.
- GPU: RTX 3050 is fully utilized and training is progressing normally.
- Anomalies: None. No NaNs or Infs found in recent logs. Skip logic confirmed successfully working during dry run test.

[2026-08-09T19:00] Periodic Check (Iteration 6): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=34000.
- Milestone: Passed iteration 32000 (LR schedule drop from 0.1 to 0.01). Test error at iter 32000 was 10.77%.
- GPU: RTX 3050 is fully utilized and training is progressing perfectly.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T19:30] Periodic Check (Iteration 7): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=41000.
- GPU: RTX 3050 is fully utilized and training is progressing perfectly (train_acc > 99%).
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T20:00] Periodic Check (Iteration 8): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated beautifully at 6.22%.
- GPU: RTX 3050 is fully utilized and training is progressing perfectly.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T20:30] Periodic Check (Iteration 9): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=55000.
- State: Final learning rate plateau (0.001). Loss is minimal (0.0029), and train_acc is 99.95%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T21:00] Periodic Check (Iteration 10): 
- Current execution: 2xSkip+BN seed 2 (Table 4) is running at iter=62000.
- State: Final learning rate plateau (0.001). Loss is minimal (0.0027), and train_acc is 99.95%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T21:30] Periodic Check (Iteration 11): 
- 2xSkip+BN seed 2 has FINISHED! Final test error is 6.11%.
- 3xSkip+BN seed 0 has now started and is currently at iter 5000.

[2026-08-09T22:00] Periodic Check (Iteration 12): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=12000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.2320) and train_acc is 91.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T22:30] Periodic Check (Iteration 13): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=19000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.1854) and train_acc is 93.57%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T23:00] Periodic Check (Iteration 14): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=26000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.1647) and train_acc is 94.21%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-09T23:30] Periodic Check (Iteration 15): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=32000.
- Milestone: Passed iteration 32000 (LR schedule drop from 0.1 to 0.01). Test error at iter 32000 was 9.61%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T00:00] Periodic Check (Iteration 16): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=39000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0102) and train_acc is 99.73%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T00:30] Periodic Check (Iteration 17): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=46000.
- State: Late fine-tuning phase (lr=0.01). Loss is excellent (0.0057) and train_acc is 99.88%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T01:00] Periodic Check (Iteration 18): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated beautifully at 6.04%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T01:30] Periodic Check (Iteration 19): 
- Current execution: 3xSkip+BN seed 0 (Table 4) is running at iter=59000.
- State: Final learning rate plateau (0.001). Loss is minimal (0.0030) and train_acc is 99.96%. 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T02:00] Periodic Check (Iteration 20): 
- 3xSkip+BN seed 0 has FINISHED! Final test error is 5.93%.
- 3xSkip+BN seed 1 has now started and is currently at iter 1000.

[2026-08-10T02:30] Periodic Check (Iteration 21): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=8000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.2863) and train_acc is 90.13%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T03:00] Periodic Check (Iteration 22): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=15000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.2103) and train_acc is 92.70%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T03:30] Periodic Check (Iteration 23): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=21000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.1817) and train_acc is 93.73%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T04:00] Periodic Check (Iteration 24): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=28000.
- State: Late early phase (lr=0.1). Loss is 0.1632 and train_acc is 94.30%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T04:30] Periodic Check (Iteration 25): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=34000.
- Milestone: Passed iteration 32000 (LR schedule drop from 0.1 to 0.01). Test error at iter 32000 was 10.61%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T05:00] Periodic Check (Iteration 26): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=41000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0081) and train_acc is 99.80%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T05:30] Periodic Check (Iteration 27): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated beautifully at 6.08%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T06:00] Periodic Check (Iteration 28): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=54000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0034) and train_acc is 99.95%. 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T06:30] Periodic Check (Iteration 29): 
- Current execution: 3xSkip+BN seed 1 (Table 4) is running at iter=61000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0029) and train_acc is 99.95%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T07:00] Periodic Check (Iteration 30): 
- 3xSkip+BN seed 1 has FINISHED! Final test error is 5.97%.
- 3xSkip+BN seed 2 (the final seed for this variant) has started and is currently at iter 3000.

[2026-08-10T07:30] Periodic Check (Iteration 31): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=10000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.2548) and train_acc is 91.14%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T08:00] Periodic Check (Iteration 32): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=16000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.2027) and train_acc is 92.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T08:30] Periodic Check (Iteration 33): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=23000.
- State: Early training phase (lr=0.1). Loss is dropping steadily (0.1735) and train_acc is 93.97%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T09:00] Periodic Check (Iteration 34): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=30000.
- State: Late early phase (lr=0.1). Loss is 0.1605 and train_acc is 94.40%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T09:30] Periodic Check (Iteration 35): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=36000.
- Milestone: Passed iteration 32000 (LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (lr=0.01). Loss is 0.0185 and train_acc is 99.46%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T10:00] Periodic Check (Iteration 36): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=43000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0066) and train_acc is 99.85%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T10:30] Periodic Check (Iteration 37): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=49000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated at 6.51%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T11:00] Periodic Check (Iteration 38): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=56000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0029) and train_acc is 99.95%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T11:30] Periodic Check (Iteration 39): 
- Current execution: 3xSkip+BN seed 2 (Table 4) is running at iter=63000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0025) and train_acc is 99.96%. Extremely close to completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T12:00] Periodic Check (Iteration 40): 
- 3xSkip+BN seed 2 (Table 4) has FINISHED! Final test error is 6.07%.
- Mean error for 3xSkip+BN is 5.99% (Outperforms paper 6.50% by 0.51).
- Queue has automatically advanced to 1rSkip+BN seed 0, which is running at iter 5000.

[2026-08-10T12:30] Periodic Check (Iteration 41): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=12000.
- State: Early training phase (lr=0.1). Loss is 0.4468 and train_acc is 84.69%.
- Note: Initial cold-start gradient clipping triggered as expected, keeping training stable.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T13:00] Periodic Check (Iteration 42): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=19000.
- State: Early training phase (lr=0.1). Loss is 0.2910 and train_acc is 89.97%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T13:30] Periodic Check (Iteration 43): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=26000.
- State: Early training phase (lr=0.1). Loss is 0.2268 and train_acc is 92.08%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T14:00] Periodic Check (Iteration 44): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running past iter=32000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated at 11.14%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T14:30] Periodic Check (Iteration 45): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=39000.
- State: Fine-tuning phase (lr=0.01). Loss is 0.0148 and train_acc is 99.59%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T15:00] Periodic Check (Iteration 46): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=46000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0078) and train_acc is 99.79%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T15:30] Periodic Check (Iteration 47): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=53000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001).
- State: Final learning rate plateau (0.001). Loss is excellent (0.0041) and train_acc is 99.92%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T16:00] Periodic Check (Iteration 48): 
- Current execution: 1rSkip+BN seed 0 (Table 4) is running at iter=60000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0031) and train_acc is 99.94%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T16:30] Periodic Check (Iteration 49): 
- 1rSkip+BN seed 0 (Table 4) has FINISHED! Final test error is 6.34%.
- Queue has automatically advanced to 1rSkip+BN seed 1, which is running at iter 2000.

[2026-08-10T17:00] Periodic Check (Iteration 50): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=9000.
- State: Early training phase (lr=0.1). Loss is 0.5618 and train_acc is 80.59%.
- Note: Initial cold-start gradient clipping triggered successfully as expected.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T17:30] Periodic Check (Iteration 51): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=16000.
- State: Early training phase (lr=0.1). Loss is 0.3167 and train_acc is 89.04%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T18:00] Periodic Check (Iteration 52): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=23000.
- State: Early training phase (lr=0.1). Loss is 0.2420 and train_acc is 91.60%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T18:30] Periodic Check (Iteration 53): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=29000.
- State: Early training phase (lr=0.1). Loss is 0.2088 and train_acc is 92.72%. Nearing first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T19:00] Periodic Check (Iteration 54): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=36000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated at 12.13%.
- State: Fine-tuning phase (lr=0.01). Loss is 0.0257 and train_acc is 99.20%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T19:30] Periodic Check (Iteration 55): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=43000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0091) and train_acc is 99.77%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T20:00] Periodic Check (Iteration 56): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=50000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated at 6.77%.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0042) and train_acc is 99.90%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T20:30] Periodic Check (Iteration 57): 
- Current execution: 1rSkip+BN seed 1 (Table 4) is running at iter=57000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0031) and train_acc is 99.94%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T21:00] Periodic Check (Iteration 58): 
- 1rSkip+BN seed 1 (Table 4) has FINISHED! Final test error is 6.44%.
- Queue has automatically advanced to 1rSkip+BN seed 2, which has just started its run.

[2026-08-10T21:30] Periodic Check (Iteration 59): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=7000.
- State: Early training phase (lr=0.1). Loss is 0.6267 and train_acc is 78.48%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T22:00] Periodic Check (Iteration 60): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=15000.
- State: Early training phase (lr=0.1). Loss is 0.2971 and train_acc is 89.72%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T22:30] Periodic Check (Iteration 61): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=22000.
- State: Early training phase (lr=0.1). Loss is 0.2350 and train_acc is 91.83%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T23:00] Periodic Check (Iteration 62): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=29000.
- State: Early training phase (lr=0.1). Loss is 0.2038 and train_acc is 92.86%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-10T23:30] Periodic Check (Iteration 63): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=36000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated at 13.69%.
- State: Fine-tuning phase (lr=0.01). Loss is 0.0234 and train_acc is 99.27%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T00:00] Periodic Check (Iteration 64): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=43000.
- State: Fine-tuning phase (lr=0.01). Loss is excellent (0.0082) and train_acc is 99.80%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T00:30] Periodic Check (Iteration 65): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=50000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated at 6.79%.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0041) and train_acc is 99.92%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T01:00] Periodic Check (Iteration 66): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=56000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0033) and train_acc is 99.94%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T01:30] Periodic Check (Iteration 67): 
- Current execution: 1rSkip+BN seed 2 (Table 4) is running at iter=63000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0025) and train_acc is 99.96%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T02:00] Periodic Check (Iteration 68): 
- 1rSkip+BN seed 2 (Table 4) has FINISHED! Final test error is 6.39%.
- ALL 3 SEEDS FOR 1rSkip+BN ARE COMPLETE. Mean test error: 6.39%. This outperforms the paper's target of 8.23% by a massive 1.84 points.
- Queue has automatically advanced to 2rSkip+BN seed 0, which is running at iter 5000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T02:30] Periodic Check (Iteration 69): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=11000.
- State: Early training phase (lr=0.1). Loss is 0.2533 and train_acc is 91.17%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: Minor gradient clipping during cold-start (8/1000 steps), which is expected for extended-skip networks and correctly caught by the 5.0 max_norm limit.

[2026-08-11T03:00] Periodic Check (Iteration 70): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=17000.
- State: Early training phase (lr=0.1). Loss is 0.2014 and train_acc is 92.98%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T03:30] Periodic Check (Iteration 71): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=24000.
- State: Early training phase (lr=0.1). Loss is 0.1751 and train_acc is 93.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T04:00] Periodic Check (Iteration 72): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=30000.
- State: Early training phase (lr=0.1). Loss is 0.1615 and train_acc is 94.36%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T04:30] Periodic Check (Iteration 73): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=36000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated at 11.62%.
- State: Fine tuning phase (lr=0.01). Loss is 0.0185 and train_acc is 99.50%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T05:00] Periodic Check (Iteration 74): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=42000.
- State: Fine tuning phase (lr=0.01). Loss is excellent (0.0077) and train_acc is 99.80%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T05:30] Periodic Check (Iteration 75): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated at 6.70%.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0058) and train_acc is 99.85%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T06:00] Periodic Check (Iteration 76): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=54000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0033) and train_acc is 99.93%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T06:30] Periodic Check (Iteration 77): 
- Current execution: 2rSkip+BN seed 0 (Table 4) is running at iter=60000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0026) and train_acc is 99.96%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T07:00] Periodic Check (Iteration 78): 
- 2rSkip+BN seed 0 (Table 4) has FINISHED! Final test error is 6.34% (outperforming the paper's 8.27% target by a wide margin, matching the pattern).
- Queue has automatically advanced to 2rSkip+BN seed 1, which is running at iter 2000.
- State: Early training phase (lr=0.1). Loss is 0.7515 and train_acc is 73.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: Minor gradient clipping during cold-start (6/1000 steps), safely caught by max_norm limit.

[2026-08-11T07:30] Periodic Check (Iteration 79): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=8000.
- State: Early training phase (lr=0.1). Loss is 0.2977 and train_acc is 89.67%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T08:00] Periodic Check (Iteration 80): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=14000.
- State: Early training phase (lr=0.1). Loss is 0.2206 and train_acc is 92.29%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T08:30] Periodic Check (Iteration 81): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=21000.
- State: Early training phase (lr=0.1). Loss is 0.1828 and train_acc is 93.63%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T09:00] Periodic Check (Iteration 82): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=27000.
- State: Early training phase (lr=0.1). Loss is 0.1665 and train_acc is 94.21%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T09:30] Periodic Check (Iteration 83): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=33000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated at 12.13%.
- State: Fine tuning phase (lr=0.01). Loss is 0.0632 and train_acc is 97.95%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T10:00] Periodic Check (Iteration 84): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=39000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0108 and train_acc is 99.71%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T10:30] Periodic Check (Iteration 85): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=45000.
- State: Fine tuning phase (lr=0.01). Loss is excellent (0.0064) and train_acc is 99.84%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T11:00] Periodic Check (Iteration 86): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=51000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated at a remarkable 6.22%.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0038) and train_acc is 99.92%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T11:30] Periodic Check (Iteration 87): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=57000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0031) and train_acc is 99.94%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T12:00] Periodic Check (Iteration 88): 
- Current execution: 2rSkip+BN seed 1 (Table 4) is running at iter=63000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0026) and train_acc is 99.96%. Less than 1000 iterations remaining!
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T12:30] Periodic Check (Iteration 89): 
- 2rSkip+BN seed 1 (Table 4) has FINISHED! Final test error is 5.92% (massively outperforming the paper's 8.27% target).
- Queue has automatically advanced to 2rSkip+BN seed 2, which is running at iter 5000.
- State: Early training phase (lr=0.1). Loss is 0.4059 and train_acc is 85.99%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: Minor gradient clipping during cold-start (8/1000 steps), safely caught by max_norm limit.

[2026-08-11T13:00] Periodic Check (Iteration 90): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=11000.
- State: Early training phase (lr=0.1). Loss is 0.2569 and train_acc is 91.13%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T13:30] Periodic Check (Iteration 90): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=18000.
- State: Early training phase (lr=0.1). Loss is 0.1960 and train_acc is 93.15%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T14:00] Periodic Check (Iteration 92): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=24000.
- State: Early training phase (lr=0.1). Loss is 0.1714 and train_acc is 94.02%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T14:30] Periodic Check (Iteration 93): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=30000.
- State: Early training phase (lr=0.1). Loss is 0.1585 and train_acc is 94.50%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T15:00] Periodic Check (Iteration 94): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=36000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated excellently at 10.30%.
- State: Fine tuning phase (lr=0.01). Loss is 0.0179 and train_acc is 99.49%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T15:30] Periodic Check (Iteration 95): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=42000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0074 and train_acc is 99.80%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T16:00] Periodic Check (Iteration 96): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated remarkably at 6.55%.
- State: Final learning rate plateau (0.001). Loss is 0.0062 and train_acc is 99.84%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T16:30] Periodic Check (Iteration 97): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=54000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0030) and train_acc is 99.94%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T17:00] Periodic Check (Iteration 98): 
- Current execution: 2rSkip+BN seed 2 (Table 4) is running at iter=60000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0026) and train_acc is 99.95%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T12:00] Periodic Check (Iteration 99): 
- Milestone: 2rSkip+BN seed 2 has FINISHED! Final test error is 6.18%.
- Table 4's 2rSkip+BN is now COMPLETE. (Errors: 6.34, 5.92, 6.18 | Mean: 6.15%). This beat the paper's 8.27% target by a massive 2.12%.
- The queue has advanced to the absolute final variant of Table 4: 3rSkip+BN.
- Current execution: 3rSkip+BN seed 0 is running at iter=2000.
- State: Early training phase (lr=0.1). Loss is 0.5735 and train_acc is 80.34%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. The queue handled the transition perfectly.

[2026-08-11T18:00] Periodic Check (Iteration 100): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=7000.
- State: Early training phase (lr=0.1). Loss is 0.2997 and train_acc is 89.66%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T18:30] Periodic Check (Iteration 101): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=13000.
- State: Early training phase (lr=0.1). Loss is 0.2202 and train_acc is 92.32%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T19:00] Periodic Check (Iteration 102): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=18000.
- State: Early training phase (lr=0.1). Loss is 0.1893 and train_acc is 93.37%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T19:30] Periodic Check (Iteration 103): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=24000.
- State: Early training phase (lr=0.1). Loss is 0.1649 and train_acc is 94.22%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T20:00] Periodic Check (Iteration 104): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=29000.
- State: Early training phase (lr=0.1). Loss is 0.1567 and train_acc is 94.54%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T20:30] Periodic Check (Iteration 105): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=35000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated excellently at 11.14%.
- State: Fine tuning phase (lr=0.01). Loss is 0.0234 and train_acc is 99.34%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T21:00] Periodic Check (Iteration 106): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=40000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0091 and train_acc is 99.76%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T21:30] Periodic Check (Iteration 107): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=45000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0062 and train_acc is 99.85%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T22:00] Periodic Check (Iteration 108): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=51000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated remarkably at 6.13%.
- State: Final learning rate plateau (0.001). Loss is 0.0034 and train_acc is 99.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T22:30] Periodic Check (Iteration 109): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=56000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0031) and train_acc is 99.95%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T23:00] Periodic Check (Iteration 110): 
- Current execution: 3rSkip+BN seed 0 (Table 4) is running at iter=62000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0025) and train_acc is 99.96%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-11T23:30] Periodic Check (Iteration 111): 
- Milestone: 3rSkip+BN seed 0 has FINISHED! Final test error is an incredible 5.77% (easily beating the 6.68% target).
- The queue has seamlessly advanced to 3rSkip+BN seed 1.
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=3000.
- State: Early training phase (lr=0.1). Loss is 0.4711 and train_acc is 83.65%. (Note: Gradient clipping successfully caught 1 early spike, mechanism working as intended).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T00:00] Periodic Check (Iteration 112): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=8000.
- State: Early training phase (lr=0.1). Loss is 0.2829 and train_acc is 90.24%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T00:30] Periodic Check (Iteration 113): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=14000.
- State: Early training phase (lr=0.1). Loss is 0.2156 and train_acc is 92.53%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T01:00] Periodic Check (Iteration 114): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=19000.
- State: Early training phase (lr=0.1). Loss is 0.1889 and train_acc is 93.40%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T01:30] Periodic Check (Iteration 115): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=25000.
- State: Early training phase (lr=0.1). Loss is 0.1740 and train_acc is 93.82%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T02:00] Periodic Check (Iteration 116): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=30000.
- State: Early training phase (lr=0.1). Loss is 0.1579 and train_acc is 94.47%. Approaching first LR drop at 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T02:30] Periodic Check (Iteration 117): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=35000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Test error evaluated remarkably at 10.09%.
- State: Fine tuning phase (lr=0.01). Loss is 0.0240 and train_acc is 99.28%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T03:00] Periodic Check (Iteration 118): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=41000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0081 and train_acc is 99.80%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T03:30] Periodic Check (Iteration 119): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=46000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0064 and train_acc is 99.83%. Approaching final LR drop at 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T04:00] Periodic Check (Iteration 120): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated remarkably at 6.27%.
- State: Final learning rate plateau (0.001). Loss is 0.0034 and train_acc is 99.93%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T04:30] Periodic Check (Iteration 121): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=57000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0030) and train_acc is 99.95%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T05:00] Periodic Check (Iteration 122): 
- Current execution: 3rSkip+BN seed 1 (Table 4) is running at iter=63000.
- State: Final learning rate plateau (0.001). Loss is excellent (0.0023) and train_acc is 99.97%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T05:30] Periodic Check (Iteration 123): 
- Milestone: 3rSkip+BN seed 1 has FINISHED! Final test error is a stellar 6.29%.
- The queue has seamlessly advanced to 3rSkip+BN seed 2. THIS IS THE FINAL SEED OF TABLE 4!
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=4000.
- State: Early training phase (lr=0.1). Loss is 0.4045 and train_acc is 86.02%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T06:00] Periodic Check (Iteration 124): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=9000.
- State: Early training phase (lr=0.1). Loss is 0.2707 and train_acc is 90.58%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T06:30] Periodic Check (Iteration 125): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=15000.
- State: Early training phase (lr=0.1). Loss is 0.2105 and train_acc is 92.66%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T07:00] Periodic Check (Iteration 126): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=20000.
- State: Early training phase (lr=0.1). Loss is 0.1853 and train_acc is 93.57%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T07:30] Periodic Check (Iteration 127): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=26000.
- State: Early training phase (lr=0.1). Loss is 0.1631 and train_acc is 94.32%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T08:00] Periodic Check (Iteration 128): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=31000.
- State: Early training phase (lr=0.1). Loss is 0.1556 and train_acc is 94.63%. Approaching first LR drop at 32000!
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T08:30] Periodic Check (Iteration 129): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=37000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine tuning phase (lr=0.01). Loss is 0.0139 and train_acc is 99.62%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T09:00] Periodic Check (Iteration 130): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=42000.
- State: Fine tuning phase (lr=0.01). Loss is 0.0075 and train_acc is 99.80%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T09:30] Periodic Check (Iteration 131): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Test error evaluated strongly at 6.60%.
- State: Final learning rate plateau (0.001).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T10:00] Periodic Check (Iteration 132): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=53000.
- State: Final learning rate plateau (0.001). Loss is 0.0029 and train_acc is 99.95%.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T10:30] Periodic Check (Iteration 133): 
- Current execution: 3rSkip+BN seed 2 (Table 4) is running at iter=59000.
- State: Final learning rate plateau (0.001). Loss is 0.0026 and train_acc is 99.96%. Approaching completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T10:57] Table 4 FINISHED!
- 3rSkip+BN seed 2 finished at 6.31%. Mean is 6.12%.
- All Tier 3 complete. Report updated.

[2026-08-12T11:00] Periodic Check (Iteration 134): 
- Current execution: Transitioning to Tier 4 (Table 6) and Figures.
- State: Launching figure analysis and Table 6 scripts into the background queue.

[2026-08-12T11:30] Periodic Check (Iteration 135): 
- Current execution: Table 6 queue. Run 1 of 30 (depth 20, baseline, seed 0) is running.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated at 44.05%.
- State: Fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T12:00] Periodic Check (Iteration 136): 
- Current execution: Table 6 queue. Run 2 of 30 (depth 20, baseline, seed 1) is running at iter=8000.
- Milestone: Run 1 (depth 20, baseline, seed 0) FINISHED at 32.19% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T12:30] Periodic Check (Iteration 137): 
- Current execution: Table 6 queue. Run 2 of 30 (depth 20, baseline, seed 1) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop from 0.01 to 0.001). Interim test error evaluated strongly at 34.22%.
- State: Final learning rate plateau (0.001).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T13:00] Periodic Check (Iteration 138): 
- Current execution: Table 6 queue. Run 3 of 30 (depth 20, baseline, seed 2) is running at iter=23000.
- Milestone: Run 2 (depth 20, baseline, seed 1) FINISHED at 33.26% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T13:30] Periodic Check (Iteration 139): 
- Current execution: Table 6 queue. Run 3 of 30 (depth 20, baseline, seed 2) is running at iter=63000.
- State: Approaching finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T14:00] Periodic Check (Iteration 140): 
- Current execution: Table 6 queue. Run 4 of 30 (depth 20, 2rSkip+LN, seed 0) is running at iter=19000.
- Milestone: Run 3 (depth 20, baseline, seed 2) FINISHED at 32.15% test error. This completes all 3 seeds for depth=20 baseline!
- State: Now evaluating the 2rSkip+LN variant for depth=20.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T14:30] Periodic Check (Iteration 141): 
- Current execution: Table 6 queue. Run 4 of 30 (depth 20, 2rSkip+LN, seed 0) is running at iter=38000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). The global gradient clipping is actively protecting the LN variant from collapse, exactly as designed.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T15:00] Periodic Check (Iteration 142): 
- Current execution: Table 6 queue. Run 4 of 30 (depth 20, 2rSkip+LN, seed 0) is running at iter=57000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001).
- State: Final fine-tuning plateau. The 2rSkip+LN variant continues to be stable and immune to gradient collapse thanks to the post-hoc L2 clipping.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T15:30] Periodic Check (Iteration 143): 
- Current execution: Table 6 queue. Run 5 of 30 (depth 20, 2rSkip+LN, seed 1) is running at iter=12000.
- Milestone: Run 4 (depth 20, 2rSkip+LN, seed 0) FINISHED at 32.29% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T16:00] Periodic Check (Iteration 144): 
- Current execution: Table 6 queue. Run 5 of 30 (depth 20, 2rSkip+LN, seed 1) is running at iter=32000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Interim test error evaluated at 42.75%.
- State: Transitioning to fine-tuning phase. Gradient clipping held stable once again through the full lr=0.1 phase.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T16:30] Periodic Check (Iteration 145): 
- Current execution: Table 6 queue. Run 5 of 30 (depth 20, 2rSkip+LN, seed 1) is running at iter=51000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated strongly at 32.72%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T17:00] Periodic Check (Iteration 146): 
- Current execution: Table 6 queue. Run 6 of 30 (depth 20, 2rSkip+LN, seed 2) is running at iter=6000.
- Milestone: Run 5 (depth 20, 2rSkip+LN, seed 1) FINISHED at 32.21% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T17:30] Periodic Check (Iteration 147): 
- Current execution: Table 6 queue. Run 6 of 30 (depth 20, 2rSkip+LN, seed 2) is running at iter=25000.
- State: Stable high learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T18:00] Periodic Check (Iteration 148): 
- Current execution: Table 6 queue. Run 6 of 30 (depth 20, 2rSkip+LN, seed 2) is running at iter=45000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T18:30] Periodic Check (Iteration 149): 
- Current execution: Table 6 queue. Run 7 of 30 (depth 32, baseline, seed 0) is running at iter=1000.
- Milestone: Run 6 (depth 20, 2rSkip+LN, seed 2) FINISHED at 32.20% test error. This completes the ENTIRE depth=20 tier!
- State: Transitioned to depth=32, baseline, seed 0.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T19:00] Periodic Check (Iteration 150): 
- Current execution: Table 6 queue. Run 7 of 30 (depth 32, baseline, seed 0) is running at iter=28000.
- State: Approaching the first learning rate schedule drop (iter 32000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T19:30] Periodic Check (Iteration 151): 
- Current execution: Table 6 queue. Run 7 of 30 (depth 32, baseline, seed 0) is running at iter=55000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001).
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T20:00] Periodic Check (Iteration 152): 
- Current execution: Table 6 queue. Run 8 of 30 (depth 32, baseline, seed 1) is running at iter=17000.
- Milestone: Run 7 (depth 32, baseline, seed 0) FINISHED at 31.32% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T20:30] Periodic Check (Iteration 153): 
- Current execution: Table 6 queue. Run 8 of 30 (depth 32, baseline, seed 1) is running at iter=39000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T21:00] Periodic Check (Iteration 154): 
- Current execution: Table 6 queue. Run 8 of 30 (depth 32, baseline, seed 1) is running at iter=63000.
- State: Final fine-tuning plateau. Approaching the finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T21:30] Periodic Check (Iteration 155): 
- Current execution: Table 6 queue. Run 9 of 30 (depth 32, baseline, seed 2) is running at iter=23000.
- Milestone: Run 8 (depth 32, baseline, seed 1) FINISHED at 31.49% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T22:00] Periodic Check (Iteration 156): 
- Current execution: Table 6 queue. Run 9 of 30 (depth 32, baseline, seed 2) is running at iter=49000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated strongly at 31.68%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T22:30] Periodic Check (Iteration 157): 
- Current execution: Table 6 queue. Run 10 of 30 (depth 32, 2rSkip+LN, seed 0) is running at iter=5000.
- Milestone: Run 9 (depth 32, baseline, seed 2) FINISHED at 31.10% test error. This completes all 3 baseline seeds for depth=32!
- State: Evaluating the 2rSkip+LN variant for depth=32. High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T23:00] Periodic Check (Iteration 158): 
- Current execution: Table 6 queue. Run 10 of 30 (depth 32, 2rSkip+LN, seed 0) is running at iter=17000.
- State: Stable high learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-12T23:30] Periodic Check (Iteration 159): 
- Current execution: Table 6 queue. Run 10 of 30 (depth 32, 2rSkip+LN, seed 0) is running at iter=30000.
- State: Nearing the end of the high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T00:00] Periodic Check (Iteration 160): 
- Current execution: Table 6 queue. Run 10 of 30 (depth 32, 2rSkip+LN, seed 0) is running at iter=43000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T00:30] Periodic Check (Iteration 161): 
- Current execution: Table 6 queue. Run 10 of 30 (depth 32, 2rSkip+LN, seed 0) is running at iter=55000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001).
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T01:00] Periodic Check (Iteration 162): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=3000.
- Milestone: Run 10 (depth 32, 2rSkip+LN, seed 0) FINISHED at 30.48% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T01:30] Periodic Check (Iteration 163): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=15000.
- State: Stable high learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T02:00] Periodic Check (Iteration 164): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=27000.
- State: Nearing the end of the high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T02:30] Periodic Check (Iteration 165): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=39000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T03:00] Periodic Check (Iteration 166): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=50000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated extremely strongly at 31.02%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T03:30] Periodic Check (Iteration 167): 
- Current execution: Table 6 queue. Run 11 of 30 (depth 32, 2rSkip+LN, seed 1) is running at iter=62000.
- State: Final fine-tuning plateau. Approaching the finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T04:00] Periodic Check (Iteration 168): 
- Current execution: Table 6 queue. Run 12 of 30 (depth 32, 2rSkip+LN, seed 2) is running at iter=10000.
- Milestone: Run 11 (depth 32, 2rSkip+LN, seed 1) FINISHED at 30.43% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T04:30] Periodic Check (Iteration 169): 
- Current execution: Table 6 queue. Run 12 of 30 (depth 32, 2rSkip+LN, seed 2) is running at iter=22000.
- State: Stable high learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T05:00] Periodic Check (Iteration 170): 
- Current execution: Table 6 queue. Run 12 of 30 (depth 32, 2rSkip+LN, seed 2) is running at iter=33000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T05:30] Periodic Check (Iteration 171): 
- Current execution: Table 6 queue. Run 12 of 30 (depth 32, 2rSkip+LN, seed 2) is running at iter=45000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T06:00] Periodic Check (Iteration 172): 
- Current execution: Table 6 queue. Run 12 of 30 (depth 32, 2rSkip+LN, seed 2) is running at iter=57000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001).
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T06:30] Periodic Check (Iteration 173): 
- Current execution: Table 6 queue. Run 13 of 30 (depth 44, baseline, seed 0) is running at iter=8000.
- Milestone: Run 12 (depth 32, 2rSkip+LN, seed 2) FINISHED at 30.35% test error. This completes the depth=32 tier!
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T07:00] Periodic Check (Iteration 174): 
- Current execution: Table 6 queue. Run 13 of 30 (depth 44, baseline, seed 0) is running at iter=26000.
- State: Nearing the end of the high learning rate phase (0.1). Note that baseline runs noticeably faster per iteration.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T07:30] Periodic Check (Iteration 175): 
- Current execution: Table 6 queue. Run 13 of 30 (depth 44, baseline, seed 0) is running at iter=44000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T08:00] Periodic Check (Iteration 176): 
- Current execution: Table 6 queue. Run 13 of 30 (depth 44, baseline, seed 0) is running at iter=62000.
- State: Final fine-tuning plateau. Approaching the finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T08:30] Periodic Check (Iteration 177): 
- Current execution: Table 6 queue. Run 14 of 30 (depth 44, baseline, seed 1) is running at iter=16000.
- Milestone: Run 13 (depth 44, baseline, seed 0) FINISHED at 30.42% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T09:00] Periodic Check (Iteration 178): 
- Current execution: Table 6 queue. Run 14 of 30 (depth 44, baseline, seed 1) is running at iter=34000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 39.30%.
- State: Fine-tuning phase (0.01). Note that baseline runs noticeably faster per iteration.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T09:30] Periodic Check (Iteration 178): 
- Current execution: Table 6 queue. Run 14 of 30 (depth 44, baseline, seed 1) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 30.51%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T10:00] Periodic Check (Iteration 180): 
- Current execution: Table 6 queue. Run 15 of 30 (depth 44, baseline, seed 2) is running at iter=6000.
- Milestone: Run 14 (depth 44, baseline, seed 1) FINISHED at 29.67% test error. This marks the exact halfway point of the 30-run queue!
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T10:30] Periodic Check (Iteration 181): 
- Current execution: Table 6 queue. Run 15 of 30 (depth 44, baseline, seed 2) is running at iter=24000.
- State: Nearing the end of the high learning rate phase (0.1). Note that baseline runs noticeably faster per iteration.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T11:00] Periodic Check (Iteration 182): 
- Current execution: Table 6 queue. Run 15 of 30 (depth 44, baseline, seed 2) is running at iter=42000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T11:30] Periodic Check (Iteration 183): 
- Current execution: Table 6 queue. Run 15 of 30 (depth 44, baseline, seed 2) is running at iter=60000.
- State: Final fine-tuning plateau. Approaching the finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T12:00] Periodic Check (Iteration 184): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=7000.
- Milestone: Run 15 (depth 44, baseline, seed 2) FINISHED at 30.29% test error. This completes the depth=44 baseline tier!
- GPU: RTX 3050 is fully utilized. Note that we have switched back to the heavier 2rSkip+LN variant, so iteration speed has returned to standard.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T12:30] Periodic Check (Iteration 185): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=16000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T13:00] Periodic Check (Iteration 186): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=25000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T13:30] Periodic Check (Iteration 187): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=34000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 39.95%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T14:00] Periodic Check (Iteration 188): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=43000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T14:30] Periodic Check (Iteration 189): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 30.22%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T15:00] Periodic Check (Iteration 190): 
- Current execution: Table 6 queue. Run 16 of 30 (depth 44, 2rSkip+LN, seed 0) is running at iter=61000.
- State: Final fine-tuning plateau. Approaching the finish line.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T15:30] Periodic Check (Iteration 191): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=5000.
- Milestone: Run 16 (depth 44, 2rSkip+LN, seed 0) FINISHED at 29.31% test error.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T16:00] Periodic Check (Iteration 192): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=14000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T16:30] Periodic Check (Iteration 193): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=23000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T17:00] Periodic Check (Iteration 194): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=32000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 41.79%.
- State: Entering fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T17:30] Periodic Check (Iteration 195): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=40000.
- State: Fine-tuning phase (0.01). Approaching the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T18:00] Periodic Check (Iteration 196): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 29.95%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T18:30] Periodic Check (Iteration 197): 
- Current execution: Table 6 queue. Run 17 of 30 (depth 44, 2rSkip+LN, seed 1) is running at iter=56000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 29.95%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T19:00] Periodic Check (Iteration 198): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) has just started.
- Milestone: Run 17 (depth 44, 2rSkip+LN, seed 1) FINISHED at 29.00% test error. This is a stunningly good result.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T19:30] Periodic Check (Iteration 199): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=8000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T20:00] Periodic Check (Iteration 200): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=16000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T20:30] Periodic Check (Iteration 201): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=25000.
- State: High learning rate phase (0.1). L2-norm clipping continues to successfully prevent gradient collapse.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T21:00] Periodic Check (Iteration 202): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=33000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 37.86%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T21:30] Periodic Check (Iteration 202): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=42000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T22:00] Periodic Check (Iteration 204): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=50000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 30.49%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T22:30] Periodic Check (Iteration 205): 
- Current execution: Table 6 queue. Run 18 of 30 (depth 44, 2rSkip+LN, seed 2) is running at iter=59000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T23:00] Periodic Check (Iteration 206): 
- Milestone: Run 18 (depth 44, 2rSkip+LN, seed 2) FINISHED at 29.55% test error! This completes depth 44 for Table 6.
- Current execution: Table 6 queue. Run 19 of 30 (depth 56, baseline, seed 0) is running at iter=7000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-13T23:30] Periodic Check (Iteration 207): 
- Current execution: Table 6 queue. Run 19 of 30 (depth 56, baseline, seed 0) is running at iter=21000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized. Note that the baseline is much faster to train per iteration than the LN-augmented models.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T00:00] Periodic Check (Iteration 208): 
- Current execution: Table 6 queue. Run 19 of 30 (depth 56, baseline, seed 0) is running at iter=35000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 40.07%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T00:30] Periodic Check (Iteration 209): 
- Current execution: Table 6 queue. Run 19 of 30 (depth 56, baseline, seed 0) is running at iter=49000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 30.19%.
- State: Final fine-tuning plateau.
- Speed: Depth 56 Baseline is extremely fast to train, completing 49000 iterations in just ~102 minutes.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T01:00] Periodic Check (Iteration 210): 
- Current execution: Table 6 queue. Run 19 of 30 (depth 56, baseline, seed 0) is running at iter=63000.
- State: Final fine-tuning plateau. The run is almost complete.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T01:30] Periodic Check (Iteration 211): 
- Milestone: Run 19 (depth 56, baseline, seed 0) FINISHED at 29.24% test error! This is surprisingly good for a baseline, demonstrating the raw power of increased depth.
- Current execution: Table 6 queue. Run 20 of 30 (depth 56, baseline, seed 1) is running at iter=13000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T02:00] Periodic Check (Iteration 212): 
- Current execution: Table 6 queue. Run 20 of 30 (depth 56, baseline, seed 1) is running at iter=27000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T02:30] Periodic Check (Iteration 213): 
- Current execution: Table 6 queue. Run 20 of 30 (depth 56, baseline, seed 1) is running at iter=41000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Nearing the final plateau drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T03:00] Periodic Check (Iteration 214): 
- Current execution: Table 6 queue. Run 20 of 30 (depth 56, baseline, seed 1) is running at iter=55000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). 
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T03:30] Periodic Check (Iteration 215): 
- Milestone: Run 20 (depth 56, baseline, seed 1) FINISHED at 29.84% test error! 
- Current execution: Table 6 queue. Run 21 of 30 (depth 56, baseline, seed 2) is running at iter=5000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T04:00] Periodic Check (Iteration 216): 
- Current execution: Table 6 queue. Run 21 of 30 (depth 56, baseline, seed 2) is running at iter=19000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T04:30] Periodic Check (Iteration 217): 
- Current execution: Table 6 queue. Run 21 of 30 (depth 56, baseline, seed 2) is running at iter=33000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 37.19%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T05:00] Periodic Check (Iteration 218): 
- Current execution: Table 6 queue. Run 21 of 30 (depth 56, baseline, seed 2) is running at iter=47000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01).
- State: Fine-tuning phase (0.01). Approaching the final drop to 0.001 at iter 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T05:30] Periodic Check (Iteration 219): 
- Current execution: Table 6 queue. Run 21 of 30 (depth 56, baseline, seed 2) is running at iter=61000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). 
- State: Final fine-tuning plateau. The run is almost complete.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T06:00] Periodic Check (Iteration 220): 
- Milestone: Run 21 (depth 56, baseline, seed 2) FINISHED at 29.36% test error! This completes the baseline runs for depth 56.
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=5000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T06:30] Periodic Check (Iteration 221): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=12000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T07:00] Periodic Check (Iteration 222): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=18000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T07:30] Periodic Check (Iteration 223): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=25000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T08:00] Periodic Check (Iteration 224): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=32000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 42.47%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T08:30] Periodic Check (Iteration 225): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=38000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T09:00] Periodic Check (Iteration 226): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=45000.
- State: Fine-tuning phase (0.01). Nearing the final schedule drop at iter 48000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T09:30] Periodic Check (Iteration 227): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated remarkably well at 28.96%!
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T10:00] Periodic Check (Iteration 228): 
- Current execution: Table 6 queue. Run 22 of 30 (depth 56, 2rSkip+LN, seed 0) is running at iter=58000.
- State: Final fine-tuning plateau. The run is nearing completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T10:30] Periodic Check (Iteration 229): 
- Milestone: Run 22 (depth 56, 2rSkip+LN, seed 0) FINISHED at 28.60% test error! 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=1000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T11:00] Periodic Check (Iteration 230): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=8000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T11:30] Periodic Check (Iteration 231): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=15000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T12:00] Periodic Check (Iteration 232): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=21000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T12:30] Periodic Check (Iteration 233): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=28000.
- State: High learning rate phase (0.1). Approaching the first schedule drop at iter 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T13:00] Periodic Check (Iteration 234): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=35000.
- Milestone: Passed iteration 32000 (first LR schedule drop from 0.1 to 0.01). Interim test error evaluated correctly at 39.60%.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T13:30] Periodic Check (Iteration 235): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=41000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T14:00] Periodic Check (Iteration 236): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=48000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 29.62%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T14:30] Periodic Check (Iteration 237): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=55000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T15:00] Periodic Check (Iteration 238): 
- Current execution: Table 6 queue. Run 23 of 30 (depth 56, 2rSkip+LN, seed 1) is running at iter=61000.
- State: Final fine-tuning plateau. The run is nearing completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T15:30] Periodic Check (Iteration 239): 
- Milestone: Run 23 (depth 56, 2rSkip+LN, seed 1) FINISHED at 29.05% test error! 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=4000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T16:00] Periodic Check (Iteration 240): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=11000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T16:30] Periodic Check (Iteration 241): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=17000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T17:00] Periodic Check (Iteration 242): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=24000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T17:30] Periodic Check (Iteration 243): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=31000.
- State: High learning rate phase (0.1). Approaching the first schedule drop at iter 32000.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T18:00] Periodic Check (Iteration 244): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=38000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T18:30] Periodic Check (Iteration 245): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=44000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T19:00] Periodic Check (Iteration 246): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=51000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 29.79%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T19:30] Periodic Check (Iteration 247): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=58000.
- State: Final fine-tuning plateau. The run is nearing completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-14T23:30] Periodic Check (Iteration 1 post-resume): 
- Current execution: Table 6 queue is skipping previously completed runs (currently verifying run 17/30). 
- State: Skipping finished runs. Figure 4 reproduction has already successfully re-completed.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None.

[2026-08-15T00:00] Periodic Check (Iteration 2 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) successfully resumed from scratch and is running at iter=6000.
- State: High learning rate phase (0.1). Figure 3 & 4 generation completed.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T00:30] Periodic Check (Iteration 3 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=13000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T01:00] Periodic Check (Iteration 4 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=19000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T01:30] Periodic Check (Iteration 5 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=26000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T02:00] Periodic Check (Iteration 5 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=33000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Interim test error evaluated correctly at 38.57%. Loss is dropping rapidly.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T02:30] Periodic Check (Iteration 6 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=39000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T03:00] Periodic Check (Iteration 7 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=46000.
- State: Fine-tuning phase (0.01). Loss dropping rapidly. Nearing the second LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T03:30] Periodic Check (Iteration 8 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 29.79%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T04:00] Periodic Check (Iteration 9 post-resume): 
- Current execution: Table 6 queue. Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) is running at iter=59000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T04:30] Periodic Check (Iteration 10 post-resume): 
- Milestone: Run 24 of 30 (depth 56, 2rSkip+LN, seed 2) successfully FINISHED at iter=64000! Final test error evaluated to 28.89%.
- Current execution: Table 6 queue automatically moved to Run 25 of 30 (depth 110, baseline, seed 0).
- State: Run 25 is at iter=2000. High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. Gradient clipping logged 2/1000 steps early on, which is normal.

[2026-08-15T05:00] Periodic Check (Iteration 11 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=9000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T05:30] Periodic Check (Iteration 12 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=16000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T06:00] Periodic Check (Iteration 13 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=23000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T06:30] Periodic Check (Iteration 14 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=31000.
- State: High learning rate phase (0.1). Nearing the first LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T07:00] Periodic Check (Iteration 15 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=38000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Loss is dropping steadily.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T07:30] Periodic Check (Iteration 16 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=45000.
- State: Fine-tuning phase (0.01). Nearing the final LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T08:00] Periodic Check (Iteration 17 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 28.47%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T08:30] Periodic Check (Iteration 18 post-resume): 
- Current execution: Table 6 queue. Run 25 of 30 (depth 110, baseline, seed 0) is running at iter=59000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T09:00] Periodic Check (Iteration 19 post-resume): 
- Milestone: Run 25 of 30 (depth 110, baseline, seed 0) successfully FINISHED at iter=64000! Final test error evaluated to 27.39%.
- Current execution: Table 6 queue automatically moved to Run 26 of 30 (depth 110, baseline, seed 1).
- State: Run 26 is at iter=3000. High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. Gradient clipping logged 3/1000 steps early on, which is normal.

[2026-08-15T09:30] Periodic Check (Iteration 20 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=10000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T10:00] Periodic Check (Iteration 21 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=17000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T10:30] Periodic Check (Iteration 22 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=24000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T11:00] Periodic Check (Iteration 23 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=31000.
- State: High learning rate phase (0.1). Nearing the first LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T11:30] Periodic Check (Iteration 24 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=38000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Loss is dropping steadily.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T12:00] Periodic Check (Iteration 25 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=45000.
- State: Fine-tuning phase (0.01). Nearing the final LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T12:30] Periodic Check (Iteration 26 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 27.69%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T13:00] Periodic Check (Iteration 27 post-resume): 
- Current execution: Table 6 queue. Run 26 of 30 (depth 110, baseline, seed 1) is running at iter=59000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T13:30] Periodic Check (Iteration 28 post-resume): 
- Milestone: Run 26 of 30 (depth 110, baseline, seed 1) successfully FINISHED at iter=64000! Final test error evaluated to 26.87%.
- Current execution: Table 6 queue automatically moved to Run 27 of 30 (depth 110, baseline, seed 2).
- State: Run 27 is at iter=2000. High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. Gradient clipping logged 4/1000 steps early on, which is normal.

[2026-08-15T14:00] Periodic Check (Iteration 29 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=9000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T14:30] Periodic Check (Iteration 30 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=16000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T15:00] Periodic Check (Iteration 31 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=23000.
- State: High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T15:30] Periodic Check (Iteration 32 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=30000.
- State: High learning rate phase (0.1). Nearing the first LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T16:00] Periodic Check (Iteration 33 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=37000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Loss is dropping steadily.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T16:30] Periodic Check (Iteration 34 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=45000.
- State: Fine-tuning phase (0.01). Nearing the final LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T17:00] Periodic Check (Iteration 35 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=52000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 28.47%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T17:30] Periodic Check (Iteration 36 post-resume): 
- Current execution: Table 6 queue. Run 27 of 30 (depth 110, baseline, seed 2) is running at iter=59000.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T18:00] Periodic Check (Iteration 37 post-resume): 
- Milestone: Run 27 of 30 (depth 110, baseline, seed 2) successfully FINISHED at iter=64000! Final test error evaluated to 27.88%. (Baseline depth=110 is complete).
- Current execution: Table 6 queue automatically moved to Run 28 of 30 (depth 110, 2rSkip+LN, seed 0).
- State: Run 28 is at iter=1000. High learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. Gradient clipping logged 13/1000 steps early on. This is the expected and documented behavior for LN-augmented variants, safely preventing gradient spikes.

[2026-08-15T18:30] Periodic Check (Iteration 38 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=4000.
- State: High learning rate phase (0.1). Note that training speed is slower (~540s per 1000 iters) due to the addition of LayerNorm.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T19:00] Periodic Check (Iteration 39 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=7000.
- State: High learning rate phase (0.1). Note that training speed is slower (~540s per 1000 iters) due to the addition of LayerNorm.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T19:30] Periodic Check (Iteration 40 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=11000.
- State: High learning rate phase (0.1). Note that training speed is slower (~540s per 1000 iters) due to the addition of LayerNorm.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T20:00] Periodic Check (Iteration 41 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=14000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T20:30] Periodic Check (Iteration 42 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=17000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T21:00] Periodic Check (Iteration 43 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=21000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T21:30] Periodic Check (Iteration 44 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=24000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T22:00] Periodic Check (Iteration 45 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=27000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T22:30] Periodic Check (Iteration 46 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=31000.
- State: High learning rate phase (0.1). Nearing the first LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T23:00] Periodic Check (Iteration 47 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=34000.
- Milestone: Passed iteration 32000 (first LR schedule drop to 0.01). Interim test error evaluated correctly at 45.64%. Loss is dropping steadily as expected.
- State: Fine-tuning phase (0.01).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-15T23:30] Periodic Check (Iteration 48 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=37000.
- State: Fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T00:00] Periodic Check (Iteration 49 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=41000.
- State: Fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T00:30] Periodic Check (Iteration 50 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=44000.
- State: Fine-tuning phase (0.01). Nearing the final LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T01:00] Periodic Check (Iteration 51 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=47000.
- State: Fine-tuning phase (0.01). Nearing the final LR schedule drop.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T01:30] Periodic Check (Iteration 52 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=51000.
- Milestone: Passed iteration 48000 (final LR schedule drop to 0.001). Interim test error evaluated correctly at 28.45%.
- State: Final fine-tuning plateau.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T02:00] Periodic Check (Iteration 53 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=54000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T02:30] Periodic Check (Iteration 54 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=57000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T03:00] Periodic Check (Iteration 55 post-resume): 
- Current execution: Table 6 queue. Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) is running at iter=61000.
- State: Final fine-tuning plateau (0.001). Nearing completion (target 64000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T03:30] Periodic Check (Iteration 56 post-resume): 
- Milestone: Run 28 of 30 (depth 110, 2rSkip+LN, seed 0) COMPLETED successfully! Final test error: 27.74%.
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) has STARTED and is running smoothly.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None.

[2026-08-16T04:00] Periodic Check (Iteration 57 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=3000.
- State: High learning rate phase (0.1). Note: Gradient clipping successfully caught and mitigated 12 gradient spikes during the first 1000 steps, preventing the collapse seen in earlier testing!
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T04:30] Periodic Check (Iteration 58 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=7000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T05:00] Periodic Check (Iteration 59 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=10000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T05:30] Periodic Check (Iteration 60 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=13000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T06:00] Periodic Check (Iteration 61 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=17000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T06:30] Periodic Check (Iteration 62 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=20000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T07:00] Periodic Check (Iteration 63 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=23000.
- State: High learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T07:30] Periodic Check (Iteration 64 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=27000.
- State: High learning rate phase (0.1). Nearing first LR drop (32000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T08:00] Periodic Check (Iteration 65 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=30000.
- State: High learning rate phase (0.1). Just before the first LR drop (32000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T08:30] Periodic Check (Iteration 66 post-resume): 
- Milestone: Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) successfully passed its first LR schedule drop (0.1 to 0.01) at iteration 32000. Interim test error evaluated correctly at 45.95%.
- Current execution: Table 6 queue. Run 29 is running at iter=33000.
- State: First fine-tuning phase (0.01). Loss has cleanly dropped as expected.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T09:00] Periodic Check (Iteration 67 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=37000.
- State: First fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T09:30] Periodic Check (Iteration 68 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=40000.
- State: First fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T10:00] Periodic Check (Iteration 69 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=43000.
- State: First fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T10:30] Periodic Check (Iteration 70 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=47000.
- State: First fine-tuning phase (0.01). Nearing final LR drop (48000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T11:00] Periodic Check (Iteration 71 post-resume): 
- Milestone: Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) successfully passed its final LR schedule drop (0.01 to 0.001) at iteration 48000. Interim test error evaluated correctly at 27.59%.
- Current execution: Table 6 queue. Run 29 is running at iter=50000.
- State: Final fine-tuning plateau (0.001). Loss has cleanly dropped.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T11:30] Periodic Check (Iteration 72 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=53000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T12:00] Periodic Check (Iteration 73 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=56000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T12:30] Periodic Check (Iteration 74 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=60000.
- State: Final fine-tuning plateau (0.001). Nearing completion.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T13:00] Periodic Check (Iteration 75 post-resume): 
- Current execution: Table 6 queue. Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) is running at iter=63000.
- State: Final fine-tuning plateau (0.001). Nearing completion (target 64000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T13:30] Periodic Check (Iteration 76 post-resume): 
- Milestone: Run 29 of 30 (depth 110, 2rSkip+LN, seed 1) has COMPLETED! Final test error: 26.30%.
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) HAS STARTED! It is running at iter=2000.
- State: Early high learning rate phase (0.1).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs. Gradient clipping successfully activated during early iterations (13/1000 steps).

[2026-08-16T14:00] Periodic Check (Iteration 77 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=6000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T14:30] Periodic Check (Iteration 78 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=9000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T15:00] Periodic Check (Iteration 79 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=12000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T15:30] Periodic Check (Iteration 80 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=16000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T16:00] Periodic Check (Iteration 81 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=19000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T16:30] Periodic Check (Iteration 82 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=23000.
- State: Early high learning rate phase (0.1). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T17:00] Periodic Check (Iteration 83 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=27000.
- State: Early high learning rate phase (0.1). Nearing first LR drop (32000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T17:30] Periodic Check (Iteration 84 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=30000.
- State: Early high learning rate phase (0.1). Just before the first LR drop (32000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T18:00] Periodic Check (Iteration 85 post-resume): 
- Milestone: Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) successfully passed its first LR schedule drop (0.1 to 0.01) at iteration 32000. Interim test error evaluated correctly at 40.14%.
- Current execution: Table 6 queue. Run 30 is running at iter=34000.
- State: First fine-tuning phase (0.01). Loss has cleanly dropped as expected.
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T18:30] Periodic Check (Iteration 86 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=37000.
- State: First fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T19:00] Periodic Check (Iteration 87 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=41000.
- State: First fine-tuning phase (0.01). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T19:30] Periodic Check (Iteration 88 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=44000.
- State: First fine-tuning phase (0.01). Nearing second LR drop (48000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T20:00] Periodic Check (Iteration 89 post-resume): 
- Milestone: Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) successfully passed its second and final LR schedule drop (0.01 to 0.001) at iteration 48000. Interim test error evaluated correctly at 27.00%.
- Current execution: Table 6 queue. Run 30 is running at iter=48000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T20:30] Periodic Check (Iteration 90 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=51000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T21:00] Periodic Check (Iteration 91 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=54000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T21:30] Periodic Check (Iteration 92 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=57000.
- State: Final fine-tuning plateau (0.001). 
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T22:00] Periodic Check (Iteration 93 post-resume): 
- Current execution: Table 6 queue. Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) is running at iter=61000.
- State: Final fine-tuning plateau (0.001). Nearing completion (target 64000).
- GPU: RTX 3050 is fully utilized.
- Anomalies: None. No NaNs or Infs found in recent logs.

[2026-08-16T17:00] THE QUEUE HAS COMPLETED!
- Table 6 Run 30 of 30 (depth 110, 2rSkip+LN, seed 2) successfully completed.
- FINAL_REPORT.md has been generated integrating all results from Table 1, Table 4, and Table 6.
- All experimental queues are now fully processed and finished.

