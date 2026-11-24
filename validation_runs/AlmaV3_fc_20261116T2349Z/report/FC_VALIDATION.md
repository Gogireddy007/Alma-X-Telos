# Alma v3 — sealed forecaster validation

28 tasks, 8300 fresh GIFT-Eval test windows (never used for training, selection or any earlier grading).

| Model | CRPS skill vs SN | MASE skill vs SN |
|---|---|---|
| alma_v3 | 35.63% | 15.65% |
| telos | 37.87% | 19.94% |
| alma_v2 | 35.56% | 15.71% |
| snaive | 0.00% | 0.00% |

| Alma v3 vs | gmean CRPS ratio | 95% CI | t | win rate | Holm-corrected beat (t>=3) |
|---|---|---|---|---|---|
| telos | 1.0362 | [1.0074, 1.0676] | -2.35 | 0.29 | no |
| alma_v2 | 0.9990 | [0.9788, 1.0187] | 0.10 | 0.61 | no |
| snaive | 0.6437 | [0.5985, 0.6927] | 12.10 | 1.00 | YES |
