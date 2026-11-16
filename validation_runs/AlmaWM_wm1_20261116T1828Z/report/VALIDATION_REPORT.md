# Validation report — Alma-WM (`alma_wm1`)

## 1. Verdict
**No claim tier is earned.** The first tier (Gates 0-3) did not fully pass.

The claim is capped at the row before Gates [0, 1, 2, 3], which did not pass.

## 2. Gate verdicts

| Gate | Verdict |
|---|---|
| 0 | PASS |
| 1 | PASS |
| 2 | NOT PASSED |
| 3 | PASS |
| 4 | NOT PASSED |
| 5 | NOT PASSED |
| 6 | NOT PASSED |
| 7 | NOT PASSED |
| 8 | PASS |
| 9 | NOT PASSED |
| 10 | NOT PASSED |
| 11 | PASS |
| 12 | NOT PASSED |
| 13 | PASS |
| 14 | NOT PASSED |

## 3. Scorecard

| Gate | Check | Result | Threshold | Status |
|---|---|---|---|---|
| 0.6 | preregistration.md exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 0.5 | model_card.md exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 0.4 | independence.md exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 0.3 | trial_log.csv exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 0.2 | FROZEN.txt exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 0.1 | preregistration.md exists and hashed before holdout outputs | `true` | exists | **PASS** |
| 1.1 | provenance card (source, hashes, licence from HF dataset cards) | `{"GiftEvalPretrain": "apache-2.0", "GiftEval": "apache-2.0"}` | all fields | **PASS** |
| 1.2 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.3 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.4 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.6 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.7 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.8 | market point-in-time / survivorship / microstructure / corporate actions | `"not securities data"` | - | **N/A** |
| 1.9 | missing-data report (2,000 sampled series) | `0.0377` | reported | **REPORTED** |
| 1.10 | leakage canary: context with the label appended is rejected | `true` | rejected | **PASS** |
| 2.4 | target leakage: encoder states before t unchanged when data/actions after t change | `0.0` | == 0 (fp tolerance 1e-5) | **PASS** |
| 2.4 | predictor: latent k unchanged when actions after k change | `3.814697265625e-06` | == 0 (1e-5) | **PASS** |
| 2.5 | latent-action shortcut | `"actions are observed inputs, not inferred latents"` | - | **N/A** |
| 2.6/2.7 | LLM contamination / comparator parity of tools | `"no LLM component"` | - | **N/A** |
| 2.1 | cutoff ordering | `"GIFT-Eval test windows are disjoint from GiftEvalPretrain by construction (per dataset authors); simulated holdout seeds disjoint"` | holdout after cutoffs | **REPORTED** |
| 2.2 | exact-match scan: sealed-window context tails (12 values) found in pretraining data | `{"tasks_with_hit": 1, "tasks": 31}` | 0 hits (else report) | **FAIL** |
| 3.6 | repeatability: identical outputs on rerun (fixed weights, CPU) | `0.0` | == 0 | **PASS** |
| 3.7 | precision stability fp32 vs bf16 (median relative difference) | `0.0004249699847723821` | < 0.05, finite | **PASS** |
| 3.9 | latency per 512-step action-conditioned rollout (CPU, batch 64) | `"5.5 ms"` | reported | **REPORTED** |
| 3.1/3.2 | identical inputs for every model; frozen configs | `"all models read the same harness/inputs_*.npz; hashes in manifest.json"` | hashes match | **PASS** |
| 5.7 | long-horizon drift (2048-step rollouts, held action): finite, median max deviation in context ranges | `{"finite": true, "median_dev_ranges": 0.24, "max_dev_ranges": 2.11}` | finite and median < 20 | **PASS** |
| 9.5 | input perturbation: CRPS ratio vs clean (sim_val) | `{"clean": 1.0, "noise_0.1sd": 1.044, "drop_10pct": 1.138, "stale_last_64": 1.323}` | descriptive (finite, graceful) | **REPORTED** |
| 4 | SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24 | `{"arx": [1.353, -10.7], "persistence": [0.597, 16.6], "snaive24": [0.398, 28.5]}` | t>=3 each (Holm) | **FAIL** |
| 5.1 | SIMULATED rollout beats every simple law in every horizon bucket | `{"1-16": {"arx": 1.353, "persistence": 0.597, "snaive24": 0.398}, "17-64": {"arx": 1.176, "persistence": 0.413, "snaive24": 0.394}, "65-256": {"arx": 1.182, "persistence": 0.357, "snaive24": 0.358}, "257-512": {"arx":...` | t>=3 every bucket | **FAIL** |
| 5.1 | error growth CRPS(257-512)/CRPS(1-16) | `2.07` | <= 10 | **PASS** |
| 5.2 | open-loop vs closed-loop (re-anchored) CRPS gap | `1.854` | reported | **REPORTED** |
| 5.5 | no mode collapse: median 80%-width ratio model/oracle | `1.664` | descriptive (~1) | **REPORTED** |
| 5.3 | PIT histogram (10 bins, expected 0.1 each) | `[0.121, 0.1, 0.093, 0.095, 0.092, 0.092, 0.09, 0.091, 0.1, 0.126]` | descriptive | **REPORTED** |
| 5.4 | distribution two-sample / discriminator test | `"not computable: model emits marginal quantiles, no joint sample paths"` | required | **FAIL** |
| 5.6 | physical constraint violations (median < 0 for non-negative systems) | `0.1541` | <= 0.01 | **FAIL** |
| 6.1 | SIMULATED action-aware beats action-blind (CRPS) | `[0.751, 15.1]` | t>=3 | **PASS** |
| 6.2 | SIMULATED effect sign accuracy (|true effect|>0.5 sd) | `{"wm": 0.954, "arx": 0.925, "n": 839}` | >= 0.90 | **PASS** |
| 6.2 | SIMULATED monotone size sweeps predicted monotone, same direction | `{"rate": 0.923, "n": 522}` | >= 0.95 | **FAIL** |
| 6.5* | SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories) | `[1.427, -8.8]` | t>=3 | **FAIL** |
| 6.3 | impact accuracy on real interventions | `"no real interventional data available"` | required | **FAIL** |
| 6.4 | natural experiments on real data | `"no real interventional data available"` | required | **FAIL** |
| 6.5 | counterfactual vs nature on real data | `"no real interventional data available"` | required | **FAIL** |
| 6.6 | randomised live checks | `"no live deployment"` | when live | **N/A** |
| 6.7 | self-consistency (split trades) | `"market-specific"` | - | **N/A** |
| 6.8 | reflexivity | `"market-specific"` | - | **N/A** |
| 7 / 4.4 | SIMULATED coverage 80% / 60% (wm_aware) | `[0.753, 0.554]` | +-0.02 | **FAIL** |
| 7.2 | uncertainty predicts error (Spearman width vs |error|) | `[0.771, 29.6]` | t>=3 | **PASS** |
| 7.3 | SIMULATED conformal 80% interval coverage (calibrated on sim_val) | `0.799` | +-0.02 | **PASS** |
| 7.4 | OOD flag (Mahalanobis of state) AUROC for top-decile errors | `{"primary_mahalanobis": 0.532, "secondary_disagreement": 0.584}` | >= 0.75 | **FAIL** |
| 7.5 | fallback (ARX) on flagged episodes no worse than the model | `{"n_flagged": 438, "gmean_model": 0.31481218543586464, "gmean_fallback": 0.2606995546452661}` | fallback <= model | **PASS** |
| 7.6 | novel family (tanks) flag rate vs in-family flag rate | `{"tanks": 0.455, "in_family": 0.73}` | descriptive | **REPORTED** |
| 7.7 | risk-coverage (gmean CRPS keeping least-OOD fraction) | `{"0.5": 0.28597498878750716, "0.7": 0.29347211207356344, "0.9": 0.28874308959778616, "1.0": 0.292159936037531}` | descriptive | **REPORTED** |
| 8.10 | placebo: true actions beat shuffled actions | `[0.637, 20.5]` | t>=3 | **PASS** |
| 8.10 | placebo: random actions do not beat action-blind | `[1.162, -7.8]` | t<3 | **PASS** |
| 8.10 | placebo: time-shifted (16 steps) predictions lose skill | `[0.677, 18.2]` | t>=3 | **PASS** |
| 9.2/9.3 | SIMULATED by system family: aware/ARX gmean ratio | `{"fopdt": 1.256, "inventory": 1.111, "lss": 2.195, "pricing": 0.716, "thermal": 1.368}` | descriptive | **REPORTED** |
| 9.7 | transfer to held-out family (tanks), gmean CRPS | `{"wm_aware": 0.152, "wm_blind": 0.196, "arx": 0.109, "persistence": 0.303, "oracle": 0.032, "aware_vs_arx": 1.402}` | descriptive | **REPORTED** |
| 10.1 | linear probes: state -> true system parameters (5-fold CV R², permutation p) | `{"thermal.a": [0.27, 0.01], "thermal.b": [0.13, 0.01], "fopdt.tau": [0.03, 0.02], "fopdt.delay": [-0.02, 0.386], "fopdt.K": [0.19, 0.01], "inventory.lag": [-0.05, 0.832], "pricing.elast": [-0.03, 0.554], "pricing.carr...` | R²>0 and p<0.01 for every probe | **FAIL** |
| 10.1 | latent effective rank (non-collapse) | `146.3` | >= 16 | **PASS** |
| 10.4 | latent stability across training seeds | `"only one training seed exists"` | >= 0.8 | **FAIL** |
| 10.2 | attribution faithfulness | `"not run"` | required | **FAIL** |
| 10.5 | disentanglement | `"not run"` | required | **FAIL** |
| 10.6/10.7 | explanation decomposition / residual load | `"no named-component decomposition"` | - | **N/A** |
| 12 | SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC | `{"vs_default": [0.652, 3.3], "vs_arx": [1.691, -5.0], "oracle_vs_default": 0.337}` | t>=3 both | **FAIL** |
| 12.1 | imagined vs real gain (sum real / sum imagined) | `1.131` | >= 0.70 | **PASS** |
| 12.2 | exploit hunt: corr(imagined gain, imagined-real gap) | `-0.084` | descriptive | **REPORTED** |
| 12.5 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.6 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.7 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.8 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 4.2 | REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small | `{"snaive": [0.645, 7.3, "wins 1.00"], "persistence": [0.514, 7.1, "wins 0.88"], "telos": [1.038, -1.4, "wins 0.41"]}` | t>=3 each (Holm) | **FAIL** |
| 4.2 | REAL non-inferiority vs Alma v2 (gmean ratio, CI upper) | `[1.0141, 1.0264]` | <= 1.01 | **FAIL** |
| 4.2 | REAL latent-rollout forecasts vs direct head (ratio) | `1.075` | descriptive | **REPORTED** |
| 4.4 | REAL coverage 80% / 60% (Alma-WM) | `{"wm_direct": [0.817, 0.623], "wm_rollout": [0.743, 0.544], "alma_v2": [0.818, 0.629], "telos": [0.799, 0.618]}` | +-0.02 | **FAIL** |
| 8.2 | effective sample (real tasks) | `32` | >=12 preliminary, >=36 frontier | **PASS (preliminary)** |
| 8.6 | Holm correction across all 'beats' claims | `{"placebo_true_vs_shuffled": true, "plan_wm_vs_arx": false, "plan_wm_vs_default": true, "real_wm_vs_persistence": true, "real_wm_vs_snaive": true, "real_wm_vs_telos": false, "sim_aware_vs_blind": true, "sim_b1-16_vs_a...` | alpha 0.05 | **REPORTED** |
| 11 | language component | `"no LLM"` | - | **N/A** |
| 13.1-13.5 | leaderboard on identical harness | `"see Leaderboard"` | Holm-corrected beats | **REPORTED** |
| 14 | live sealed forward record >= 6 months | `"no live deployment"` | >= 6 months | **FAIL** |
| 3.3 | sealing proof: outputs hashed before truth loaded | `"grade.py refuses unsealed outputs; manifest has seal and truth-load times"` | - | **PASS** |

## 4. Reproducibility
- commit `af3da472c5212b4c6d9eeb4e20016b1d0e0b00d3`
- manifest sha256 `b04b12e9af98c1f3f390c1e128c10d90920ca39ff29eaa03cdaa9a6eb030a553`
- rerun: `make validate RUN=AlmaWM_wm1_20261116T1828Z`

## 5. Interpretation (written after grading; verdicts above are unchanged)

**Earned claim: none.** The pre-registered first tier (Gates 0-3) failed on one check, 2.2.

**About 2.2.** The single hit is `car_parts_with_missing/short`. The matched 12-value context tails are sparse 0/1 demand patterns such as `[0 1 0 0 0 0 0 1 0 0 0 0]`, which recur by chance in any intermittent-demand corpus. This is almost certainly a coincidental match, not leakage: only context values were scanned, and labels were never scanned. The pre-registered rule still says FAIL, so the verdict stands. The next run's scan will require an entropy floor (at least 6 distinct values) and will also scan labels.

**What the model already does well** (sealed simulator test, n = 600):
- Actions matter: action-aware vs action-blind CRPS ratio 0.751 (t = 15.1).
- Effect sign accuracy is 95.4%, vs 92.5% for ARX.
- All three placebos behave correctly:
  - shuffled actions are worse (0.637);
  - random actions are no better than blind (1.16);
  - time-shifted inputs lose skill.
- It beats persistence (0.60) and seasonal-naive (0.40) in every horizon bucket.
- Conformal 80% coverage is 0.799, and interval width predicts error (Spearman 0.77).
- The ARX fallback is safe.
- The latent space has not collapsed (effective rank 146).
- Imagined planning gains are realised (1.13×).
- Leakage controls pass, and results are repeatable to fp32 / bf16 precision.

**Where it fails, with the likely cause:**

| Gate | Result | Likely cause |
|---|---|---|
| 4 / 5.1 / 6.5* | 1.35× / 1.18× / 1.43× worse than ARX(24,24) | The simulators are mostly linear, so ARX is close to the true model class. 8k steps on 36M parameters is far short of convergence. The lss family is the worst at 2.2×. |
| 5.6 | 15% negative medians | No positivity constraint on the output head |
| 6.2 monotone | 0.923 | Just below the 0.95 threshold |
| 7 raw coverage | 0.753 / 0.554 | Rollout intervals are under-dispersed; the conformal version passes |
| 7.4 OOD | AUROC 0.53 | The Mahalanobis distance on the latent state does not track error |
| 10.1 probes | R² ≤ 0.27 | System parameters are not linearly decodable from the latent state |
| 12 MPC | 1.69× ARX-MPC cost | Follows from the gap to ARX in one-step accuracy |
| 4.2 real | Telos 1.038×; v2 non-inferiority CI upper bound 1.026 | Joint control training pulled the shared encoder away from forecasting |
| 6.3-6.5 / 10.2 / 10.4 / 10.5 / 14 | not run | No real interventional data, single seed, attribution not run, no live record |

**Real data** (8,998 sealed GIFT windows, 32 tasks): CRPS skill vs seasonal-naive:

| Model | CRPS skill |
|---|---|
| Telos-Small | 37.9% |
| Alma v2 | 36.4% |
| Alma-WM (direct) | 35.5% |
| Alma-WM (rollout) | 30.7% |

On this subset, which is weighted toward the large, high-frequency tasks, Telos-Small is ahead of Alma v2 as well. This differs from the full 97-config GIFT-Eval comparison, where v2 led (38.76% vs 38.33%).

**Independence:** the model builder and the validator are the same agent. All conclusions are provisional until an independent party re-runs `make validate`.
