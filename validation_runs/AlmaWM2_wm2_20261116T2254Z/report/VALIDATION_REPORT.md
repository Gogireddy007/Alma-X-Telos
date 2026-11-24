# Validation report — Alma-WM2 (`alma_wm2`, ema_28000)

## 1. Verdict
Highest claim earned (Section F wording): **The evaluation is leak-free and reproducible.**

The claim is capped at the row before Gates [4], which did not pass.

## 2. Gate verdicts

| Gate | Verdict |
|---|---|
| 0 | PASS |
| 1 | PASS |
| 2 | PASS |
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
| 2.4 | predictor: latent k unchanged when actions after k change | `3.337860107421875e-06` | == 0 (1e-5) | **PASS** |
| 2.5 | latent-action shortcut | `"actions are observed inputs, not inferred latents"` | - | **N/A** |
| 2.6/2.7 | LLM contamination / comparator parity of tools | `"no LLM component"` | - | **N/A** |
| 2.1 | cutoff ordering | `"GIFT-Eval test windows are disjoint from GiftEvalPretrain by construction (per dataset authors); simulated holdout seeds disjoint"` | holdout after cutoffs | **REPORTED** |
| 2.2 | exact-match scan: 12-value context tails and label heads (>= 6 distinct) in pretraining | `{"queries": 83, "skipped": 33, "context_hit_tasks": [], "label_hit_tasks": []}` | 0 hits | **PASS** |
| 3.6 | repeatability: identical outputs on rerun (fixed weights, CPU) | `0.0` | == 0 | **PASS** |
| 3.7 | precision stability fp32 vs bf16 (median relative difference) | `5.2913672810820335e-05` | < 0.05, finite | **PASS** |
| 3.9 | latency per 512-step action-conditioned rollout (CPU, batch 64) | `"5.5 ms"` | reported | **REPORTED** |
| 3.1/3.2 | identical inputs for every model; frozen configs | `"all models read the same harness/inputs_*.npz; hashes in manifest.json"` | hashes match | **PASS** |
| 5.7 | long-horizon drift (2048-step rollouts, held action): finite, median max deviation in context ranges | `{"finite": true, "median_dev_ranges": 0.22, "max_dev_ranges": 1.99}` | finite and median < 20 | **PASS** |
| 9.5 | input perturbation: CRPS ratio vs clean (sim_val) | `{"clean": 1.0, "noise_0.1sd": 1.019, "drop_10pct": 1.497, "stale_last_64": 1.265}` | descriptive (finite, graceful) | **REPORTED** |
| 4 | SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24 | `{"arx": [1.048, -2.8], "persistence": [0.534, 24.9], "snaive24": [0.358, 35.8]}` | t>=3 each (Holm) | **FAIL** |
| 5.1 | SIMULATED rollout beats every simple law in every horizon bucket | `{"1-16": {"arx": 1.048, "persistence": 0.534, "snaive24": 0.358}, "17-64": {"arx": 0.915, "persistence": 0.377, "snaive24": 0.349}, "65-256": {"arx": 0.868, "persistence": 0.319, "snaive24": 0.311}, "257-512": {"arx":...` | t>=3 every bucket | **FAIL** |
| 5.1 | error growth CRPS(257-512)/CRPS(1-16) | `1.83` | <= 10 | **PASS** |
| 5.2 | open-loop vs closed-loop (re-anchored) CRPS gap | `2.147` | reported | **REPORTED** |
| 5.5 | no mode collapse: median 80%-width ratio model/oracle | `1.208` | descriptive (~1) | **REPORTED** |
| 5.3 | PIT histogram (10 bins, expected 0.1 each) | `[0.107, 0.099, 0.097, 0.093, 0.094, 0.097, 0.095, 0.095, 0.099, 0.124]` | descriptive | **REPORTED** |
| 5.4 | distribution two-sample / discriminator test | `"not computable: model emits marginal quantiles, no joint sample paths"` | required | **FAIL** |
| 5.6 | physical constraint violations (median < 0 for non-negative systems) | `0.1915` | <= 0.01 | **FAIL** |
| 6.1 | SIMULATED action-aware beats action-blind (CRPS) | `[0.634, 20.4]` | t>=3 | **PASS** |
| 6.2 | SIMULATED effect sign accuracy (|true effect|>0.5 sd) | `{"wm": 0.971, "arx": 0.922, "n": 1122}` | >= 0.90 | **PASS** |
| 6.2 | SIMULATED monotone size sweeps predicted monotone, same direction | `{"rate": 0.954, "n": 680}` | >= 0.95 | **PASS** |
| 6.5* | SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories) | `[0.981, 0.7]` | t>=3 | **FAIL** |
| 6.3 | impact accuracy on real interventions | `"no real interventional data available"` | required | **FAIL** |
| 6.4 | natural experiments on real data | `"no real interventional data available"` | required | **FAIL** |
| 6.5 | counterfactual vs nature on real data | `"no real interventional data available"` | required | **FAIL** |
| 6.6 | randomised live checks | `"no live deployment"` | when live | **N/A** |
| 6.7 | self-consistency (split trades) | `"market-specific"` | - | **N/A** |
| 6.8 | reflexivity | `"market-specific"` | - | **N/A** |
| 7 / 4.4 | SIMULATED coverage 80% / 60% (wm_aware) | `[0.769, 0.571]` | +-0.02 | **FAIL** |
| 7.2 | uncertainty predicts error (Spearman width vs |error|) | `[0.886, 53.8]` | t>=3 | **PASS** |
| 7.3 | SIMULATED conformal 80% interval coverage (calibrated on sim_val) | `0.796` | +-0.02 | **PASS** |
| 7.4 | OOD flag (Mahalanobis of state) AUROC for top-decile errors | `{"primary_mahalanobis": 0.525, "secondary_disagreement": 0.688}` | >= 0.75 | **FAIL** |
| 7.5 | fallback (ARX) on flagged episodes no worse than the model | `{"n_flagged": 637, "gmean_model": 0.27843118975488285, "gmean_fallback": 0.3132292457200606}` | fallback <= model | **FAIL** |
| 7.6 | novel family (tanks) flag rate vs in-family flag rate | `{"tanks": 0.565, "in_family": 0.796}` | descriptive | **REPORTED** |
| 7.7 | risk-coverage (gmean CRPS keeping least-OOD fraction) | `{"0.5": 0.2781048005193886, "0.7": 0.27612899562764476, "0.9": 0.2727129611908753, "1.0": 0.2749657254863075}` | descriptive | **REPORTED** |
| 8.10 | placebo: true actions beat shuffled actions | `[0.53, 26.5]` | t>=3 | **PASS** |
| 8.10 | placebo: random actions do not beat action-blind | `[1.359, -12.5]` | t<3 | **PASS** |
| 8.10 | placebo: time-shifted (16 steps) predictions lose skill | `[0.645, 20.2]` | t>=3 | **PASS** |
| 9.2/9.3 | SIMULATED by system family: aware/ARX gmean ratio | `{"bilinear": 0.855, "fopdt": 1.013, "hammer": 0.951, "inventory": 0.822, "logistic": 0.915, "lss": 1.316, "pricing": 0.626, "thermal": 0.983}` | descriptive | **REPORTED** |
| 9.7 | transfer to held-out family (tanks), gmean CRPS | `{"wm_aware": 0.119, "wm_blind": 0.211, "arx": 0.13, "persistence": 0.376, "oracle": 0.032, "aware_vs_arx": 0.919}` | descriptive | **REPORTED** |
| 10.1 | linear probes: state -> true system parameters (5-fold CV R², permutation p) | `{"thermal.a": [0.66, 0.01], "thermal.b": [0.28, 0.01], "fopdt.tau": [0.23, 0.01], "fopdt.delay": [-0.04, 0.634], "fopdt.K": [0.32, 0.01], "inventory.lag": [0.17, 0.01], "pricing.elast": [0.07, 0.02], "pricing.carry": ...` | R²>0 and p<0.01 for every probe | **FAIL** |
| 10.1 | latent effective rank (non-collapse) | `177.6` | >= 16 | **PASS** |
| 10.4 | latent stability across training seeds | `"only one training seed exists"` | >= 0.8 | **FAIL** |
| 10.2 | attribution faithfulness | `"not run"` | required | **FAIL** |
| 10.5 | disentanglement | `"not run"` | required | **FAIL** |
| 10.6/10.7 | explanation decomposition / residual load | `"no named-component decomposition"` | - | **N/A** |
| 12 | SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC | `{"vs_default": [0.418, 7.9], "vs_arx": [1.087, -1.4], "oracle_vs_default": 0.332}` | t>=3 both | **FAIL** |
| 12.1 | imagined vs real gain (sum real / sum imagined) | `1.078` | >= 0.70 | **PASS** |
| 12.2 | exploit hunt: corr(imagined gain, imagined-real gap) | `-0.204` | descriptive | **REPORTED** |
| 12.5 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.6 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.7 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 12.8 | execution / portfolio / risk / deflated Sharpe | `"no trading component"` | - | **N/A** |
| 4.2 | REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small | `{"snaive": [0.638, 11.1, "wins 1.00"], "persistence": [0.553, 6.9, "wins 0.93"], "telos": [1.037, -1.8, "wins 0.24"]}` | t>=3 each (Holm) | **FAIL** |
| 4.2 | REAL non-inferiority vs Alma v2 (gmean ratio, CI upper) | `[1.0, 1.0]` | <= 1.01 | **PASS** |
| 4.2 | REAL latent-rollout forecasts vs direct head (ratio) | `1.027` | descriptive | **REPORTED** |
| 4.4 | REAL coverage 80% / 60% (Alma-WM) | `{"wm_direct": [0.812, 0.623], "wm_rollout": [0.754, 0.563], "alma_v2": [0.812, 0.623], "telos": [0.793, 0.609]}` | +-0.02 | **FAIL** |
| 8.2 | effective sample (real tasks) | `29` | >=12 preliminary, >=36 frontier | **PASS (preliminary)** |
| 8.6 | Holm correction across all 'beats' claims | `{"placebo_true_vs_shuffled": true, "plan_wm_vs_arx": false, "plan_wm_vs_default": true, "real_wm_vs_persistence": true, "real_wm_vs_snaive": true, "real_wm_vs_telos": false, "sim_aware_vs_blind": true, "sim_b1-16_vs_a...` | alpha 0.05 | **REPORTED** |
| 11 | language component | `"no LLM"` | - | **N/A** |
| 13.1-13.5 | leaderboard on identical harness | `"see Leaderboard"` | Holm-corrected beats | **REPORTED** |
| 14 | live sealed forward record >= 6 months | `"no live deployment"` | >= 6 months | **FAIL** |
| 3.3 | sealing proof: outputs hashed before truth loaded | `"grade.py refuses unsealed outputs; manifest has seal and truth-load times"` | - | **PASS** |

## 4. Reproducibility
- commit `ec952e7a2e543980292031f7a8b5c1d8c7c30fa4`
- manifest sha256 `09e63745cfafc78e7dba59cb75e7661249347e08455b5c6e6e6d8c99b1c38190`
- rerun: `make validate RUN=AlmaWM2_wm2_20261116T2254Z`

---

## Interpretation (written after grading; nothing above was changed)

**Claim earned: "The evaluation is leak-free and reproducible"** (Gates 0-3). This is one tier above wm1,
which failed Gate 2.2 on a chance overlap. Gate 4 (one-step vs ARX) did not pass, so the claim is capped
before Gate 4. (The report header label was corrected from `alma_wm1` to `alma_wm2`; this was a template
string only.)

### Pre-registered targets

| Target | wm1 | wm2 aim | **wm2 sealed result** | Met? |
|---|---|---|---|---|
| Gate 2.2 overlap | 1 chance hit | 0 hits | 0 hits | **yes** |
| Gate 4: one-step CRPS vs ARX | 1.353 | ≤ 1.0, t ≥ 3 | 1.048 (t = −2.8) | no, but the gap shrank from 35% to 5% |
| 6.5*: counterfactual vs ARX | 1.427 | ≤ 1.0 | **0.981** (t = 0.7) | ratio yes, significance no |
| 5.6: constraint violations | 15.4% | 0% | 19.1% | gate mis-specified (amendment A3) |
| 12: MPC cost vs ARX-MPC | 1.691 | ≤ 1.0 | 1.087 (t = −1.4, a tie) | no, but the gap shrank from 69% to 9% |
| 9.7: tanks (held out) vs ARX | 1.402 | < 1.0 | **0.919** | **yes** |
| 4.2: non-inferiority vs v2 | 1.014 | 1.000 | 1.000 | **yes** |
| 6.1 / 6.2 / 8.10 | pass | keep passing | all pass, larger t | **yes** |

### Rollout vs ARX by horizon (lower is better; below 1 means wm2 beats ARX)
1-16: 1.048 · 17-64: **0.915** · 65-256: **0.868** · 257-512: **0.929**.
wm2 beats ARX at every horizon beyond 16 steps. It also beats ARX in 6 of 8 system families
(bilinear 0.86, hammer 0.95, inventory 0.82, logistic 0.92, pricing 0.63, thermal 0.98). It is even with ARX
on fopdt (1.01) and loses only on linear state-space (lss, 1.32), where ARX is the correct model class.

### Regressions and open problems
1. **Calibration still slightly narrow:** 80% intervals cover 76.9% (wm1: 75.3%; tolerance ±2 pp). Conformal calibration on sim_val
   fixes this (7.3: 79.6%). Next: ship the conformal layer as part of the model.
2. **OOD flag:** Mahalanobis AUROC is 0.53 (wm1: 0.53, also failed). The secondary disagreement measure improved from 0.58 to 0.69. The
   latent space grew richer (effective rank 178), so a single Gaussian no longer describes it. Next: a k-NN
   or per-family density.
3. **One-step gap (1.048):** concentrated in lss. Next: let the ARX gate act per horizon, or start the
   prior gate more open than −6.
4. **Real data:** wm2's direct path *is* v2 (frozen), so vs Telos it stays 1.037 (wins 24% of tasks).
   Beating Telos is v3's job.

### Next round (wm3)
v3 forecaster as the frozen path · conformal calibration layer · k-NN OOD head · per-horizon prior gate ·
a fixed Gate 5.6 using truly bounded families · multiple training seeds (10.4).
