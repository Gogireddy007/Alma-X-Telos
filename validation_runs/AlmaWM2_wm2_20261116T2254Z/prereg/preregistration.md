# Pre-registration — Alma-WM2 (run `alma_wm2`)

Written and hashed **before** any wm2 holdout output exists. The wm1 pre-registration, reproduced at the end, still applies except where this section changes it. **Thresholds are unchanged** (`thresholds.yaml` is a byte-identical copy of the wm1 file). The bar has not been lowered.

## What changed in the model (each change aimed at a failed wm1 gate)
1. **Frozen forecasting path.** `fc_base` is a frozen copy of Alma v2 @6k. Action-free forecasts come from it, so the wm2 direct forecast is identical to v2 by construction. This is disclosed as by-construction, not as a learned result.
2. **Linear-dynamics prior.**
   - For each origin, a ridge ARX(24, 24) is fitted in closed form on the last 768 context steps. It is rolled forward under the planned actions, with a stability guard.
   - The decoder sees that prior and outputs alpha·prior + residual, where alpha is a learned per-step gate.
   - The JEPA core is unchanged: context encoder, action-conditioned latent predictor, EMA target encoder with stop-gradient, L1 latent loss and variance anti-collapse.
3. **Output constraint.** If the context is non-negative, the forecast is clipped to be non-negative.
4. **Data.** Training uses 8 simulated families: the 5 from wm1 plus hammer (Hammerstein-Wiener), logistic and bilinear. The share of control data rises from 35% to 50%. Tanks stays held out.
5. **Training.** Warm start from wm1 @8k, then 28,000 steps. **The final checkpoint `ema_28000.pt` is pre-declared as the graded one.** There is no checkpoint selection on any holdout. Intermediate checkpoints may only be checked on `DEBUG_harness` (development data).

## Holdouts (new, never graded before)
- **Simulated:**
  - sim_val: seeds 501000-501239;
  - sim_test: 901000-901799, 8 families × 100, with 64-draw Monte Carlo oracle;
  - sim_tanks: 911000-911199;
  - plan_test: 921000-921099.
- **Real:** GIFT-Eval test windows that are outside the seed-0 development sample **and** outside the 9,600 windows graded for wm1 (seed 777). Seed 778, up to 300 per task.

## Amendment A2 (before sealing)
- **Gate 2.2 overlap scan.** Queries are 12-value context tails **and** 12-value label heads. A query needs at least 6 distinct values; low-entropy queries are skipped and counted. Reason: wm1's only hit was a chance match of a sparse 0/1 pattern.
- **Gate 5.6.** The non-negative families are inventory, pricing and logistic.
- **Gate 4.2.** wm2's direct forecast equals Alma v2 by construction, so the non-inferiority result is reported as by-construction. The test that matters for the world model is the **rollout path**, which is reported alongside it.

## Pre-registered targets for wm2 (expectations, not thresholds)

| Target | wm1 result | wm2 aim |
|---|---|---|
| Gate 2.2 overlap | 1 chance hit | 0 hits → Gates 0-3 pass |
| Gate 4: one-step CRPS vs ARX | 1.353 | ≤ 1.0 with t ≥ 3 |
| 6.5*: counterfactual vs ARX | 1.427 | ≤ 1.0 |
| 5.6: constraint violations | 15.4% | 0% |
| 12: MPC cost vs ARX-MPC | 1.691 | ≤ 1.0 |
| 9.7: tanks (held out) vs ARX | 1.402 | < 1.0 |
| 4.2: non-inferiority vs v2 | 1.014 (CI upper 1.026) | 1.000 by construction |
| 6.1 / 6.2 / 8.10 | pass | keep passing |

**Expected to still fail:**
- 6.3-6.5 (no real interventional data);
- 10.2, 10.4 and 10.5 (one seed, not run);
- 14 (no live record);
- 4.2 vs Telos-Small, because v2 itself trails Telos on these high-frequency tasks.

---
# Pre-registration — Alma-WM (run `alma_wm1`)

Written 2026-11-16 while `alma_wm1` was training and **before any Alma-WM prediction, rollout or score existed**.
Frozen by `prereg/FROZEN.txt` (git commit + UTC time). Any later change goes in `prereg/amendments.md`.

## Section A — what is being tested

```yaml
model:
  name: Alma-WM
  version_id: runs/alma_wm1/ema_8000.pt   # sha256 recorded in manifest.json at grading time
  type: NUMERIC_WM   # generic time-series world model (NOT a market model; see "Adaptations")
  components:
    - context encoder E_theta: Alma v2 backbone (8 layers, d 384, patch 16) + action embedding
    - target encoder E_xi: EMA (0.998) copy of E_theta, stop-gradient (JEPA target)
    - latent predictor P_psi: 4-layer causal transformer, 8 patches (128 steps) per chunk, chained
    - decoder D: latent + origin state -> 9 quantiles per step (grounding)
    - direct forecast head: Alma's per-token 1024-step quantile head (action-free forecast)
  entrypoint: model_wm.py (AlmaWM.forecast, AlmaWM.state_rollout), forecast.py (direct head)
  inference_config: deterministic (no sampling); context 2048; rollout horizon <= 2048; fp32; seed 0
  data_cutoffs:
    base_llm_pretraining: N/A (no LLM)
    world_model_training: GiftEvalPretrain (Salesforce, published 2024; series end dates vary by source)
    finetune_rl_rewards: N/A
  harness_format_seen_in_training: false

what_it_predicts:
  state_variables: the target series value (univariate), plus a latent state vector (384-d)
  levels: any frequency (5-minute to yearly)
  horizons: GIFT-Eval short/medium/long (6 to 900 steps); simulated rollouts 16 to 2048 steps
  actions_it_conditions_on: 2 channels - controllable input u0, exogenous driver u1 (known in advance)
  outputs: 9 quantiles (0.1..0.9) per future step; latent state trajectory
  universe: GIFT-Eval 97 configurations (real); controlled simulated systems (simulated)
  decision_time: forecast origin = end of the supplied context; nothing after it is visible

data:
  sources: Salesforce/GiftEvalPretrain (licence per its HF card - NOT verified, Gate 1.1) 94 sources, 433,528 series, 1.20B points;
           Salesforce/GiftEval test (licence NOT verified) for evaluation only; simulators.py (own code)
  delisted_names_included: N/A (not equities)
  point_in_time_timestamps: N/A (no revised/published fields; series values only)
  ground_truth_footprints: simulated systems only (true actions, true counterfactuals, true parameters)

periods:
  development: GIFT-Eval windows in the seed-0 1000-window sample per task (USED for v1/v2 checkpoint
               selection -> burned as a holdout for any "beats" claim)
  validation: simulated episodes with seeds 500000-500999 (may be used to calibrate conformal intervals)
  sealed_holdout:
    real: GIFT-Eval windows NOT in the seed-0 sample, for the 32 tasks with >1000 windows; up to 300 per
          task drawn with seed 777. Same series as development windows (different windows) - weaker holdout.
    simulated: seeds 900000-900599 (TRAIN_KINDS) and 910000-910199 (held-out family "tanks")
  live_forward_start: none (no live deployment) -> Gate 14 cannot pass

comparators:
  simple_laws: [seasonal_naive, persistence (last value), ARX(24,24) fitted by least squares on the context]
  learned_baselines: [Alma v2 @6k (own lineage, direct), Alma-WM action-blind (actions masked), Telos-Small]
  llm_baselines: none (no LLM component)
  external_claims: Telos-Small model card 43.21% GIFT CRPS skill (different protocol; labelled)
  own_lineage: [Alma v1 @20k, Alma v2 @6k]
  oracle (simulated only): true simulator with true parameters, Monte-Carlo future noise (lower bound)

trial_log_path: prereg/trial_log.csv (reconstructed after the fact from results/ and runs/; flagged)
compute_budget: Apple M4 laptop, <= 6 h for validation
```

## Model type and gates (Section B, column NUMERIC_WM)

| Gate | Applies | Notes |
|---|---|---|
| 0 Governance | yes | this file |
| 1 Data integrity | yes, partly | market items (survivorship, corporate actions, PIT membership, microstructure) N/A: no securities |
| 2 Contamination | yes (◐) | no LLM: 2.6 N/A. Overlap scan, causality (target-leakage) and action-shortcut tests run |
| 3 Harness | yes | |
| 4 One-step fidelity | yes | 4.5 cross-sectional ranking N/A (no ranking output); 4.6 joint accuracy N/A (univariate) |
| 5 Rollout fidelity | yes | market stylised facts replaced by dynamical-system checks listed below |
| 6 Action & causal | yes | simulated systems only; real-data impact / natural experiments are **not available -> FAIL** |
| 7 Uncertainty & fallback | yes | |
| 8 Statistics | yes | 8.7 deflated Sharpe / 8.8 PBO apply to trading strategies: N/A (none) |
| 9 Robustness | yes | 9.6 adversarial news N/A; 9.7 transfer to held-out family |
| 10 Interpretability | yes | probes against true simulator parameters; 10.4 needs >= 2 training seeds |
| 11 Language | N/A | no LLM component |
| 12 Decision quality | yes (◐) | planning (MPC) on simulated systems; portfolios/execution N/A |
| 13 Benchmarking | yes | |
| 14 Live record | yes | no live deployment exists -> not passed; caps the claim |

## Adaptations of the engine (disclosed up front)

1. **The engine is written for market world models.** Alma-WM is a generic time-series world model.
   Market-specific checks are N/A with the reason given in the table above.
2. **Rule C.6 (real data only).** Real data contain no recorded interventions, so action and causal
   claims can only be tested on simulated controlled systems, where the true counterfactual is known.
   Every such result is labelled **SIMULATED**, and any claim from it says "on simulated controlled
   systems". It is never presented as real-world causal validity. Real-data causal checks (6.3–6.5) are
   recorded as **FAIL (no data)**.
3. **Independence (0.4).** The same agent (Claude) built the model and wrote the validation code.
   This is not independent validation; a human reviewer is needed before any external claim.
4. **GIFT-Eval development windows are burned** (used to pick v1/v2 checkpoints). "Beats Telos" claims
   are graded only on the sealed unseen windows.

## Hypotheses and claims (each graded against `thresholds.yaml`)

- **H1 (Gate 4, real, sealed GIFT windows).** Alma-WM's forecasts beat Seasonal Naive, persistence and
  Telos-Small on CRPS (paired t over tasks of the log CRPS ratio, t >= 3, Holm-corrected), and are not
  worse than Alma v2 by more than 1% (non-inferiority).
- **H2 (Gate 5, SIMULATED).** Open-loop latent rollouts, decoded, beat every simple law on CRPS in every
  horizon bucket (1–16, 17–64, 65–256, 257–512 steps). CRPS at 257–512 is <= 10x CRPS at 1–16.
  Decoded rollouts stay finite and within physical constraints.
- **H3 (Gate 6, SIMULATED).**
  - Action-aware beats action-blind (t >= 3).
  - Counterfactual forecasts under a changed plan beat ARX on the true counterfactual trajectory (t >= 3).
  - Effect sign is right in >= 90% of cases with |true effect| > 0.5 context sd.
  - >= 95% of the size sweeps that are truly monotone are predicted monotone in the same direction.
- **H4 (Gate 7).**
  - 80% and 60% interval coverage within +/-2 pp (sealed real; SIMULATED).
  - Uncertainty predicts error (Spearman > 0, t >= 3).
  - OOD flag AUROC >= 0.75 for the top-decile errors.
  - Fallback is no worse than the model on flagged cases.
- **H5 (Gate 8, placebos).**
  - Shuffled actions (from another episode) lose skill vs true actions (t >= 3).
  - Time-shifted forecasts lose skill.
  - Random actions do not beat the action-blind model.
- **H6 (Gate 10, SIMULATED).**
  - Linear probes from the context state recover true system parameters out of sample (R² > 0,
    permutation p < 0.01).
  - Latent effective rank >= 16.
- **H7 (Gate 12, SIMULATED).**
  - Setpoint-tracking MPC with Alma-WM gets lower real cost than ARX-MPC and than the default plan
    (t >= 3).
  - Real gain >= 70% of imagined gain.
- **Transfer (9.7).** The held-out family ("tanks") is reported separately and descriptively; no claim.

## Grading formulas

- **CRPS** = mean weighted quantile loss over q in {0.1..0.9}, normalised by sum |y| (GIFT-Eval form).
  For simulated series, each episode is normalised by its context sd.
- **Relative CRPS** = model / baseline on identical windows, aggregated by geometric mean.
- **Coverage(80)** = share of outcomes in [q0.1, q0.9]; **Coverage(60)** = share in [q0.2, q0.8].
- **Effects:**
  - True effect = y_cf − y_f (same noise).
  - Predicted effect = median_cf − median_f.
  - Both are averaged over the 256 steps after the intervention starts and expressed in context sd.
- **Tests:**
  - Paired t on per-unit log-ratios (unit = task for real data, episode for simulated data).
  - Block bootstrap, 2,000 resamples, block = 1 unit.
  - Holm correction across all "beats" claims in one family.

## Failure policy, sampling and seeds

- **Failures:** a malformed or non-finite output is a failure and is never retried. The failure rate
  must be <= 1%.
- **Sealing:**
  - The predict step gets contexts and action plans only.
  - Ground truth is written by the builder to a separate file and loaded only by the grader.
  - The grader refuses outputs that are missing from the sealed manifest.

## Kill / re-validation triggers (0.6)

Re-validate on any of the following:
- Any change to the weights, data mix, simulators or harness.
- The sealed real holdout is touched a second time.
- Coverage drifts > 5 pp on new data.
- The failure rate is > 1%.

## Built holdout sizes (recorded at freeze)
- sim_val: 240 episodes; sim_test: 800; sim_tanks: 200; plan_test: 100.
- real: 8,607 windows over 29 tasks. Three tasks have no unused windows left after excluding the dev sample and wm1's set, and are skipped: electricity/W/short, SZ_TAXI/15T/short and solar/10T/long.
