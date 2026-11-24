[2026-11-16 14:54:53] built sim_val: 240 episodes, kinds=['bilinear', 'fopdt', 'hammer', 'inventory', 'logistic', 'lss', 'pricing', 'thermal']
[2026-11-16 14:56:59] built sim_test: 800 episodes, kinds=['bilinear', 'fopdt', 'hammer', 'inventory', 'logistic', 'lss', 'pricing', 'thermal']
[2026-11-16 14:57:37] built sim_tanks: 200 episodes, kinds=['tanks']
[2026-11-16 14:57:42] built plan_test: 100 episodes
[2026-11-16 14:57:46] real holdout m4_yearly/short: 300 unseen windows (of 22974)
[2026-11-16 14:57:47] real holdout m4_quarterly/short: 300 unseen windows (of 24000)
[2026-11-16 14:57:49] real holdout m4_monthly/short: 300 unseen windows (of 48000)
[2026-11-16 14:57:49] real holdout m4_daily/short: 300 unseen windows (of 4227)
[2026-11-16 14:57:50] real holdout electricity/15T/short: 300 unseen windows (of 7400)
[2026-11-16 14:57:50] real holdout electricity/H/short: 300 unseen windows (of 7400)
[2026-11-16 14:57:50] real holdout electricity/D/short: 300 unseen windows (of 1850)
[2026-11-16 14:57:50] real holdout electricity/W/short: skipped, only 0 unused windows
[2026-11-16 14:57:50] real holdout solar/10T/short: 300 unseen windows (of 2740)
[2026-11-16 14:57:50] real holdout solar/H/short: 300 unseen windows (of 2603)
[2026-11-16 14:57:52] real holdout temperature_rain_with_missing/short: 300 unseen windows (of 96216)
[2026-11-16 14:57:52] real holdout kdd_cup_2018_with_missing/H/short: 300 unseen windows (of 5400)
[2026-11-16 14:57:53] real holdout car_parts_with_missing/short: 300 unseen windows (of 2674)
[2026-11-16 14:57:53] real holdout LOOP_SEATTLE/5T/short: 300 unseen windows (of 6460)
[2026-11-16 14:57:53] real holdout LOOP_SEATTLE/H/short: 300 unseen windows (of 6137)
[2026-11-16 14:57:53] real holdout SZ_TAXI/15T/short: skipped, only 0 unused windows
[2026-11-16 14:57:54] real holdout bitbrains_fast_storage/5T/short: 300 unseen windows (of 45000)
[2026-11-16 14:57:54] real holdout bitbrains_fast_storage/H/short: 300 unseen windows (of 5000)
[2026-11-16 14:57:55] real holdout bitbrains_rnd/5T/short: 300 unseen windows (of 18000)
[2026-11-16 14:57:55] real holdout bitbrains_rnd/H/short: 300 unseen windows (of 2000)
[2026-11-16 14:57:55] real holdout electricity/15T/medium: 300 unseen windows (of 7400)
[2026-11-16 14:57:55] real holdout electricity/15T/long: 300 unseen windows (of 7400)
[2026-11-16 14:57:56] real holdout electricity/H/medium: 300 unseen windows (of 2960)
[2026-11-16 14:57:56] real holdout electricity/H/long: 300 unseen windows (of 1850)
[2026-11-16 14:57:56] real holdout solar/10T/medium: 207 unseen windows (of 1507)
[2026-11-16 14:57:56] real holdout solar/10T/long: skipped, only 0 unused windows
[2026-11-16 14:57:56] real holdout LOOP_SEATTLE/5T/medium: 300 unseen windows (of 6460)
[2026-11-16 14:57:56] real holdout LOOP_SEATTLE/5T/long: 300 unseen windows (of 4845)
[2026-11-16 14:57:57] real holdout bitbrains_fast_storage/5T/medium: 300 unseen windows (of 5000)
[2026-11-16 14:57:57] real holdout bitbrains_fast_storage/5T/long: 300 unseen windows (of 5000)
[2026-11-16 14:58:00] real holdout bitbrains_rnd/5T/medium: 300 unseen windows (of 2000)
[2026-11-16 14:58:00] real holdout bitbrains_rnd/5T/long: 300 unseen windows (of 2000)
[2026-11-17 07:06:10] predict: ckpt=runs/alma_wm2/ema_28000.pt suites=['sim_val', 'sim_test', 'sim_tanks', 'plan_test', 'real']
[2026-11-17 07:06:25] sim_val: neural predictions done in 15s
[2026-11-17 07:06:33] sim_val: ARX done in 8s
[2026-11-17 07:06:37] sealed pred_sim_val; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-17 07:07:28] sim_test: neural predictions done in 51s
[2026-11-17 07:07:56] sim_test: ARX done in 28s
[2026-11-17 07:08:07] sealed pred_sim_test; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-17 07:08:20] sim_tanks: neural predictions done in 13s
[2026-11-17 07:08:26] sim_tanks: ARX done in 7s
[2026-11-17 07:08:29] sealed pred_sim_tanks; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-17 07:08:50] sealed pred_plan_test
[2026-11-17 07:08:58] real m4_yearly/short: 300 windows (2s)
[2026-11-17 07:09:00] real m4_quarterly/short: 300 windows (5s)
[2026-11-17 07:09:03] real m4_monthly/short: 300 windows (8s)
[2026-11-17 07:09:14] real m4_daily/short: 300 windows (19s)
[2026-11-17 07:09:28] real electricity/15T/short: 300 windows (32s)
[2026-11-17 07:09:41] real electricity/H/short: 300 windows (46s)
[2026-11-17 07:09:49] real electricity/D/short: 300 windows (53s)
[2026-11-17 07:10:02] real solar/10T/short: 300 windows (67s)
[2026-11-17 07:10:16] real solar/H/short: 300 windows (80s)
[2026-11-17 07:10:20] real temperature_rain_with_missing/short: 300 windows (85s)
[2026-11-17 07:10:34] real kdd_cup_2018_with_missing/H/short: 300 windows (98s)
[2026-11-17 07:10:36] real car_parts_with_missing/short: 300 windows (100s)
[2026-11-17 07:10:49] real LOOP_SEATTLE/5T/short: 300 windows (114s)
[2026-11-17 07:11:03] real LOOP_SEATTLE/H/short: 300 windows (127s)
[2026-11-17 07:11:16] real bitbrains_fast_storage/5T/short: 300 windows (141s)
[2026-11-17 07:11:21] real bitbrains_fast_storage/H/short: 300 windows (145s)
[2026-11-17 07:11:34] real bitbrains_rnd/5T/short: 300 windows (159s)
[2026-11-17 07:11:39] real bitbrains_rnd/H/short: 300 windows (163s)
[2026-11-17 07:11:53] real electricity/15T/medium: 300 windows (178s)
[2026-11-17 07:12:08] real electricity/15T/long: 300 windows (193s)
[2026-11-17 07:12:23] real electricity/H/medium: 300 windows (207s)
[2026-11-17 07:12:38] real electricity/H/long: 300 windows (222s)
[2026-11-17 07:12:47] real solar/10T/medium: 207 windows (232s)
[2026-11-17 07:13:02] real LOOP_SEATTLE/5T/medium: 300 windows (246s)
[2026-11-17 07:13:17] real LOOP_SEATTLE/5T/long: 300 windows (261s)
[2026-11-17 07:13:31] real bitbrains_fast_storage/5T/medium: 300 windows (276s)
[2026-11-17 07:13:46] real bitbrains_fast_storage/5T/long: 300 windows (291s)
[2026-11-17 07:14:01] real bitbrains_rnd/5T/medium: 300 windows (305s)
[2026-11-17 07:14:16] real bitbrains_rnd/5T/long: 300 windows (320s)
[2026-11-17 07:14:33] sealed pred_real
[2026-11-17 07:14:35] [PASS] 1.1 provenance card (source, hashes, licence from HF dataset cards): {'GiftEvalPretrain': 'apache-2.0', 'GiftEval': 'apache-2.0'}
[2026-11-17 07:14:35] [N/A] 1.2 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:35] [N/A] 1.3 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:35] [N/A] 1.4 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:35] [N/A] 1.6 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:35] [N/A] 1.7 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:35] [N/A] 1.8 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-17 07:14:36] [REPORTED] 1.9 missing-data report (2,000 sampled series): 0.0377
[2026-11-17 07:14:36] [PASS] 1.10 leakage canary: context with the label appended is rejected: True
[2026-11-17 07:14:36] [PASS] 2.4 target leakage: encoder states before t unchanged when data/actions after t change: 0.0
[2026-11-17 07:14:36] [PASS] 2.4 predictor: latent k unchanged when actions after k change: 3.337860107421875e-06
[2026-11-17 07:14:36] [N/A] 2.5 latent-action shortcut: actions are observed inputs, not inferred latents
[2026-11-17 07:14:36] [N/A] 2.6/2.7 LLM contamination / comparator parity of tools: no LLM component
[2026-11-17 07:14:36] [REPORTED] 2.1 cutoff ordering: GIFT-Eval test windows are disjoint from GiftEvalPretrain by construction (per dataset authors); simulated holdout seeds disjoint
[2026-11-17 07:16:58] [PASS] 2.2 exact-match scan: 12-value context tails and label heads (>= 6 distinct) in pretraining: {'queries': 83, 'skipped': 33, 'context_hit_tasks': [], 'label_hit_tasks': []}
[2026-11-17 07:16:58] [PASS] 3.6 repeatability: identical outputs on rerun (fixed weights, CPU): 0.0
[2026-11-17 07:16:58] [PASS] 3.7 precision stability fp32 vs bf16 (median relative difference): 5.2913672810820335e-05
[2026-11-17 07:16:59] [REPORTED] 3.9 latency per 512-step action-conditioned rollout (CPU, batch 64): 5.5 ms
[2026-11-17 07:16:59] [PASS] 3.1/3.2 identical inputs for every model; frozen configs: all models read the same harness/inputs_*.npz; hashes in manifest.json
[2026-11-17 07:16:59] [PASS] 5.7 long-horizon drift (2048-step rollouts, held action): finite, median max deviation in context ranges: {'finite': True, 'median_dev_ranges': 0.22, 'max_dev_ranges': 1.99}
[2026-11-17 07:17:01] [REPORTED] 9.5 input perturbation: CRPS ratio vs clean (sim_val): {'clean': 1.0, 'noise_0.1sd': 1.019, 'drop_10pct': 1.497, 'stale_last_64': 1.265}
[2026-11-17 07:17:01] static tests done
[2026-11-17 07:17:05] [FAIL] 4 SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24: {'arx': (1.048, -2.8), 'persistence': (0.534, 24.9), 'snaive24': (0.358, 35.8)}
[2026-11-17 07:17:05] [FAIL] 5.1 SIMULATED rollout beats every simple law in every horizon bucket: {'1-16': {'arx': 1.048, 'persistence': 0.534, 'snaive24': 0.358}, '17-64': {'arx': 0.915, 'persistence': 0.377, 'snaive24': 0.349}, '65-256': {'arx': 0.868, 'persistence': 0.319, 'snaive24': 0.311}, '257-512': {'arx': 0.929, 'persistence': 0.341, 'snaive24': 0.335}}
[2026-11-17 07:17:05] [PASS] 5.1 error growth CRPS(257-512)/CRPS(1-16): 1.83
[2026-11-17 07:17:13] [REPORTED] 5.2 open-loop vs closed-loop (re-anchored) CRPS gap: 2.147
[2026-11-17 07:17:13] [REPORTED] 5.5 no mode collapse: median 80%-width ratio model/oracle: 1.208
[2026-11-17 07:17:13] [REPORTED] 5.3 PIT histogram (10 bins, expected 0.1 each): [0.107, 0.099, 0.097, 0.093, 0.094, 0.097, 0.095, 0.095, 0.099, 0.124]
[2026-11-17 07:17:13] [FAIL] 5.4 distribution two-sample / discriminator test: not computable: model emits marginal quantiles, no joint sample paths
[2026-11-17 07:17:13] [FAIL] 5.6 physical constraint violations (median < 0 for non-negative systems): 0.1915
[2026-11-17 07:17:13] [PASS] 6.1 SIMULATED action-aware beats action-blind (CRPS): (0.634, 20.4)
[2026-11-17 07:17:13] [PASS] 6.2 SIMULATED effect sign accuracy (|true effect|>0.5 sd): {'wm': 0.971, 'arx': 0.922, 'n': 1122}
[2026-11-17 07:17:13] [PASS] 6.2 SIMULATED monotone size sweeps predicted monotone, same direction: {'rate': 0.954, 'n': 680}
[2026-11-17 07:17:13] [FAIL] 6.5* SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories): (0.981, 0.7)
[2026-11-17 07:17:13] [FAIL] 6.3 impact accuracy on real interventions: no real interventional data available
[2026-11-17 07:17:13] [FAIL] 6.4 natural experiments on real data: no real interventional data available
[2026-11-17 07:17:13] [FAIL] 6.5 counterfactual vs nature on real data: no real interventional data available
[2026-11-17 07:17:13] [N/A] 6.6 randomised live checks: no live deployment
[2026-11-17 07:17:13] [N/A] 6.7 self-consistency (split trades): market-specific
[2026-11-17 07:17:13] [N/A] 6.8 reflexivity: market-specific
[2026-11-17 07:17:13] [FAIL] 7 / 4.4 SIMULATED coverage 80% / 60% (wm_aware): (0.769, 0.571)
[2026-11-17 07:17:13] [PASS] 7.2 uncertainty predicts error (Spearman width vs |error|): (0.886, 53.8)
[2026-11-17 07:17:13] [PASS] 7.3 SIMULATED conformal 80% interval coverage (calibrated on sim_val): 0.796
[2026-11-17 07:17:16] [FAIL] 7.4 OOD flag (Mahalanobis of state) AUROC for top-decile errors: {'primary_mahalanobis': 0.525, 'secondary_disagreement': 0.688}
[2026-11-17 07:17:16] [FAIL] 7.5 fallback (ARX) on flagged episodes no worse than the model: {'n_flagged': 637, 'gmean_model': 0.27843118975488285, 'gmean_fallback': 0.3132292457200606}
[2026-11-17 07:17:16] [REPORTED] 7.6 novel family (tanks) flag rate vs in-family flag rate: {'tanks': 0.565, 'in_family': 0.796}
[2026-11-17 07:17:16] [REPORTED] 7.7 risk-coverage (gmean CRPS keeping least-OOD fraction): {'0.5': 0.2781048005193886, '0.7': 0.27612899562764476, '0.9': 0.2727129611908753, '1.0': 0.2749657254863075}
[2026-11-17 07:17:16] [PASS] 8.10 placebo: true actions beat shuffled actions: (0.53, 26.5)
[2026-11-17 07:17:16] [PASS] 8.10 placebo: random actions do not beat action-blind: (1.359, -12.5)
[2026-11-17 07:17:16] [PASS] 8.10 placebo: time-shifted (16 steps) predictions lose skill: (0.645, 20.2)
[2026-11-17 07:17:16] [REPORTED] 9.2/9.3 SIMULATED by system family: aware/ARX gmean ratio: {'bilinear': 0.855, 'fopdt': 1.013, 'hammer': 0.951, 'inventory': 0.822, 'logistic': 0.915, 'lss': 1.316, 'pricing': 0.626, 'thermal': 0.983}
[2026-11-17 07:17:16] [REPORTED] 9.7 transfer to held-out family (tanks), gmean CRPS: {'wm_aware': 0.119, 'wm_blind': 0.211, 'arx': 0.13, 'persistence': 0.376, 'oracle': 0.032, 'aware_vs_arx': 0.919}
[2026-11-17 07:17:23] [FAIL] 10.1 linear probes: state -> true system parameters (5-fold CV R², permutation p): {'thermal.a': (0.66, 0.01), 'thermal.b': (0.28, 0.01), 'fopdt.tau': (0.23, 0.01), 'fopdt.delay': (-0.04, 0.634), 'fopdt.K': (0.32, 0.01), 'inventory.lag': (0.17, 0.01), 'pricing.elast': (0.07, 0.02), 'pricing.carry': (0.05, 0.02)}
[2026-11-17 07:17:23] [PASS] 10.1 latent effective rank (non-collapse): 177.6
[2026-11-17 07:17:23] [FAIL] 10.4 latent stability across training seeds: only one training seed exists
[2026-11-17 07:17:23] [FAIL] 10.2 attribution faithfulness: not run
[2026-11-17 07:17:23] [FAIL] 10.5 disentanglement: not run
[2026-11-17 07:17:23] [N/A] 10.6/10.7 explanation decomposition / residual load: no named-component decomposition
[2026-11-17 07:17:24] [FAIL] 12 SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC: {'vs_default': (0.418, 7.9), 'vs_arx': (1.087, -1.4), 'oracle_vs_default': 0.332}
[2026-11-17 07:17:24] [PASS] 12.1 imagined vs real gain (sum real / sum imagined): 1.078
[2026-11-17 07:17:24] [REPORTED] 12.2 exploit hunt: corr(imagined gain, imagined-real gap): -0.204
[2026-11-17 07:17:24] [N/A] 12.5 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-17 07:17:24] [N/A] 12.6 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-17 07:17:24] [N/A] 12.7 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-17 07:17:24] [N/A] 12.8 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-17 07:17:29] [FAIL] 4.2 REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small: {'snaive': (0.638, 11.1, 'wins 1.00'), 'persistence': (0.553, 6.9, 'wins 0.93'), 'telos': (1.037, -1.8, 'wins 0.24')}
[2026-11-17 07:17:29] [PASS] 4.2 REAL non-inferiority vs Alma v2 (gmean ratio, CI upper): (1.0, 1.0)
[2026-11-17 07:17:29] [REPORTED] 4.2 REAL latent-rollout forecasts vs direct head (ratio): 1.027
[2026-11-17 07:17:29] [FAIL] 4.4 REAL coverage 80% / 60% (Alma-WM): {'wm_direct': (0.812, 0.623), 'wm_rollout': (0.754, 0.563), 'alma_v2': (0.812, 0.623), 'telos': (0.793, 0.609)}
[2026-11-17 07:17:29] [PASS (preliminary)] 8.2 effective sample (real tasks): 29
[2026-11-17 07:17:29] [REPORTED] 8.6 Holm correction across all 'beats' claims: {'placebo_true_vs_shuffled': True, 'plan_wm_vs_arx': False, 'plan_wm_vs_default': True, 'real_wm_vs_persistence': True, 'real_wm_vs_snaive': True, 'real_wm_vs_telos': False, 'sim_aware_vs_blind': True, 'sim_b1-16_vs_arx': False, 'sim_b1-16_vs_persistence': True, 'sim_b1-16_vs_snaive24': True, 'sim_b17-64_vs_arx': True, 'sim_b17-64_vs_persistence': True, 'sim_b17-64_vs_snaive24': True, 'sim_b257-512_vs_arx': True, 'sim_b257-512_vs_persistence': True, 'sim_b257-512_vs_snaive24': True, 'sim_b65-256_vs_arx': True, 'sim_b65-256_vs_persistence': True, 'sim_b65-256_vs_snaive24': True, 'sim_cf_vs_arx': False}
[2026-11-17 07:17:29] grading done
