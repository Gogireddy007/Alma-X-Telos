[2026-11-16 10:31:59] built sim_val: 200 episodes, kinds=['fopdt', 'inventory', 'lss', 'pricing', 'thermal']
[2026-11-16 10:33:22] built sim_test: 600 episodes, kinds=['fopdt', 'inventory', 'lss', 'pricing', 'thermal']
[2026-11-16 10:34:01] built sim_tanks: 200 episodes, kinds=['tanks']
[2026-11-16 10:34:06] built plan_test: 100 episodes
[2026-11-16 10:34:11] real holdout m4_yearly/short: 300 unseen windows (of 22974)
[2026-11-16 10:34:12] real holdout m4_quarterly/short: 300 unseen windows (of 24000)
[2026-11-16 10:34:14] real holdout m4_monthly/short: 300 unseen windows (of 48000)
[2026-11-16 10:34:15] real holdout m4_daily/short: 300 unseen windows (of 4227)
[2026-11-16 10:34:15] real holdout electricity/15T/short: 300 unseen windows (of 7400)
[2026-11-16 10:34:15] real holdout electricity/H/short: 300 unseen windows (of 7400)
[2026-11-16 10:34:15] real holdout electricity/D/short: 300 unseen windows (of 1850)
[2026-11-16 10:34:15] real holdout electricity/W/short: 110 unseen windows (of 1110)
[2026-11-16 10:34:15] real holdout solar/10T/short: 300 unseen windows (of 2740)
[2026-11-16 10:34:15] real holdout solar/H/short: 300 unseen windows (of 2603)
[2026-11-16 10:34:17] real holdout temperature_rain_with_missing/short: 300 unseen windows (of 96216)
[2026-11-16 10:34:17] real holdout kdd_cup_2018_with_missing/H/short: 300 unseen windows (of 5400)
[2026-11-16 10:34:18] real holdout car_parts_with_missing/short: 300 unseen windows (of 2674)
[2026-11-16 10:34:18] real holdout LOOP_SEATTLE/5T/short: 300 unseen windows (of 6460)
[2026-11-16 10:34:18] real holdout LOOP_SEATTLE/H/short: 300 unseen windows (of 6137)
[2026-11-16 10:34:18] real holdout SZ_TAXI/15T/short: 92 unseen windows (of 1092)
[2026-11-16 10:34:19] real holdout bitbrains_fast_storage/5T/short: 300 unseen windows (of 45000)
[2026-11-16 10:34:19] real holdout bitbrains_fast_storage/H/short: 300 unseen windows (of 5000)
[2026-11-16 10:34:19] real holdout bitbrains_rnd/5T/short: 300 unseen windows (of 18000)
[2026-11-16 10:34:20] real holdout bitbrains_rnd/H/short: 300 unseen windows (of 2000)
[2026-11-16 10:34:20] real holdout electricity/15T/medium: 300 unseen windows (of 7400)
[2026-11-16 10:34:20] real holdout electricity/15T/long: 300 unseen windows (of 7400)
[2026-11-16 10:34:20] real holdout electricity/H/medium: 300 unseen windows (of 2960)
[2026-11-16 10:34:20] real holdout electricity/H/long: 300 unseen windows (of 1850)
[2026-11-16 10:34:20] real holdout solar/10T/medium: 300 unseen windows (of 1507)
[2026-11-16 10:34:20] real holdout solar/10T/long: 96 unseen windows (of 1096)
[2026-11-16 10:34:21] real holdout LOOP_SEATTLE/5T/medium: 300 unseen windows (of 6460)
[2026-11-16 10:34:21] real holdout LOOP_SEATTLE/5T/long: 300 unseen windows (of 4845)
[2026-11-16 10:34:22] real holdout bitbrains_fast_storage/5T/medium: 300 unseen windows (of 5000)
[2026-11-16 10:34:22] real holdout bitbrains_fast_storage/5T/long: 300 unseen windows (of 5000)
[2026-11-16 10:34:22] real holdout bitbrains_rnd/5T/medium: 300 unseen windows (of 2000)
[2026-11-16 10:34:23] real holdout bitbrains_rnd/5T/long: 300 unseen windows (of 2000)
[2026-11-16 14:15:55] predict: ckpt=runs/alma_wm1/ema_8000.pt suites=['sim_val', 'sim_test', 'sim_tanks', 'plan_test', 'real']
[2026-11-16 14:16:08] sim_val: neural predictions done in 13s
[2026-11-16 14:16:15] sim_val: ARX done in 7s
[2026-11-16 14:16:18] sealed pred_sim_val; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-16 14:16:55] sim_test: neural predictions done in 37s
[2026-11-16 14:17:16] sim_test: ARX done in 21s
[2026-11-16 14:17:25] sealed pred_sim_test; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-16 14:17:37] sim_tanks: neural predictions done in 12s
[2026-11-16 14:17:44] sim_tanks: ARX done in 7s
[2026-11-16 14:17:47] sealed pred_sim_tanks; non-finite rows per model: {'wm_aware': 0, 'state': 0, 'wm_blind': 0, 'wm_shuffled': 0, 'wm_random': 0, 'wm_cf': 0, 'wm_direct': 0, 'alma_v2': 0, 'persistence': 0, 'snaive24': 0, 'arx': 0, 'arx_cf': 0}
[2026-11-16 14:18:07] sealed pred_plan_test
[2026-11-16 14:18:17] real m4_yearly/short: 300 windows (3s)
[2026-11-16 14:18:20] real m4_quarterly/short: 300 windows (5s)
[2026-11-16 14:18:23] real m4_monthly/short: 300 windows (8s)
[2026-11-16 14:18:34] real m4_daily/short: 300 windows (20s)
[2026-11-16 14:18:48] real electricity/15T/short: 300 windows (34s)
[2026-11-16 14:19:02] real electricity/H/short: 300 windows (47s)
[2026-11-16 14:19:09] real electricity/D/short: 300 windows (55s)
[2026-11-16 14:19:10] real electricity/W/short: 110 windows (56s)
[2026-11-16 14:19:24] real solar/10T/short: 300 windows (70s)
[2026-11-16 14:19:38] real solar/H/short: 300 windows (84s)
[2026-11-16 14:19:43] real temperature_rain_with_missing/short: 300 windows (88s)
[2026-11-16 14:19:56] real kdd_cup_2018_with_missing/H/short: 300 windows (102s)
[2026-11-16 14:19:58] real car_parts_with_missing/short: 300 windows (104s)
[2026-11-16 14:20:12] real LOOP_SEATTLE/5T/short: 300 windows (118s)
[2026-11-16 14:20:25] real LOOP_SEATTLE/H/short: 300 windows (131s)
[2026-11-16 14:20:29] real SZ_TAXI/15T/short: 92 windows (135s)
[2026-11-16 14:20:43] real bitbrains_fast_storage/5T/short: 300 windows (149s)
[2026-11-16 14:20:47] real bitbrains_fast_storage/H/short: 300 windows (153s)
[2026-11-16 14:21:01] real bitbrains_rnd/5T/short: 300 windows (167s)
[2026-11-16 14:21:05] real bitbrains_rnd/H/short: 300 windows (171s)
[2026-11-16 14:21:20] real electricity/15T/medium: 300 windows (186s)
[2026-11-16 14:21:36] real electricity/15T/long: 300 windows (201s)
[2026-11-16 14:21:50] real electricity/H/medium: 300 windows (216s)
[2026-11-16 14:22:06] real electricity/H/long: 300 windows (231s)
[2026-11-16 14:22:20] real solar/10T/medium: 300 windows (246s)
[2026-11-16 14:22:25] real solar/10T/long: 96 windows (251s)
[2026-11-16 14:22:41] real LOOP_SEATTLE/5T/medium: 300 windows (266s)
[2026-11-16 14:22:56] real LOOP_SEATTLE/5T/long: 300 windows (281s)
[2026-11-16 14:23:10] real bitbrains_fast_storage/5T/medium: 300 windows (296s)
[2026-11-16 14:23:28] real bitbrains_fast_storage/5T/long: 300 windows (314s)
[2026-11-16 14:23:45] real bitbrains_rnd/5T/medium: 300 windows (331s)
[2026-11-16 14:24:02] real bitbrains_rnd/5T/long: 300 windows (348s)
[2026-11-16 14:24:22] sealed pred_real
[2026-11-16 14:24:25] [PASS] 1.1 provenance card (source, hashes, licence from HF dataset cards): {'GiftEvalPretrain': 'apache-2.0', 'GiftEval': 'apache-2.0'}
[2026-11-16 14:24:25] [N/A] 1.2 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:25] [N/A] 1.3 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:25] [N/A] 1.4 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:25] [N/A] 1.6 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:25] [N/A] 1.7 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:25] [N/A] 1.8 market point-in-time / survivorship / microstructure / corporate actions: not securities data
[2026-11-16 14:24:26] [REPORTED] 1.9 missing-data report (2,000 sampled series): 0.0377
[2026-11-16 14:24:26] [PASS] 1.10 leakage canary: context with the label appended is rejected: True
[2026-11-16 14:24:26] [PASS] 2.4 target leakage: encoder states before t unchanged when data/actions after t change: 0.0
[2026-11-16 14:24:26] [PASS] 2.4 predictor: latent k unchanged when actions after k change: 3.814697265625e-06
[2026-11-16 14:24:26] [N/A] 2.5 latent-action shortcut: actions are observed inputs, not inferred latents
[2026-11-16 14:24:26] [N/A] 2.6/2.7 LLM contamination / comparator parity of tools: no LLM component
[2026-11-16 14:24:26] [REPORTED] 2.1 cutoff ordering: GIFT-Eval test windows are disjoint from GiftEvalPretrain by construction (per dataset authors); simulated holdout seeds disjoint
[2026-11-16 14:26:30] [FAIL] 2.2 exact-match scan: sealed-window context tails (12 values) found in pretraining data: {'tasks_with_hit': 1, 'tasks': 31}
[2026-11-16 14:26:30] [PASS] 3.6 repeatability: identical outputs on rerun (fixed weights, CPU): 0.0
[2026-11-16 14:26:31] [PASS] 3.7 precision stability fp32 vs bf16 (median relative difference): 0.0004249699847723821
[2026-11-16 14:26:31] [REPORTED] 3.9 latency per 512-step action-conditioned rollout (CPU, batch 64): 5.5 ms
[2026-11-16 14:26:31] [PASS] 3.1/3.2 identical inputs for every model; frozen configs: all models read the same harness/inputs_*.npz; hashes in manifest.json
[2026-11-16 14:26:31] [PASS] 5.7 long-horizon drift (2048-step rollouts, held action): finite, median max deviation in context ranges: {'finite': True, 'median_dev_ranges': 0.24, 'max_dev_ranges': 2.11}
[2026-11-16 14:26:34] [REPORTED] 9.5 input perturbation: CRPS ratio vs clean (sim_val): {'clean': 1.0, 'noise_0.1sd': 1.044, 'drop_10pct': 1.138, 'stale_last_64': 1.323}
[2026-11-16 14:26:34] static tests done
[2026-11-16 14:26:38] [FAIL] 4 SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24: {'arx': (1.353, -10.7), 'persistence': (0.597, 16.6), 'snaive24': (0.398, 28.5)}
[2026-11-16 14:26:38] [FAIL] 5.1 SIMULATED rollout beats every simple law in every horizon bucket: {'1-16': {'arx': 1.353, 'persistence': 0.597, 'snaive24': 0.398}, '17-64': {'arx': 1.176, 'persistence': 0.413, 'snaive24': 0.394}, '65-256': {'arx': 1.182, 'persistence': 0.357, 'snaive24': 0.358}, '257-512': {'arx': 1.26, 'persistence': 0.392, 'snaive24': 0.398}}
[2026-11-16 14:26:38] [PASS] 5.1 error growth CRPS(257-512)/CRPS(1-16): 2.07
[2026-11-16 14:26:43] [REPORTED] 5.2 open-loop vs closed-loop (re-anchored) CRPS gap: 1.854
[2026-11-16 14:26:43] [REPORTED] 5.5 no mode collapse: median 80%-width ratio model/oracle: 1.664
[2026-11-16 14:26:43] [REPORTED] 5.3 PIT histogram (10 bins, expected 0.1 each): [0.121, 0.1, 0.093, 0.095, 0.092, 0.092, 0.09, 0.091, 0.1, 0.126]
[2026-11-16 14:26:43] [FAIL] 5.4 distribution two-sample / discriminator test: not computable: model emits marginal quantiles, no joint sample paths
[2026-11-16 14:26:43] [FAIL] 5.6 physical constraint violations (median < 0 for non-negative systems): 0.1541
[2026-11-16 14:26:43] [PASS] 6.1 SIMULATED action-aware beats action-blind (CRPS): (0.751, 15.1)
[2026-11-16 14:26:43] [PASS] 6.2 SIMULATED effect sign accuracy (|true effect|>0.5 sd): {'wm': 0.954, 'arx': 0.925, 'n': 839}
[2026-11-16 14:26:43] [FAIL] 6.2 SIMULATED monotone size sweeps predicted monotone, same direction: {'rate': 0.923, 'n': 522}
[2026-11-16 14:26:43] [FAIL] 6.5* SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories): (1.427, -8.8)
[2026-11-16 14:26:43] [FAIL] 6.3 impact accuracy on real interventions: no real interventional data available
[2026-11-16 14:26:43] [FAIL] 6.4 natural experiments on real data: no real interventional data available
[2026-11-16 14:26:43] [FAIL] 6.5 counterfactual vs nature on real data: no real interventional data available
[2026-11-16 14:26:43] [N/A] 6.6 randomised live checks: no live deployment
[2026-11-16 14:26:43] [N/A] 6.7 self-consistency (split trades): market-specific
[2026-11-16 14:26:43] [N/A] 6.8 reflexivity: market-specific
[2026-11-16 14:26:43] [FAIL] 7 / 4.4 SIMULATED coverage 80% / 60% (wm_aware): (0.753, 0.554)
[2026-11-16 14:26:43] [PASS] 7.2 uncertainty predicts error (Spearman width vs |error|): (0.771, 29.6)
[2026-11-16 14:26:44] [PASS] 7.3 SIMULATED conformal 80% interval coverage (calibrated on sim_val): 0.799
[2026-11-16 14:26:46] [FAIL] 7.4 OOD flag (Mahalanobis of state) AUROC for top-decile errors: {'primary_mahalanobis': 0.532, 'secondary_disagreement': 0.584}
[2026-11-16 14:26:46] [PASS] 7.5 fallback (ARX) on flagged episodes no worse than the model: {'n_flagged': 438, 'gmean_model': 0.31481218543586464, 'gmean_fallback': 0.2606995546452661}
[2026-11-16 14:26:46] [REPORTED] 7.6 novel family (tanks) flag rate vs in-family flag rate: {'tanks': 0.455, 'in_family': 0.73}
[2026-11-16 14:26:46] [REPORTED] 7.7 risk-coverage (gmean CRPS keeping least-OOD fraction): {'0.5': 0.28597498878750716, '0.7': 0.29347211207356344, '0.9': 0.28874308959778616, '1.0': 0.292159936037531}
[2026-11-16 14:26:46] [PASS] 8.10 placebo: true actions beat shuffled actions: (0.637, 20.5)
[2026-11-16 14:26:46] [PASS] 8.10 placebo: random actions do not beat action-blind: (1.162, -7.8)
[2026-11-16 14:26:46] [PASS] 8.10 placebo: time-shifted (16 steps) predictions lose skill: (0.677, 18.2)
[2026-11-16 14:26:46] [REPORTED] 9.2/9.3 SIMULATED by system family: aware/ARX gmean ratio: {'fopdt': 1.256, 'inventory': 1.111, 'lss': 2.195, 'pricing': 0.716, 'thermal': 1.368}
[2026-11-16 14:26:46] [REPORTED] 9.7 transfer to held-out family (tanks), gmean CRPS: {'wm_aware': 0.152, 'wm_blind': 0.196, 'arx': 0.109, 'persistence': 0.303, 'oracle': 0.032, 'aware_vs_arx': 1.402}
[2026-11-16 14:26:59] [FAIL] 10.1 linear probes: state -> true system parameters (5-fold CV R², permutation p): {'thermal.a': (0.27, 0.01), 'thermal.b': (0.13, 0.01), 'fopdt.tau': (0.03, 0.02), 'fopdt.delay': (-0.02, 0.386), 'fopdt.K': (0.19, 0.01), 'inventory.lag': (-0.05, 0.832), 'pricing.elast': (-0.03, 0.554), 'pricing.carry': (0.11, 0.01)}
[2026-11-16 14:26:59] [PASS] 10.1 latent effective rank (non-collapse): 146.3
[2026-11-16 14:26:59] [FAIL] 10.4 latent stability across training seeds: only one training seed exists
[2026-11-16 14:26:59] [FAIL] 10.2 attribution faithfulness: not run
[2026-11-16 14:26:59] [FAIL] 10.5 disentanglement: not run
[2026-11-16 14:26:59] [N/A] 10.6/10.7 explanation decomposition / residual load: no named-component decomposition
[2026-11-16 14:26:59] [FAIL] 12 SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC: {'vs_default': (0.652, 3.3), 'vs_arx': (1.691, -5.0), 'oracle_vs_default': 0.337}
[2026-11-16 14:26:59] [PASS] 12.1 imagined vs real gain (sum real / sum imagined): 1.131
[2026-11-16 14:26:59] [REPORTED] 12.2 exploit hunt: corr(imagined gain, imagined-real gap): -0.084
[2026-11-16 14:26:59] [N/A] 12.5 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-16 14:26:59] [N/A] 12.6 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-16 14:26:59] [N/A] 12.7 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-16 14:26:59] [N/A] 12.8 execution / portfolio / risk / deflated Sharpe: no trading component
[2026-11-16 14:27:05] [FAIL] 4.2 REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small: {'snaive': (0.645, 7.3, 'wins 1.00'), 'persistence': (0.514, 7.1, 'wins 0.88'), 'telos': (1.038, -1.4, 'wins 0.41')}
[2026-11-16 14:27:05] [FAIL] 4.2 REAL non-inferiority vs Alma v2 (gmean ratio, CI upper): (1.0141, 1.0264)
[2026-11-16 14:27:05] [REPORTED] 4.2 REAL latent-rollout forecasts vs direct head (ratio): 1.075
[2026-11-16 14:27:05] [FAIL] 4.4 REAL coverage 80% / 60% (Alma-WM): {'wm_direct': (0.817, 0.623), 'wm_rollout': (0.743, 0.544), 'alma_v2': (0.818, 0.629), 'telos': (0.799, 0.618)}
[2026-11-16 14:27:05] [PASS (preliminary)] 8.2 effective sample (real tasks): 32
[2026-11-16 14:27:05] [REPORTED] 8.6 Holm correction across all 'beats' claims: {'placebo_true_vs_shuffled': True, 'plan_wm_vs_arx': False, 'plan_wm_vs_default': True, 'real_wm_vs_persistence': True, 'real_wm_vs_snaive': True, 'real_wm_vs_telos': False, 'sim_aware_vs_blind': True, 'sim_b1-16_vs_arx': False, 'sim_b1-16_vs_persistence': True, 'sim_b1-16_vs_snaive24': True, 'sim_b17-64_vs_arx': False, 'sim_b17-64_vs_persistence': True, 'sim_b17-64_vs_snaive24': True, 'sim_b257-512_vs_arx': False, 'sim_b257-512_vs_persistence': True, 'sim_b257-512_vs_snaive24': True, 'sim_b65-256_vs_arx': False, 'sim_b65-256_vs_persistence': True, 'sim_b65-256_vs_snaive24': True, 'sim_cf_vs_arx': False}
[2026-11-16 14:27:05] grading done
