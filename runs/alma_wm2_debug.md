## step 4000 (Mon Nov 16 17:30:19 PST 2026)
[FAIL] 4 SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24: {'arx': (1.243, -2.3), 'persistence': (0.535, 5.6), 'snaive24': (0.327, 9.4)}
[FAIL] 5.1 SIMULATED rollout beats every simple law in every horizon bucket: {'1-16': {'arx': 1.243, 'persistence': 0.535, 'snaive24': 0.327}, '17-64': {'arx': 1.063, 'persistence': 0.314, 'snaive24': 0.308}, '65-256': {'arx': 1.141, 'persistence': 0.289, 'snaive24': 0.293}, '2
[PASS] 5.1 error growth CRPS(257-512)/CRPS(1-16): 2.27
[FAIL] 5.6 physical constraint violations (median < 0 for non-negative systems): 0.1418
[PASS] 6.1 SIMULATED action-aware beats action-blind (CRPS): (0.624, 6.1)
[PASS] 6.2 SIMULATED effect sign accuracy (|true effect|>0.5 sd): {'wm': 0.906, 'arx': 0.847, 'n': 85}
[FAIL] 6.2 SIMULATED monotone size sweeps predicted monotone, same direction: {'rate': 0.944, 'n': 54}
[FAIL] 6.5* SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories): (1.23, -1.6)
[PASS] 7 / 4.4 SIMULATED coverage 80% / 60% (wm_aware): (0.785, 0.589)
[FAIL] 7.4 OOD flag (Mahalanobis of state) AUROC for top-decile errors: {'primary_mahalanobis': 0.668, 'secondary_disagreement': 0.481}
[REPORTED] 9.2/9.3 SIMULATED by system family: aware/ARX gmean ratio: {'fopdt': 1.093, 'inventory': 1.359, 'lss': 1.867, 'pricing': 0.612, 'thermal': 1.749}
[REPORTED] 9.7 transfer to held-out family (tanks), gmean CRPS: {'wm_aware': 0.147, 'wm_blind': 0.267, 'arx': 0.138, 'persistence': 0.378, 'oracle': 0.053, 'aware_vs_arx': 1.065}
[FAIL] 12 SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC: {'vs_default': (0.534, 3.0), 'vs_arx': (1.054, -0.3), 'oracle_vs_default': 0.393}
[PASS] 12.1 imagined vs real gain (sum real / sum imagined): 1.082
[FAIL] 4.2 REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small: {'snaive': (0.518, 13.9, 'wins 1.00'), 'persistence': (0.379, 10.4, 'wins 1.00'), 'telos': (nan, -inf, 'wins nan')}
[PASS] 4.2 REAL non-inferiority vs Alma v2 (gmean ratio, CI upper): (1.0, 1.0)
[REPORTED] 4.2 REAL latent-rollout forecasts vs direct head (ratio): 1.124
## step 12000 (Mon Nov 16 23:25:28 PST 2026)
[FAIL] 4 SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24: {'arx': (1.132, -1.7), 'persistence': (0.487, 7.0), 'snaive24': (0.298, 10.4)}
[FAIL] 5.1 SIMULATED rollout beats every simple law in every horizon bucket: {'1-16': {'arx': 1.132, 'persistence': 0.487, 'snaive24': 0.298}, '17-64': {'arx': 0.961, 'persistence': 0.284, 'snaive24': 0.278}, '65-256': {'arx': 0.979, 'persistence': 0.248, 'snaive24': 0.252}, '2
[PASS] 5.1 error growth CRPS(257-512)/CRPS(1-16): 2.22
[FAIL] 5.6 physical constraint violations (median < 0 for non-negative systems): 0.1444
[PASS] 6.1 SIMULATED action-aware beats action-blind (CRPS): (0.554, 6.6)
[PASS] 6.2 SIMULATED effect sign accuracy (|true effect|>0.5 sd): {'wm': 0.976, 'arx': 0.847, 'n': 85}
[PASS] 6.2 SIMULATED monotone size sweeps predicted monotone, same direction: {'rate': 0.963, 'n': 54}
[FAIL] 6.5* SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories): (1.092, -0.8)
[PASS] 7 / 4.4 SIMULATED coverage 80% / 60% (wm_aware): (0.79, 0.594)
[PASS] 7.4 OOD flag (Mahalanobis of state) AUROC for top-decile errors: {'primary_mahalanobis': 0.793, 'secondary_disagreement': 0.78}
[REPORTED] 9.2/9.3 SIMULATED by system family: aware/ARX gmean ratio: {'fopdt': 1.016, 'inventory': 1.129, 'lss': 1.523, 'pricing': 0.581, 'thermal': 1.527}
[REPORTED] 9.7 transfer to held-out family (tanks), gmean CRPS: {'wm_aware': 0.131, 'wm_blind': 0.271, 'arx': 0.138, 'persistence': 0.378, 'oracle': 0.053, 'aware_vs_arx': 0.949}
[FAIL] 12 SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC: {'vs_default': (0.509, 3.4), 'vs_arx': (1.005, -0.0), 'oracle_vs_default': 0.393}
[PASS] 12.1 imagined vs real gain (sum real / sum imagined): 1.086
[FAIL] 4.2 REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small: {'snaive': (0.518, 13.9, 'wins 1.00'), 'persistence': (0.379, 10.4, 'wins 1.00'), 'telos': (nan, -inf, 'wins nan')}
[PASS] 4.2 REAL non-inferiority vs Alma v2 (gmean ratio, CI upper): (1.0, 1.0)
[REPORTED] 4.2 REAL latent-rollout forecasts vs direct head (ratio): 1.091
## step 20000 (Tue Nov 17 03:15:40 PST 2026)
[FAIL] 4 SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24: {'arx': (1.078, -1.1), 'persistence': (0.464, 7.7), 'snaive24': (0.284, 10.7)}
[FAIL] 5.1 SIMULATED rollout beats every simple law in every horizon bucket: {'1-16': {'arx': 1.078, 'persistence': 0.464, 'snaive24': 0.284}, '17-64': {'arx': 0.904, 'persistence': 0.268, 'snaive24': 0.262}, '65-256': {'arx': 0.921, 'persistence': 0.233, 'snaive24': 0.237}, '2
[PASS] 5.1 error growth CRPS(257-512)/CRPS(1-16): 2.2
[FAIL] 5.6 physical constraint violations (median < 0 for non-negative systems): 0.1344
[PASS] 6.1 SIMULATED action-aware beats action-blind (CRPS): (0.527, 6.9)
[PASS] 6.2 SIMULATED effect sign accuracy (|true effect|>0.5 sd): {'wm': 1.0, 'arx': 0.847, 'n': 85}
[PASS] 6.2 SIMULATED monotone size sweeps predicted monotone, same direction: {'rate': 1.0, 'n': 54}
[FAIL] 6.5* SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories): (1.041, -0.3)
[FAIL] 7 / 4.4 SIMULATED coverage 80% / 60% (wm_aware): (0.776, 0.579)
[FAIL] 7.4 OOD flag (Mahalanobis of state) AUROC for top-decile errors: {'primary_mahalanobis': 0.693, 'secondary_disagreement': 0.792}
[REPORTED] 9.2/9.3 SIMULATED by system family: aware/ARX gmean ratio: {'fopdt': 0.99, 'inventory': 1.091, 'lss': 1.466, 'pricing': 0.541, 'thermal': 1.38}
[REPORTED] 9.7 transfer to held-out family (tanks), gmean CRPS: {'wm_aware': 0.134, 'wm_blind': 0.257, 'arx': 0.138, 'persistence': 0.378, 'oracle': 0.053, 'aware_vs_arx': 0.971}
[FAIL] 12 SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC: {'vs_default': (0.549, 3.0), 'vs_arx': (1.084, -0.5), 'oracle_vs_default': 0.393}
[PASS] 12.1 imagined vs real gain (sum real / sum imagined): 0.988
[FAIL] 4.2 REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small: {'snaive': (0.518, 13.9, 'wins 1.00'), 'persistence': (0.379, 10.4, 'wins 1.00'), 'telos': (nan, -inf, 'wins nan')}
[PASS] 4.2 REAL non-inferiority vs Alma v2 (gmean ratio, CI upper): (1.0, 1.0)
[REPORTED] 4.2 REAL latent-rollout forecasts vs direct head (ratio): 1.074
