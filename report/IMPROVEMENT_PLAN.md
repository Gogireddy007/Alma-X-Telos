# Alma-WM improvement plan (Section G loop) — for `alma_wm2`

Each change targets one failed gate. Every change is logged in `trial_log.csv` before training starts, and the next run gets a new pre-registration and new sealed seeds.

| # | Change | Targets | Expected effect |
|---|---|---|---|
| 1 | Freeze Alma v2's forecasting path. The world-model branch gets its own adapter layers, so the forecast head is bit-identical to v2. | 4.2 non-inferiority | Real-data skill equals v2's by construction. |
| 2 | Train longer and on more control data: 30k+ steps, 50% control batches, 3 more simulator families (nonlinear, saturating, delayed). | 4, 5.1, 6.5*, 12 | Closes the gap to ARX; the target is ≤ 1.0× on lss and thermal. |
| 3 | Add a residual linear-dynamics head inside the predictor (learned ARX-style prior plus a neural correction). | 4, 5.1, 6.5*, 12 | The model can never be worse than the linear law it should contain. |
| 4 | Add a positivity / bounds-aware output transform when the context is non-negative. | 5.6 | Constraint violations go to 0%. |
| 5 | Add a joint path sampler: latent noise with a CRPS-trained ensemble decoder, so trajectories can be sampled. | 5.4, 5.5, raw 7 coverage | Two-sample tests become computable and intervals are wider. |
| 6 | Add an auxiliary system-identification head (predict parameters) and an OOD score based on predicted error (a learned error model, not Mahalanobis distance). | 10.1, 7.4 | Probe R² above 0.5 and OOD AUROC of at least 0.75. |
| 7 | Train 3 seeds, and run integrated-gradients attribution and disentanglement tests. | 10.2, 10.4, 10.5 | Makes those gates computable. |
| 8 | Fix the overlap scan: entropy floor of at least 6 distinct values, and also scan labels. | 2.2 | Removes coincidental 0/1 matches. |
| 9 | Run on real interventional data, e.g. building HVAC control logs or public A/B pricing data. | 6.3-6.5 | Turns the simulated-only causal claims into real-data claims. |

Order of work: 1, 8, 4 (cheap, about 1 hour), then 3 and 2 (one long training run of about 15 h on the M4), then 5, 6 and 7. Real interventional data (9) needs a data source chosen by the user.
