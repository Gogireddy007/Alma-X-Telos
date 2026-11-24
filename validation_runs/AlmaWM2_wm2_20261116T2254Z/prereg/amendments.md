# Amendments (made before any Alma-WM prediction or score existed)

## A1 — 2026-11-16, while building the harness; no results seen

These operational definitions were left open in preregistration.md.

**OOD flag (7.4)**
- Primary score: Mahalanobis distance (Ledoit-Wolf covariance) of the context-end latent state from
  the sim_val states.
- Secondary score: disagreement between the direct head and the rollout. It is reported, but never used
  for pass/fail.
- "Large error" means the top decile of per-episode normalised CRPS on pooled sim_test + sim_tanks.

**Fallback (7.5)**
- An episode is flagged when its score exceeds the 90th percentile of the sim_val scores.
- On flagged episodes the fallback is ARX on simulated data and Seasonal Naive on real data.

**Planning (12)**
- Imagined cost of a plan is the mean over quantile levels of (q − target)², over the first 256 steps,
  divided by the context variance.
- Real cost is the same quantity computed on the true trajectory.
- Candidate 0 (hold the last action) is the default plan.

**Effect window (6.2)**
- Effects are averaged over the first 256 steps after the intervention.

**Thresholds not set in Section D**
- Dispersion (5.5), PIT shape (5.3), perturbation degradation (9.5) and transfer (9.7) are reported
  descriptively, with no pass threshold. Each therefore counts as **not passed** for the claim table
  rather than PASS.

**Holm family for "beats" claims**
- Real: wm_direct vs {snaive, persistence, telos}.
- SIMULATED: wm_aware vs {persistence, snaive24, arx} in each horizon bucket, aware vs blind, and
  cf vs arx_cf.
- Planning: WM-MPC vs {default plan, ARX-MPC}.

## A2 (wm2, before sealing)
See preregistration.md §Amendment A2: the 2.2 overlap scan gets an entropy floor and a label scan, 5.6 adds logistic to the non-negative families, and 4.2 non-inferiority is reported as by-construction.

## A3 (wm2, 2026-11-16 17:45 PST, before sealing; based on development data only)
**Gate 5.6 is mis-specified, so its result will not count toward any claim tier.**
- On the DEBUG harness (development data, not a sealed holdout), the *ground truth* itself is below
  zero 4.9% of the time for inventory and 20.3% for pricing. The simulators observe deviations with
  noise, so these series are not physically non-negative.
- Rates at the step-4,000 checkpoint, below zero:

  | | wm2 | ARX | v2 | Truth |
  |---|---|---|---|---|
  | inventory | 8.6% | 8.4% | 6.5% | 4.9% |
  | pricing | 19.8% | 20.2% | 21.2% | 20.3% |

  The `<= 0.01` threshold cannot be met by a perfect model.
- **Action:** the grading code and threshold stay unchanged. 5.6 is still computed and printed in the
  sealed report, and the truth's own rate is reported next to it. It counts as **not passed**
  (uninformative) for the claim table and cannot help any tier.
- **Next round (wm3):** replace it with a test against families whose latent state is truly bounded,
  defined before any data is drawn.
