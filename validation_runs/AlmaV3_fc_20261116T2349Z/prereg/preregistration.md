# Pre-registration — Alma v3 forecaster (run `alma_v3`)

Written and hashed before any v3 checkpoint exists.

## Model
- 12 × 512, 61.8M parameters, plus a 7.65M JEPA predictor head that is used only in training.
- Grown exactly from Alma v2 @6k. Verified: v3 output equals v2's to within 8e-6 on contexts up to 4,096 steps.
- Stride-32 coarse path for contexts up to 8,192 steps.
- Long-horizon balanced loss: per-step mean, weighted by (1 + h/H)^0.5.
- JEPA latent objective with an EMA target encoder (0.998), weight 0.5, using the same Predictor module as Alma-WM.

## Training and selection
- 20,000 steps, batch 32, lr 3e-4 with cosine decay, 35% of batches at 8k context.
- **Checkpoint selection:** the EMA snapshot (every 2,000 steps) with the best GIFT quick-dev rel-CRPS. Quick-dev is the first 64 windows of the seed-0 development sample of all 97 configs, scored at 8k context. No other data is used for selection.

## Sealed test (never used for training, selection or any earlier grading)
- GIFT-Eval test windows outside the seed-0 dev sample and outside the wm1 (seed 777) and wm2 (seed 778) sets. Seed 779, up to 300 per task.
- **Built:** 8300 28 (windows, tasks).
- Four tasks were skipped because no unused windows remain: electricity/W/short, SZ_TAXI/15T/short, solar/10T/medium and solar/10T/long. Two of these were among v2's weakest tasks, which is disclosed.
- **Inference settings:**
  - v3: 8,192-step context, symmetric averaging.
  - v2: 2,048-step context, as in its published results.
  - Telos-Small: its own default.

## Claims and thresholds
- Per-task GIFT-Eval CRPS and MASE, relative to Seasonal Naive.
- "v3 beats X" requires a paired t on per-task log CRPS ratios of at least 3, passing Holm correction across {Telos-Small, Alma v2, Seasonal Naive}.
- The full public GIFT-Eval benchmark (97 configs, up to 400 windows each) is also reported. Its windows overlap the quick-dev sample used for selection, so it is a leaderboard number, not a sealed test.

## Pre-registered targets
- Beat Telos-Small on the sealed set with a Holm-corrected t ≥ 3.
- Beat Alma v2 the same way.
- Full GIFT-Eval CRPS skill > 38.76% (v2) and > 38.33% (Telos-Small).
