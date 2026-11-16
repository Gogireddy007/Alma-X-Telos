# Alma v3 forecaster plan (runs after wm2 frees the GPU)

**Goal:** beat Telos-Small on a fresh sealed real-data test, by a significant margin after Holm correction. wm3 then inherits the v3 forecaster as its frozen forecasting path.

## Where v2 loses (wm1 sealed real windows; now burned for testing, used here for design only)

Ratio is v2 CRPS / Telos CRPS, so above 1 means v2 is worse. Geometric mean 1.024; v2 wins 14 of 32 tasks.

| Pattern | Tasks (v2 / Telos) |
|---|---|
| **Long and medium horizons** (worst group) | bitbrains_fast_storage/5T/long 1.32, solar/10T/long 1.25, electricity/H/long 1.16, solar/10T/medium 1.15, electricity/15T/long 1.12, LOOP_SEATTLE/5T/medium and long 1.12 / 1.10 |
| **Short, low-frequency series** | m4_yearly 1.35, m4_monthly 1.05, m4_quarterly 1.05 |
| **v2 wins** | short horizons at high frequency: electricity/D 0.62, bitbrains_fast_storage/5T/short 0.77, solar/10T/short 0.91, electricity/15T/short 0.92 |

## Capacity gap

| | Alma v2 | Telos-Small |
|---|---|---|
| Layers × width | 8 × 384 | 12 × 512 |
| Parameters | 27.9M | about 2-3× more |
| Patch | 16 | 32 |
| Training context | 4,096 | 4,096 (8,192 max) |
| Horizon per shot | 1,024 | 1,024, plus a JEPA latent predictor |

## Changes, in order of expected value per GPU hour
1. **Scale to 12 × 512,** about 60M parameters, warm-started by width/depth growth from v2. This targets the long-horizon gap, which looks like a capacity and steps problem.
2. **Train longer:** 40k steps, against v2's 30k total. Oversample long-horizon targets: weight the horizon loss so steps 256-1024 are not swamped by near-term steps.
3. **Two patch resolutions.** Patch 16 for short and low-frequency series, and patch 32 tokens for long contexts (a stride-32 stem) so 8,192-step contexts are affordable. This targets m4_yearly and the long tasks.
4. **JEPA auxiliary for the forecaster** (Telos does this too). Predict future patch latents with an EMA target encoder. It reuses the wm2 machinery, so the forecaster and the world model share one JEPA backbone.
5. **Evaluation discipline.**
   - Select checkpoints on the GIFT quick-dev sample only.
   - Do the final sealed comparison on GIFT windows excluding the seed-0 dev sample and the seed-777/778 sets (seed 779).
   - Also run the full 97-config GIFT-Eval leaderboard.

## Cost on the M4
About 2× slower per step than v2, so 40k steps is about 40 h. A cheaper first pass is 12 × 512 for 15k steps (about 15 h), checked on quick-dev before committing to the full run.
