# Checkpoints (Git LFS)

| File | Model | dev2 vs Telos (all / sealed28) | Load with |
|---|---|---|---|
| `alma_v3_ema_10000.pt` | Alma v3, 61.8M, EMA weights at 10k steps (+ JEPA head) | 0.981 / 1.033 (ctx 8192) | `AlmaForecaster.load(path, context_len=8192)` |
| `alma_small_D_bucket_mix_ema_3000.pt` | Alma v2-size 27.9M, bucket sampler + mix loss, 3k steps from v2@6k | 0.984 / 1.019 (ctx 2048) | `AlmaForecaster.load(path, context_len=2048)` |
| `alma_small_C_bucket_series_ema_3000.pt` | same, series-normalised loss | 0.985 / 1.019 (ctx 2048) | idem |

Best result: ensemble of v3 + D (`python dev_eval.py --model ens --members checkpoints/alma_v3_ema_10000.pt@8192,checkpoints/alma_small_D_bucket_mix_ema_3000.pt@2048`): 0.974 / 1.017.
Fetch with `git lfs pull`.
