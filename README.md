# Alma × Telos

Alma is a time-series foundation model trained on a laptop (Apple-silicon MPS) to compete with
**Telos-Small** (50.1M params) on GIFT-Eval.

| Model | Params | Notes |
|---|---|---|
| Alma v2 | 27.9M | 8 layers × 384 |
| Alma v3 | 61.8M | 12 × 512, coarse path to 8k context, JEPA auxiliary head (`model_v3.py`, `train_v3.py`) |

## Current standing (dev2, 256 random windows per task, seed 1; ratio vs Telos, < 1 = Alma better)

| Model | All 97 tasks | 28 sealed-task names |
|---|---|---|
| Ensemble v3@10k + D (quantile averaging) | **0.974** | 1.017 |
| Honest 2-fold ensemble search (`ens_search.py`) | 0.975 | 1.021 |
| Alma v3 @ 10k steps (ctx 8192) | 0.981 | 1.033 |
| Small model D (bucket sampler, mix loss) | 0.984 | 1.019 |

Alma beats Telos on the full dev set but is still ~2% behind on the sealed-task subset.
The gap is concentrated in hourly traffic (LOOP_SEATTLE), dense traffic (M_DENSE) and short M4
yearly/quarterly series, mostly at the first horizon steps (point accuracy). It is the same for every
data/recipe/seed variant tried, which points to an architecture change as the next step.
Full ranking: `results/dev2/progress_summary.txt`; experiment notes: `results/overnight_notes.md`;
per-family breakdown: `python fam_report.py`.

## What was tried (stages)

- **A**: sampler (sqrt vs frequency bucket) × loss normalisation (token / series / mix). Bucket + mix best.
- **B**: short ETS synthetic series; v3 fine-tune; checkpoint soups. No gain.
- **C/ens**: quantile-averaged ensembles. Best lever so far.
- **D**: +1.5B new GiftEvalPretrain points (`prep_pretrain2.py`). Neutral at 3k steps.
- **E**: v3 continued pretraining on both corpora. Neutral at 2k, worse at 4k; stopped.
- **F**: traffic/cloud source up-weighting (`ALMA_BOOST`), seed replicate. No gain (seed noise ≈ 0.25%).
- Post-hoc spread calibration and disabling sign-symmetric inference: no gain.

## Layout

- `model.py`, `model_v3.py`, `data.py`, `train.py`, `train_v3.py`, `forecast.py`: model, data stream, training, inference
- `dev_eval.py` (dev2 scoring, `--save-preds`), `dev_rank.py`, `dev_compare.py`, `ens_search.py`, `calib_search.py`, `fam_report.py`
- `run_stage*.sh`: the chained experiment drivers; `resume_all.sh` resumes the paused Stage F/G
- `results/`: all dev2 result JSONs, logs, ensemble searches; `runs/`: training logs (`log.jsonl`, `*.train.log`)
- `validation_runs/`: pre-registered sealed validation runs; `report/`: written reports

Not in the repo (size): training data (`data/`, ~18 GB), checkpoints (`*.pt`), saved prediction arrays (`*.npz`).
Sealed windows (seeds 777–779) were used only for pre-registered tests; seed 780 is reserved.
