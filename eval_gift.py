"""GIFT-Eval harness: official 97 task configs, fixed window subsample, metrics normalised by Seasonal Naive.

Every model is scored on exactly the same windows, so `rel_crps` / `rel_mase` are directly comparable.
CRPS  = mean weighted quantile loss over q in 0.1..0.9 (GluonTS `mean_weighted_sum_quantile_loss`).
MASE  = MASE of the median, seasonal error from each window's own history (GluonTS `MASE[0.5]`).
Aggregate = geometric mean over tasks of (model / seasonal naive); skill % = 100 * (1 - gmean).

Usage:
  python eval_gift.py --model telos  --out results/telos.json
  python eval_gift.py --model alma --ckpt runs/x/ema.pt --out results/alma.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SHORT = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
    "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H "
    "ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application "
    "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
TASKS = [(d, "short") for d in SHORT] + [(d, t) for d in MED_LONG for t in ("medium", "long")]


# ------------------------------------------------------------------------------------------------
# Task loading (cached as compact npz so later runs skip GluonTS entirely)
# ------------------------------------------------------------------------------------------------


def load_task(name: str, term: str, max_windows: int, cache_dir: Path, seed: int = 0) -> dict:
    key = f"{name.replace('/', '_')}__{term}__{max_windows}.npz"
    path = cache_dir / key
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {"contexts": list(z["contexts"]), "labels": z["labels"], "season": int(z["season"]),
                "freq": str(z["freq"]), "H": int(z["H"]), "total_windows": int(z["total_windows"])}

    from gift_eval.data import Dataset
    from gluonts.time_feature import get_seasonality

    to_uni = Dataset(name=name, term=term, to_univariate=False).target_dim != 1
    ds = Dataset(name=name, term=term, to_univariate=to_uni)
    H = ds.prediction_length
    season = get_seasonality(ds.freq)
    pairs = list(ds.test_data)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    if len(idx) > max_windows:
        idx = np.sort(rng.choice(idx, size=max_windows, replace=False))
    contexts, labels = [], []
    for i in idx:
        inp, lab = pairs[i]
        contexts.append(np.asarray(inp["target"], dtype=np.float32))
        labels.append(np.asarray(lab["target"], dtype=np.float32)[:H])
    labels = np.stack(labels)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, contexts=np.array(contexts, dtype=object), labels=labels, season=season,
             freq=ds.freq, H=H, total_windows=len(pairs))
    return {"contexts": contexts, "labels": labels, "season": season, "freq": ds.freq, "H": H,
            "total_windows": len(pairs)}


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------


def _ffill(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(ok)
    x[:first] = x[first]
    return x


def seasonal_naive(contexts: list, H: int, season: int) -> np.ndarray:
    out = np.zeros((len(contexts), H), dtype=np.float64)
    for i, c in enumerate(contexts):  # mirrors gluonts SeasonalNaivePredictor (last-value imputation, mean fallback)
        raw = c.astype(np.float64)
        c = _ffill(raw)
        if len(c) >= season:
            last = c[-season:]
            out[i] = np.tile(last, math.ceil(H / season))[:H]
        else:
            out[i] = np.nanmean(raw) if np.isfinite(raw).any() else 0.0
    return out


def seasonal_error(c: np.ndarray, season: int) -> float:
    c = c.astype(np.float64)
    m = season if len(c) > season else 1
    d = np.abs(c[m:] - c[:-m])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else np.nan


def score(contexts: list, labels: np.ndarray, qpred: np.ndarray, season: int) -> dict:
    """qpred: (N, H, 9) quantiles at QUANTILES."""
    y = labels.astype(np.float64)
    mask = np.isfinite(y)
    y0 = np.where(mask, y, 0.0)
    denom = np.abs(y0).sum()
    wql = []
    for k, q in enumerate(QUANTILES):
        f = qpred[..., k]
        loss = np.where(mask, np.maximum(q * (y0 - f), (q - 1) * (y0 - f)), 0.0).sum()
        wql.append(2 * loss / max(denom, 1e-12))
    med = qpred[..., QUANTILES.index(0.5)]
    mases = []
    for i, c in enumerate(contexts):
        se = seasonal_error(c, season)
        m = mask[i]
        if not m.any() or not np.isfinite(se) or se == 0:
            continue
        mases.append(np.abs(y0[i][m] - med[i][m]).mean() / se)
    return {"crps": float(np.mean(wql)), "mase": float(np.mean(mases)) if mases else np.nan}


# ------------------------------------------------------------------------------------------------
# Predictors: fn(contexts: list[np.ndarray], H: int) -> (N, H, 9)
# ------------------------------------------------------------------------------------------------


def make_telos(device: str):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("magilabsco/Telos-Small", trust_remote_code=True).to(device).eval()
    torch.set_num_threads(int(os.environ.get("EVAL_THREADS", os.cpu_count() or 8)))

    def fn(contexts, H):
        out = model.predict_series_batch(contexts, horizon=H, quantile_levels=QUANTILES, batch_size=64)
        return out[:, :, 1:]

    return fn


def make_alma(ckpt: str, device: str, context_len: int, symmetric: bool):
    from forecast import AlmaForecaster

    f = AlmaForecaster.load(ckpt, device=device, context_len=context_len, symmetric=symmetric)
    return lambda contexts, H: f.predict(contexts, H, QUANTILES)


# ------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["telos", "alma", "snaive"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--no-symmetric", action="store_true")
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--tasks", default="all", help="'all', or comma list of indices / 'name:term'")
    ap.add_argument("--cache", default="data/gift_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = TASKS
    if args.tasks != "all":
        sel = args.tasks.split(",")
        tasks = [t for i, t in enumerate(TASKS) if str(i) in sel or f"{t[0]}:{t[1]}" in sel]

    if args.model == "telos":
        fn = make_telos(args.device)
    elif args.model == "alma":
        fn = make_alma(args.ckpt, args.device, args.context_len, not args.no_symmetric)
    else:
        fn = None

    results = []
    t0 = time.time()
    for i, (name, term) in enumerate(tasks):
        task = load_task(name, term, args.max_windows, Path(args.cache))
        ctx, lab, H, m = task["contexts"], task["labels"], task["H"], task["season"]
        sn = seasonal_naive(ctx, H, m)
        sn_s = score(ctx, lab, np.repeat(sn[..., None], len(QUANTILES), -1), m)
        ts = time.time()
        q = np.repeat(sn[..., None], len(QUANTILES), -1) if fn is None else np.asarray(fn(ctx, H), dtype=np.float64)
        q = np.sort(q, axis=-1)
        s = score(ctx, lab, q, m)
        r = {"task": f"{name}/{term}", "H": H, "season": m, "windows": len(ctx), "total_windows": task["total_windows"],
             "crps": s["crps"], "mase": s["mase"], "sn_crps": sn_s["crps"], "sn_mase": sn_s["mase"],
             "rel_crps": s["crps"] / sn_s["crps"], "rel_mase": s["mase"] / sn_s["mase"], "secs": time.time() - ts}
        results.append(r)
        print(f"[{i + 1:3d}/{len(tasks)}] {r['task']:42s} H={H:4d} n={len(ctx):4d} relCRPS={r['rel_crps']:.3f} "
              f"relMASE={r['rel_mase']:.3f} ({r['secs']:.1f}s)", flush=True)

    rc = np.array([r["rel_crps"] for r in results])
    rm = np.array([r["rel_mase"] for r in results])
    ok_c, ok_m = np.isfinite(rc) & (rc > 0), np.isfinite(rm) & (rm > 0)
    summary = {
        "model": args.model, "ckpt": args.ckpt, "tasks": len(results), "max_windows": args.max_windows,
        "gmean_rel_crps": float(np.exp(np.log(rc[ok_c]).mean())),
        "gmean_rel_mase": float(np.exp(np.log(rm[ok_m]).mean())),
        "elapsed_s": time.time() - t0,
    }
    summary["crps_skill_pct"] = 100 * (1 - summary["gmean_rel_crps"])
    summary["mase_skill_pct"] = 100 * (1 - summary["gmean_rel_mase"])
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "per_task": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
