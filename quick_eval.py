"""Quick proxy: first N cached windows of every GIFT-Eval task, gmean rel CRPS/MASE vs Seasonal Naive.

  python quick_eval.py --model telos
  python quick_eval.py --model alma --ckpt runs/meridian_s/ema_2000.pt [--context-lens 512,1024,2048]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from eval_gift import QUANTILES, score, seasonal_naive


def run(fn, n_win: int, cache: str = "data/gift_cache") -> dict:
    per = []
    for p in sorted(glob.glob(f"{cache}/*__1000.npz")):
        z = np.load(p, allow_pickle=True)
        ctx, lab, H, m = list(z["contexts"])[:n_win], z["labels"][:n_win], int(z["H"]), int(z["season"])
        sn = np.repeat(seasonal_naive(ctx, H, m)[..., None], 9, -1)
        s_sn = score(ctx, lab, sn, m)
        q = np.sort(np.asarray(fn(ctx, H), dtype=np.float64), -1)
        s = score(ctx, lab, q, m)
        per.append({"task": os.path.basename(p).split("__1000")[0], "rel_crps": s["crps"] / s_sn["crps"],
                    "rel_mase": s["mase"] / s_sn["mase"]})
    rc = np.array([r["rel_crps"] for r in per])
    rm = np.array([r["rel_mase"] for r in per])
    g = lambda a: float(np.exp(np.log(a[np.isfinite(a) & (a > 0)]).mean()))
    return {"tasks": len(per), "rel_crps": g(rc), "rel_mase": g(rm), "per_task": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["telos", "alma"])
    ap.add_argument("--ckpt")
    ap.add_argument("--n-win", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--context-lens", default="2048")
    ap.add_argument("--no-symmetric", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.model == "telos":
        from eval_gift import make_telos

        fn = make_telos(a.device)
    else:
        from forecast import AlmaForecaster

        fs = [AlmaForecaster.load(a.ckpt, device=a.device, context_len=int(c), symmetric=not a.no_symmetric)
              for c in a.context_lens.split(",")]
        fn = lambda ctx, H: np.mean([f.predict(ctx, H, QUANTILES) for f in fs], 0)
    r = run(fn, a.n_win)
    print(json.dumps({k: v for k, v in r.items() if k != "per_task"}))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
