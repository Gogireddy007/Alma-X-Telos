"""dev2: development score on RANDOM windows of the seed-0 dev sample (the __1000 cache).

quick-dev (first 64 windows per task) is biased: the cache is sorted, so the first windows come from a
few series. dev2 draws n random windows per task with a fixed seed. The seed-0 sample is excluded from
every sealed set (seeds 777/778/779), so dev2 never touches sealed windows.

  python dev_eval.py --model telos --out results/dev2/telos.json
  python dev_eval.py --model alma --ckpt runs/alma_v3/ema_10000.pt --ctx 8192 --out results/dev2/v3_10k.json
  python dev_eval.py --model soup --ckpts runs/alma_v3/ema_8000.pt,runs/alma_v3/ema_10000.pt --ctx 8192 --out ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import zlib

import numpy as np
import torch

from eval_gift import QUANTILES, score, seasonal_naive

def windows(n: int, seed: int, cache: str = "data/gift_cache"):
    for p in sorted(glob.glob(f"{cache}/*__1000.npz")):
        z = np.load(p, allow_pickle=True)
        C, L = list(z["contexts"]), z["labels"]
        rng = np.random.default_rng([seed, zlib.crc32(os.path.basename(p).encode())])
        idx = np.sort(rng.choice(len(C), size=min(n, len(C)), replace=False))
        yield (os.path.basename(p).split("__1000")[0], [C[i] for i in idx], L[idx], int(z["H"]), int(z["season"]),
               str(z["freq"]))


def soup(ckpts: list[str], out: str) -> str:
    sds = [torch.load(c, map_location="cpu", weights_only=False) for c in ckpts]
    base = sds[0]
    key = "ema" if "ema" in base else "model"
    avg = {k: sum(s[key][k].float() for s in sds) / len(sds) if base[key][k].is_floating_point() else base[key][k]
           for k in base[key]}
    torch.save({**{k: v for k, v in base.items() if k not in ("ema", "model", "jepa", "opt")}, key: avg,
                "soup_of": ckpts}, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["telos", "alma", "soup", "ens"])
    ap.add_argument("--ckpt")
    ap.add_argument("--ckpts")
    ap.add_argument("--members", help="ens: comma list of ckpt@ctx; quantiles are averaged (Vincentization)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.model == "telos":
        from eval_gift import make_telos

        fn = make_telos("cpu")
    elif a.model == "ens":
        from forecast import AlmaForecaster

        fs = []
        for spec in a.members.split(","):
            ck, cx = spec.rsplit("@", 1)
            fs.append(AlmaForecaster.load(ck, device=a.device, context_len=int(cx), symmetric=True))
        fn = lambda ctx, H: np.mean([np.sort(np.asarray(f.predict(ctx, H, QUANTILES), dtype=np.float64), -1)  # noqa: E731
                                     for f in fs], 0)
    else:
        from forecast import AlmaForecaster

        ck = a.ckpt
        if a.model == "soup":
            ck = soup(a.ckpts.split(","), os.path.splitext(a.out)[0] + "_soup.pt")
        f = AlmaForecaster.load(ck, device=a.device, context_len=a.ctx, symmetric=True)
        fn = lambda ctx, H: f.predict(ctx, H, QUANTILES)  # noqa: E731
    per, t0 = [], time.time()
    for task, C, L, H, m, freq in windows(a.n, a.seed):
        sn = np.repeat(seasonal_naive(C, H, m)[..., None], 9, -1)
        s_sn = score(C, L, sn, m)
        q = np.sort(np.asarray(fn(C, H), dtype=np.float64), -1)
        s = score(C, L, q, m)
        per.append({"task": task, "freq": freq, "rel_crps": s["crps"] / s_sn["crps"], "rel_mase": s["mase"] / s_sn["mase"]})
        print(f"{task:45s} {per[-1]['rel_crps']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if torch.backends.mps.is_available():  # MPS caches every batch shape; without this memory grows to swap
            import gc

            gc.collect()
            torch.mps.empty_cache()
    g = lambda v: float(np.exp(np.log([x for x in v if np.isfinite(x) and x > 0]).mean()))  # noqa: E731
    res = {"model": a.model, "ckpt": a.ckpt or a.ckpts or a.members, "ctx": a.ctx, "n": a.n, "seed": a.seed,
           "rel_crps": g([r["rel_crps"] for r in per]), "rel_mase": g([r["rel_mase"] for r in per]),
           "secs": time.time() - t0, "per_task": per}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}))


if __name__ == "__main__":
    main()
