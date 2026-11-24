"""Offline ensemble search over saved dev2 predictions (results/preds/*.npz), with honest 2-fold selection.

Every task's 256 dev2 windows are split into two halves (even / odd position). Members and weights are chosen on
one half by greedy forward selection with replacement (Caruana et al.), then scored on the OTHER half, and vice
versa. The reported "cv" number is therefore never scored on windows used to pick it. The score is the geometric
mean over tasks of CRPS(ensemble)/CRPS(Telos) on the same windows. Telos is only the reference, never a member.
Ensemble = average of sorted quantiles (Vincentization).

  python ens_search.py                      # global ensemble
  python ens_search.py --route              # also a separate ensemble per frequency bucket (routing)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from data import freq_bucket
from dev_eval import windows
from eval_gift import QUANTILES

SEALED = "validation_runs/AlmaV3_fc_20261116T2349Z/metrics/fc.json"


def wql(y, q):
    """CRPS proxy exactly as eval_gift.score (mean weighted quantile loss) for one set of windows."""
    mask = np.isfinite(y)
    y0 = np.where(mask, y, 0.0)
    d = max(np.abs(y0).sum(), 1e-12)
    tot = 0.0
    for k, a in enumerate(QUANTILES):
        e = y0 - q[..., k]
        tot += 2 * np.where(mask, np.maximum(a * e, (a - 1) * e), 0.0).sum() / d
    return tot / len(QUANTILES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/preds")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--exclude", default="")
    a = ap.parse_args()
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{a.dir}/*.npz"))
    names = [n for n in names if n != "telos" and n not in a.exclude.split(",")]
    P = {n: np.load(f"{a.dir}/{n}.npz") for n in names + ["telos"]}
    sealed = set()
    if os.path.exists(SEALED):
        fc = json.load(open(SEALED))
        sealed = {k.replace("/", "_", k.count("/") - 1).replace("/", "__") for k in fc["per_task_rel_crps"]["telos"]}
    tasks = []
    for task, C, L, H, m, freq in windows(256, 1):
        if not all(task in P[n].files for n in names + ["telos"]):
            continue
        L = np.asarray(L, dtype=np.float64)
        halves = [np.arange(0, len(L), 2), np.arange(1, len(L), 2)]
        Q = {n: P[n][task].astype(np.float64) for n in names}
        T = P["telos"][task].astype(np.float64)
        tasks.append({"task": task, "b": freq_bucket(freq), "s": task in sealed, "L": L, "Q": Q, "T": T, "h": halves})
    print(f"{len(tasks)} tasks, members: {', '.join(names)}")

    def logr(t, combo, h):
        """log CRPS ratio vs Telos of the equal-weight average of combo (a list, repeats = weight) on half h."""
        idx = t["h"][h] if h is not None else slice(None)
        q = np.mean([t["Q"][n][idx] for n in combo], 0)
        return np.log(wql(t["L"][idx], q) / max(wql(t["L"][idx], t["T"][idx]), 1e-12))

    # cache per (task, member, half) the member's quantiles is cheap; greedy needs sums, so cache running sums
    def greedy(ts, h):
        combo, best = [], None
        for _ in range(a.steps):
            cand = {}
            for n in names:
                cand[n] = np.mean([logr(t, combo + [n], h) for t in ts])
            n, v = min(cand.items(), key=lambda kv: kv[1])
            if best is not None and v >= best - 1e-5:
                break
            combo, best = combo + [n], v
        return combo

    def report(label, pick):
        """pick(h) -> {bucket or '*': combo} chosen on half h; score each half with the other half's choice."""
        ch = [pick(0), pick(1)]
        d = {t["task"]: np.mean([logr(t, ch[1 - h].get(t["b"], ch[1 - h]["*"]), h) for h in (0, 1)]) for t in tasks}
        v = np.array(list(d.values()))
        s = np.array([d[t["task"]] for t in tasks if t["s"]])
        print(f"{label:28s} cv all {np.exp(v.mean()):.4f}  sealed28 {np.exp(s.mean()):.4f}  "
              f"win {np.mean(v < 0):.2f}  t {v.mean() / (v.std(ddof=1) / np.sqrt(len(v))):+.2f}")
        for h in (0, 1):
            print(f"   picked on half {h}: " + "; ".join(f"{k}: {dict(Counter(c))}" for k, c in ch[h].items()))
        return d

    for n in names:  # singles, for reference (no selection involved)
        v = np.array([logr(t, [n], None) for t in tasks])
        s = np.array([logr(t, [n], None) for t in tasks if t["s"]])
        print(f"  single {n:20s} all {np.exp(v.mean()):.4f}  sealed28 {np.exp(s.mean()):.4f}")
    report("greedy global", lambda h: {"*": greedy(tasks, h)})
    if a.route:
        def per_bucket(h):
            out = {"*": greedy(tasks, h)}
            for b in sorted({t["b"] for t in tasks}):
                ts = [t for t in tasks if t["b"] == b]
                if len(ts) >= 3:
                    out[b] = greedy(ts, h)
            return out
        report("greedy routed by bucket", per_bucket)


if __name__ == "__main__":
    main()
