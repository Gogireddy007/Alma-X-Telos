"""Rank dev2 result files against a reference (default Telos n=256): geometric-mean CRPS ratio on all tasks and on
the 28 sealed-task names, paired t over tasks, win rate.  --best prints only the path of the best file (by 'all')."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEALED = "validation_runs/AlmaV3_fc_20261116T2349Z/metrics/fc.json"


def load(f):
    return {r["task"]: r["rel_crps"] for r in json.load(open(f))["per_task"]}


def stats(x, ref, keys=None):
    ks = [k for k in ref if k in x and x[k] > 0 and ref[k] > 0 and (keys is None or k in keys)]
    d = np.array([np.log(x[k] / ref[k]) for k in ks])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else float("nan")
    return float(np.exp(d.mean())), float(t), float(np.mean(d < 0)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default="results/dev2/telos_n256.json")
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args()
    ref = load(a.ref)
    sealed = None
    if os.path.exists(SEALED):
        fc = json.load(open(SEALED))
        sealed = {k.replace("/", "_", k.count("/") - 1).replace("/", "__") for k in fc["per_task_rel_crps"]["telos"]}
    rows = []
    for f in a.files:
        if not os.path.exists(f):
            continue
        x = load(f)
        r, t, w, n = stats(x, ref)
        rs = stats(x, ref, sealed)[0] if sealed else float("nan")
        rows.append((r, rs, t, w, n, f))
    rows.sort()
    if a.best:
        print(rows[0][-1] if rows else "")
        return
    print(f"{'all':>7} {'sealed28':>8} {'t':>6} {'win':>5} {'n':>3}  file   (ref {a.ref})")
    for r, rs, t, w, n, f in rows:
        print(f"{r:7.4f} {rs:8.4f} {t:6.2f} {w:5.2f} {n:3d}  {f}")


if __name__ == "__main__":
    main()
