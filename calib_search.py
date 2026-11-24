"""Post-hoc spread calibration q' = med + s_b * (q - med), s_b per frequency bucket, honest 2-fold
(s chosen on one half of each task's dev2 windows, scored on the other). Usage: python calib_search.py D v3_10k+D"""
import sys
import numpy as np
from data import freq_bucket
from dev_eval import windows
from ens_search import wql, SEALED
import json
fc = json.load(open(SEALED)); S = {k.replace("/", "_", k.count("/") - 1).replace("/", "__") for k in fc["per_task_rel_crps"]["telos"]}
SG = np.round(np.arange(0.6, 1.21, 0.05), 2)
specs = sys.argv[1:]
P = {n: np.load(f"results/preds/{n}.npz") for s in specs for n in s.split("+")}
T = np.load("results/preds/telos.npz")
tasks = []
for task, C, L, H, m, f in windows(256, 1):
    L = np.asarray(L, float); tasks.append((task, freq_bucket(f), L, T[task].astype(float), [np.arange(0, len(L), 2), np.arange(1, len(L), 2)]))
for spec in specs:
    Q = {t[0]: np.mean([P[n][t[0]].astype(float) for n in spec.split("+")], 0) for t in tasks}
    # lr[task][h][si] = log ratio vs telos on half h with scale SG[si]
    lr = {}
    for task, b, L, Tq, hv in tasks:
        q = Q[task]; med = q[..., 4:5]
        lr[task] = np.array([[np.log(wql(L[i], med[i] + s * (q[i] - med[i])) / wql(L[i], Tq[i])) for s in SG] for i in hv])
    B = sorted({t[1] for t in tasks})
    def pick(h, b):
        return int(np.argmin(np.mean([lr[t[0]][h] for t in tasks if t[1] == b], 0)))
    out = {}
    for task, b, *_ in tasks:
        out[task] = np.mean([lr[task][h][pick(1 - h, b)] for h in (0, 1)])
    base = {t[0]: lr[t[0]][:, list(SG).index(1.0)].mean() for t in tasks}
    g = lambda d, kk: np.exp(np.mean([d[k] for k in kk]))
    allk = [t[0] for t in tasks]; sk = [k for k in allk if k in S]
    print(f"{spec:20s} base all {g(base, allk):.4f} sealed {g(base, sk):.4f} | calibrated cv all {g(out, allk):.4f} sealed {g(out, sk):.4f}")
    print("   s per bucket (half0/half1):", {b: (float(SG[pick(0, b)]), float(SG[pick(1, b)])) for b in B})
