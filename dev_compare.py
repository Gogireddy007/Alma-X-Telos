"""Compare dev2 results: overall, on the 28 sealed-task names, and by frequency bucket.
python dev_compare.py results/dev2/A.json results/dev2/B.json   (ratio A/B)"""
import json, math, sys
from data import freq_bucket

def load(p): return {d["task"]: d for d in json.load(open(p))["per_task"]}
g = lambda x: math.exp(sum(map(math.log, x)) / len(x))
fc = json.load(open("validation_runs/AlmaV3_fc_20261116T2349Z/metrics/fc.json"))["per_task_rel_crps"]["telos"]
SEALED = [k.replace("/", "_", k.count("/") - 1).replace("/", "__") for k in fc]

def compare(a, b, metric="rel_crps"):
    A, B = load(a), load(b)
    ks = [k for k in A if k in B and A[k][metric] > 0 and B[k][metric] > 0]
    out = {"all": (g([A[k][metric] / B[k][metric] for k in ks]), len(ks)),
           "sealed28": (g([A[k][metric] / B[k][metric] for k in ks if k in SEALED]), sum(k in SEALED for k in ks))}
    for k in ks:
        out.setdefault("f:" + freq_bucket(A[k]["freq"]), []).append(A[k][metric] / B[k][metric])
    return {k: (v if isinstance(v, tuple) else (g(v), len(v))) for k, v in out.items()}

if __name__ == "__main__":
    for k, (r, n) in compare(sys.argv[1], sys.argv[2]).items():
        print(f"{k:14s} {r:.4f}  (n={n})")
