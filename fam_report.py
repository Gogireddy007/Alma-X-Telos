"""Per-family CRPS ratio vs Telos (dev2) for every results/preds/*.json: shows where each model wins/loses."""
import glob, json, os, re
import numpy as np
from dev_rank import SEALED
fc = json.load(open(SEALED))
S = {k.replace("/", "_", k.count("/") - 1).replace("/", "__") for k in fc["per_task_rel_crps"]["telos"]}
L = lambda f: {r["task"]: r["rel_crps"] for r in json.load(open(f))["per_task"]}
T = L("results/preds/telos.json")
fam = lambda k: re.split(r"_(?:\d+)?(?:[A-Z]|5T|10T|15T|H|D|W|M|Q|A)", k)[0].split("__")[0]
FAMS = ["LOOP_SEATTLE", "bitbrains", "m4", "electricity", "solar", "SZ_TAXI", "M_DENSE", "ett", "jena", "hospital"]
files = sorted(f for f in glob.glob("results/preds/*.json") if not f.endswith("telos.json"))
print(f"{'model':14s} {'all':>6s} {'seal':>6s} " + " ".join(f"{x[:9]:>9s}" for x in FAMS))
for f in files:
    x = L(f); ks = [k for k in T if k in x]
    g = lambda kk: np.exp(np.mean([np.log(x[k] / T[k]) for k in kk])) if kk else float("nan")
    row = [g(ks), g([k for k in ks if k in S])] + [g([k for k in ks if k.lower().startswith(fm.lower())]) for fm in FAMS]
    print(f"{os.path.basename(f)[:-5]:14s} " + " ".join(f"{v:6.3f}" if i < 2 else f"{v:9.3f}" for i, v in enumerate(row)))
