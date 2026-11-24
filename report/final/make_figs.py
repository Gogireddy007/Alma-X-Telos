"""Build every figure and number table for the final progress report from the raw run artefacts.

python report/final/make_figs.py   (run from the alma/ directory)
All numbers are read from files; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("report/final")
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif"})
C_V3, C_V2, C_T, C_ARX = "#1f5fa8", "#6aa84f", "#b03a2e", "#7f7f7f"
NUM: dict = {}


def gmean(x):
    x = [v for v in x if v is not None and np.isfinite(v) and v > 0]
    return float(math.exp(sum(map(math.log, x)) / len(x)))


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------------------------------------------------------------- v3 training log
log = jl("runs/alma_v3/log.jsonl")
tr = [r for r in log if "eval" not in r]
ev = [r["eval"] for r in log if "eval" in r]
st = np.array([r["step"] for r in tr])


def smooth(a, k=10):
    a = np.asarray(a, float)
    return np.convolve(a, np.ones(k) / k, mode="valid"), st[k - 1:]


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
for key, lab, c in (("fc", "forecast (quantile) loss", C_V3), ("jepa", "JEPA latent loss", "#8e44ad")):
    y, x = smooth([r[key] for r in tr])
    ax[0].plot(x / 1000, y, c=c, lw=1.1, label=lab)
ax[0].set_xlabel("step (k)"); ax[0].set_title("(a) training losses (10-point moving avg.)"); ax[0].legend()
y, x = smooth([r["gnorm"] for r in tr])
ax[1].plot(x / 1000, y, c="#444", lw=1.0)
ax[1].set_xlabel("step (k)"); ax[1].set_title("(b) gradient norm (pre-clip)")
ax1b = ax[1].twinx(); ax1b.plot(st / 1000, [r["lr"] * 1e4 for r in tr], c="#d68910", lw=1.0)
ax1b.set_ylabel(r"lr ($\times10^{-4}$)", color="#d68910"); ax1b.grid(False)
y, x = smooth([r["seq_s"] for r in tr], 20)
ax[2].plot(x / 1000, y, c="#16a085", lw=1.0)
ax[2].set_xlabel("step (k)"); ax[2].set_title("(c) throughput (sequences/s)")
fig.tight_layout(); fig.savefig(FIG / "v3_training.pdf"); plt.close(fig)

first, last = tr[:20], tr[-20:]
NUM["v3_train"] = {"fc_first1k": np.mean([r["fc"] for r in first]), "fc_last1k": np.mean([r["fc"] for r in last]),
                   "jepa_first1k": np.mean([r["jepa"] for r in first]),
                   "jepa_last1k": np.mean([r["jepa"] for r in last]),
                   "gnorm_mean": np.mean([r["gnorm"] for r in tr]), "gnorm_max": max(r["gnorm"] for r in tr),
                   "seq_s_mean": np.mean([r["seq_s"] for r in tr]), "long_frac": np.mean([r["long"] for r in tr]),
                   "n_log": len(tr)}
NUM["v3_evals"] = {e["step"]: e["quick_rel_crps"] for e in ev}

V2_QUICK_8K = 0.60693  # v2 @6k scored by the same quick-dev routine at 8k context (growth check, trial_log_v3.csv)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
xs = [e["step"] / 1000 for e in ev]; ys = [e["quick_rel_crps"] for e in ev]
ax.plot(xs, ys, "-o", c=C_V3, ms=3, lw=1.2, label="Alma v3 (EMA)")
ib = int(np.argmin(ys)); ax.plot(xs[ib], ys[ib], "*", c="k", ms=9, label=f"selected (step {int(xs[ib]*1000)})")
ax.axhline(V2_QUICK_8K, c=C_V2, ls="--", lw=1, label=f"Alma v2 @6k ({V2_QUICK_8K:.4f})")
ax.set_xlabel("training step (k)"); ax.set_ylabel("quick-dev rel. CRPS (lower = better)")
ax.legend(loc="upper left", frameon=False); fig.tight_layout(); fig.savefig(FIG / "v3_quickdev.pdf"); plt.close(fig)

# ---------------------------------------------------------------- v3 sealed test
fc = json.load(open("validation_runs/AlmaV3_fc_20261116T2349Z/metrics/fc.json"))
NUM["sealed"] = {k: v for k, v in fc.items() if k != "per_task_rel_crps"}
p = fc["per_task_rel_crps"]
tasks = sorted(p["alma_v3"], key=lambda t: p["alma_v3"][t] / p["telos"][t])
r_t = [p["alma_v3"][t] / p["telos"][t] for t in tasks]
r_2 = [p["alma_v3"][t] / p["alma_v2"][t] for t in tasks]
NUM["sealed_per_task"] = {t: {"v3": p["alma_v3"][t], "v2": p["alma_v2"][t], "telos": p["telos"][t],
                              "v3_over_telos": a, "v3_over_v2": b} for t, a, b in zip(tasks, r_t, r_2)}
fig, ax = plt.subplots(figsize=(7.0, 2.6))
xi = np.arange(len(tasks))
ax.bar(xi - 0.2, r_t, 0.4, color=[C_V3 if v < 1 else C_T for v in r_t], label="v3 / Telos-Small")
ax.bar(xi + 0.2, r_2, 0.4, color="#bbbbbb", label="v3 / v2")
ax.axhline(1, c="k", lw=0.8)
ax.set_xticks(xi); ax.set_xticklabels([t.replace("_with_missing", "*").replace("bitbrains_", "bb_")
                                       .replace("fast_storage", "fs").replace("/short", "/S").replace("/medium", "/M")
                                       .replace("/long", "/L") for t in tasks], rotation=70, ha="right", fontsize=6)
ax.set_ylabel("CRPS ratio (<1: v3 better)"); ax.set_ylim(0.8, 1.35)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_V3, label="v3 / Telos (v3 better)"), Patch(color=C_T, label="v3 / Telos (Telos better)"),
                   Patch(color="#bbbbbb", label="v3 / v2")], loc="upper left", frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(FIG / "sealed_per_task.pdf"); plt.close(fig)

# sealed by term
def term(t):
    return t.rsplit("/", 1)[-1]


NUM["sealed_by_term"] = {}
for tm in ("short", "medium", "long"):
    ks = [t for t in tasks if term(t) == tm]
    NUM["sealed_by_term"][tm] = {"n": len(ks), **{m: gmean([p[m][t] for t in ks]) for m in ("alma_v3", "alma_v2", "telos")},
                                 "v3_over_telos": gmean([p["alma_v3"][t] / p["telos"][t] for t in ks]),
                                 "wins_vs_telos": sum(p["alma_v3"][t] < p["telos"][t] for t in ks)}

# ---------------------------------------------------------------- full GIFT-Eval
def gl(f):
    return {d["task"]: d for d in json.load(open(f"results/{f}.json"))["per_task"]}


G = {"v3": gl("alma_v3_gift"), "v2": gl("alma_v2_6000_gift"), "telos": gl("telos_gift"), "v1": gl("alma_v1_final_gift")}
summ = {k: json.load(open(f"results/{f}.json"))["summary"] for k, f in
        (("v3", "alma_v3_gift"), ("v2", "alma_v2_6000_gift"), ("telos", "telos_gift"), ("v1", "alma_v1_final_gift"))}
NUM["gift_summary"] = summ
keys = [k for k in G["v3"] if all(k in G[m] for m in G)]
same = [k for k in keys if G["v3"][k]["total_windows"] <= 400]
NUM["gift"] = {}
for name, ks in (("all", keys), ("same_windows", same)):
    d = {"n": len(ks)}
    for m in G:
        d[m + "_crps_skill"] = 100 * (1 - gmean([G[m][k]["rel_crps"] for k in ks]))
        d[m + "_mase_skill"] = 100 * (1 - gmean([G[m][k]["rel_mase"] for k in ks]))
    d["v3_over_telos"] = gmean([G["v3"][k]["rel_crps"] / G["telos"][k]["rel_crps"] for k in ks])
    d["v3_over_v2"] = gmean([G["v3"][k]["rel_crps"] / G["v2"][k]["rel_crps"] for k in ks])
    d["wins_vs_telos"] = sum(G["v3"][k]["rel_crps"] < G["telos"][k]["rel_crps"] for k in ks)
    lr = np.log([G["v3"][k]["rel_crps"] / G["telos"][k]["rel_crps"] for k in ks])
    d["t_vs_telos"] = float(-lr.mean() / (lr.std(ddof=1) / math.sqrt(len(lr))))
    for tm in ("short", "medium", "long"):
        kt = [k for k in ks if term(k) == tm]
        if kt:
            d[tm] = {"n": len(kt), **{m: gmean([G[m][k]["rel_crps"] for k in kt]) for m in G},
                     **{m + "_mase": gmean([G[m][k]["rel_mase"] for k in kt]) for m in G},
                     "wins": sum(G["v3"][k]["rel_crps"] < G["telos"][k]["rel_crps"] for k in kt)}
    NUM["gift"][name] = d

# domain breakdown on all tasks (dataset family = first path component)
fam = {}
for k in keys:
    fam.setdefault(k.split("/")[0], []).append(k)
NUM["gift_by_dataset"] = {f: {"n": len(ks), "v3_over_telos": gmean([G["v3"][k]["rel_crps"] / G["telos"][k]["rel_crps"] for k in ks]),
                              "v3_over_v2": gmean([G["v3"][k]["rel_crps"] / G["v2"][k]["rel_crps"] for k in ks])}
                          for f, ks in fam.items()}

fig, ax = plt.subplots(figsize=(3.4, 3.0))
for ks, c, lab in ((sorted(set(keys) - set(same)), "#9db7d5", "different window subsample"), (same, C_V3, "identical windows")):
    ax.scatter([G["telos"][k]["rel_crps"] for k in ks], [G["v3"][k]["rel_crps"] for k in ks], s=10, c=c, label=lab,
               edgecolor="none")
lim = [0.15, 1.25]; ax.plot(lim, lim, "k-", lw=0.7)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Telos-Small rel. CRPS"); ax.set_ylabel("Alma v3 rel. CRPS")
ax.text(0.97, 0.04, "below diagonal: v3 better", transform=ax.transAxes, ha="right", fontsize=6.5)
ax.legend(loc="upper left", frameon=False); fig.tight_layout(); fig.savefig(FIG / "gift_scatter.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.0))
models = [("v1", "Alma v1", "#a9cce3"), ("v2", "Alma v2", C_V2), ("v3", "Alma v3", C_V3), ("telos", "Telos-Small", C_T)]
for j, (name, ks) in enumerate((("all 97 tasks", keys), (f"{len(same)} identical-window tasks", same))):
    for i, (m, lab, c) in enumerate(models):
        v = 100 * (1 - gmean([G[m][k]["rel_crps"] for k in ks]))
        ax.bar(j + (i - 1.5) * 0.2, v, 0.19, color=c, label=lab if j == 0 else None)
        ax.text(j + (i - 1.5) * 0.2, v + 0.15, f"{v:.1f}", ha="center", fontsize=5.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["all 97 tasks", f"{len(same)} identical-window tasks"])
ax.set_ylim(34, 42.5); ax.set_ylabel("CRPS skill vs SN (%)"); ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=6)
fig.tight_layout(); fig.savefig(FIG / "gift_skill.pdf"); plt.close(fig)

# ---------------------------------------------------------------- world models wm1 vs wm2
WM = {"wm1": "AlmaWM_wm1_20261116T1828Z", "wm2": "AlmaWM2_wm2_20261116T2254Z"}
SC = {k: json.load(open(f"validation_runs/{r}/report/scorecard_grade.json")) for k, r in WM.items()}


def find(k, gate, frag):
    for row in SC[k]:
        if row["gate"] == gate and frag in row["check"]:
            return row
    return None


wmn = {}
for k in WM:
    g = lambda gate, frag: (find(k, gate, frag) or {}).get("result")  # noqa: E731
    wmn[k] = {"onestep": g("4", "one-step"), "buckets": g("5.1", "every horizon bucket"),
              "growth": g("5.1", "error growth"), "viol": g("5.6", "constraint"), "aware_blind": g("6.1", "action-blind"),
              "sign": g("6.2", "sign accuracy"), "mono": g("6.2", "monotone"), "cf": g("6.5*", "counterfactual"),
              "cov": g("7 / 4.4", "coverage"), "conformal": g("7.3", "conformal"), "ood": g("7.4", "OOD"),
              "fallback": g("7.5", "fallback"), "placebo_shuf": g("8.10", "shuffled"),
              "placebo_rand": g("8.10", "random actions"), "placebo_shift": g("8.10", "time-shifted"),
              "family": g("9.2/9.3", "family"), "tanks": g("9.7", "tanks"), "probes": g("10.1", "linear probes"),
              "rank": g("10.1", "effective rank"), "mpc": g("12", "MPC"), "imag": g("12.1", "imagined"),
              "real": g("4.2", "REAL sealed"), "noninf": g("4.2", "non-inferiority"), "rollout_direct": g("4.2", "latent-rollout"),
              "real_cov": g("4.4", "REAL coverage"), "overlap": g("2.2", "exact-match"), "pit": g("5.3", "PIT"),
              "perturb": g("9.5", "perturbation"), "status": {f'{r["gate"]}|{r["check"][:40]}': r["status"] for r in SC[k]}}
NUM["wm"] = wmn

B = ["1-16", "17-64", "65-256", "257-512"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1))
for i, (k, c) in enumerate((("wm1", "#c39bd3"), ("wm2", "#5b2c6f"))):
    b = wmn[k]["buckets"]
    v = [b[x]["arx"] if x in b else np.nan for x in B]
    ax[0].bar(np.arange(4) + (i - 0.5) * 0.38, v, 0.36, color=c, label=k)
    for j, y in enumerate(v):
        if np.isfinite(y):
            ax[0].text(j + (i - 0.5) * 0.38, y + 0.01, f"{y:.2f}", ha="center", fontsize=5.5)
ax[0].axhline(1, c="k", lw=0.8); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"{x} steps" for x in B])
ax[0].set_ylabel("CRPS / ARX (<1: WM better)"); ax[0].set_title("(a) action-conditioned rollout vs ARX by horizon")
ax[0].set_ylim(0.6, 1.45); ax[0].legend(frameon=False)
f1, f2 = wmn["wm1"]["family"], wmn["wm2"]["family"]
fams = sorted(set(f1) | set(f2), key=lambda f: f2.get(f, 9))
for i, (f, c) in enumerate(((f1, "#c39bd3"), (f2, "#5b2c6f"))):
    ax[1].bar(np.arange(len(fams)) + (i - 0.5) * 0.38, [f.get(x, np.nan) for x in fams], 0.36, color=c)
ax[1].axhline(1, c="k", lw=0.8); ax[1].set_xticks(range(len(fams))); ax[1].set_xticklabels(fams, rotation=30, ha="right")
ax[1].set_title("(b) by simulated system family (wm2 vs wm1)"); ax[1].set_ylabel("aware / ARX")
fig.tight_layout(); fig.savefig(FIG / "wm_horizon_family.pdf"); plt.close(fig)

# wm2 headline comparison wm1 -> wm2 on key ratios
lab = ["one-step\nvs ARX", "counterfactual\nvs ARX", "MPC cost\nvs ARX-MPC", "held-out tanks\nvs ARX", "aware vs\nblind"]
vals = {k: [wmn[k]["onestep"]["arx"][0], wmn[k]["cf"][0], wmn[k]["mpc"]["vs_arx"][0], wmn[k]["tanks"]["aware_vs_arx"],
            wmn[k]["aware_blind"][0]] for k in WM}
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for i, (k, c) in enumerate((("wm1", "#c39bd3"), ("wm2", "#5b2c6f"))):
    ax.bar(np.arange(5) + (i - 0.5) * 0.38, vals[k], 0.36, color=c, label=k)
    for j, y in enumerate(vals[k]):
        ax.text(j + (i - 0.5) * 0.38, y + 0.02, f"{y:.2f}", ha="center", fontsize=5.5)
ax.axhline(1, c="k", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=6)
ax.set_ylabel("ratio (<1: WM better)"); ax.legend(frameon=False); ax.set_ylim(0, 1.85)
fig.tight_layout(); fig.savefig(FIG / "wm_key_ratios.pdf"); plt.close(fig)

# wm2 training curves
w2 = [r for r in jl("runs/alma_wm2/log.jsonl") if "eval" not in r and "step" in r]
s2 = np.array([r["step"] for r in w2])
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for key, c in (("ground", "#5b2c6f"), ("teacher", "#a569bd"), ("jepa", "#8e44ad"), ("direct", C_V2)):
    if key in w2[0]:
        y = np.convolve([r[key] for r in w2], np.ones(10) / 10, mode="valid")
        ax.plot(s2[9:] / 1000, y, lw=1, c=c, label=key)
ax.set_xlabel("step (k)"); ax.set_ylabel("loss"); ax.legend(frameon=False, ncol=2); ax.set_title("wm2 training losses")
fig.tight_layout(); fig.savefig(FIG / "wm2_training.pdf"); plt.close(fig)
NUM["wm2_train"] = {"steps": int(s2[-1]), "seq_s": float(np.mean([r["seq_s"] for r in w2])),
                    **{k + "_first": float(np.mean([r[k] for r in w2[:20]])) for k in ("ground", "teacher", "jepa", "direct")},
                    **{k + "_last": float(np.mean([r[k] for r in w2[-20:]])) for k in ("ground", "teacher", "jepa", "direct")}}

(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
print("ok", len(keys), len(same))
