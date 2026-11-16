"""Sealed forecaster validation: Alma v3 vs Telos-Small vs Alma v2 on fresh GIFT-Eval windows.

  VALRUN=validation_runs/<run> python validate/fc_validate.py build            # before training ends
  VALRUN=... python validate/fc_validate.py predict --ckpt runs/alma_v3/ema_X.pt  # seals outputs
  VALRUN=... python validate/fc_validate.py grade                                 # only after sealing

Metric: GIFT-Eval per-task CRPS (mean weighted quantile loss) and MASE, relative to Seasonal Naive.
Claims: paired t on per-task log CRPS ratios (t >= 3), Holm-corrected across the comparisons below.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common import RUN, holm, load_sealed, load_truth, log, manifest_add, paired_t, seal, t_to_p

COMPARATORS = ("telos", "alma_v2", "snaive")


def build() -> None:
    from build_suite import build_real

    (RUN / "harness").mkdir(parents=True, exist_ok=True)
    build_real(seed=779, exclude_seeds=(777, 778))


def predict(ckpt: str, v3_ctx: int, no_telos: bool) -> None:
    from eval_gift import make_telos, seasonal_naive
    from forecast import AlmaForecaster

    z = np.load(RUN / "harness/inputs_real.npz", allow_pickle=True)
    C, meta = list(z["contexts"]), z["meta"]
    tasks = list(dict.fromkeys(m[0] for m in meta))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    v3 = AlmaForecaster.load(ckpt, device=dev, context_len=v3_ctx, symmetric=True)
    v2 = AlmaForecaster.load("runs/alma_v2/ema_6000.pt", device=dev, context_len=2048, symmetric=True)
    tel = None if no_telos else make_telos("cpu")
    manifest_add("weights", __import__("pathlib").Path(ckpt).resolve())
    out = {k: [] for k in ("alma_v3", "alma_v2", "telos", "snaive")}
    t0 = time.time()
    for tk in tasks:
        idx = [i for i, m in enumerate(meta) if m[0] == tk]
        H, season = int(meta[idx[0]][1]), int(meta[idx[0]][2])
        ctx = [np.asarray(C[i], np.float32) for i in idx]
        out["alma_v3"].append(np.sort(v3.predict(ctx, H, [0.1 * k for k in range(1, 10)]), -1))
        out["alma_v2"].append(np.sort(v2.predict(ctx, H, [0.1 * k for k in range(1, 10)]), -1))
        out["telos"].append(np.sort(tel(ctx, H), -1) if tel else np.full((len(ctx), H, 9), np.nan))
        out["snaive"].append(np.repeat(seasonal_naive(ctx, H, season)[..., None], 9, -1))
        log(f"fc predict {tk}: {len(idx)} windows ({time.time() - t0:.0f}s)")

    def objarr(lst):
        a = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            a[i] = v
        return a

    seal("pred_fc", tasks=np.array(tasks), **{k: objarr(v) for k, v in out.items()})
    log("sealed pred_fc")


def grade() -> None:
    from eval_gift import score

    P = load_sealed("pred_fc")
    z = np.load(RUN / "harness/inputs_real.npz", allow_pickle=True)
    C, meta = list(z["contexts"]), z["meta"]
    lab = load_truth("real")["labels"]
    tasks = list(P["tasks"])
    crps = {k: [] for k in ("alma_v3", "alma_v2", "telos", "snaive")}
    mase = {k: [] for k in crps}
    for ti, tk in enumerate(tasks):
        idx = [i for i, m in enumerate(meta) if m[0] == tk]
        season = int(meta[idx[0]][2])
        ctx = [np.asarray(C[i], np.float64) for i in idx]
        H = int(meta[idx[0]][1])
        L = np.stack([np.asarray(lab[i], np.float64)[:H] for i in idx])
        for k in crps:
            s = score(ctx, L, np.asarray(P[k][ti], np.float64), season)
            crps[k].append(s["crps"])
            mase[k].append(s["mase"])
    crps = {k: np.array(v) for k, v in crps.items()}
    mase = {k: np.array(v) for k, v in mase.items()}
    rel = {k: crps[k] / crps["snaive"] for k in crps}
    relm = {k: mase[k] / mase["snaive"] for k in mase}
    skill = {k: float(100 * (1 - np.exp(np.log(rel[k]).mean()))) for k in rel}
    skillm = {k: float(100 * (1 - np.exp(np.nanmean(np.log(relm[k]))))) for k in relm}
    tests = {o: paired_t(crps["alma_v3"], crps[o]) for o in COMPARATORS}
    tests_m = {o: paired_t(mase["alma_v3"], mase[o]) for o in COMPARATORS}
    pv = {o: t_to_p(t["t"], t["n"]) for o, t in tests.items()}
    hp = holm(pv)
    res = {"n_tasks": len(tasks), "n_windows": int(len(meta)), "crps_skill_pct": skill, "mase_skill_pct": skillm,
           "v3_vs": {o: {**tests[o], "p": pv[o], "holm_pass": hp[o] and tests[o]["t"] is not None and tests[o]["t"] >= 3}
                     for o in COMPARATORS},
           "v3_vs_mase": tests_m,
           "per_task_rel_crps": {k: dict(zip(tasks, map(float, v))) for k, v in rel.items()}}
    (RUN / "metrics").mkdir(exist_ok=True)
    (RUN / "metrics/fc.json").write_text(json.dumps(res, indent=1, default=float))
    lines = ["# Alma v3 — sealed forecaster validation", "",
             f"{len(tasks)} tasks, {len(meta)} fresh GIFT-Eval test windows (never used for training, selection or any "
             "earlier grading).", "", "| Model | CRPS skill vs SN | MASE skill vs SN |", "|---|---|---|"]
    for k in ("alma_v3", "telos", "alma_v2", "snaive"):
        lines.append(f"| {k} | {skill[k]:.2f}% | {skillm[k]:.2f}% |")
    lines += ["", "| Alma v3 vs | gmean CRPS ratio | 95% CI | t | win rate | Holm-corrected beat (t>=3) |",
              "|---|---|---|---|---|---|"]
    for o in COMPARATORS:
        t = res["v3_vs"][o]
        lines.append(f"| {o} | {t['gmean_ratio']:.4f} | [{t['ci95_ratio'][0]:.4f}, {t['ci95_ratio'][1]:.4f}] | "
                     f"{t['t']:.2f} | {t['win_rate']:.2f} | {'YES' if t['holm_pass'] else 'no'} |")
    (RUN / "report").mkdir(exist_ok=True)
    (RUN / "report/FC_VALIDATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "predict", "grade"])
    ap.add_argument("--ckpt")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--no-telos", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "predict":
        predict(a.ckpt, a.ctx, a.no_telos)
    else:
        grade()
