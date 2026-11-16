"""Grade sealed outputs against truth (Gates 4-10, 12, 13). Writes metrics/*.json and report/scorecard.json.

python validate/grade.py --ckpt runs/alma_wm1/ema_8000.pt --last runs/alma_wm1/last.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy import stats

from common import CTX, HZ, PATCH, QS, RUN, coverage, holm, load_sealed, load_truth, log, paired_t, t_to_p, wql

TH = None
CARD: list[dict] = []
BEATS: dict = {}  # Holm family: name -> (t, n)


def card(gate: str, check: str, result, threshold, status: str, script: str = "validate/grade.py") -> None:
    CARD.append({"gate": gate, "check": check, "result": result, "threshold": threshold, "status": status,
                 "script": script})
    log(f"[{status}] {gate} {check}: {result}")


def save(name: str, obj) -> None:
    (RUN / "metrics" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def beats(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    r = paired_t(a, b)
    BEATS[name] = r
    return r


def tstat(r: dict) -> float:
    return r["t"] if r["t"] is not None else -np.inf


# ------------------------------------------------------------------------------------------------
def ep_crps(y: np.ndarray, q: np.ndarray, x: np.ndarray, sl=slice(None)) -> np.ndarray:
    """Per-episode CRPS normalised by context sd. y: (n,H), q: (n,H,9), x: (n,CTX)."""
    sd = x.std(1) + 1e-8
    return np.array([wql(y[i, sl], q[i, sl], norm=sd[i]) for i in range(len(y))])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    return float(stats.mannwhitneyu(score[label], score[~label]).statistic / (label.sum() * (~label).sum()))


# ------------------------------------------------------------------------------------------------
def grade_sim(ckpt: str, last: str) -> dict:
    P = load_sealed("pred_sim_test")
    inp = np.load(RUN / "harness/inputs_sim_test.npz", allow_pickle=True)
    T = load_truth("sim_test")
    x, u, kind = inp["x"], inp["u"], inp["kind"]
    y, ycf, oq, params = T["y"], T["y_cf"], T["oracle_q"], T["params"]
    n = len(y)
    sd = x.std(1) + 1e-8
    out = {"n": n}
    models = ["wm_aware", "wm_blind", "wm_direct", "alma_v2", "arx", "persistence", "snaive24"]
    full = {m: ep_crps(y, P[m], x) for m in models}
    full["oracle"] = ep_crps(y, oq, x)
    out["crps_full_gmean"] = {m: float(np.exp(np.log(v).mean())) for m, v in full.items()}

    # ---- Gate 4 / 5: buckets -------------------------------------------------------------------
    buckets = TH["rollout"]["buckets"]
    bres = {}
    for lo, hi in buckets:
        sl = slice(lo - 1, hi)
        b = {m: ep_crps(y, P[m], x, sl) for m in ["wm_aware", "arx", "persistence", "snaive24"]}
        b["oracle"] = ep_crps(y, oq, x, sl)
        tests = {o: beats(f"sim_b{lo}-{hi}_vs_{o}", b["wm_aware"], b[o]) for o in ["arx", "persistence", "snaive24"]}
        bres[f"{lo}-{hi}"] = {"gmean": {m: float(np.exp(np.log(v).mean())) for m, v in b.items()}, "tests": tests}
    out["buckets"] = bres
    g1 = bres[f"{buckets[0][0]}-{buckets[0][1]}"]["tests"]
    card("4", "SIMULATED one-step (1-16) CRPS beats ARX/persistence/SN24",
         {k: (round(v["gmean_ratio"], 3), round(tstat(v), 1)) for k, v in g1.items()}, "t>=3 each (Holm)",
         "PASS" if all(tstat(v) >= 3 for v in g1.values()) else "FAIL")
    allb = all(tstat(t) >= 3 for b in bres.values() for t in b["tests"].values())
    card("5.1", "SIMULATED rollout beats every simple law in every horizon bucket",
         {k: {o: round(v["gmean_ratio"], 3) for o, v in b["tests"].items()} for k, b in bres.items()},
         "t>=3 every bucket", "PASS" if allb else "FAIL")
    g_first = bres[f"{buckets[0][0]}-{buckets[0][1]}"]["gmean"]["wm_aware"]
    g_last = bres[f"{buckets[-1][0]}-{buckets[-1][1]}"]["gmean"]["wm_aware"]
    growth = g_last / g_first
    card("5.1", "error growth CRPS(257-512)/CRPS(1-16)", round(growth, 2), "<= 10",
         "PASS" if growth <= TH["rollout"]["max_error_growth_ratio"] else "FAIL")

    # ---- 5.2 open vs closed loop -----------------------------------------------------------------
    from model_wm import AlmaA
    from model_wm2 import load_any
    from data import prescale
    from data_wm import prescale_actions

    model = load_any(ckpt, "cpu")
    ck = torch.load(last, map_location="cpu", weights_only=False)
    tgt = AlmaA(model.c)
    tgt.load_state_dict(ck["target"])
    tgt.eval()
    m_sub = min(n, 200)
    cl, ol = [], []
    with torch.no_grad():
        for i in range(m_sub):
            full_y = np.concatenate([x[i], y[i]])
            ms = np.ones(CTX + HZ, bool)
            z, loc, sc = prescale(full_y, ms)
            uu = np.nan_to_num(u[i])
            um = np.isfinite(u[i])
            us = prescale_actions(uu, um)
            xt = torch.tensor(z[None], dtype=torch.float32)
            mt = torch.tensor(ms[None])
            ut, umt = torch.tensor(us[None], dtype=torch.float32), torch.tensor(um[None])
            h, mu, sdd = model.enc.encode(xt[:, :CTX], mt[:, :CTX], ut[:, :CTX], umt[:, :CTX])
            zt, _, _ = tgt.encode(xt, mt, ut, umt)
            nf = HZ // PATCH
            zf = zt[:, CTX // PATCH : CTX // PATCH + nf]
            from model import from_frame

            if hasattr(model, "decode_states"):  # wm2: decoder also needs the ARX prior
                qcl = model.decode_states(zf, xt[:, :CTX], mt[:, :CTX], ut, umt)[0].numpy()
            else:
                qcl = model.decode(zf, h[:, -1]).flatten(1, 2)
                qcl = from_frame(qcl, mu[:, -1, None, None], sdd[:, -1, None, None], model.c.asinh_clip)[0].numpy()
            cl.append(wql(y[i], qcl * sc + loc, norm=sd[i]))
            ol.append(full["wm_aware"][i])
    cl, ol = np.array(cl), np.array(ol)
    out["open_vs_closed"] = paired_t(ol, cl)
    card("5.2", "open-loop vs closed-loop (re-anchored) CRPS gap", round(out["open_vs_closed"]["gmean_ratio"], 3),
         "reported", "REPORTED")

    # ---- 5.5 dispersion / 5.3 calibration vs oracle -----------------------------------------------
    w_m = (P["wm_aware"][..., 8] - P["wm_aware"][..., 0]).mean(1) / sd
    w_o = (oq[..., 8] - oq[..., 0]).mean(1) / sd
    disp = float(np.exp(np.median(np.log(w_m / np.maximum(w_o, 1e-9)))))
    card("5.5", "no mode collapse: median 80%-width ratio model/oracle", round(disp, 3), "descriptive (~1)",
         "REPORTED")
    pit = np.array([[(y[i, h] > P["wm_aware"][i, h]).sum() for h in range(HZ)] for i in range(n)]).ravel()
    hist = np.bincount(pit, minlength=10) / len(pit)
    out["pit_hist_deciles"] = hist.tolist()
    chi = stats.chisquare(np.bincount(pit, minlength=10)).pvalue
    card("5.3", "PIT histogram (10 bins, expected 0.1 each)", [round(h, 3) for h in hist], "descriptive",
         "REPORTED")
    card("5.4", "distribution two-sample / discriminator test", "not computable: model emits marginal quantiles,"
         " no joint sample paths", "required", "FAIL")

    # ---- 5.6 constraints --------------------------------------------------------------------------
    nonneg = np.isin(kind, ["inventory", "pricing", "logistic"])
    viol = float((P["wm_aware"][nonneg][..., 4] < 0).mean()) if nonneg.any() else 0.0
    card("5.6", "physical constraint violations (median < 0 for non-negative systems)", round(viol, 4),
         "<= 0.01", "PASS" if viol <= TH["rollout"]["constraint_violation_rate_max"] else "FAIL")

    # ---- Gate 6 -----------------------------------------------------------------------------------
    r_ab = beats("sim_aware_vs_blind", full["wm_aware"], full["wm_blind"])
    card("6.1", "SIMULATED action-aware beats action-blind (CRPS)", (round(r_ab["gmean_ratio"], 3), round(tstat(r_ab), 1)),
         "t>=3", "PASS" if tstat(r_ab) >= 3 else "FAIL")
    W = 256
    med_f = P["wm_aware"][:, :W, 4]
    e_true = (ycf[:, :, :W] - y[:, None, :W]).mean(-1) / sd[:, None]
    e_pred = (P["wm_cf"][:, :, :W, 4] - med_f[:, None]).mean(-1) / sd[:, None]
    e_arx = (P["arx_cf"][:, :, :W, 4] - P["arx"][:, None, :W, 4]).mean(-1) / sd[:, None]
    big = np.abs(e_true) > 0.5
    sign_acc = float((np.sign(e_pred[big]) == np.sign(e_true[big])).mean()) if big.any() else float("nan")
    sign_arx = float((np.sign(e_arx[big]) == np.sign(e_true[big])).mean()) if big.any() else float("nan")
    corr = float(stats.spearmanr(e_true.ravel(), e_pred.ravel()).statistic)
    out["effects"] = {"n_big": int(big.sum()), "sign_acc_wm": sign_acc, "sign_acc_arx": sign_arx,
                      "spearman_wm": corr, "spearman_arx": float(stats.spearmanr(e_true.ravel(), e_arx.ravel()).statistic),
                      "mae_wm": float(np.abs(e_pred - e_true).mean()), "mae_arx": float(np.abs(e_arx - e_true).mean())}
    card("6.2", "SIMULATED effect sign accuracy (|true effect|>0.5 sd)",
         {"wm": round(sign_acc, 3), "arx": round(sign_arx, 3), "n": int(big.sum())}, ">= 0.90",
         "PASS" if sign_acc >= TH["action"]["effect_sign_accuracy_min"] else "FAIL")
    tr = np.concatenate([e_true[:, :2], np.zeros((n, 1)), e_true[:, 2:]], 1)  # deltas -1,-.5,0,.5,1
    pr = np.concatenate([e_pred[:, :2], np.zeros((n, 1)), e_pred[:, 2:]], 1)
    dtr, dpr = np.diff(tr, axis=1), np.diff(pr, axis=1)
    mono_true = (np.all(dtr > 0, 1) | np.all(dtr < 0, 1)) & (np.abs(tr[:, -1] - tr[:, 0]) > 0.1)
    same = np.where(np.all(dtr > 0, 1), np.all(dpr > 0, 1), np.all(dpr < 0, 1))
    mono = float(same[mono_true].mean()) if mono_true.any() else float("nan")
    card("6.2", "SIMULATED monotone size sweeps predicted monotone, same direction",
         {"rate": round(mono, 3), "n": int(mono_true.sum())}, ">= 0.95",
         "PASS" if mono >= TH["action"]["monotone_sweep_rate_min"] else "FAIL")
    cf_wm = np.array([np.mean([wql(ycf[i, k], P["wm_cf"][i, k], norm=sd[i]) for k in range(ycf.shape[1])]) for i in range(n)])
    cf_ar = np.array([np.mean([wql(ycf[i, k], P["arx_cf"][i, k], norm=sd[i]) for k in range(ycf.shape[1])]) for i in range(n)])
    r_cf = beats("sim_cf_vs_arx", cf_wm, cf_ar)
    card("6.5*", "SIMULATED counterfactual CRPS beats ARX (true counterfactual trajectories)",
         (round(r_cf["gmean_ratio"], 3), round(tstat(r_cf), 1)), "t>=3",
         "PASS" if tstat(r_cf) >= 3 else "FAIL")
    for c, nm in (("6.3", "impact accuracy on real interventions"), ("6.4", "natural experiments on real data"),
                  ("6.5", "counterfactual vs nature on real data")):
        card(c, nm, "no real interventional data available", "required", "FAIL")
    card("6.6", "randomised live checks", "no live deployment", "when live", "N/A")
    card("6.7", "self-consistency (split trades)", "market-specific", "-", "N/A")
    card("6.8", "reflexivity", "market-specific", "-", "N/A")

    # ---- Gate 7 -----------------------------------------------------------------------------------
    c80, c60 = np.mean([coverage(y[i], P["wm_aware"][i]) for i in range(n)], 0)
    tol = TH["coverage_tolerance_pp"] / 100
    card("7 / 4.4", "SIMULATED coverage 80% / 60% (wm_aware)", (round(c80, 3), round(c60, 3)), "+-0.02",
         "PASS" if abs(c80 - 0.8) <= tol and abs(c60 - 0.6) <= tol else "FAIL")
    width = (P["wm_aware"][..., 8] - P["wm_aware"][..., 0]).mean(1) / sd
    err = np.abs(P["wm_aware"][..., 4] - y).mean(1) / sd
    rs = stats.spearmanr(width, err).statistic
    tt = rs * np.sqrt((n - 2) / max(1 - rs**2, 1e-9))
    card("7.2", "uncertainty predicts error (Spearman width vs |error|)", (round(float(rs), 3), round(float(tt), 1)),
         "t>=3", "PASS" if tt >= 3 else "FAIL")
    # conformal on sim_val
    Pv = load_sealed("pred_sim_val")
    iv = np.load(RUN / "harness/inputs_sim_val.npz", allow_pickle=True)
    Tv = load_truth("sim_val")
    sdv = iv["x"].std(1)[:, None] + 1e-8
    rv = (Tv["y"] - Pv["wm_aware"][..., 4]) / sdv
    off_lo = np.quantile(((Pv["wm_aware"][..., 0] - Pv["wm_aware"][..., 4]) / sdv - rv), 0.9)
    lo_c = P["wm_aware"][..., 0] - np.maximum(off_lo, 0) * sd[:, None]
    off_hi = np.quantile((rv - (Pv["wm_aware"][..., 8] - Pv["wm_aware"][..., 4]) / sdv), 0.9)
    hi_c = P["wm_aware"][..., 8] + np.maximum(off_hi, 0) * sd[:, None]
    c80c = float(np.mean((y >= lo_c) & (y <= hi_c)))
    card("7.3", "SIMULATED conformal 80% interval coverage (calibrated on sim_val)", round(c80c, 3), "+-0.02",
         "PASS" if abs(c80c - 0.8) <= tol else "FAIL")
    # OOD
    St = load_sealed("pred_sim_tanks")
    it = np.load(RUN / "harness/inputs_sim_tanks.npz", allow_pickle=True)
    Tt = load_truth("sim_tanks")
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(Pv["state"])
    maha = lambda s: lw.mahalanobis(s)
    s_test, s_tank = maha(P["state"]), maha(St["state"])
    crps_tank = ep_crps(Tt["y"], St["wm_aware"], it["x"])
    pool_s = np.concatenate([s_test, s_tank])
    pool_e = np.concatenate([full["wm_aware"], crps_tank])
    lab = pool_e >= np.quantile(pool_e, 0.9)
    au = auroc(pool_s, lab)
    dis_test = np.abs(P["wm_direct"][..., 4] - P["wm_aware"][..., 4]).mean(1) / sd
    dis_tank = np.abs(St["wm_direct"][..., 4] - St["wm_aware"][..., 4]).mean(1) / (it["x"].std(1) + 1e-8)
    au2 = auroc(np.concatenate([dis_test, dis_tank]), lab)
    card("7.4", "OOD flag (Mahalanobis of state) AUROC for top-decile errors",
         {"primary_mahalanobis": round(au, 3), "secondary_disagreement": round(au2, 3)}, ">= 0.75",
         "PASS" if au >= TH["uncertainty"]["ood_auroc_min"] else "FAIL")
    thr = np.quantile(maha(Pv["state"]), 0.9)
    flag = s_test > thr
    fb = full["arx"][flag]
    md = full["wm_aware"][flag]
    ok_fb = bool(flag.sum() == 0 or np.exp(np.log(fb).mean()) <= np.exp(np.log(md).mean()))
    card("7.5", "fallback (ARX) on flagged episodes no worse than the model",
         {"n_flagged": int(flag.sum()), "gmean_model": float(np.exp(np.log(md).mean())) if flag.any() else None,
          "gmean_fallback": float(np.exp(np.log(fb).mean())) if flag.any() else None}, "fallback <= model",
         "PASS" if ok_fb else "FAIL")
    card("7.6", "novel family (tanks) flag rate vs in-family flag rate",
         {"tanks": round(float((s_tank > thr).mean()), 3), "in_family": round(float(flag.mean()), 3)},
         "descriptive", "REPORTED")
    order = np.argsort(s_test)
    rc = {f"{c:.1f}": float(np.exp(np.log(full["wm_aware"][order[: max(1, int(c * n))]]).mean()))
          for c in (0.5, 0.7, 0.9, 1.0)}
    card("7.7", "risk-coverage (gmean CRPS keeping least-OOD fraction)", rc, "descriptive", "REPORTED")

    # ---- Gate 8 placebos -------------------------------------------------------------------------
    r_sh = beats("placebo_true_vs_shuffled", full["wm_aware"], ep_crps(y, P["wm_shuffled"], x))
    card("8.10", "placebo: true actions beat shuffled actions", (round(r_sh["gmean_ratio"], 3), round(tstat(r_sh), 1)),
         "t>=3", "PASS" if tstat(r_sh) >= 3 else "FAIL")
    rnd = ep_crps(y, P["wm_random"], x)
    r_rb = paired_t(rnd, full["wm_blind"])
    card("8.10", "placebo: random actions do not beat action-blind",
         (round(r_rb["gmean_ratio"], 3), round(tstat(r_rb), 1)), "t<3", "PASS" if tstat(r_rb) < 3 else "FAIL")
    k = 16
    shifted = np.array([wql(y[i, k:], P["wm_aware"][i, :-k], norm=sd[i]) for i in range(n)])
    unshift = np.array([wql(y[i, k:], P["wm_aware"][i, k:], norm=sd[i]) for i in range(n)])
    r_ts = paired_t(unshift, shifted)
    card("8.10", "placebo: time-shifted (16 steps) predictions lose skill",
         (round(r_ts["gmean_ratio"], 3), round(tstat(r_ts), 1)), "t>=3", "PASS" if tstat(r_ts) >= 3 else "FAIL")

    # ---- Gate 9 by family + transfer ------------------------------------------------------------
    fam = {}
    for kd in sorted(set(kind)):
        s = kind == kd
        fam[kd] = {m: float(np.exp(np.log(full[m][s]).mean())) for m in ("wm_aware", "wm_blind", "arx", "oracle")}
        fam[kd]["aware_vs_blind_t"] = tstat(paired_t(full["wm_aware"][s], full["wm_blind"][s]))
    out["by_family"] = fam
    worst = max(fam.items(), key=lambda kv: kv[1]["wm_aware"] / kv[1]["arx"])
    card("9.2/9.3", "SIMULATED by system family: aware/ARX gmean ratio",
         {k: round(v["wm_aware"] / v["arx"], 3) for k, v in fam.items()}, "descriptive", "REPORTED")
    ty = Tt["y"]
    xt_ = it["x"]
    tr_ = {m: float(np.exp(np.log(ep_crps(ty, St[m], xt_)).mean())) for m in ("wm_aware", "wm_blind", "arx", "persistence")}
    tr_["oracle"] = float(np.exp(np.log(ep_crps(ty, Tt["oracle_q"], xt_)).mean()))
    tr_["aware_vs_arx"] = paired_t(ep_crps(ty, St["wm_aware"], xt_), ep_crps(ty, St["arx"], xt_))
    out["transfer_tanks"] = tr_
    card("9.7", "transfer to held-out family (tanks), gmean CRPS",
         {k: (round(v, 3) if isinstance(v, float) else round(v["gmean_ratio"], 3)) for k, v in tr_.items()},
         "descriptive", "REPORTED")

    # ---- Gate 10 probes -----------------------------------------------------------------------------
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    targets = {"thermal": ["a", "b"], "fopdt": ["tau", "delay", "K"], "inventory": ["lag"], "pricing": ["elast", "carry"]}
    probes = {}
    S_ = P["state"]
    rng = np.random.default_rng(0)
    for kd, keys in targets.items():
        s = np.where(kind == kd)[0]
        for key in keys:
            yv = np.array([params[i][key] for i in s], float)
            yv = np.log(np.abs(yv) + 1e-6) if key in ("tau", "a", "b", "elast") else yv
            pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), S_[s], yv, cv=5)
            r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
            null = []
            for _ in range(100):
                yp = rng.permutation(yv)
                pp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), S_[s], yp, cv=5)
                null.append(1 - ((yp - pp) ** 2).sum() / ((yp - yp.mean()) ** 2).sum())
            pval = float((np.sum(np.array(null) >= r2) + 1) / 101)
            probes[f"{kd}.{key}"] = {"r2": float(r2), "p": pval, "n": int(len(s))}
    out["probes"] = probes
    okp = sum(v["r2"] > 0 and v["p"] < 0.01 for v in probes.values())
    card("10.1", "linear probes: state -> true system parameters (5-fold CV R², permutation p)",
         {k: (round(v["r2"], 2), round(v["p"], 3)) for k, v in probes.items()},
         "R²>0 and p<0.01 for every probe", "PASS" if okp == len(probes) else "FAIL")
    sv = np.linalg.svd(S_ - S_.mean(0), compute_uv=False)
    pnorm = sv / sv.sum()
    erank = float(np.exp(-(pnorm * np.log(pnorm + 1e-12)).sum()))
    card("10.1", "latent effective rank (non-collapse)", round(erank, 1), ">= 16",
         "PASS" if erank >= TH["probes"]["effective_rank_min"] else "FAIL")
    card("10.4", "latent stability across training seeds", "only one training seed exists", ">= 0.8", "FAIL")
    card("10.2", "attribution faithfulness", "not run", "required", "FAIL")
    card("10.5", "disentanglement", "not run", "required", "FAIL")
    card("10.6/10.7", "explanation decomposition / residual load", "no named-component decomposition", "-", "N/A")
    save("sim_test", out)
    return out


# ------------------------------------------------------------------------------------------------
def grade_plan() -> dict:
    P = load_sealed("pred_plan_test")
    inp = np.load(RUN / "harness/inputs_plan_test.npz", allow_pickle=True)
    T = load_truth("plan_test")
    x, tgt, yc = inp["x"], inp["target"], T["y_cand"]
    W = 256
    var = x.var(1)[:, None] + 1e-8
    real = ((yc[:, :, :W] - tgt[:, None, None]) ** 2).mean(-1) / var
    im_wm = ((P["wm"][:, :, :W] - tgt[:, None, None, None]) ** 2).mean((-1, -2)) / var
    im_ar = ((P["arx"][:, :, :W] - tgt[:, None, None, None]) ** 2).mean((-1, -2)) / var
    n = len(x)
    ch_wm, ch_ar = im_wm.argmin(1), im_ar.argmin(1)
    i = np.arange(n)
    c_wm, c_ar, c_def, c_or = real[i, ch_wm], real[i, ch_ar], real[:, 0], real.min(1)
    r1 = beats("plan_wm_vs_default", c_wm, c_def)
    r2 = beats("plan_wm_vs_arx", c_wm, c_ar)
    card("12", "SIMULATED MPC real cost: WM vs default plan and vs ARX-MPC",
         {"vs_default": (round(r1["gmean_ratio"], 3), round(tstat(r1), 1)), "vs_arx": (round(r2["gmean_ratio"], 3), round(tstat(r2), 1)),
          "oracle_vs_default": round(float(np.exp(np.log(c_or / c_def).mean())), 3)}, "t>=3 both",
         "PASS" if tstat(r1) >= 3 and tstat(r2) >= 3 else "FAIL")
    img_gain = im_wm[:, 0] - im_wm[i, ch_wm]
    real_gain = c_def - c_wm
    ratio = float(real_gain.sum() / max(img_gain.sum(), 1e-9))
    card("12.1", "imagined vs real gain (sum real / sum imagined)", round(ratio, 3), ">= 0.70",
         "PASS" if ratio >= TH["planning"]["real_vs_imagined_gain_min"] else "FAIL")
    gap = img_gain - real_gain
    card("12.2", "exploit hunt: corr(imagined gain, imagined-real gap)",
         round(float(stats.spearmanr(img_gain, gap).statistic), 3), "descriptive", "REPORTED")
    for c in ("12.5", "12.6", "12.7", "12.8"):
        card(c, "execution / portfolio / risk / deflated Sharpe", "no trading component", "-", "N/A")
    res = {"cost_gmean": {k: float(np.exp(np.log(v).mean())) for k, v in
                          {"wm": c_wm, "arx": c_ar, "default": c_def, "oracle": c_or}.items()}, "gain_ratio": ratio}
    save("planning", res)
    return res


# ------------------------------------------------------------------------------------------------
def grade_real() -> dict:
    from eval_gift import score

    P = load_sealed("pred_real")
    inp = np.load(RUN / "harness/inputs_real.npz", allow_pickle=True)
    T = load_truth("real")
    C, meta, L = list(inp["contexts"]), inp["meta"], T["labels"]
    tasks = list(P["tasks"])
    models = ["wm_direct", "wm_rollout", "alma_v2", "telos", "snaive", "persistence"]
    per = {m: {"crps": [], "mase": []} for m in models}
    cov = {m: [] for m in ("wm_direct", "wm_rollout", "alma_v2", "telos")}
    for ti, tk in enumerate(tasks):
        idx = list(P["idx"][ti])
        season = int(meta[idx[0]][2])
        ctx = [C[i] for i in idx]
        lab = np.stack([L[i] for i in idx]).astype(np.float64)
        for m in models:
            q = P[m][ti]
            s = score(ctx, lab, q, season)
            per[m]["crps"].append(s["crps"])
            per[m]["mase"].append(s["mase"])
            if m in cov:
                ok = np.isfinite(lab)
                c80 = ((lab >= q[..., 0]) & (lab <= q[..., 8]))[ok].mean()
                c60 = ((lab >= q[..., 1]) & (lab <= q[..., 7]))[ok].mean()
                cov[m].append((c80, c60, ok.sum()))
    arr = {m: {k: np.array(v) for k, v in d.items()} for m, d in per.items()}
    rel = {m: arr[m]["crps"] / arr["snaive"]["crps"] for m in models}
    relm = {m: arr[m]["mase"] / arr["snaive"]["mase"] for m in models}
    out = {"tasks": tasks, "n_windows": int(sum(len(i) for i in P["idx"])),
           "skill_crps_pct": {m: float(100 * (1 - np.exp(np.log(rel[m]).mean()))) for m in models},
           "skill_mase_pct": {m: float(100 * (1 - np.exp(np.nanmean(np.log(relm[m]))))) for m in models}}
    tests = {o: beats(f"real_wm_vs_{o}", arr["wm_direct"]["crps"], arr[o]["crps"]) for o in ("snaive", "persistence", "telos")}
    out["tests"] = tests
    card("4.2", "REAL sealed GIFT windows: Alma-WM CRPS beats SN / persistence / Telos-Small",
         {o: (round(v["gmean_ratio"], 3), round(tstat(v), 1), f"wins {v['win_rate']:.2f}") for o, v in tests.items()},
         "t>=3 each (Holm)", "PASS" if all(tstat(v) >= 3 for v in tests.values()) else "FAIL")
    ni = paired_t(arr["wm_direct"]["crps"], arr["alma_v2"]["crps"])
    out["noninferiority_v2"] = ni
    card("4.2", "REAL non-inferiority vs Alma v2 (gmean ratio, CI upper)",
         (round(ni["gmean_ratio"], 4), round(ni["ci95_ratio"][1], 4)), "<= 1.01",
         "PASS" if ni["ci95_ratio"][1] <= 1 + TH["noninferiority_vs_alma_v2"] else "FAIL")
    roll = paired_t(arr["wm_rollout"]["crps"], arr["wm_direct"]["crps"])
    out["rollout_vs_direct"] = roll
    card("4.2", "REAL latent-rollout forecasts vs direct head (ratio)", round(roll["gmean_ratio"], 3), "descriptive",
         "REPORTED")
    tol = TH["coverage_tolerance_pp"] / 100
    cv = {}
    for m, v in cov.items():
        a = np.array(v)
        w = a[:, 2] / a[:, 2].sum()
        cv[m] = (float((a[:, 0] * w).sum()), float((a[:, 1] * w).sum()))
    out["coverage"] = cv
    c80, c60 = cv["wm_direct"]
    card("4.4", "REAL coverage 80% / 60% (Alma-WM)", {m: (round(a, 3), round(b, 3)) for m, (a, b) in cv.items()},
         "+-0.02", "PASS" if abs(c80 - 0.8) <= tol and abs(c60 - 0.6) <= tol else "FAIL")
    card("8.2", "effective sample (real tasks)", len(tasks), ">=12 preliminary, >=36 frontier",
         "PASS (preliminary)" if len(tasks) >= 12 else "FAIL")
    out["per_task_rel_crps"] = {m: dict(zip(tasks, rel[m].tolist())) for m in models}
    save("real", out)
    return out


# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    TH = yaml.safe_load(open(RUN / "prereg/thresholds.yaml"))
    torch.set_num_threads(8)
    res = {"sim": grade_sim(a.ckpt, a.last), "plan": grade_plan(), "real": grade_real()}
    pv = {k: t_to_p(v["t"], v["n"]) for k, v in BEATS.items()}
    hol = holm(pv)
    card("8.6", "Holm correction across all 'beats' claims",
         {k: hol[k] for k in sorted(hol)}, "alpha 0.05", "REPORTED")
    save("beats_tests", {k: {**BEATS[k], "p": pv[k], "holm_pass": hol[k]} for k in BEATS})
    (RUN / "report/scorecard_grade.json").write_text(json.dumps(CARD, indent=1, default=float))
    log("grading done")
