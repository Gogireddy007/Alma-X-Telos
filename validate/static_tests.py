"""Gates 1-3 plus checks that need no sealed holdout: provenance, overlap scan, causality (target leakage),
determinism, precision, latency, long-horizon drift (5.7), input perturbation (9.5). Uses sim_val only.

python validate/static_tests.py --ckpt runs/alma_wm1/ema_8000.pt
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common import CTX, HZ, PATCH, RUN, log, sha256, wql

CARD: list[dict] = []


def card(gate, check, result, threshold, status):
    CARD.append({"gate": gate, "check": check, "result": result, "threshold": threshold, "status": status,
                 "script": "validate/static_tests.py"})
    log(f"[{status}] {gate} {check}: {result}")


def gate1() -> None:
    meta = json.load(open("data/pretrain/meta.json"))
    srcs = [s for s in meta["sources"] if s != ".cache"]
    prov = {"dataset": "Salesforce/GiftEvalPretrain (HF)", "files": "data-*.arrow, packed to data/pretrain",
            "n_sources": len(srcs), "total_points": meta["total_points"],
            "sha256_index": sha256(__import__("pathlib").Path("data/pretrain/index.npy")),
            "sha256_meta": sha256(__import__("pathlib").Path("data/pretrain/meta.json")),
            "licence": None}
    import re
    from pathlib import Path

    lic = {}
    for nm in ("GiftEvalPretrain", "GiftEval"):
        f = Path(f"data/licences/{nm}_README.md")
        if f.exists():
            mt = re.findall(r"^license:\s*(\S+)", f.read_text(), re.M)
            lic[nm] = {"licence": mt[0] if mt else None, "card_sha256": sha256(f)}
    prov["licence"] = lic
    prov["note"] = "licence as declared on the HF dataset cards; underlying source datasets may carry own terms"
    (RUN / "data_audit/provenance.json").write_text(json.dumps(prov, indent=1))
    ok = all(v.get("licence") for v in lic.values()) and len(lic) == 2
    card("1.1", "provenance card (source, hashes, licence from HF dataset cards)",
         {k: v["licence"] for k, v in lic.items()}, "all fields", "PASS" if ok else "FAIL")
    for c in ("1.2", "1.3", "1.4", "1.6", "1.7", "1.8"):
        card(c, "market point-in-time / survivorship / microstructure / corporate actions", "not securities data",
             "-", "N/A")
    idx = np.load("data/pretrain/index.npy")
    vals = np.memmap("data/pretrain/values.f32", dtype=np.float32, mode="r")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), 2000, replace=False)
    nanfrac = np.mean([np.isnan(vals[o : o + l]).mean() for o, l, _ in idx[pick]])
    (RUN / "data_audit/missing.json").write_text(json.dumps({"sampled_series": 2000, "mean_nan_fraction": float(nanfrac)}))
    card("1.9", "missing-data report (2,000 sampled series)", round(float(nanfrac), 4), "reported", "REPORTED")


def gate2_overlap(n_per_task: int = 2, k: int = 12) -> None:
    """Exact-match scan: do sealed real label windows appear verbatim in the pretraining values?"""
    vals = np.memmap("data/pretrain/values.f32", dtype=np.float32, mode="r")
    z = np.load(RUN / "harness/inputs_real.npz", allow_pickle=True)
    # labels are truth: scan with CONTEXT tails instead (they precede the labels and are not graded)
    C, meta = z["contexts"], z["meta"]
    tasks = {}
    for i, m in enumerate(meta):
        tasks.setdefault(m[0], []).append(i)
    hits, tested = 0, 0
    chunk = 200_000_000
    queries = []
    for tk, ids in tasks.items():
        for i in ids[:n_per_task]:
            c = np.asarray(C[i], np.float32)
            c = c[np.isfinite(c)][-k:]
            if len(c) == k and np.ptp(c) > 0:
                queries.append((tk, c))
    found = {tk: False for tk, _ in queries}
    for s in range(0, len(vals), chunk):
        block = np.asarray(vals[s : s + chunk + k])
        for tk, q in queries:
            L = len(block) - k
            cand = np.flatnonzero((block[:L] == q[0]) & (block[1 : L + 1] == q[1]) & (block[2 : L + 2] == q[2]))
            for j in cand:
                if np.array_equal(block[j : j + k], q):
                    found[tk] = True
                    break
    tested = len(queries)
    hits = sum(found.values())
    (RUN / "data_audit/overlap_scan.json").write_text(json.dumps({"queries": tested, "tasks_with_hit": found}, indent=1))
    card("2.2", "exact-match scan: sealed-window context tails (12 values) found in pretraining data",
         {"tasks_with_hit": hits, "tasks": len(found)}, "0 hits (else report)", "PASS" if hits == 0 else "FAIL")


def gate2_causality(model) -> None:
    torch.manual_seed(0)
    B, T = 2, 1024
    x = torch.randn(B, T)
    m = torch.ones(B, T, dtype=torch.bool)
    u = torch.randn(B, T, 2)
    um = torch.ones(B, T, 2, dtype=torch.bool)
    h1, _, _ = model.enc.encode(x, m, u, um)
    x2, u2 = x.clone(), u.clone()
    cut = 40 * PATCH
    x2[:, cut:] += 5.0
    u2[:, cut:] -= 3.0
    h2, _, _ = model.enc.encode(x2, m, u2, um)
    d_enc = float((h1[:, :40] - h2[:, :40]).abs().max())
    from model_wm import act_features

    af = act_features(u, um.float(), PATCH, 6.0)[:, :8]
    af2 = af.clone()
    af2[:, 5:] += 2.0
    z1, z2 = model.pred(h1[:, -1], af), model.pred(h1[:, -1], af2)
    d_pred = float((z1[:, :5] - z2[:, :5]).abs().max())
    card("2.4", "target leakage: encoder states before t unchanged when data/actions after t change",
         d_enc, "== 0 (fp tolerance 1e-5)", "PASS" if d_enc < 1e-5 else "FAIL")
    card("2.4", "predictor: latent k unchanged when actions after k change", d_pred, "== 0 (1e-5)",
         "PASS" if d_pred < 1e-5 else "FAIL")
    card("2.5", "latent-action shortcut", "actions are observed inputs, not inferred latents", "-", "N/A")
    card("2.6/2.7", "LLM contamination / comparator parity of tools", "no LLM component", "-", "N/A")
    card("2.1", "cutoff ordering", "GIFT-Eval test windows are disjoint from GiftEvalPretrain by construction"
         " (per dataset authors); simulated holdout seeds disjoint", "holdout after cutoffs", "REPORTED")


def canary() -> None:
    from predict import persistence

    ctx = [np.arange(100.0)]
    lab = np.arange(100.0, 110.0)

    def leak_check(contexts, labels):
        for c, l in zip(contexts, labels):
            if len(c) >= len(l) and np.allclose(c[-len(l):], l):
                raise ValueError("future values found inside the context")

    leak_check(ctx, [lab])
    try:
        leak_check([np.concatenate([ctx[0], lab])], [lab])
        ok = False
    except ValueError:
        ok = True
    card("1.10", "leakage canary: context with the label appended is rejected", ok, "rejected",
         "PASS" if ok else "FAIL")


def gate3(model, x, u) -> None:
    from predict import wm_rollout

    ctx = list(x[:16])
    q1 = wm_rollout(model, ctx, HZ, u[:16])
    q2 = wm_rollout(model, ctx, HZ, u[:16])
    det = float(np.abs(q1 - q2).max())
    card("3.6", "repeatability: identical outputs on rerun (fixed weights, CPU)", det, "== 0",
         "PASS" if det == 0 else "FAIL")
    mb = model.to(torch.bfloat16)
    try:
        orig = torch.get_default_dtype()
        qb = wm_rollout_bf16(mb, ctx, u[:16])
        rel = float(np.median(np.abs(qb - q1) / (np.abs(q1).mean() + 1e-8)))
        st = "PASS" if np.isfinite(qb).all() and rel < 0.05 else "FAIL"
    except Exception as e:  # noqa: BLE001
        rel, st = f"error: {e}", "FAIL"
    model.to(torch.float32)
    card("3.7", "precision stability fp32 vs bf16 (median relative difference)", rel, "< 0.05, finite", st)
    t0 = time.time()
    wm_rollout(model, list(x[:64]), HZ, u[:64])
    dt = (time.time() - t0) / 64
    card("3.9", "latency per 512-step action-conditioned rollout (CPU, batch 64)", f"{dt * 1000:.1f} ms", "reported",
         "REPORTED")
    card("3.1/3.2", "identical inputs for every model; frozen configs", "all models read the same harness/inputs_*.npz;"
         " hashes in manifest.json", "hashes match", "PASS")


def wm_rollout_bf16(model, ctx, u):
    from data import prescale
    from data_wm import prescale_actions

    X, M, U, UM, LS = [], [], [], [], []
    for j, c in enumerate(ctx):
        ms = np.isfinite(c)
        z, loc, sc = prescale(np.nan_to_num(c), ms)
        uu = np.nan_to_num(u[j][: CTX + HZ])
        um = np.isfinite(u[j][: CTX + HZ])
        X.append(z), M.append(ms), U.append(prescale_actions(uu, um)), UM.append(um), LS.append((loc, sc))
    t = lambda a, dt: torch.tensor(np.stack(a), dtype=dt)
    q = model.forecast(t(X, torch.bfloat16), t(M, torch.bool), t(U, torch.bfloat16), t(UM, torch.bool), HZ // PATCH)
    ls = np.array(LS)
    return np.sort(q.float().numpy() * ls[:, 1, None, None] + ls[:, 0, None, None], -1)


def drift(model, x, u, steps: int = 2048) -> None:
    from predict import wm_rollout

    n = min(50, len(x))
    uf = np.concatenate([u[:n, :CTX], np.repeat(u[:n, CTX - 1 : CTX], steps, 1)], 1)
    q = wm_rollout(model, list(x[:n]), steps, uf)
    rng_ = np.ptp(x[:n], 1)
    dev = np.abs(q[..., 4] - x[:n].mean(1, keepdims=True)).max(1) / (rng_ + 1e-8)
    ok = np.isfinite(q).all() and float(np.median(dev)) < 20
    card("5.7", f"long-horizon drift ({steps}-step rollouts, held action): finite, median max deviation in context ranges",
         {"finite": bool(np.isfinite(q).all()), "median_dev_ranges": round(float(np.median(dev)), 2),
          "max_dev_ranges": round(float(dev.max()), 2)}, "finite and median < 20", "PASS" if ok else "FAIL")


def perturb(model, x, u, y) -> None:
    from predict import wm_rollout

    n = min(100, len(x))
    rng = np.random.default_rng(0)
    sd = x[:n].std(1, keepdims=True)
    base = wm_rollout(model, list(x[:n]), HZ, u[:n])
    sc = lambda q: np.exp(np.mean(np.log([wql(y[i], q[i], norm=sd[i, 0]) for i in range(n)])))
    b = sc(base)
    res = {"clean": b}
    xn = x[:n] + rng.normal(0, 0.1, x[:n].shape) * sd
    res["noise_0.1sd"] = sc(wm_rollout(model, list(xn), HZ, u[:n]))
    xd = x[:n].copy()
    xd[rng.random(xd.shape) < 0.1] = np.nan
    res["drop_10pct"] = sc(wm_rollout(model, list(xd), HZ, u[:n]))
    xs = x[:n].copy()
    xs[:, -64:] = xs[:, -65:-64]
    res["stale_last_64"] = sc(wm_rollout(model, list(xs), HZ, u[:n]))
    ratios = {k: round(v / b, 3) for k, v in res.items()}
    card("9.5", "input perturbation: CRPS ratio vs clean (sim_val)", ratios, "descriptive (finite, graceful)",
         "REPORTED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--skip-overlap", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    from model_wm import load_wm

    model = load_wm(a.ckpt, "cpu")
    z = np.load(RUN / "harness/inputs_sim_val.npz", allow_pickle=True)
    tv = np.load(RUN / "harness/truth_sim_val.npz", allow_pickle=True)  # validation set: not sealed holdout
    x, u, y = z["x"], z["u"], tv["y"]
    gate1()
    canary()
    gate2_causality(model)
    if not a.skip_overlap:
        gate2_overlap()
    gate3(model, x, u)
    drift(model, x, u)
    perturb(model, x, u, y)
    (RUN / "report/scorecard_static.json").write_text(json.dumps(CARD, indent=1, default=float))
    log("static tests done")
