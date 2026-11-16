"""Generate and SEAL every prediction. Reads only harness/inputs_*.npz - never truth.

python validate/predict.py --ckpt runs/alma_wm1/ema_8000.pt --suites sim_val sim_test sim_tanks plan_test real
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import CTX, HZ, PATCH, QS, RUN, log, seal

from data import prescale
from data_wm import prescale_actions
from forecast import AlmaForecaster
from model_wm import load_wm

DEV = "cpu"


# ------------------------------------------------------------------------------------------------
# Alma-WM wrappers
# ------------------------------------------------------------------------------------------------


def wm_rollout(model, contexts: list, H: int, u_full: np.ndarray | None = None, bs: int = 64,
               return_state: bool = False):
    """contexts: list of raw 1-D arrays; u_full: (B, len(ctx)+H, 2) raw actions (NaN = absent) or None.
    Uses the last CTX steps of each context. Returns quantiles (B, H, 9) in data units [, states]."""
    nf = -(-H // PATCH)
    F = nf * PATCH
    outs, states = [], []
    for s in range(0, len(contexts), bs):
        X, M, U, UM, LS = [], [], [], [], []
        for j, c in enumerate(contexts[s : s + bs]):
            c = np.asarray(c, np.float64)[-CTX:]
            xs, ms = np.zeros(CTX), np.zeros(CTX, bool)
            xs[CTX - len(c):] = np.nan_to_num(c)
            ms[CTX - len(c):] = np.isfinite(c)
            z, loc, sc = prescale(xs, ms)
            us, ums = np.zeros((CTX + F, 2)), np.zeros((CTX + F, 2), bool)
            if u_full is not None:
                uf = u_full[s + j]
                L = len(np.asarray(contexts[s + j]))
                ctx_u = uf[:L][-CTX:]
                fut_u = uf[L : L + H]
                us[CTX - len(ctx_u) : CTX] = np.nan_to_num(ctx_u)
                ums[CTX - len(ctx_u) : CTX] = np.isfinite(ctx_u)
                us[CTX : CTX + len(fut_u)] = np.nan_to_num(fut_u)
                ums[CTX : CTX + len(fut_u)] = np.isfinite(fut_u)
                if len(fut_u) < F:  # hold the last planned action to fill the final patch
                    us[CTX + len(fut_u):] = us[CTX + len(fut_u) - 1]
                    ums[CTX + len(fut_u):] = ums[CTX + len(fut_u) - 1]
            X.append(z)
            M.append(ms)
            U.append(prescale_actions(us, ums))
            UM.append(ums)
            LS.append((loc, sc))
        t = lambda a, dt=torch.float32: torch.tensor(np.stack(a), dtype=dt)
        x, m, u, um = t(X), t(M, torch.bool), t(U), t(UM, torch.bool)
        q = model.forecast(x, m, u, um, nf).numpy()[:, :H]
        ls = np.array(LS)
        outs.append(q * ls[:, 1, None, None] + ls[:, 0, None, None])
        if return_state:
            h, _, _ = model.enc.encode(x, m, u[:, :CTX], um[:, :CTX])
            states.append(h[:, -1].numpy())
    q = np.sort(np.concatenate(outs), -1)
    return (q, np.concatenate(states)) if return_state else q


def direct(fc: AlmaForecaster, contexts: list, H: int) -> np.ndarray:
    return np.sort(fc.predict([np.asarray(c, np.float32) for c in contexts], H, list(QS)), -1)


# ------------------------------------------------------------------------------------------------
# Simple laws
# ------------------------------------------------------------------------------------------------


def persistence(contexts, H):
    out = []
    for c in contexts:
        c = np.asarray(c, np.float64)
        v = c[np.isfinite(c)][-1] if np.isfinite(c).any() else 0.0
        out.append(np.full((H, 9), v))
    return np.stack(out)


def snaive(contexts, H, m):
    from eval_gift import seasonal_naive

    return np.repeat(seasonal_naive([np.asarray(c, np.float32) for c in contexts], H, m)[..., None], 9, -1)


def arx(y: np.ndarray, u: np.ndarray, H: int, plans: list[np.ndarray], p: int = 24, n_boot: int = 32,
        seed: int = 0) -> list[np.ndarray]:
    """ARX(p, p) by ridge least squares on the context; residual-bootstrap rollouts under each future plan.
    y: (CTX,), u: (CTX, 2) context actions, plans: list of (H, 2) future actions. Returns [(H, 9)] per plan."""
    mu, sd = y.mean(), y.std() + 1e-8
    z = (y - mu) / sd
    use = [j for j in range(u.shape[1]) if np.isfinite(u[:, j]).all()]
    um = {j: (np.nanmean(u[:, j]), np.nanstd(u[:, j]) + 1e-8) for j in use}
    U = np.stack([(u[:, j] - um[j][0]) / um[j][1] for j in use], 1) if use else np.zeros((len(y), 0))
    rows = []
    for t in range(p, len(z)):
        rows.append(np.concatenate([z[t - p : t][::-1], U[t - p : t][::-1].ravel(), [1.0]]))
    A = np.array(rows)
    b = z[p:]
    lam = 1e-3 * len(b)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    res = b - A @ w
    out = []
    rng0 = np.random.default_rng(seed)
    eps = res[rng0.integers(0, len(res), (n_boot, H))]
    for plan in plans:
        Uf = np.stack([(plan[:, j] - um[j][0]) / um[j][1] for j in use], 1) if use else np.zeros((H, 0))
        Uall = np.concatenate([U, Uf])
        paths = np.empty((n_boot, H))
        zz = np.tile(np.concatenate([z, np.zeros(H)]), (n_boot, 1))
        for h in range(H):
            t = len(z) + h
            feat = np.concatenate([Uall[t - p : t][::-1].ravel(), [1.0]])
            v = zz[:, t - p : t][:, ::-1] @ w[:p] + feat @ w[p:] + eps[:, h]
            v = np.clip(v, -50, 50)
            zz[:, t] = v
            paths[:, h] = v
        out.append(np.quantile(paths * sd + mu, QS, axis=0).T)
    return out


# ------------------------------------------------------------------------------------------------
# Suites
# ------------------------------------------------------------------------------------------------


def run_sim(model, v2: AlmaForecaster, name: str) -> None:
    z = np.load(RUN / f"harness/inputs_{name}.npz", allow_pickle=True)
    x, u, ucf = z["x"], z["u"], z["u_cf"]
    n = len(x)
    ctxs = list(x)
    t0 = time.time()
    rng = np.random.default_rng(12345)
    res = {}
    res["wm_aware"], res["state"] = wm_rollout(model, ctxs, HZ, u, return_state=True)
    blind = np.full_like(u, np.nan)
    res["wm_blind"] = wm_rollout(model, ctxs, HZ, blind)
    shuf = u[np.roll(np.arange(n), 1)]
    res["wm_shuffled"] = wm_rollout(model, ctxs, HZ, shuf)
    import simulators as S

    rnd = u.copy()
    for i in range(n):
        if np.isfinite(rnd[i, :, 0]).all():
            r = S.sample_actions(rng, HZ) * np.nanstd(u[i, :CTX, 0]) + np.nanmean(u[i, :CTX, 0])
            rnd[i, CTX:, 0] = r
    res["wm_random"] = wm_rollout(model, ctxs, HZ, rnd)
    res["wm_cf"] = np.stack([wm_rollout(model, ctxs, HZ, ucf[:, k]) for k in range(ucf.shape[1])], 1)
    fc = AlmaForecaster(model.enc, DEV, context_len=CTX, symmetric=True)
    res["wm_direct"] = direct(fc, ctxs, HZ)
    res["alma_v2"] = direct(v2, ctxs, HZ)
    res["persistence"] = persistence(ctxs, HZ)
    res["snaive24"] = snaive(ctxs, HZ, 24)
    log(f"{name}: neural predictions done in {time.time() - t0:.0f}s")
    t0 = time.time()
    A, ACF = [], []
    for i in range(n):
        plans = [u[i, CTX:]] + [ucf[i, k, CTX:] for k in range(ucf.shape[1])]
        r = arx(x[i], u[i, :CTX], HZ, plans, seed=i)
        A.append(r[0])
        ACF.append(np.stack(r[1:]))
    res["arx"], res["arx_cf"] = np.stack(A), np.stack(ACF)
    log(f"{name}: ARX done in {time.time() - t0:.0f}s")
    bad = {k: int((~np.isfinite(v)).any(axis=tuple(range(1, v.ndim))).sum()) for k, v in res.items()}
    seal(f"pred_{name}", **res, nonfinite=np.array(list(bad.items()), dtype=object))
    log(f"sealed pred_{name}; non-finite rows per model: {bad}")


def run_plan(model, name: str = "plan_test") -> None:
    z = np.load(RUN / f"harness/inputs_{name}.npz", allow_pickle=True)
    x, uc = z["x"], z["u_cand"]
    n, C = uc.shape[:2]
    ctxs = list(x)
    wm = np.stack([wm_rollout(model, ctxs, HZ, uc[:, k]) for k in range(C)], 1)  # (n, C, HZ, 9)
    ar = np.stack([np.stack(arx(x[i], uc[i, 0, :CTX], HZ, [uc[i, k, CTX:] for k in range(C)], seed=i))
                   for i in range(n)])
    seal(f"pred_{name}", wm=wm, arx=ar)
    log(f"sealed pred_{name}")


def run_real(model, v2: AlmaForecaster, telos: bool = True) -> None:
    z = np.load(RUN / "harness/inputs_real.npz", allow_pickle=True)
    C, meta = list(z["contexts"]), z["meta"]
    tasks = sorted(set(m[0] for m in meta), key=lambda s: [m[0] for m in meta].index(s))
    fc = AlmaForecaster(model.enc, DEV, context_len=CTX, symmetric=True)
    tel = None
    if telos:
        from eval_gift import make_telos

        tel = make_telos("cpu")
    out = {k: [] for k in ("wm_direct", "wm_rollout", "alma_v2", "telos", "snaive", "persistence")}
    idx_all = []
    t0 = time.time()
    for tk in tasks:
        idx = [i for i, m in enumerate(meta) if m[0] == tk]
        H, season = int(meta[idx[0]][1]), int(meta[idx[0]][2])
        ctx = [C[i] for i in idx]
        out["wm_direct"].append(direct(fc, ctx, H))
        out["wm_rollout"].append(wm_rollout(model, ctx, H))
        out["alma_v2"].append(direct(v2, ctx, H))
        out["telos"].append(np.sort(tel(ctx, H), -1) if tel else np.full((len(ctx), H, 9), np.nan))
        out["snaive"].append(snaive(ctx, H, season))
        out["persistence"].append(persistence(ctx, H))
        idx_all.append(np.array(idx))
        log(f"real {tk}: {len(idx)} windows ({time.time() - t0:.0f}s)")
    def objarr(lst):
        a = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            a[i] = v
        return a

    arrays = {k: objarr(v) for k, v in out.items()}
    seal("pred_real", tasks=np.array(tasks), idx=objarr(idx_all), **arrays)
    log("sealed pred_real")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suites", nargs="+", default=["sim_val", "sim_test", "sim_tanks", "plan_test", "real"])
    ap.add_argument("--no-telos", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = load_wm(a.ckpt, DEV)
    v2 = AlmaForecaster.load("runs/alma_v2/ema_6000.pt", device=DEV, context_len=CTX, symmetric=True)
    log(f"predict: ckpt={a.ckpt} suites={a.suites}")
    from common import manifest_add
    from pathlib import Path

    manifest_add("weights", Path(a.ckpt).resolve())
    for s in a.suites:
        if s.startswith("sim"):
            run_sim(model, v2, s)
        elif s.startswith("plan"):
            run_plan(model, s)
        elif s == "real":
            run_real(model, v2, telos=not a.no_telos)
