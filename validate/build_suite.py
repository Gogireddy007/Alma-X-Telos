"""Build every evaluation input. Inputs (contexts, action plans) go to harness/inputs_*.npz;
ground truth goes to harness/truth_*.npz, which only grade.py may open (and it logs when it does)."""

from __future__ import annotations

import numpy as np

from common import CTX, HZ, RUN, log, manifest_add

import simulators as S

DELTAS = np.array([-1.0, -0.5, 0.5, 1.0])  # counterfactual shifts of u0 (in context sd of u0)
N_MC = 64


def clip_kind(kind: str, u0: np.ndarray) -> np.ndarray:
    return np.clip(u0, 0, None) if kind in ("thermal", "inventory", "tanks") else u0


def mc_future(p, u, nz, rng, n=N_MC):
    """True predictive distribution of the future: resample future noise, keep the past."""
    out = np.empty((n, HZ))
    for i in range(n):
        z = {k: v.copy() for k, v in nz.items()}
        z["proc"][CTX:] = rng.normal(0, 1, len(z["proc"]) - CTX)
        z["obs"][CTX:] = rng.normal(0, 1, len(z["obs"]) - CTX)
        out[i] = S.simulate(p, u, z)[CTX:]
    return out


def build_sim(name: str, seeds: range, kinds: tuple, with_mc: bool = True, plan: bool = False) -> None:
    T = CTX + HZ
    X, U, KIND, PAR = [], [], [], []
    Y, Ycf, Ucf, MCq, PARAMS = [], [], [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        kind = kinds[sd % len(kinds)]
        p = S.sample_system(rng, kind)
        nz = S.sample_noise(rng, T)
        u = S.default_actions(rng, p, T)
        y = S.simulate(p, u, nz)
        u0sd = float(np.nanstd(u[:CTX, 0])) or 1.0
        cfs, ucfs = [], []
        for d in DELTAS:
            u2 = u.copy()
            u2[CTX:, 0] = clip_kind(kind, u2[CTX:, 0] + d * u0sd)
            ucfs.append(u2)
            cfs.append(S.simulate(p, u2, nz)[CTX:])
        X.append(y[:CTX])
        U.append(u)
        KIND.append(kind)
        Y.append(y[CTX:])
        Ycf.append(np.stack(cfs))
        Ucf.append(np.stack(ucfs))
        if with_mc:
            MCq.append(np.quantile(mc_future(p, u, nz, rng), np.arange(1, 10) / 10, axis=0).T)
        PARAMS.append(p)
    np.savez_compressed(RUN / f"harness/inputs_{name}.npz", x=np.stack(X), u=np.stack(U), u_cf=np.stack(Ucf),
                        kind=np.array(KIND), deltas=DELTAS)
    np.savez_compressed(RUN / f"harness/truth_{name}.npz", y=np.stack(Y), y_cf=np.stack(Ycf),
                        oracle_q=np.stack(MCq) if MCq else np.zeros(0), params=np.array(PARAMS, dtype=object))
    for f in ("inputs", "truth"):
        manifest_add("harness", RUN / f"harness/{f}_{name}.npz")
    log(f"built {name}: {len(X)} episodes, kinds={sorted(set(KIND))}")


def build_planning(name: str, seeds: range, kinds: tuple, n_cand: int = 16) -> None:
    """Setpoint tracking: choose a constant u0 for the 256 future steps to bring y to ctx mean + 1 sd.
    Inputs: context + candidate plans. Truth: the real outcome of every candidate (same noise)."""
    T = CTX + HZ
    X, Uc, KIND, TGT, Yc = [], [], [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        kind = kinds[sd % len(kinds)]
        p = S.sample_system(rng, kind)
        nz = S.sample_noise(rng, T)
        u = S.default_actions(rng, p, T)
        y = S.simulate(p, u, nz)
        u0 = u[:CTX, 0]
        lv = np.linspace(np.nanmin(u0) - np.nanstd(u0), np.nanmax(u0) + np.nanstd(u0), n_cand - 1)
        lv = np.concatenate([[u0[-1]], lv])  # candidate 0 = default plan: hold the last action
        cands, outs = [], []
        for v in lv:
            u2 = u.copy()
            u2[CTX:, 0] = clip_kind(kind, np.full(HZ, v))
            cands.append(u2)
            outs.append(S.simulate(p, u2, nz)[CTX:])
        X.append(y[:CTX])
        Uc.append(np.stack(cands))
        KIND.append(kind)
        TGT.append(y[:CTX].mean() + y[:CTX].std())
        Yc.append(np.stack(outs))
    np.savez_compressed(RUN / f"harness/inputs_{name}.npz", x=np.stack(X), u_cand=np.stack(Uc), kind=np.array(KIND),
                        target=np.array(TGT))
    np.savez_compressed(RUN / f"harness/truth_{name}.npz", y_cand=np.stack(Yc))
    for f in ("inputs", "truth"):
        manifest_add("harness", RUN / f"harness/{f}_{name}.npz")
    log(f"built {name}: {len(X)} episodes")


def build_real(per_task: int = 300, seed: int = 777) -> None:
    """Sealed GIFT-Eval windows: not in the seed-0 development sample, tasks with > 1000 windows."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    from gift_eval.data import Dataset
    from gluonts.time_feature import get_seasonality

    from eval_gift import TASKS

    C, L, META = [], [], []
    for name, term in TASKS:
        to_uni = Dataset(name=name, term=term, to_univariate=False).target_dim != 1
        ds = Dataset(name=name, term=term, to_univariate=to_uni)
        pairs = list(ds.test_data)
        if len(pairs) <= 1000:
            continue
        dev = set(np.random.default_rng(0).choice(np.arange(len(pairs)), size=1000, replace=False).tolist())
        rest = np.array([i for i in range(len(pairs)) if i not in dev])
        pick = np.sort(np.random.default_rng(seed).choice(rest, size=min(per_task, len(rest)), replace=False))
        H, season = ds.prediction_length, get_seasonality(ds.freq)
        for i in pick:
            inp, lab = pairs[i]
            C.append(np.asarray(inp["target"], np.float32))
            L.append(np.asarray(lab["target"], np.float32)[:H])
            META.append((f"{name}/{term}", H, season))
        log(f"real holdout {name}/{term}: {len(pick)} unseen windows (of {len(pairs)})")
    np.savez_compressed(RUN / "harness/inputs_real.npz", contexts=np.array(C, dtype=object),
                        meta=np.array(META, dtype=object))
    np.savez_compressed(RUN / "harness/truth_real.npz", labels=np.array(L, dtype=object))
    for f in ("inputs", "truth"):
        manifest_add("harness", RUN / f"harness/{f}_real.npz")


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["sim", "real"]
    if "sim" in which:
        build_sim("sim_val", range(500000, 500200), S.TRAIN_KINDS, with_mc=False)
        build_sim("sim_test", range(900000, 900600), S.TRAIN_KINDS)
        build_sim("sim_tanks", range(910000, 910200), S.HELDOUT_KINDS)
        build_planning("plan_test", range(920000, 920100), ("thermal", "fopdt", "lss", "pricing"))
    if "real" in which:
        build_real()
