"""Controlled dynamical systems with known actions and exact counterfactuals.

Every system is a deterministic function  y = simulate(kind, params, u, noise)  of
  * params  - drawn once per episode by sample_system(),
  * u       - (T, 2) action array: channel 0 = controllable input, channel 1 = exogenous driver
              (NaN marks a channel the system does not use),
  * noise   - drawn once per episode by sample_noise().
Re-running with the SAME params and noise but a DIFFERENT u gives the true counterfactual trajectory,
which is what Gate 6 grades against.

TRAIN_KINDS are used in training. HELDOUT_KINDS are never seen in training (transfer test, Gate 9.7).
"""

from __future__ import annotations

import numpy as np

A = 2  # action channels
TRAIN_KINDS = ("lss", "thermal", "inventory", "pricing", "fopdt")
HELDOUT_KINDS = ("tanks",)


# ------------------------------------------------------------------------------------------------
# Action sequences
# ------------------------------------------------------------------------------------------------


def sample_actions(rng: np.random.Generator, T: int, kind: str = "any") -> np.ndarray:
    """One action channel of length T, roughly O(1) scale."""
    kind = kind if kind != "any" else rng.choice(["piecewise", "prbs", "steps", "sine", "pulses", "ramp", "ou"])
    t = np.arange(T)
    if kind == "piecewise":
        n = int(rng.integers(2, 12))
        cuts = np.sort(rng.integers(0, T, n))
        lv = rng.normal(0, 1, n + 1)
        return lv[np.searchsorted(cuts, t, side="right")]
    if kind == "prbs":
        hold = int(rng.integers(4, 96))
        bits = rng.integers(0, 2, T // hold + 2) * 2.0 - 1.0
        return np.repeat(bits, hold)[:T]
    if kind == "steps":
        s = int(rng.integers(T // 4, T))
        return np.where(t >= s, rng.normal(0, 1.5), rng.normal(0, 0.5))
    if kind == "sine":
        return np.sin(2 * np.pi * t / rng.uniform(24, 800) + rng.uniform(0, 6.3)) * rng.uniform(0.5, 1.5)
    if kind == "pulses":
        u = np.zeros(T)
        for s in rng.integers(0, T, int(rng.integers(1, 10))):
            u[s : s + int(rng.integers(1, 48))] = rng.uniform(0.5, 2.0)
        return u
    if kind == "ramp":
        s, e = sorted(rng.integers(0, T, 2))
        u = np.zeros(T)
        u[s:e] = np.linspace(0, rng.normal(0, 1.5), e - s)
        u[e:] = u[e - 1] if e > s else 0.0
        return u
    # ou
    u = np.zeros(T)
    th, sg = rng.uniform(0.005, 0.1), rng.uniform(0.05, 0.3)
    e = rng.normal(0, sg, T)
    for i in range(1, T):
        u[i] = u[i - 1] * (1 - th) + e[i]
    return u


def seasonal(rng: np.random.Generator, T: int) -> np.ndarray:
    t = np.arange(T)
    p = rng.choice([24, 48, 96, 168, 7, 12, 52])
    return np.sin(2 * np.pi * t / p + rng.uniform(0, 6.3)) + 0.3 * np.sin(4 * np.pi * t / p + rng.uniform(0, 6.3))


# ------------------------------------------------------------------------------------------------
# Systems
# ------------------------------------------------------------------------------------------------


def sample_system(rng: np.random.Generator, kind: str | None = None) -> dict:
    kind = kind or str(rng.choice(TRAIN_KINDS))
    p: dict = {"kind": kind, "obs_sd": float(rng.uniform(0.01, 0.15)), "level": float(rng.normal(0, 5)),
               "scale": float(np.exp(rng.uniform(-2, 4)))}
    if kind == "lss":
        d = int(rng.integers(1, 4))
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        eig = rng.uniform(0.6, 0.995, d)
        p.update(Am=(Q @ np.diag(eig) @ Q.T).tolist(), B=rng.normal(0, 0.3, (d, 2)).tolist(),
                 C=rng.normal(0, 1, d).tolist(), proc_sd=float(rng.uniform(0.01, 0.2)),
                 uses_exo=bool(rng.random() < 0.5))
    elif kind == "thermal":
        p.update(a=float(rng.uniform(0.005, 0.1)), b=float(rng.uniform(0.05, 1.0)),
                 proc_sd=float(rng.uniform(0.01, 0.1)))
    elif kind == "inventory":
        p.update(lag=int(rng.integers(1, 40)), demand=float(rng.uniform(0.5, 2.0)),
                 season=float(rng.uniform(0, 0.5)), proc_sd=float(rng.uniform(0.05, 0.4)))
    elif kind == "pricing":
        p.update(elast=float(rng.uniform(0.2, 2.0)), lift=float(rng.uniform(0.1, 1.5)),
                 carry=float(rng.uniform(0.0, 0.9)), season=float(rng.uniform(0, 0.6)),
                 proc_sd=float(rng.uniform(0.02, 0.2)))
    elif kind == "fopdt":
        p.update(K=float(rng.normal(0, 2)), tau=float(rng.uniform(2, 200)), delay=int(rng.integers(0, 60)),
                 gain=float(rng.uniform(0.3, 2.0)), exo=float(rng.normal(0, 0.5)),
                 proc_sd=float(rng.uniform(0.01, 0.15)), ar=float(rng.uniform(0, 0.95)))
    elif kind == "tanks":  # held out
        p.update(k1=float(rng.uniform(0.02, 0.15)), k2=float(rng.uniform(0.02, 0.15)),
                 gain=float(rng.uniform(0.05, 0.4)), proc_sd=float(rng.uniform(0.005, 0.05)))
    return p


def sample_noise(rng: np.random.Generator, T: int) -> dict:
    return {"proc": rng.normal(0, 1, T), "obs": rng.normal(0, 1, T), "seas": seasonal(rng, T),
            "x0": rng.normal(0, 1, 4)}


def uses(p: dict) -> tuple[bool, bool]:
    """Which action channels a system uses (controllable, exogenous)."""
    k = p["kind"]
    if k == "lss":
        return True, p["uses_exo"]
    if k in ("thermal", "pricing", "fopdt"):
        return True, True
    return True, False  # inventory, tanks


def default_actions(rng: np.random.Generator, p: dict, T: int) -> np.ndarray:
    u = np.full((T, A), np.nan)
    c, e = uses(p)
    k = p["kind"]
    if c:
        u[:, 0] = sample_actions(rng, T)
        if k == "thermal":
            u[:, 0] = np.clip(0.5 + 0.5 * u[:, 0], 0, 1.5)  # heater power >= 0
        if k == "inventory":
            u[:, 0] = np.clip(p["demand"] * (1 + 0.5 * u[:, 0]), 0, None)  # orders >= 0
        if k == "tanks":
            u[:, 0] = np.clip(1 + 0.5 * u[:, 0], 0, None)  # inflow >= 0
    if e:
        if k == "thermal":
            u[:, 1] = seasonal(rng, T) * rng.uniform(0.5, 2) + rng.normal(0, 0.3)  # outdoor temperature
        elif k == "pricing":
            u[:, 1] = (sample_actions(rng, T, "pulses") > 0).astype(float)  # promotions
        else:
            u[:, 1] = sample_actions(rng, T)
    return u


def simulate(p: dict, u: np.ndarray, nz: dict) -> np.ndarray:
    """Deterministic trajectory y (T,) for params p, actions u (T,2), noise nz."""
    T = len(u)
    k = p["kind"]
    e = nz["proc"][:T] * p.get("proc_sd", 0.05)
    u0 = np.nan_to_num(u[:, 0])
    u1 = np.nan_to_num(u[:, 1])
    if k == "lss":
        Am, B, C = np.array(p["Am"]), np.array(p["B"]), np.array(p["C"])
        s = nz["x0"][: len(C)] * 0.5
        y = np.empty(T)
        for t in range(T):
            y[t] = C @ s
            s = Am @ s + B @ np.array([u0[t], u1[t] if p["uses_exo"] else 0.0]) + e[t]
    elif k == "thermal":
        Tm = float(nz["x0"][0])
        y = np.empty(T)
        for t in range(T):
            y[t] = Tm
            Tm = Tm + p["a"] * (u1[t] - Tm) + p["b"] * p["a"] * 5 * u0[t] + e[t]
    elif k == "inventory":
        L = p["lag"]
        dem = np.clip(p["demand"] * (1 + p["season"] * nz["seas"][:T]) + e, 0, None)
        inv = 5.0 + float(nz["x0"][0])
        y = np.empty(T)
        for t in range(T):
            y[t] = inv
            arr = u0[t - L] if t >= L else p["demand"]
            inv = max(inv + arr - dem[t], 0.0)  # lost sales when stocked out
    elif k == "pricing":
        ad = np.zeros(T)
        for t in range(T):
            ad[t] = u1[t] + (p["carry"] * ad[t - 1] if t else 0.0)
        base = 1 + p["season"] * nz["seas"][:T]
        y = np.clip(base, 0.05, None) * np.exp(-p["elast"] * 0.3 * u0) * (1 + p["lift"] * ad) * np.exp(e)
    elif k == "fopdt":
        d = p["delay"]
        ud = np.concatenate([np.full(d, u0[0]), u0])[:T] if d else u0
        y = np.empty(T)
        v, n = float(nz["x0"][0]), 0.0
        for t in range(T):
            n = p["ar"] * n + e[t]
            y[t] = v + n
            v = v + (p["K"] * np.tanh(p["gain"] * ud[t]) + p["exo"] * u1[t] - v) / p["tau"]
    elif k == "tanks":
        h1, h2 = 1.0 + 0.2 * abs(nz["x0"][0]), 1.0 + 0.2 * abs(nz["x0"][1])
        y = np.empty(T)
        for t in range(T):
            y[t] = h2
            f1 = p["k1"] * np.sqrt(max(h1, 0.0))
            f2 = p["k2"] * np.sqrt(max(h2, 0.0))
            h1 = max(h1 + p["gain"] * u0[t] - f1 + e[t], 0.0)
            h2 = max(h2 + f1 - f2, 0.0)
    else:
        raise ValueError(k)
    y = p["level"] + p["scale"] * y
    return y + p["obs_sd"] * p["scale"] * nz["obs"][:T]  # noise does not depend on u -> exact counterfactuals


def episode(rng: np.random.Generator, T: int, kind: str | None = None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    p = sample_system(rng, kind)
    nz = sample_noise(rng, T)
    u = default_actions(rng, p, T)
    return simulate(p, u, nz), u, p, nz
