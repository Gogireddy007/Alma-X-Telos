"""Training stream: real GiftEvalPretrain windows + TSMixup + synthetic series, generated on the fly by CPU workers.

Every sample is a fixed-length window (WINDOW steps) with an observation mask, affinely pre-scaled in float64
using only its first PRESCALE_N observed points (so no future information enters the scaling).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

WINDOW = 2560
PATCH = 16
PRESCALE_N = 512

COMMON_PERIODS = [4, 5, 6, 7, 12, 24, 48, 52, 60, 96, 144, 168, 288, 336, 365, 672, 1008]


# ------------------------------------------------------------------------------------------------
# Synthetic generators
# ------------------------------------------------------------------------------------------------


def _kernel(kind: str, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = t[:, None] - t[None, :]
    if kind == "rbf":
        ls = rng.uniform(0.02, 0.5)
        return np.exp(-0.5 * (d / ls) ** 2)
    if kind == "rq":
        ls, a = rng.uniform(0.02, 0.5), rng.uniform(0.5, 5)
        return (1 + d**2 / (2 * a * ls**2)) ** (-a)
    if kind == "per":
        p = rng.uniform(0.01, 0.5)
        ls = rng.uniform(0.3, 2.0)
        return np.exp(-2 * np.sin(np.pi * np.abs(d) / p) ** 2 / ls**2)
    if kind == "lin":
        c = rng.uniform(-0.5, 1.5)
        return (t[:, None] - c) * (t[None, :] - c)
    if kind == "white":
        return np.eye(len(t)) * rng.uniform(0.01, 0.2)
    return np.ones((len(t), len(t))) * rng.uniform(0.1, 1.0)  # constant


def kernel_synth(n: int, rng: np.random.Generator) -> np.ndarray:
    """KernelSynth-style GP sample: random sum/product of 1-4 kernels on a coarse grid, linearly upsampled."""
    g = min(n, int(rng.choice([64, 128, 192, 256])))
    t = np.linspace(0, 1, g)
    kinds = ["rbf", "rq", "per", "per", "per", "lin", "white", "const"]
    K = _kernel(rng.choice(kinds), t, rng)
    for _ in range(int(rng.integers(0, 4))):
        K2 = _kernel(rng.choice(kinds), t, rng)
        K = K + K2 if rng.random() < 0.6 else K * K2
    K += 1e-6 * np.eye(g)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(K)
        L = V * np.sqrt(np.clip(w, 0, None))
    y = L @ rng.standard_normal(g)
    return np.interp(np.linspace(0, g - 1, n), np.arange(g), y)


def _ar_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    p = int(rng.integers(1, 4))
    phi = rng.uniform(-0.5, 0.95, p)
    phi *= 0.98 / max(1.0, np.abs(phi).sum())
    df = float(rng.choice([2.5, 4.0, 10.0, 1e6]))
    eps = rng.standard_t(df, n) if df < 1e5 else rng.standard_normal(n)
    from scipy.signal import lfilter

    return lfilter([1.0], np.concatenate([[1.0], -phi]), eps)


def _seasonal(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    for _ in range(int(rng.integers(1, 4))):
        per = float(rng.choice(COMMON_PERIODS)) if rng.random() < 0.7 else rng.uniform(2.5, 500)
        per = min(per, n / 2)
        k = int(rng.integers(1, 6))
        amp = rng.lognormal(0, 0.7)
        s = np.zeros(n)
        for h in range(1, k + 1):
            a = rng.standard_normal() / h ** rng.uniform(0.5, 2)
            s += a * np.sin(2 * np.pi * h * t / per + rng.uniform(0, 2 * np.pi))
        if rng.random() < 0.3:  # slowly varying amplitude
            s *= 1 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(n / 4, 4 * n) + rng.uniform(0, 6.3))
        out += amp * s
    return out


def _trend(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    kind = rng.choice(["none", "lin", "pw", "logistic", "exp", "rw"], p=[0.2, 0.25, 0.2, 0.1, 0.1, 0.15])
    if kind == "none":
        return np.zeros(n)
    if kind == "lin":
        return rng.normal(0, 3) * t
    if kind == "pw":
        cps = np.sort(rng.uniform(0, 1, int(rng.integers(1, 5))))
        slopes = rng.normal(0, 3, len(cps) + 1)
        seg = np.searchsorted(cps, t)
        d = np.diff(np.concatenate([[0], cps, [1]]))
        base = np.concatenate([[0], np.cumsum(slopes[:-1] * d[:-1])])
        starts = np.concatenate([[0], cps])
        return base[seg] + slopes[seg] * (t - starts[seg])
    if kind == "logistic":
        return rng.uniform(1, 6) / (1 + np.exp(-rng.uniform(5, 30) * (t - rng.uniform(0.2, 0.8))))
    if kind == "exp":
        return np.expm1(rng.uniform(-2, 3) * t)
    return np.cumsum(rng.standard_normal(n)) * rng.uniform(0.02, 0.3)


def structural_synth(n: int, rng: np.random.Generator) -> np.ndarray:
    """Trend + multi-seasonality + AR / heavy-tailed noise + regime events; optional intermittent / count."""
    if rng.random() < 0.12:  # intermittent demand
        p = rng.uniform(0.02, 0.6)
        lam = rng.lognormal(1, 1)
        season = 1 + 0.5 * _seasonal(n, rng) / 3 if rng.random() < 0.5 else 1.0
        occ = rng.random(n) < np.clip(p * season, 0, 1)
        return occ * rng.poisson(lam, n).clip(min=1).astype(np.float64)
    x = _trend(n, rng) * rng.lognormal(0, 0.5)
    if rng.random() < 0.8:
        x = x + _seasonal(n, rng)
    noise = _ar_noise(n, rng) * rng.uniform(0.05, 0.8)
    if rng.random() < 0.2:  # heteroscedastic
        noise *= np.exp(0.5 * np.sin(2 * np.pi * np.arange(n) / rng.uniform(20, n)))
    x = x + noise
    if rng.random() < 0.2:  # level shifts
        for _ in range(int(rng.integers(1, 4))):
            x[int(rng.integers(0, n)) :] += rng.normal(0, 2)
    if rng.random() < 0.15:  # spikes
        k = int(rng.integers(1, max(2, n // 100)))
        x[rng.integers(0, n, k)] += rng.normal(0, 6, k)
    if rng.random() < 0.3:  # multiplicative / positive series
        x = np.exp(0.4 * (x - x.mean()) / (x.std() + 1e-8))
        if rng.random() < 0.3:
            x = np.round(x * rng.lognormal(2, 1))
    elif rng.random() < 0.15:
        x = np.clip(x, 0, None)
    return x


def ets_synth(n: int, rng: np.random.Generator) -> np.ndarray:
    """Short low-frequency series from an ETS-style state-space model (yearly/quarterly/monthly-like).

    Level + (damped) trend + optional additive/multiplicative season of period 4 or 12, with
    error-driven state updates, occasional level shifts and a positive, multiplicative variant.
    """
    m = int(rng.choice([1, 1, 4, 12, 12]))
    a = rng.uniform(0.05, 0.9)                      # level smoothing
    b = rng.uniform(0.0, 0.3) * a                   # trend smoothing
    g = rng.uniform(0.0, 0.3) * (1 - a)             # season smoothing
    phi = 1.0 if rng.random() < 0.4 else rng.uniform(0.8, 0.99)
    sig = rng.lognormal(np.log(0.03), 0.8)          # relative noise
    lvl = 1.0
    tr = rng.normal(0.02, 0.05) if rng.random() < 0.85 else 0.0
    mult = rng.random() < 0.5
    s = np.zeros(max(m, 1))
    if m > 1:
        s = rng.normal(0, rng.uniform(0.02, 0.3), m)
        s -= s.mean()
    burn = int(rng.integers(0, 20))
    out = np.empty(n + burn)
    for t in range(n + burn):
        si = s[t % m] if m > 1 else 0.0
        e = rng.standard_t(5) * sig if rng.random() < 0.3 else rng.standard_normal() * sig
        base = lvl + phi * tr
        y = base * (1 + si) * (1 + e) if mult else base + si + e
        out[t] = y
        lvl = base + a * (base * e if mult else e)
        tr = phi * tr + b * (base * e if mult else e)
        if m > 1:
            s[t % m] = si + g * e
        if rng.random() < 0.01:
            lvl += rng.normal(0, 0.1) * abs(lvl)
    x = out[burn:]
    if rng.random() < 0.5:
        x = x * rng.lognormal(3, 2)
    if rng.random() < 0.1:
        x = np.round(x)
    return x


# ------------------------------------------------------------------------------------------------
# Real data
# ------------------------------------------------------------------------------------------------


def freq_bucket(f: str | None) -> str:
    f = (f or "").upper()
    if f.startswith(("A", "Y")):
        return "yearly"
    if f.startswith("Q"):
        return "quarterly"
    if f.startswith("M") and not f.startswith("MIN"):
        return "monthly"
    if f.startswith("W"):
        return "weekly"
    if f.startswith("D"):
        return "daily"
    if f.startswith("H") or f.endswith("H"):
        return "hourly"
    return "subhourly"


class RealStore:
    """sampler='sqrt': source p ~ sqrt(points) (v1-v3 behaviour).
    sampler='bucket': frequency bucket p ~ (series in bucket)^alpha, then source p ~ sqrt(points) inside it.
    This lifts yearly/quarterly/monthly series from ~1% to ~12% of real samples (alpha=0.5)."""

    def __init__(self, root: str, sampler: str = "sqrt", alpha: float = 0.5) -> None:
        # root may be "a,b,...": stores are merged by source name (same source in two stores = one source)
        self.vals, idx, self.sources, stats = [], [], [], {}
        for k, r in enumerate(Path(x) for x in root.split(",")):
            self.vals.append(np.memmap(r / "values.f32", dtype=np.float32, mode="r"))
            ix = np.load(r / "index.npy").reshape(-1, 3)
            meta = json.load(open(r / "meta.json"))
            remap = []
            for s in meta["sources"]:
                if s not in self.sources:
                    self.sources.append(s)
                remap.append(self.sources.index(s))
                st = meta["stats"].get(s)
                if st and (s not in stats or not stats[s].get("freq")):
                    stats[s] = st
            remap = np.asarray(remap, dtype=np.int64)
            idx.append(np.column_stack([ix[:, 0], ix[:, 1], remap[ix[:, 2]] if len(ix) else ix[:, 2],
                                        np.full(len(ix), k, dtype=np.int64)]))
        self.index = np.concatenate(idx)
        meta = {"stats": stats}
        sid = self.index[:, 2]
        self.by_src = [np.where(sid == s)[0] for s in range(len(self.sources))]
        pts = np.array([self.index[ix, 1].sum() if len(ix) else 0 for ix in self.by_src], dtype=np.float64)
        w = np.sqrt(pts)
        if sampler == "bucket":
            stats = meta["stats"]
            bk = [freq_bucket(stats[s]["freq"]) if s in stats else "subhourly" for s in self.sources]
            n_ser = np.array([len(ix) for ix in self.by_src], dtype=np.float64)
            tot = {}
            for b, n in zip(bk, n_ser):
                tot[b] = tot.get(b, 0.0) + n
            bw = {b: v**alpha for b, v in tot.items() if v > 0}
            z = sum(bw.values())
            inb = {b: sum(wi for wi, bi in zip(w, bk) if bi == b) for b in bw}
            w = np.array([bw[b] / z * wi / inb[b] if b in bw and inb[b] > 0 else 0.0 for wi, b in zip(w, bk)])
            self.bucket_p = {b: bw[b] / z for b in bw}
        self.src_p = w / w.sum()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        s = rng.choice(len(self.src_p), p=self.src_p)
        i = self.by_src[s][rng.integers(len(self.by_src[s]))]
        off, ln, _, k = self.index[i]
        v = self.vals[k]
        if ln > n:
            st = int(rng.integers(0, ln - n + 1))
            return np.asarray(v[off + st : off + st + n], dtype=np.float64)
        return np.asarray(v[off : off + ln], dtype=np.float64)


# ------------------------------------------------------------------------------------------------
# Stream
# ------------------------------------------------------------------------------------------------


def _std_obs(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan_to_num(x)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def _downsample(x: np.ndarray, k: int) -> np.ndarray:
    n = len(x) // k * k
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x[:n].reshape(-1, k), axis=1)


def prescale(x: np.ndarray, m: np.ndarray, n_ref: int = PRESCALE_N) -> tuple[np.ndarray, float, float]:
    obs = x[m]
    ref = obs[:n_ref] if len(obs) else np.zeros(1)
    loc = float(ref.mean())
    sc = float(ref.std())
    if not np.isfinite(sc) or sc < 1e-8 * (1 + abs(loc)):
        sc = float(np.abs(obs).mean()) if len(obs) else 1.0
        sc = sc if sc > 0 else 1.0
    z = np.clip((x - loc) / sc, -1e6, 1e6)
    return np.where(m, z, 0.0), loc, sc


class TrainStream(IterableDataset):
    def __init__(self, real_root: str | None, seed: int = 0, p_real: float = 0.55, p_mix: float = 0.15,
                 window: int = WINDOW, sampler: str = "sqrt", alpha: float = 0.5, p_short: float = 0.0) -> None:
        self.real_root, self.seed, self.p_real, self.p_mix = real_root, seed, p_real, p_mix
        self.p_short = p_short
        self.window, self.sampler, self.alpha = window, sampler, alpha

    def _raw(self, rng: np.random.Generator, real: RealStore | None) -> np.ndarray:
        if self.p_short > 0 and rng.random() < self.p_short:
            return ets_synth(int(np.exp(rng.uniform(np.log(12), np.log(320)))), rng)
        u = rng.random()
        if real is not None and u < self.p_real:
            x = real.sample(rng, self.window * 4 if rng.random() < 0.2 else self.window)
            if len(x) > self.window:  # sampled long for aggregation
                k = int(rng.choice([2, 3, 4]))
                x = _downsample(x, k)[-self.window:]
            return x
        if real is not None and u < self.p_real + self.p_mix:
            k = int(rng.integers(2, 4))
            n = self.window if rng.random() < 0.7 else int(rng.integers(128, self.window))
            parts = [real.sample(rng, n) for _ in range(k)]
            n = min(len(p) for p in parts)
            w = rng.dirichlet(np.ones(k))
            return sum(wi * _std_obs(p[-n:]) for wi, p in zip(w, parts))
        n = self.window if rng.random() < 0.6 else int(rng.integers(64, self.window))
        return kernel_synth(n, rng) if rng.random() < 0.4 else structural_synth(n, rng)

    def sample(self, rng: np.random.Generator, real: RealStore | None) -> tuple[np.ndarray, np.ndarray]:
        x = self._raw(rng, real)
        x = x[: self.window - PATCH]
        m = np.isfinite(x)
        if rng.random() < 0.1 and len(x) > 64:  # synthetic missing blocks
            for _ in range(int(rng.integers(1, 5))):
                s = int(rng.integers(0, len(x)))
                m[s : s + int(rng.integers(1, 64))] = False
        lead = int(rng.integers(0, PATCH))  # vary patch alignment
        xs = np.zeros(self.window)
        ms = np.zeros(self.window, dtype=bool)
        xs[lead : lead + len(x)] = np.where(m, x, 0.0)
        ms[lead : lead + len(x)] = m
        z, _, _ = prescale(xs, ms)
        return z.astype(np.float32), ms

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed()) % (2**31)])
        real = (RealStore(self.real_root, self.sampler, self.alpha)
                if self.real_root and all(Path(r, "index.npy").exists() for r in self.real_root.split(",")) else None)
        while True:
            x, m = self.sample(rng, real)
            yield torch.from_numpy(x), torch.from_numpy(m)
