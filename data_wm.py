"""Training stream for Alma-WM: (x, m, u, um).

Mix: real GiftEvalPretrain windows / TSMixup / plain synthetic (no actions, um = 0) as in Alma v2, plus
controlled systems from simulators.py (TRAIN_KINDS only) with their true actions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from data import PATCH, PRESCALE_N, RealStore, TrainStream, prescale
from simulators import A, episode


def prescale_actions(u: np.ndarray, um: np.ndarray, n_ref: int = PRESCALE_N) -> np.ndarray:
    """Per channel: centre/scale by the first n_ref present values (context only, never the future)."""
    out = np.zeros_like(u)
    for j in range(u.shape[1]):
        v = u[um[:, j], j]
        if not len(v):
            continue
        ref = v[:n_ref]
        loc = float(np.median(ref))
        sc = float(ref.std())
        sc = max(sc, 0.1 * abs(loc), 1e-3) if sc > 1e-9 else max(abs(loc), 1.0)
        out[:, j] = np.where(um[:, j], (u[:, j] - loc) / sc, 0.0)
    return out


class WMStream(IterableDataset):
    def __init__(self, real_root: str | None, seed: int = 0, window: int = 4096, p_ctrl: float = 0.35) -> None:
        self.real_root, self.seed, self.window, self.p_ctrl = real_root, seed, window, p_ctrl
        self.base = TrainStream(real_root, seed=seed, window=window)

    def sample(self, rng: np.random.Generator, real: RealStore | None):
        W = self.window
        if rng.random() >= self.p_ctrl:
            x, m = self.base.sample(rng, real)
            return x, m, np.zeros((W, A), np.float32), np.zeros((W, A), bool)
        n = W - PATCH if rng.random() < 0.7 else int(rng.integers(512, W - PATCH))
        y, u, _, _ = episode(rng, n)
        um = np.isfinite(u)
        if rng.random() < 0.1:  # sometimes hide an action channel: model must marginalise it
            um[:, int(rng.integers(0, A))] = False
        m = np.isfinite(y)
        if rng.random() < 0.1:
            for _ in range(int(rng.integers(1, 5))):
                s = int(rng.integers(0, n))
                m[s : s + int(rng.integers(1, 64))] = False
        lead = int(rng.integers(0, PATCH))
        xs, ms = np.zeros(W), np.zeros(W, bool)
        us, ums = np.zeros((W, A)), np.zeros((W, A), bool)
        xs[lead : lead + n] = np.where(m, y, 0.0)
        ms[lead : lead + n] = m
        us[lead : lead + n] = np.nan_to_num(u)
        ums[lead : lead + n] = um
        z, _, _ = prescale(xs, ms)
        return z.astype(np.float32), ms, prescale_actions(us, ums).astype(np.float32), ums

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = np.random.default_rng([self.seed, wid, int(torch.initial_seed()) % (2**31)])
        real = RealStore(self.real_root) if self.real_root and Path(self.real_root, "index.npy").exists() else None
        while True:
            x, m, u, um = self.sample(rng, real)
            yield torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(u), torch.from_numpy(um)
