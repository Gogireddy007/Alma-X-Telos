"""Zero-shot inference wrapper for Alma checkpoints."""

from __future__ import annotations

import math

import numpy as np
import torch

from data import prescale
from model import Alma, config_from_dict


class AlmaForecaster:
    def __init__(self, model: Alma, device: str, context_len: int = 2048, symmetric: bool = True,
                 batch_size: int = 256) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.P = model.c.patch
        self.context_len = context_len // self.P * self.P
        self.symmetric = symmetric
        self.batch_size = batch_size

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kw) -> "AlmaForecaster":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["config"].get("arch") == "v3":
            from model_v3 import load_v3

            return cls(load_v3(path, "cpu"), device, **kw)
        model = Alma(config_from_dict(ck["config"]))
        model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        return cls(model, device, **kw)

    @torch.inference_mode()
    def _run(self, x: np.ndarray, m: np.ndarray, H: int) -> np.ndarray:
        """x, m: (B, T) prescaled, T % P == 0. Returns (B, H, Q) in prescaled units; block-AR past `horizon`."""
        hz = self.model.c.horizon
        xt = torch.from_numpy(x.astype(np.float32)).to(self.device)
        mt = torch.from_numpy(m).to(self.device)
        out = []
        done = 0
        while done < H:
            q = self.model.last_token_quantiles(xt, mt)  # (B, hz, Q)
            out.append(q)
            done += hz
            if done >= H:
                break
            med = q[..., q.shape[-1] // 2]
            xt = torch.cat([xt, med], 1)[:, -self.context_len :]
            mt = torch.cat([mt, torch.ones_like(med, dtype=torch.bool)], 1)[:, -self.context_len :]
        return torch.cat(out, 1)[:, :H].float().cpu().numpy()

    def predict(self, contexts: list, H: int, quantiles: list) -> np.ndarray:
        qs = list(self.model.c.quantiles)
        N = len(contexts)
        res = np.zeros((N, H, len(qs)))
        lens = [min(len(c), self.context_len) for c in contexts]
        order = np.argsort(lens)
        for s in range(0, N, self.batch_size):
            idx = order[s : s + self.batch_size]
            T = max(self.P, math.ceil(max(lens[i] for i in idx) / self.P) * self.P)
            xb = np.zeros((len(idx), T))
            mb = np.zeros((len(idx), T), dtype=bool)
            locs, scs, nonneg = [], [], []
            for j, i in enumerate(idx):
                c = np.asarray(contexts[i], dtype=np.float64)[-self.context_len :]
                m = np.isfinite(c)
                # reference stats over the most recent observed points (all of it is past)
                obs = c[m]
                z, loc, sc = prescale(np.where(m, c, 0.0), m, n_ref=10**9) if len(obs) else (np.zeros_like(c), 0.0, 1.0)
                xb[j, T - len(c) :] = z
                mb[j, T - len(c) :] = m
                locs.append(loc)
                scs.append(sc)
                nonneg.append(len(obs) > 0 and obs.min() >= 0)
            q = self._run(xb, mb, H)
            if self.symmetric:
                qm = self._run(-xb, mb, H)
                q = 0.5 * (q + (-qm)[..., ::-1])
            q = q * np.array(scs)[:, None, None] + np.array(locs)[:, None, None]
            nn_ = np.array(nonneg)
            q[nn_] = np.clip(q[nn_], 0, None)
            res[idx] = q
        if list(quantiles) == qs:
            return res
        out = np.zeros((N, H, len(quantiles)))
        for k, lv in enumerate(quantiles):
            out[..., k] = np.apply_along_axis(lambda r: np.interp(lv, qs, r), -1, res)
        return out
